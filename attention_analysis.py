"""
Attention-based attribution for GRM (Qwen3VL).

思路来源：
  - Kang et al., "Your Large Vision-Language Model Only Needs A Few Attention
    Heads For Visual Grounding" (arXiv:2503.06287)
  - Gandikota & Bau, "Gaze Heads: How VLMs Look at What They Describe"
    (arXiv:2606.14703)

核心做法：
  对每次 score 评估，提取 LLM 内部"最后一个 prompt token"对所有 image
  token 的 post-softmax attention 权重。这些权重本身就是一张空间热力图，
  反映模型在准备生成 score 时关注图像的哪些区域。

  与 perturbation / gradient 方法相比，attention 方法：
    - 不需要反向传播，单次前向即可
    - 没有 OOD 问题（输入始终是真实图像）
    - 没有梯度饱和问题
  注意：attention 权重是"模型在看哪里"的相关性信号，不等于"如果遮住这块
  score 会变多少"的因果重要性。

输出：
  - 每个 sample 的热力图 PNG（原图 / attention 热力图 / 叠加）
  - 全 head × 全层的 attention 统计（attention_sum / 空间熵）的 npz
  - 汇总视频，追踪 attention 随任务推进的变化
"""

import os
import sys
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# moviepy 可选（缺失时退回 cv2.VideoWriter）
_MOVIEPY_AVAILABLE = False
try:
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    _MOVIEPY_AVAILABLE = True
except ImportError:
    pass

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 复用 gradient_attention.py 的公共逻辑（帧提取 / 消息构建 / score 解析 / 可视化）
from gradient_attention import (
    SYSTEM_PROMPT,
    get_frame_count,
    make_sample_indices,
    save_frames,
    build_samples,
    parse_score,
    build_messages,
    patches_to_2d,
)


# ============================================================
# 配置区
# ============================================================

MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'

DATA_DIR = "/home/dais/workspace/Robo-Dopamine/aligned_data/pick3suc_1_carrot"

TASK_INSTRUCTION = "pick the white cube and put it on the plate"
GOAL_IMAGE = "./examples/blank_goal.png"

FRAME_INTERVAL = 20
EVAL_MODE = "incremental"

# Attention 分析参数
# 归因哪些图（支持多图，会同时对所有目标图算 attention 并联合排序/平均）
# 8 张图顺序：[0]REF_START [1]REF_END [2]BEFORE_high [3]BEFORE_left [4]BEFORE_right
#             [5]AFTER_high  [6]AFTER_left  [7]AFTER_right
TARGET_VIEW_INDICES = [5, 6, 7]   # AFTER 的三个视角

# 输出
OUTPUT_ROOT = "./results/attention_analysis_cube_score_sum_new"
VIDEO_FPS = 2.0
OVERLAY_ALPHA = 0.45   # 热力图叠加透明度（0=只原图，1=只热力图；0.45 接近 scan 脚本）

# 前向模式：
#   'generate'        : 正常自回归生成 score（逐 token，attention 取自生成过程）
#   'teacher_forcing' : 先 generate 拿 score 文本，再 teacher-force 一次前向
FORWARD_MODE = "generate"

# Query 选择：
#   'last_prompt' : 最后一个 prompt token（对齐 2503.06287 的"最后一个输入文本 token"）
#   'score_digit' : <score> 标签里的数字 token（生成时正在看的）
#   注意：FORWARD_MODE='generate' 时，query 自动取生成过程中最后一个数字 token，
#         此参数只影响 teacher_forcing 模式
QUERY_MODE = "score_digit"

# Head 聚合策略：
#   'mean' : 所有 head 平均（sanity check）
#   'topk' : 按 HEAD_CRITERION 选 top-k head 再平均
HEAD_AGG = "topk"
HEAD_TOPK = 5

# Head 排序公式（HEAD_AGG='topk' 时生效，多图时先对所有 target 的指标取平均再排序）：
#   'attn_sum'     : 仅按 attention sum 排序（论文标准 1）
#   'low_entropy'  : 仅按空间熵升序排序（论文标准 2）
#   'localization' : s_norm * (1 - e_norm)，两个标准联合（论文完整标准，推荐）
#                    先用 ATTN_SUM_FLOOR 淘汰不看图的 head，再在剩余 head 里按联合分排序
HEAD_CRITERION = "localization"

# 论文细节：排除前 N 层（论文排除前 2 层，因为早期层行为不同）
EXCLUDE_FIRST_LAYERS = 2

# 论文标准 1：attention sum 阈值 τ 的选取方式
#   'floor'        : 用固定值 ATTN_SUM_FLOOR（推荐，单样本稳定）
#   'max_curvature': 最大曲率法（论文默认，但需多样本才稳定）
ATTN_SUM_THRESHOLD = "floor"
ATTN_SUM_FLOOR = 0.01    # s_img 低于此值的 head 直接淘汰（对齐 scan_localization_heads_best.py）

# 论文标准 2：spatial entropy 的计算方式
#   'connected' : 论文原文——二值化 + 连通分量 entropy（推荐）
#   'shannon'   : 简单 Shannon entropy
SPATIAL_ENTROPY_MODE = "connected"

# 是否为固定 top-K head 输出单独视频
# True  : 跑完所有 sample 后，在聚合（多 sample 平均）的指标上选固定 HEAD_TOPK 个 head，
#         每个 head 对每个 target 视图各输出一个完整视频
# False : 不输出 per-head 视频
OUTPUT_PER_HEAD_VIDEOS = True

# Head 筛选时"熵低"的分位数（仅 ATTN_SUM_THRESHOLD='percentile' 旧逻辑兼容用，默认不用）
LOW_ENTROPY_PERCENTILE = 30

# 视频输出内容（主视频）：
#   'selected' : 用 HEAD_AGG/HEAD_CRITERION 筛出的 head 平均
#   'mean'     : 强制用全 head 平均（独立于 HEAD_AGG）
VIDEO_CONTENT = "selected"

# 是否保存全 head 的 attention 统计（npz，用于后续离线 head 分析）
SAVE_ALL_HEAD_STATS = False

# head_stats_overview / head_scatter 的统计方式：
#   'mean'  : 所有 sample 的 attn_sum / 熵 取平均（推荐，更稳定）
#   'first' : 只用第一个 sample（快但不稳定）
HEAD_STATS_AGG = "mean"


MIN_PIXELS = 12544
MAX_PIXELS = 76800


# ============================================================
# 模型封装：Qwen3VL + attention 抽取
# ============================================================

class AttentionAttributor:
    """
    封装 Qwen3VL，支持：
    1. 前向推理拿 score（teacher forcing，输入 prompt + score 文本）
    2. 抽取 LLM 内部指定 query token → 所有 image token 的 attention 权重
    """

    def __init__(self, model_path: str, min_pixels: int = 12544, max_pixels: int = 76800):
        print(f"Loading model from {model_path} ...")
        # 关键：eager 才能拿到 attention 权重（sdpa / flash 不返回）
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            min_pixels=min_pixels, max_pixels=max_pixels,
        )
        self.tok = self.processor.tokenizer

        # Qwen3-VL 的 image placeholder token id（<|image_pad|>）
        # 用 convert_tokens_to_ids 拿，避免硬编码
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        self.vision_start_id = self.tok.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.tok.convert_tokens_to_ids("<|vision_end|>")
        print(f"  <|image_pad|> id = {self.image_pad_id}")
        print(f"  <|vision_start|> id = {self.vision_start_id}")
        print(f"  <|vision_end|> id = {self.vision_end_id}")
        print(f"  Model loaded.\n")

    @torch.no_grad()
    def generate_score(self, image_paths: List[str], task: str,
                       max_new_tokens: int = 64) -> Tuple[str, float]:
        """普通推理拿 score 文本。"""
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        ).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
        gen = out[0, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        return text, parse_score(text)

    def _find_image_token_spans(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        在 input_ids 里找出每张图对应的 image token 范围。

        Qwen3-VL 的格式：<|vision_start|> <|image_pad|>×N <|vision_end|>
        每张图是一段连续的 <|image_pad|>，返回 [(start, end_exclusive), ...]。

        end - start = (t * h * w) / merge^2，即 LLM 实际看到的 token 数。
        """
        ids = input_ids[0].tolist() if input_ids.dim() == 2 else input_ids.tolist()
        spans = []
        i = 0
        n = len(ids)
        while i < n:
            if ids[i] == self.image_pad_id:
                j = i
                while j < n and ids[j] == self.image_pad_id:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        return spans

    def _find_query_positions(self, input_ids: torch.Tensor, response_ids: List[int],
                              query_mode: str) -> List[int]:
        """
        找 query token 在完整序列里的绝对位置（用于从 attention 矩阵取行）。
        """
        n = input_ids.shape[1]
        if query_mode == "last_prompt":
            # 最后一个 prompt token（即第一个 response token 之前那个）
            return [n - 1]
        elif query_mode == "score_digit":
            # 找 response 里 <score>...</score> 之间的数字 token
            open_start, open_end = _find_token_span_by_decode(
                self.tok, response_ids, "<score>"
            )
            close_start, close_end = _find_token_span_by_decode(
                self.tok, response_ids, "</score>"
            )
            if open_start < 0 or close_start < 0:
                return [n - 1]
            prompt_len = n - len(response_ids)
            inner = list(range(open_end, close_start))
            # 只保留数字和小数点
            digit_pos = [
                prompt_len + p for p in inner
                if all(c.isdigit() or c == '.' for c in self.tok.decode([response_ids[p]]).strip())
            ]
            return digit_pos if digit_pos else [n - 1]
        else:
            return [n - 1]

    def build_forward_inputs(
        self,
        image_paths: List[str],
        task: str,
        score_text: str,
    ) -> Tuple[Dict, List[Tuple[int, int]], int]:
        """
        构造 teacher-forcing 前向输入（prompt + score 文本）。
        返回:
            inputs: model() 输入
            image_spans: 每张图在 LLM 序列里的 token 范围
            prompt_len: prompt 长度（不含 response）
        """
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        )

        response_ids = self.tok.encode(score_text, add_special_tokens=False)
        response_tensor = torch.tensor([response_ids], dtype=torch.long)

        full_input_ids = torch.cat(
            [inputs["input_ids"], response_tensor], dim=1
        ).to(self.model.device)
        full_attention_mask = torch.ones_like(full_input_ids)

        prompt_len = inputs["input_ids"].shape[1]

        inputs_ready = {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": inputs["pixel_values"].to(self.model.device, dtype=self.model.dtype),
            "image_grid_thw": inputs["image_grid_thw"].to(self.model.device),
        }
        if "position_ids" in inputs:
            inputs_ready["position_ids"] = inputs["position_ids"].to(self.model.device)

        image_spans = self._find_image_token_spans(full_input_ids)
        return inputs_ready, image_spans, prompt_len

    @torch.no_grad()
    def compute_attention(
        self,
        image_paths: List[str],
        task: str,
        target_view_indices: List[int],
        query_mode: str = "last_prompt",
        forward_mode: str = "generate",
    ) -> Tuple[Dict[int, np.ndarray], float, Dict[int, Tuple[int, int]], Dict]:
        """
        抽取 query token → 多张目标图 image token 的 attention 权重。

        forward_mode:
            'generate'        : 正常自回归生成，attention 来自生成过程
            'teacher_forcing' : teacher-force 一次前向，attention 来自重放

        返回:
            agg_maps:   {target_view_idx: [num_target_tokens]} 每个 target 的聚合 attention
            score:      原始 score
            target_ranges: {target_view_idx: (start, end)}
            stats:      全 head 统计（含多 target）
        """
        if forward_mode == "generate":
            return self._compute_attention_generate(
                image_paths, task, target_view_indices
            )
        else:
            return self._compute_attention_teacher_forcing(
                image_paths, task, target_view_indices, query_mode
            )

    @torch.no_grad()
    def _compute_attention_generate(
        self,
        image_paths: List[str],
        task: str,
        target_view_indices: List[int],
    ) -> Tuple[Dict[int, np.ndarray], float, Dict[int, Tuple[int, int]], Dict]:
        """
        正常自回归生成模式。

        用 model.generate(output_attentions=True, return_dict_in_generate=True)。
        attention 来自模型真实的逐步生成过程。

        取 query = 最后一个生成的数字 token（score 最后一位数字），
        它的 attention 反映"模型在确定 score 数值时看了哪里"。
        同时支持多个 target 视图，对所有 target 联合排序选出统一的 selected_heads。
        """
        messages = build_messages(task)
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[prompt_text], images=images,
            return_tensors="pt", padding=True,
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[1]

        # 定位 image spans（基于 prompt 的 input_ids，生成后位置不变）
        image_spans = self._find_image_token_spans(inputs["input_ids"])
        for tvi in target_view_indices:
            if tvi >= len(image_spans):
                raise ValueError(
                    f"target_view_idx={tvi} 超出图像数 {len(image_spans)}"
                )

        # 生成 + 收集 attention
        print(f"    [generate mode] generating with output_attentions ...")
        out = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False, temperature=None, top_p=None,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        # out.attentions: tuple[len = num_generated_steps]
        #   第 0 步（prefill）: 每层 [1, H, prompt_len, prompt_len]
        #   第 k 步（k>=1, decode）: 每层 [1, H, 1, prompt_len+k]
        gen_ids = out.sequences[0, prompt_len:]
        score_text = self.tok.decode(gen_ids, skip_special_tokens=True)
        score = parse_score(score_text)
        print(f"    Score text: {score_text!r}  score={score}")

        # 找最后一个数字 token 的生成步
        digit_step = None
        for k in range(len(gen_ids) - 1, -1, -1):
            tok_str = self.tok.decode([int(gen_ids[k])]).strip()
            if tok_str and all(c.isdigit() or c == '.' for c in tok_str):
                digit_step = k
                break
        if digit_step is None:
            digit_step = len(gen_ids) - 1
        # out.attentions[0] 是 prefill，out.attentions[1] 是第 1 个生成 token
        attn_idx = digit_step + 1
        print(f"    Last digit token at gen step {digit_step}, attn idx {attn_idx}")
        print(f"    Last digit token: {self.tok.decode([int(gen_ids[digit_step])])!r}")

        attentions = out.attentions[attn_idx]   # tuple[num_layers], 每层 [1, H, 1, T]
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        print(f"    Got attentions: {num_layers} layers × {num_heads} heads")

        # query 行：每个 (layer, head) 在 query token（最后一个生成 token）位置上的 attention 分布
        # generate 模式下 query 就是 attention 矩阵的最后一行（1, H, 1, T -> 取 [0])
        query_rows = np.zeros((num_layers, num_heads, attentions[0].shape[-1]), dtype=np.float32)
        for l in range(num_layers):
            query_rows[l] = attentions[l][0, :, 0, :].float().cpu().numpy()

        del out, attentions
        torch.cuda.empty_cache()

        image_grid_thw = inputs["image_grid_thw"]
        agg_maps, selected_heads, agg_info, per_target = _aggregate_heads_multi(
            query_rows, image_spans, target_view_indices, image_grid_thw, self.processor,
            query_positions=[prompt_len + digit_step],
        )

        target_ranges = {tvi: (int(per_target[tvi]["start"]), int(per_target[tvi]["end"]))
                         for tvi in target_view_indices}
        stats = {
            "per_target": per_target,
            "selected_heads": selected_heads,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "query_positions": [prompt_len + digit_step],
            "forward_mode": "generate",
            "agg_info": agg_info,
            "target_view_indices": list(target_view_indices),
        }
        return agg_maps, score, target_ranges, stats

    @torch.no_grad()
    def _compute_attention_teacher_forcing(
        self,
        image_paths: List[str],
        task: str,
        target_view_indices: List[int],
        query_mode: str = "last_prompt",
    ) -> Tuple[Dict[int, np.ndarray], float, Dict[int, Tuple[int, int]], Dict]:
        """
        Teacher-forcing 模式：先 generate 拿 score 文本，再 teacher-force 一次前向。
        支持多个 target 视图，联合排序选 head。
        """
        # Step 1: 拿 score 文本
        score_text, score = self.generate_score(image_paths, task)
        response_ids = self.tok.encode(score_text, add_special_tokens=False)
        print(f"    Score text: {score_text!r}  score={score}")

        # Step 2: teacher forcing 前向
        inputs, image_spans, prompt_len = self.build_forward_inputs(
            image_paths, task, score_text
        )
        for tvi in target_view_indices:
            if tvi >= len(image_spans):
                raise ValueError(
                    f"target_view_idx={tvi} 超出图像数 {len(image_spans)}"
                )
        print(f"    Image spans: {len(image_spans)} images")
        for tvi in target_view_indices:
            s, e = image_spans[tvi]
            print(f"    Target image[{tvi}] token range: [{s}, {e})  ({e - s} tokens)")

        # 找 query token 位置
        query_positions = self._find_query_positions(
            inputs["input_ids"], response_ids, query_mode
        )
        print(f"    Query mode: {query_mode}, positions: {query_positions}")

        # 前向
        outputs = self.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
            use_cache=False,
        )
        attentions = outputs.attentions
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        print(f"    Got attentions: {num_layers} layers × {num_heads} heads")

        # query 行：每个 (layer, head) 在 query_positions 行上的平均
        T_full = attentions[0].shape[-1]
        query_rows = np.zeros((num_layers, num_heads, T_full), dtype=np.float32)
        for l in range(num_layers):
            attn_lh = attentions[l][0].float().cpu().numpy()    # [H, T_full, T_full]
            query_rows[l] = attn_lh[:, query_positions, :].mean(axis=1)

        del outputs, attentions
        torch.cuda.empty_cache()

        image_grid_thw = inputs["image_grid_thw"]
        agg_maps, selected_heads, agg_info, per_target = _aggregate_heads_multi(
            query_rows, image_spans, target_view_indices, image_grid_thw, self.processor,
            query_positions=query_positions,
        )

        target_ranges = {tvi: (int(per_target[tvi]["start"]), int(per_target[tvi]["end"]))
                         for tvi in target_view_indices}
        stats = {
            "per_target": per_target,
            "selected_heads": selected_heads,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "query_positions": query_positions,
            "forward_mode": "teacher_forcing",
            "agg_info": agg_info,
            "target_view_indices": list(target_view_indices),
        }
        return agg_maps, score, target_ranges, stats


# ============================================================
# 辅助：decode-based 子串定位（复用 gradient_attention 的逻辑）
# ============================================================

def _find_token_span_by_decode(tok, token_ids_list: List[int], sub_str: str) -> Tuple[int, int]:
    """
    在 token_ids_list 里找 sub_str 第一次出现的 token 起止位置。
    用累积 decode 逐 token 比对。返回 (start, end_exclusive)，找不到返回 (-1, -1)。
    """
    char_offset_start = -1
    for i in range(len(token_ids_list) + 1):
        if char_offset_start < 0:
            if i == len(token_ids_list):
                break
            new_text = tok.decode(token_ids_list[:i + 1])
            if sub_str in new_text:
                char_offset_start = new_text.find(sub_str)
                start_token = i
                char_offset_end = char_offset_start + len(sub_str)
                for j in range(i + 1, len(token_ids_list) + 1):
                    full = tok.decode(token_ids_list[:j])
                    if len(full) >= char_offset_end:
                        return start_token, j
                return start_token, len(token_ids_list)
    return -1, -1


def _grid_hw_from_thw(grid_thw: List[int], processor) -> Tuple[int, int]:
    """
    从 image_grid_thw (t, h, w) 计算 LLM 看到的 2D 网格 (hm, wm)。
    hm = h // merge_size, wm = w // merge_size
    """
    merge = getattr(processor.image_processor, "merge_size", 2) if hasattr(processor, "image_processor") else 2
    t, h, w = grid_thw
    return h // merge, w // merge


def _safe_reshape_2d(seg_1d: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 1D attention 安全 reshape 成 2D 网格（处理长度不匹配）。"""
    expected = h * w
    if seg_1d.shape[0] == expected:
        return seg_1d.reshape(h, w)
    if seg_1d.shape[0] > expected:
        return seg_1d[:expected].reshape(h, w)
    # 不足则填充
    padded = np.zeros(expected, dtype=seg_1d.dtype)
    padded[:seg_1d.shape[0]] = seg_1d
    return padded.reshape(h, w)


def _spatial_entropy_connected(seg_2d: np.ndarray) -> float:
    """
    论文 2503.06287 的 spatial entropy 实现。

    步骤：
    1. 把 attention map 二值化（>均值=1，否则=0）
    2. 找 8-连通分量
    3. 对连通分量大小分布算 Shannon entropy

    低 entropy = attention 聚集在少数大块（聚焦）
    高 entropy = attention 碎成很多小块（分散）
    """
    seg = seg_2d.astype(np.float32)
    mean_val = seg.mean()
    binary = (seg > mean_val).astype(np.uint8)
    if binary.sum() == 0:
        return float(seg.size)  # 全 0，最大熵

    # 找连通分量（8-连通）
    num_labels, labels, sizes, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # sizes[0] 是背景，忽略
    component_sizes = sizes[1:]
    if len(component_sizes) == 0:
        return float(seg.size)

    total = component_sizes.sum()
    probs = component_sizes / total
    entropy = -float((probs * np.log(probs + 1e-12)).sum())
    return entropy


def _max_curvature_threshold(sorted_values: np.ndarray) -> float:
    """
    论文 2503.06287 的阈值 τ 选取：最大曲率法（knee detection）。

    输入：升序排列的 attention sum 值
    输出：阈值 τ，曲率最大的点

    原理：把排序后的值画成曲线，找"拐点"——
    曲线从平缓变陡峭（或从陡峭变平缓）的位置。
    """
    n = len(sorted_values)
    if n < 3:
        return float(sorted_values[-1])

    x = np.arange(n, dtype=np.float64)
    y = sorted_values.astype(np.float64)

    # 归一化到 [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 对每个内部点，算它到首尾连线的距离（knee detection 的标准做法）
    # 首点 (x_norm[0], y_norm[0])，尾点 (x_norm[-1], y_norm[-1])
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    distances = np.zeros(n)
    for i in range(n):
        p = np.array([x_norm[i], y_norm[i]])
        # 点到直线的距离
        line_vec = p2 - p1
        point_vec = p - p1
        line_len = np.linalg.norm(line_vec) + 1e-12
        distance = np.abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0]) / line_len
        distances[i] = distance

    # 最大距离的点就是 knee
    knee_idx = int(np.argmax(distances))
    # 阈值取 knee 点附近的值（论文用 knee 点的 sum 值作为 τ）
    return float(sorted_values[knee_idx])


def _extract_per_target(
    query_rows: np.ndarray,
    target_start: int,
    target_end: int,
    grid_thw: List[int],
    processor,
) -> Dict:
    """
    从 query_rows [L, H, T_full] 切出某张 target 图的片段，计算 q_to_target / attn_sum / spatial_entropy。

    返回 dict:
        q_to_target: [L, H, T_target]
        attn_sum:    [L, H]
        spatial_entropy: [L, H]
        start, end, grid_thw, grid_h, grid_w
    """
    L, H, _ = query_rows.shape
    seg = query_rows[:, :, target_start:target_end]          # [L, H, T_target]
    attn_sum = seg.sum(axis=2)                                # [L, H]
    grid_h, grid_w = _grid_hw_from_thw(grid_thw, processor)

    spatial_entropy = np.zeros((L, H), dtype=np.float32)
    if SPATIAL_ENTROPY_MODE == "connected" and grid_h > 0 and grid_w > 0:
        for l in range(L):
            for h in range(H):
                total = float(attn_sum[l, h])
                if total > 1e-12:
                    seg_2d = _safe_reshape_2d(seg[l, h], grid_h, grid_w)
                    spatial_entropy[l, h] = _spatial_entropy_connected(seg_2d)
                else:
                    spatial_entropy[l, h] = float(seg.shape[-1])
    else:  # shannon
        for l in range(L):
            for h in range(H):
                total = float(attn_sum[l, h])
                if total > 1e-12:
                    p = seg[l, h] / total
                    p_nz = p[p > 1e-12]
                    spatial_entropy[l, h] = -float((p_nz * np.log(p_nz)).sum())
                else:
                    spatial_entropy[l, h] = float(seg.shape[-1]) * np.log(max(seg.shape[-1], 1))

    return {
        "q_to_target": seg,
        "attn_sum": attn_sum,
        "spatial_entropy": spatial_entropy,
        "start": int(target_start),
        "end": int(target_end),
        "grid_thw": list(grid_thw),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
    }


def _aggregate_heads_multi(
    query_rows: np.ndarray,
    image_spans: List[Tuple[int, int]],
    target_view_indices: List[int],
    image_grid_thw: torch.Tensor,
    processor,
    query_positions: Optional[List[int]] = None,
) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int]], Dict, Dict[int, Dict]]:
    """
    多 target 联合 head 选择 + 每 target 独立聚合。

    流程：
      1. 对每个 target 切出 q_to_target / attn_sum / spatial_entropy
      2. 对所有 target 的 attn_sum / spatial_entropy 取平均，得到 [L, H] 的联合指标
      3. 按 HEAD_CRITERION 在联合指标上排序选出统一 selected_heads（排除前 EXCLUDE_FIRST_LAYERS 层）
      4. 用同一份 selected_heads 对每个 target 的 q_to_target 做平均，得到 per-target agg_map

    返回:
        agg_maps:      {target_view_idx: [T_target]}
        selected_heads:[(layer, head), ...]
        info:          诊断信息
        per_target:    {target_view_idx: {q_to_target, attn_sum, spatial_entropy, start, end, grid_thw, ...}}
    """
    num_layers, num_heads = query_rows.shape[:2]
    info = {
        "excluded_layers": list(range(EXCLUDE_FIRST_LAYERS)),
        "criterion": HEAD_CRITERION,
        "num_targets": len(target_view_indices),
    }

    per_target: Dict[int, Dict] = {}
    for tvi in target_view_indices:
        s, e = image_spans[tvi]
        grid_thw = image_grid_thw[tvi].tolist()
        per_target[tvi] = _extract_per_target(query_rows, s, e, grid_thw, processor)

    # 联合指标：所有 target 的 sum / entropy 取平均
    sum_stack = np.stack([per_target[tvi]["attn_sum"] for tvi in target_view_indices], axis=0)
    ent_stack = np.stack([per_target[tvi]["spatial_entropy"] for tvi in target_view_indices], axis=0)
    sum_mean = sum_stack.mean(axis=0)     # [L, H]
    ent_mean = ent_stack.mean(axis=0)     # [L, H]
    info["sum_mean"] = sum_mean
    info["entropy_mean"] = ent_mean

    # mean 模式：直接用全 head 平均
    if HEAD_AGG == "mean":
        agg_maps = {}
        for tvi in target_view_indices:
            agg_maps[tvi] = per_target[tvi]["q_to_target"].mean(axis=(0, 1))
        print(f"    Aggregation: mean over all {num_layers * num_heads} heads "
              f"({len(target_view_indices)} targets)")
        return agg_maps, [], info, per_target

    # topk 模式：构造候选池（排除前 N 层）
    layer_mask = np.array([l >= EXCLUDE_FIRST_LAYERS for l in range(num_layers)])
    valid_indices = []   # (flat_idx, layer, head)
    for l in range(num_layers):
        if not layer_mask[l]:
            continue
        for h in range(num_heads):
            valid_indices.append((l * num_heads + h, l, h))
    print(f"    Candidate pool: {len(valid_indices)} heads "
          f"(excluded first {EXCLUDE_FIRST_LAYERS} layers)")

    flat_sum = np.array([sum_mean[l, h] for (_, l, h) in valid_indices])
    flat_ent = np.array([ent_mean[l, h] for (_, l, h) in valid_indices])

    # 决定 sum 阈值 τ（floor / max_curvature / percentile）
    if ATTN_SUM_THRESHOLD == "floor":
        tau = float(ATTN_SUM_FLOOR)
    elif ATTN_SUM_THRESHOLD == "max_curvature":
        tau = _max_curvature_threshold(np.sort(flat_sum))
    else:  # percentile
        tau = float(np.percentile(flat_sum, 100 - LOW_ENTROPY_PERCENTILE))
    info["tau"] = tau

    # 按 criterion 排序
    if HEAD_CRITERION == "attn_sum":
        score = flat_sum.copy()
        print(f"    Criterion: attn_sum (τ={tau:.4f} ignored for pure-sum ranking)")
    elif HEAD_CRITERION == "low_entropy":
        score = -flat_ent.copy()   # 熵越小分越高
        print(f"    Criterion: low spatial entropy ({SPATIAL_ENTROPY_MODE})")
    elif HEAD_CRITERION == "localization":
        # 标准 1+2 联合：s_norm * (1 - e_norm)，先用 floor 淘汰不看图的 head
        mask_pass = flat_sum >= tau
        if not mask_pass.any():
            print(f"    [WARN] no head passes s_img_floor={tau:.4f}; disabling floor")
            mask_pass = np.ones_like(flat_sum, dtype=bool)
        s_eff = flat_sum.copy()
        e_eff = flat_ent.copy()
        s_eff[~mask_pass] = float(flat_sum.min())
        finite_e = np.where(np.isfinite(e_eff), e_eff, -1.0)
        e_max = float(finite_e.max()) if finite_e.size else 0.0
        e_for_norm = np.where(np.isfinite(e_eff), e_eff, e_max)
        s_norm = (s_eff - s_eff.min()) / max(float(s_eff.max() - s_eff.min()), 1e-12)
        e_norm = (e_for_norm - e_for_norm.min()) / max(float(e_for_norm.max() - e_for_norm.min()), 1e-12)
        score = s_norm * (1.0 - e_norm)
        score[~mask_pass] = -np.inf
        info["pass_mask_count"] = int(mask_pass.sum())
        print(f"    Criterion: localization = s_norm*(1-e_norm), "
              f"τ={tau:.4f}, {int(mask_pass.sum())} heads pass floor")
    else:
        print(f"    [WARN] unknown HEAD_CRITERION={HEAD_CRITERION}, fallback to attn_sum")
        score = flat_sum.copy()

    topk_local = np.argsort(score)[::-1][:HEAD_TOPK]
    selected_heads = [valid_indices[i][1:] for i in topk_local]
    print(f"    Aggregation: top-{HEAD_TOPK} by {HEAD_CRITERION}")
    print(f"    Selected heads (layer, head): {selected_heads}")

    # 用统一 selected_heads 对每个 target 聚合
    agg_maps = {}
    for tvi in target_view_indices:
        sel_maps = np.stack([per_target[tvi]["q_to_target"][l, h] for (l, h) in selected_heads])
        agg_maps[tvi] = sel_maps.mean(axis=0)

    return agg_maps, selected_heads, info, per_target


def _rank_aggregated(
    sum_mat: np.ndarray,
    ent_mat: np.ndarray,
) -> Dict:
    """
    在聚合后的 [L, H] sum/entropy 矩阵上按 HEAD_CRITERION 选 top-K head。

    与 _aggregate_heads_multi 内的排序逻辑保持一致：
      - 排除前 EXCLUDE_FIRST_LAYERS 层
      - ATTN_SUM_THRESHOLD 决定 τ（floor / max_curvature / percentile）
      - HEAD_CRITERION: attn_sum / low_entropy / localization

    返回: {"selected_heads": [(l,h),...], "tau": float|None}
    用于 head_stats_overview / head_scatter 高亮，以及 per-head 固定视频的 head 选择。
    """
    num_layers, num_heads = sum_mat.shape
    layer_mask = np.array([l >= EXCLUDE_FIRST_LAYERS for l in range(num_layers)])
    valid_indices = [(l * num_heads + h, l, h)
                     for l in range(num_layers) if layer_mask[l]
                     for h in range(num_heads)]
    if not valid_indices:
        return {"selected_heads": [], "tau": None}

    flat_sum = np.array([sum_mat[l, h] for (_, l, h) in valid_indices])
    flat_ent = np.array([ent_mat[l, h] for (_, l, h) in valid_indices])

    if ATTN_SUM_THRESHOLD == "floor":
        tau = float(ATTN_SUM_FLOOR)
    elif ATTN_SUM_THRESHOLD == "max_curvature":
        tau = _max_curvature_threshold(np.sort(flat_sum))
    else:
        tau = float(np.percentile(flat_sum, 100 - LOW_ENTROPY_PERCENTILE))

    if HEAD_CRITERION == "attn_sum":
        score = flat_sum.copy()
    elif HEAD_CRITERION == "low_entropy":
        score = -flat_ent.copy()
    elif HEAD_CRITERION == "localization":
        mask_pass = flat_sum >= tau
        if not mask_pass.any():
            mask_pass = np.ones_like(flat_sum, dtype=bool)
        s_eff = flat_sum.copy()
        e_eff = flat_ent.copy()
        s_eff[~mask_pass] = float(flat_sum.min())
        finite_e = np.where(np.isfinite(e_eff), e_eff, -1.0)
        e_max = float(finite_e.max()) if finite_e.size else 0.0
        e_for_norm = np.where(np.isfinite(e_eff), e_eff, e_max)
        s_norm = (s_eff - s_eff.min()) / max(float(s_eff.max() - s_eff.min()), 1e-12)
        e_norm = (e_for_norm - e_for_norm.min()) / max(float(e_for_norm.max() - e_for_norm.min()), 1e-12)
        score = s_norm * (1.0 - e_norm)
        score[~mask_pass] = -np.inf
    else:
        score = flat_sum.copy()

    topk_local = np.argsort(score)[::-1][:HEAD_TOPK]
    selected_heads = [valid_indices[i][1:] for i in topk_local]
    return {"selected_heads": selected_heads, "tau": tau}


# ============================================================
# 可视化
# ============================================================

def _normalize_heatmap(grid: np.ndarray) -> np.ndarray:
    """减去最小值后归一化到 [0,1]，让对比度更强（背景归零）。"""
    arr = np.asarray(grid, dtype=np.float64)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr)
    if denom <= 0:
        return np.zeros_like(arr)
    return arr / denom


def render_attention_frame(
    image_path: str,
    attention_map: np.ndarray,
    grid_thw_target: List[int],
    orig_score: float,
    step_idx: int,
    frame_idx: int,
    task: str,
    out_path: str,
    alpha: float = 0.45,
    head_label: str = "",
) -> np.ndarray:
    """单张原图分辨率 overlay（对齐 scan_localization_heads_best.py 的视觉风格）。"""
    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB")).copy()
        img_w, img_h = im.size

    # 热力图归一化（减最小值）+ PIL BICUBIC 上采样到原图尺寸
    heatmap_2d = patches_to_2d(attention_map, grid_thw_target)
    max_val = float(np.nanmax(heatmap_2d))
    heat_norm = _normalize_heatmap(heatmap_2d)
    heat_pil = Image.fromarray(np.uint8(np.clip(heat_norm, 0, 1) * 255))
    heat_pil = heat_pil.resize((img_w, img_h), resample=Image.Resampling.BICUBIC)
    heat_resized = np.asarray(heat_pil, dtype=np.float32) / 255.0

    heat_u8 = np.uint8(np.clip(heat_resized, 0, 1) * 255)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = np.clip(
        (1.0 - alpha) * image.astype(np.float32) + alpha * heat_color.astype(np.float32),
        0, 255,
    ).astype(np.uint8)

    # 标题栏叠在图顶部（半透明黑底白字），不另起画布行，保持原图分辨率
    title = (f"[{head_label}] " if head_label else "") + \
            f"Step {step_idx}  Frame {frame_idx}  |  Score: {orig_score:+.1f}%  |  Max attn: {max_val:.4f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, img_w / 1400.0)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), baseline = cv2.getTextSize(title, font, scale, thickness)
    bar_h = th + baseline + 8
    cv2.rectangle(overlay, (0, 0), (min(img_w, tw + 12), bar_h), (0, 0, 0), -1)
    cv2.putText(overlay, title, (6, th + 4), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # task 栏贴在底部
    task_txt = f"Task: {task}"
    max_chars = max(24, img_w // 8)
    if len(task_txt) > max_chars:
        task_txt = task_txt[: max_chars - 3] + "..."
    scale_t = max(0.4, img_w / 1600.0)
    thick_t = max(1, int(round(scale_t * 2)))
    (tw2, th2), base2 = cv2.getTextSize(task_txt, font, scale_t, thick_t)
    bar_h2 = th2 + base2 + 8
    cv2.rectangle(overlay, (0, img_h - bar_h2), (min(img_w, tw2 + 12), img_h), (0, 0, 0), -1)
    cv2.putText(overlay, task_txt, (6, img_h - base2 - 2), font, scale_t, (220, 220, 220), thick_t, cv2.LINE_AA)

    if out_path:
        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100)
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    return overlay


def plot_head_stats(stats: Dict, out_path: str):
    """画 attention_sum 和空间熵的 layer×head 热力图（用于 head 分析）。"""
    attn_sum = stats["attn_sum"]           # [L, H]
    entropy = stats["spatial_entropy"]     # [L, H]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(attn_sum, aspect="auto", cmap="viridis")
    axes[0].set_title("Attention Sum (query → target image)")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(entropy, aspect="auto", cmap="viridis_r")
    axes[1].set_title("Spatial Entropy (low = focused)")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    plt.colorbar(im1, ax=axes[1])

    # 标记选中的 head
    for (l, h) in stats.get("selected_heads", []):
        axes[0].plot(h, l, "rx", markersize=10, mew=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_head_scatter(
    attn_sum: np.ndarray,
    spatial_entropy: np.ndarray,
    selected_heads: List[Tuple[int, int]],
    tau: Optional[float],
    out_path: str,
):
    """
    论文 2503.06287 风格的 scatter plot：
    横坐标 = spatial entropy（低=聚焦）
    纵坐标 = attention sum（高=看图）

    理想 head 在右上角（论文的"高 sum + 低 entropy"），
    习惯上把 entropy 放横轴、sum 放纵轴，理想区在左上。

    selected_heads 用红色高亮，并标注 "L{layer}H{head}"。
    tau 用水平虚线标出（attention sum 阈值）。
    """
    L, H = attn_sum.shape
    # 展开成点列表
    sums = attn_sum.flatten()
    ents = spatial_entropy.flatten()
    labels = [(l, h) for l in range(L) for h in range(H)]
    selected_set = set(selected_heads)

    fig, ax = plt.subplots(figsize=(10, 7))

    # 按是否被选中分两组画
    is_sel = np.array([labels[i] in selected_set for i in range(len(labels))])

    # 排除前 EXCLUDE_FIRST_LAYERS 层的点用浅色
    layer_arr = np.array([l for (l, h) in labels])
    is_excluded = layer_arr < EXCLUDE_FIRST_LAYERS

    # 普通 head（未选中、未排除）
    mask_normal = (~is_sel) & (~is_excluded)
    ax.scatter(ents[mask_normal], sums[mask_normal],
               c="lightgray", s=20, alpha=0.6, label=f"others (excl. first {EXCLUDE_FIRST_LAYERS} layers)")

    # 被排除层的 head
    mask_excluded = (~is_sel) & is_excluded
    ax.scatter(ents[mask_excluded], sums[mask_excluded],
               c="#DDDDDD", s=15, alpha=0.4, marker="x", label=f"excluded first {EXCLUDE_FIRST_LAYERS} layers")

    # 选中的 head
    mask_sel = is_sel
    ax.scatter(ents[mask_sel], sums[mask_sel],
               c="red", s=120, alpha=0.9, edgecolors="darkred", linewidths=1.5,
               label=f"selected top-{len(selected_heads)} heads", zorder=5)

    # 给选中的 head 标注名字
    for i in range(len(labels)):
        if mask_sel[i]:
            l, h = labels[i]
            ax.annotate(f"L{l}H{h}", (ents[i], sums[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=9, color="darkred", fontweight="bold")

    # τ 阈值线
    if tau is not None:
        ax.axhline(y=tau, color="blue", linestyle="--", alpha=0.6, label=f"τ={tau:.3f} (max curvature)")

    ax.set_xlabel(f"Spatial Entropy ({SPATIAL_ENTROPY_MODE})  ←  lower = more focused")
    ax.set_ylabel("Attention Sum  ←  higher = attends to image")
    ax.set_title("Head Selection: Attention Sum vs Spatial Entropy\n"
                 "(ideal heads: top-left = high sum + low entropy)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"\n{'=' * 70}")
    print("Attention-based Attribution (Qwen3VL)")
    print(f"{'=' * 70}")
    print(f"Model:           {MODEL_PATH}")
    print(f"Data:            {DATA_DIR}")
    print(f"Task:            {TASK_INSTRUCTION}")
    print(f"Eval mode:       {EVAL_MODE}")
    print(f"Target views:    images{TARGET_VIEW_INDICES}")
    print(f"Forward mode:    {FORWARD_MODE}")
    print(f"Query mode:      {QUERY_MODE} (仅 teacher_forcing 模式生效)")
    print(f"Head aggregation:{HEAD_AGG} (topk={HEAD_TOPK}, criterion={HEAD_CRITERION})")
    print(f"Exclude layers:  first {EXCLUDE_FIRST_LAYERS} layers")
    print(f"Attn sum thresh: {ATTN_SUM_THRESHOLD}" + (f" (floor={ATTN_SUM_FLOOR})" if ATTN_SUM_THRESHOLD == "floor" else ""))
    print(f"Spatial entropy: {SPATIAL_ENTROPY_MODE}")
    print(f"Video content:   {VIDEO_CONTENT}")
    print(f"Per-head videos: {OUTPUT_PER_HEAD_VIDEOS}")
    print(f"Save npz:        {SAVE_ALL_HEAD_STATS}")
    print(f"Head stats agg:  {HEAD_STATS_AGG}")
    print(f"{'=' * 70}\n")

    ts = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    run_root = Path(OUTPUT_ROOT) / f"{ts}_{EVAL_MODE}_mode"
    cache_root = run_root / ".cache"
    heatmaps_dir = run_root / "heatmaps"
    run_root.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(exist_ok=True)

    # --- 1. 帧提取 ---
    print("[1/5] Extracting frames ...")
    cam_dirs = {
        "cam_high":        cache_root / "cam_high",
        "cam_left_wrist":  cache_root / "cam_left_wrist",
        "cam_right_wrist": cache_root / "cam_right_wrist",
    }
    for d in cam_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    paths = [
        Path(os.path.join(DATA_DIR, "cam_high.mp4")),
        Path(os.path.join(DATA_DIR, "cam_left_wrist.mp4")),
        Path(os.path.join(DATA_DIR, "cam_right_wrist.mp4")),
    ]
    types_counts = [get_frame_count(p) for p in paths]
    counts = [tc[1] for tc in types_counts]
    if len(set(counts)) != 1:
        raise ValueError(f"Frame count mismatch: {counts}")
    total_frames = counts[0]
    indices = make_sample_indices(total_frames, FRAME_INTERVAL)
    print(f"  Total frames: {total_frames}, Sampled: {len(indices)} (interval={FRAME_INTERVAL})")
    for p, key, (stype, _) in zip(paths, cam_dirs.keys(), types_counts):
        save_frames(p, cam_dirs[key], indices, stype)

    if GOAL_IMAGE and os.path.exists(GOAL_IMAGE):
        ref_end_path = str(cache_root / "ref_end.png")
        shutil.copy(GOAL_IMAGE, ref_end_path)
    else:
        ref_end_path = str(cam_dirs["cam_high"] / f"frame_{total_frames - 1:06d}.png")

    # --- 2. 构建 samples ---
    print("\n[2/5] Building samples ...")
    samples = build_samples(cache_root, TASK_INSTRUCTION, indices, ref_end_path, mode=EVAL_MODE)
    print(f"  Samples: {len(samples)}\n")

    # --- 3. 加载模型 ---
    print("[3/5] Loading model (eager attention) ...")
    attributor = AttentionAttributor(MODEL_PATH, MIN_PIXELS, MAX_PIXELS)

    # --- 4. 逐帧 attention 抽取 ---
    print("\n[4/5] Running attention extraction ...")
    # 主视频：每个 target view 一个视频，按 target 顺序拼接帧列表
    # video_frames_by_target[tvi] = [frame_rgb, ...]
    video_frames_by_target: Dict[int, List[np.ndarray]] = {tvi: [] for tvi in TARGET_VIEW_INDICES}
    all_records = []
    # 累积全 head 统计（用于最后的 head_stats_overview / scatter），按 target 分开
    accumulated_sum_by_target: Dict[int, List[np.ndarray]] = {tvi: [] for tvi in TARGET_VIEW_INDICES}
    accumulated_ent_by_target: Dict[int, List[np.ndarray]] = {tvi: [] for tvi in TARGET_VIEW_INDICES}
    accumulated_tau: List[float] = []
    # per-sample per-target 的完整 head 数据，用于跑完后统一渲染 per-head 视频
    # per_sample_data[step_idx] = {tvi: {image_path, q_to_target_full, grid_thw, ...}, "score", ...}
    per_sample_data: List[Dict] = []
    total_steps = len(samples)

    for step_idx, item in enumerate(samples):
        t0 = time.time()
        frame_idx = indices[step_idx + 1]
        print(f"\n  [Step {step_idx + 1}/{total_steps}] frame_idx={frame_idx}")

        try:
            agg_maps, orig_score, target_ranges, stats = attributor.compute_attention(
                image_paths=item["image"],
                task=item["task"],
                target_view_indices=TARGET_VIEW_INDICES,
                query_mode=QUERY_MODE,
                forward_mode=FORWARD_MODE,
            )
        except Exception as e:
            print(f"    [ERROR] attention extraction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        print(f"    Score: {orig_score:+.1f}%  |  Time: {elapsed:.1f}s")

        # 累积每个 target 的全 head 统计
        per_target = stats["per_target"]
        for tvi in TARGET_VIEW_INDICES:
            accumulated_sum_by_target[tvi].append(per_target[tvi]["attn_sum"])
            accumulated_ent_by_target[tvi].append(per_target[tvi]["spatial_entropy"])
        if stats.get("agg_info", {}).get("tau") is not None:
            accumulated_tau.append(stats["agg_info"]["tau"])

        # 每个 target 渲染一帧热力图，加入对应 target 的主视频
        step_sample_data: Dict = {
            "score": orig_score,
            "step_idx": step_idx,
            "frame_idx": frame_idx,
            "selected_heads": stats.get("selected_heads", []),
        }
        for tvi in TARGET_VIEW_INDICES:
            attention_map = agg_maps[tvi]
            target_image_path = item["image"][tvi]
            grid_thw_target = per_target[tvi]["grid_thw"]
            max_attn = float(attention_map.max()) if attention_map.size else 0.0
            print(f"    [target {tvi}] max attn: {max_attn:.4f}")

            heatmap_png = str(heatmaps_dir / f"heatmap_step{step_idx:04d}_frame{frame_idx:06d}_t{tvi}.png")
            frame_rgb = render_attention_frame(
                image_path=target_image_path,
                attention_map=attention_map,
                grid_thw_target=grid_thw_target,
                orig_score=orig_score,
                step_idx=step_idx,
                frame_idx=frame_idx,
                task=TASK_INSTRUCTION,
                out_path=heatmap_png,
                alpha=OVERLAY_ALPHA,
                head_label=f"t{tvi} top-{HEAD_TOPK} {HEAD_CRITERION}",
            )
            video_frames_by_target[tvi].append(frame_rgb)

            # 保存完整 head 数据用于 per-head 视频（仅开启时保留 q_to_target）
            if OUTPUT_PER_HEAD_VIDEOS or SAVE_ALL_HEAD_STATS:
                step_sample_data[tvi] = {
                    "image_path": target_image_path,
                    "q_to_target_full": per_target[tvi]["q_to_target"],   # [L, H, T_target]
                    "grid_thw_target": grid_thw_target,
                }

            record = {
                "step": step_idx,
                "frame_idx": frame_idx,
                "target_view_idx": tvi,
                "orig_score": orig_score,
                "max_attn": max_attn,
                "mean_attn": float(attention_map.mean()) if attention_map.size else 0.0,
                "attention_map": attention_map.tolist(),
                "grid_thw_target": grid_thw_target,
                "target_token_range": [target_ranges[tvi][0], target_ranges[tvi][1]],
            }
            all_records.append(record)

        if OUTPUT_PER_HEAD_VIDEOS or SAVE_ALL_HEAD_STATS:
            per_sample_data.append(step_sample_data)

        # 保存这一步的全 head 统计（npz，可选）
        if SAVE_ALL_HEAD_STATS:
            for tvi in TARGET_VIEW_INDICES:
                npz_path = run_root / f"head_stats_step{step_idx:04d}_t{tvi}.npz"
                save_dict = {
                    "attn_sum": per_target[tvi]["attn_sum"],
                    "spatial_entropy": per_target[tvi]["spatial_entropy"],
                }
                if (OUTPUT_PER_HEAD_VIDEOS or SAVE_ALL_HEAD_STATS):
                    save_dict["q_to_target_full"] = per_target[tvi]["q_to_target"]
                np.savez_compressed(str(npz_path), **save_dict)

    print(f"\n  Done. "
          + ", ".join(f"t{tvi}={len(fs)} frames" for tvi, fs in video_frames_by_target.items())
          + "\n")

    # --- 5. 保存 ---
    print("[5/5] Saving results ...")

    # 计算聚合后的 sum/entropy 矩阵（多 target 取平均 + 跨 sample 取平均/首帧）
    agg_sum_mat = None
    agg_ent_mat = None
    if any(accumulated_sum_by_target[tvi] for tvi in TARGET_VIEW_INDICES):
        # 每个 target 内部先跨 sample 聚合
        per_target_sum = {}
        per_target_ent = {}
        for tvi in TARGET_VIEW_INDICES:
            sums = accumulated_sum_by_target[tvi]
            ents = accumulated_ent_by_target[tvi]
            if not sums:
                continue
            if HEAD_STATS_AGG == "first":
                per_target_sum[tvi] = sums[0]
                per_target_ent[tvi] = ents[0]
            else:
                per_target_sum[tvi] = np.mean(np.stack(sums), axis=0)
                per_target_ent[tvi] = np.mean(np.stack(ents), axis=0)
        if per_target_sum:
            # 多 target 再平均
            agg_sum_mat = np.mean(np.stack(list(per_target_sum.values())), axis=0)
            agg_ent_mat = np.mean(np.stack(list(per_target_ent.values())), axis=0)

    # JSON
    json_path = run_root / "attention_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "method": "Attention (query token → target images)",
                "model_path": MODEL_PATH,
                "data_dir": DATA_DIR,
                "task": TASK_INSTRUCTION,
                "goal_image": GOAL_IMAGE,
                "frame_interval": FRAME_INTERVAL,
                "eval_mode": EVAL_MODE,
                "target_view_indices": TARGET_VIEW_INDICES,
                "forward_mode": FORWARD_MODE,
                "query_mode": QUERY_MODE,
                "head_agg": HEAD_AGG,
                "head_topk": HEAD_TOPK,
                "head_criterion": HEAD_CRITERION,
                "exclude_first_layers": EXCLUDE_FIRST_LAYERS,
                "attn_sum_threshold": ATTN_SUM_THRESHOLD,
                "attn_sum_floor": ATTN_SUM_FLOOR,
                "spatial_entropy_mode": SPATIAL_ENTROPY_MODE,
                "video_content": VIDEO_CONTENT,
                "per_head_videos": OUTPUT_PER_HEAD_VIDEOS,
                "save_all_head_stats": SAVE_ALL_HEAD_STATS,
                "head_stats_agg": HEAD_STATS_AGG,
            },
            "records": all_records,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Results JSON: {json_path}")

    # 头统计汇总：layer×head 热力图 + scatter plot（用聚合 sum/ent 矩阵）
    if agg_sum_mat is not None:
        try:
            # 用聚合后的指标重新选一次 head（与每步排序方式一致），用于在图上高亮
            agg_info = _rank_aggregated(agg_sum_mat, agg_ent_mat)
            common_heads = agg_info["selected_heads"]

            # 热力图
            plot_head_stats(
                {
                    "attn_sum": agg_sum_mat,
                    "spatial_entropy": agg_ent_mat,
                    "selected_heads": common_heads,
                },
                str(run_root / "head_stats_overview.png"),
            )
            print(f"  Head stats overview ({HEAD_STATS_AGG} over samples, "
                  f"mean over {len(TARGET_VIEW_INDICES)} targets): "
                  f"{run_root / 'head_stats_overview.png'}")

            # scatter plot（论文 Fig 6 风格）
            tau_for_plot = agg_info.get("tau")
            plot_head_scatter(
                attn_sum=agg_sum_mat,
                spatial_entropy=agg_ent_mat,
                selected_heads=common_heads,
                tau=tau_for_plot,
                out_path=str(run_root / "head_scatter.png"),
            )
            print(f"  Head scatter (sum vs entropy): {run_root / 'head_scatter.png'}")
            if tau_for_plot is not None:
                print(f"    τ used for ranking: {tau_for_plot:.4f}")
        except Exception as e:
            print(f"  [WARN] head stats plot failed: {e}")
            import traceback
            traceback.print_exc()

    # 主视频：每个 target 一个
    content_desc = {
        "selected": f"top-{HEAD_TOPK} heads by {HEAD_CRITERION}" if HEAD_AGG == "topk" else "mean of all heads",
        "mean": "mean of all heads (forced)",
    }.get(VIDEO_CONTENT, VIDEO_CONTENT)
    for tvi in TARGET_VIEW_INDICES:
        frames = video_frames_by_target[tvi]
        if not frames:
            continue
        video_path = run_root / f"attention_video_t{tvi}_{VIDEO_CONTENT}.mp4"
        if _MOVIEPY_AVAILABLE:
            clip = ImageSequenceClip(frames, fps=VIDEO_FPS)
            clip.write_videofile(str(video_path), logger=None)
        else:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h))
            for frm in frames:
                writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
            writer.release()
        print(f"  Attention video t{tvi} ({VIDEO_CONTENT}: {content_desc}): {video_path}")

    # 每个固定 top-K head 单独的视频（对每个 target 各输出一个）
    # 流程：
    # 1. 跑完所有 sample 后，在聚合 sum/entropy 上按 HEAD_CRITERION 选固定 HEAD_TOPK 个 head
    # 2. 用这些固定 head 对每个 sample 渲染一帧，每个 head × 每个 target 一个视频
    if OUTPUT_PER_HEAD_VIDEOS and per_sample_data and agg_sum_mat is not None:
        per_head_dir = run_root / "per_head_videos"
        per_head_dir.mkdir(exist_ok=True)

        fixed_heads_info = _rank_aggregated(agg_sum_mat, agg_ent_mat)
        fixed_heads = fixed_heads_info["selected_heads"]
        print(f"  Fixed heads (ranked by {HEAD_CRITERION} on aggregated stats, top-{HEAD_TOPK}):")
        for lh in fixed_heads:
            print(f"    L{lh[0]:02d}H{lh[1]:02d}")

        for (l, h) in fixed_heads:
            for tvi in TARGET_VIEW_INDICES:
                frames = []
                for sd in per_sample_data:
                    if tvi not in sd:
                        continue
                    q_full = sd[tvi]["q_to_target_full"]   # [L, H, T_target]
                    head_map = q_full[l, h]
                    head_frame = render_attention_frame(
                        image_path=sd[tvi]["image_path"],
                        attention_map=head_map,
                        grid_thw_target=sd[tvi]["grid_thw_target"],
                        orig_score=sd["score"],
                        step_idx=sd["step_idx"],
                        frame_idx=sd["frame_idx"],
                        task=TASK_INSTRUCTION,
                        out_path=None,
                        alpha=OVERLAY_ALPHA,
                        head_label=f"L{l:02d}H{h:02d} t{tvi}",
                    )
                    frames.append(head_frame)
                if not frames:
                    continue

                head_video_path = per_head_dir / f"L{l:02d}H{h:02d}_t{tvi}.mp4"
                if _MOVIEPY_AVAILABLE:
                    clip = ImageSequenceClip(frames, fps=VIDEO_FPS)
                    clip.write_videofile(str(head_video_path), logger=None)
                else:
                    h_, w_ = frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(head_video_path), fourcc, VIDEO_FPS, (w_, h_))
                    for frm in frames:
                        writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
                    writer.release()
                print(f"    → {head_video_path}")

    # score / max_attn 曲线（每个 target 一条 max_attn 线）
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps_sorted = sorted({r["step"] for r in all_records})
        score_by_step = {}
        for r in all_records:
            score_by_step.setdefault(r["step"], []).append(r["orig_score"])
        scores = [float(np.mean(score_by_step[s])) for s in steps_sorted]

        ax2 = ax.twinx()
        ax.plot(steps_sorted, scores, "o-", color="#2196F3", label="Score (%)")
        colors = ["#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
        for ci, tvi in enumerate(TARGET_VIEW_INDICES):
            maxes_by_step = {}
            for r in all_records:
                if r["target_view_idx"] == tvi:
                    maxes_by_step.setdefault(r["step"], []).append(r["max_attn"])
            maxes = [float(np.mean(maxes_by_step[s])) if s in maxes_by_step else np.nan
                     for s in steps_sorted]
            ax2.plot(steps_sorted, maxes, "s-", color=colors[ci % len(colors)],
                     label=f"Max attn t{tvi}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Original Score (%)", color="#2196F3")
        ax2.set_ylabel("Max attention")
        ax.set_title(f"Score vs Attention Over Time\nTask: {TASK_INSTRUCTION}")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.95))
        fig.tight_layout()
        curve_path = run_root / "score_attention_curve.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        print(f"  Curve:           {curve_path}")
    except Exception as e:
        print(f"  [WARN] curve plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"ALL DONE. Results saved under: {run_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
