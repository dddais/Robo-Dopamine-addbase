"""Minimal inference-only Robometer wrapper with exact checkpoint head loading.

Architecture and extraction follow robometer/models/{rbm,heads}.py. Training,
servers, PEFT and Unsloth are deliberately unnecessary for local inference.
"""
import torch
from torch import nn
from transformers import PreTrainedModel,Qwen3VLConfig,Qwen3VLModel

class RobometerModel(PreTrainedModel):
    config_class=Qwen3VLConfig
    base_model_prefix=''
    _supports_sdpa=True
    _no_split_modules=['Qwen3VLTextDecoderLayer','Qwen3VLVisionBlock']
    def __init__(self,config):
        super().__init__(config)
        self.model=Qwen3VLModel(config);h=config.text_config.hidden_size
        for name,n in [('progress_head',10),('success_head',1),('preference_head',1),('similarity_head',1)]:
            setattr(self,name,nn.Sequential(nn.Linear(h,h//2),nn.LayerNorm(h//2),nn.GELU(),nn.Dropout(0.1),nn.Linear(h//2,n)))
        self.frame_pool_attn=nn.Linear(h,1,bias=False)
        self.post_init()
    def forward(self,input_ids,**kwargs):
        out=self.model(input_ids=input_ids,**kwargs,use_cache=False,return_dict=True)
        hidden=out.last_hidden_state[0,input_ids[0]==151673,:]
        if hidden.shape[0]==0:raise RuntimeError('No trained progress token found')
        logits=self.progress_head(hidden);success=self.success_head(hidden).squeeze(-1).sigmoid()
        # Stored legacy bfloat16 derived readout. The authoritative offline
        # scorer reconstructs the official CPU-float32 readout from logits;
        # see meter_eval/readout.py. Keep this field stable for v1 provenance.
        progress=(logits.softmax(-1)*torch.linspace(0,1,10,device=logits.device,dtype=logits.dtype)).sum(-1)
        return {'progress':progress,'success':success,'progress_logits':logits}
