"""Equal-size, same-time-panel wrong-region controls for SOLE montages."""
import math
from mydata_bench.attention_eval.masking import bbox_to_token_positions,matched_wrong_position_set

def montage_wrong_control(inp,target,alignment):
    span=inp['spans'][0];gh=span.grid_thw[1]//2;gw=span.grid_thw[2]//2
    target_set=set(target);wrong=[]
    for entry in alignment:
        slot=entry['history_slot'];size=tuple(entry['image_size'])
        cells=bbox_to_token_positions(span,entry['mapped_bbox'],size,2)
        local=[p-span.start for p in cells];rr=[p//gw for p in local];cc=[p%gw for p in local]
        height=max(rr)-min(rr)+1;width=max(cc)-min(cc)+1
        if height*width!=len(local):raise ValueError('Nonrectangular target mapping')
        left=slot*389/1162;right=(slot*389+384)/1162
        # Official montage width = 3*384+2*5 = 1162, not 1172.
        allowed=[c for c in range(gw) if left<=(c+.5)/gw<right]
        options=[];center=((min(rr)+max(rr))/2,(min(cc)+max(cc))/2)
        for row in range(gh-height+1):
            for col in allowed:
                if col+width-1 not in allowed:continue
                selected={span.start+(row+y)*gw+col+x for y in range(height) for x in range(width)}
                if selected&(target_set|set(wrong)):continue
                distance=(row+(height-1)/2-center[0])**2+(col+(width-1)/2-center[1])**2
                options.append((distance,row,col,selected))
        if not options:raise ValueError('No same-panel equal-area disjoint wrong region')
        wrong.extend(sorted(max(options,key=lambda x:x[:3])[3]))
    if len(wrong)!=len(target_set) or set(wrong)&target_set:raise ValueError('Wrong-region matching violated')
    return sorted(wrong)

def wrong_control(inp,target,alignment):
    if inp['protocol'] in {'image_text','text_image'}:return montage_wrong_control(inp,target,alignment)
    result=[]
    for span in inp['spans']:
        local=[p for p in target if span.start<=p<span.end]
        if not local:continue
        selected=matched_wrong_position_set(span,local,spatial_merge_size=2)
        if selected is None:raise ValueError('No same-frame equal-area disjoint wrong region')
        result+=selected
    assert len(result)==len(set(target)) and not set(result)&set(target)
    return result
