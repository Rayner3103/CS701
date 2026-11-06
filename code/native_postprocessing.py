"""F2 inference settings with no transferred coordinate shift on native-size inputs.

Objectness weighting, soft-NMS, stage-box means and classifier/detector fusion
come from the copied F2 code. They are not the sampler contribution.
"""
import torch

FINAL_POST={'score_thresh':1e-5,'min_keep':30,'per_class':100,'total':100,'fusion':'none',
            'nms':'soft','soft_sigma':.75,'obj_pow':.25,'shift':0.,'stage_box':'mean'}


def fused_class(logits, output):
    raw=output['raw_logits']
    if not len(raw):return int(logits.argmax())
    log_quality=torch.log(raw.float().softmax(-1)[:,1:].amax(0)+1e-6)
    return int((logits.float().log_softmax(0)+log_quality).argmax())
