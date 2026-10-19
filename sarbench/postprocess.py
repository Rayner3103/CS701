"""Turn the RoI head's raw outputs into detections. One function serves training-time evaluation, the offline
study of post-processing settings on saved raw outputs, and predict.py, so all three score boxes the same way.

Every image has one class and all its boxes are of that class, so the image classifier's probabilities can
re-score the detector's boxes ("fusion"). This uses only the model's own outputs for the same single forward pass
of the same image: one model, no test-time augmentation."""
import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops


def detections_from_raw(boxes, class_logits, image_probs=None, *, score_thresh=0.05, nms_thresh=0.5,
                        per_class=100, total=None, fusion='none', gamma=1.0, topk=9, min_size=1e-2,
                        image_size=(512, 512)):
    """boxes: [N, C, 4] decoded XYXY boxes per foreground class (C = 9), class_logits: [N, C + 1] (background
    first), image_probs: [C] softmax of the image classifier (needed for fusion).
    fusion: 'none'  -> score(c) = p_det(c)                         (torchvision's default)
            'mult'  -> score(c) = p_det(c) * p_img(c) ** gamma
            'agn'   -> score(c) = (1 - p_det(bg)) * p_img(c) ** gamma   (class-agnostic objectness x image class)
            'hard'  -> only the argmax image class, score = (1 - p_det(bg)); boxes of that class
    Only the topk most likely image classes keep boxes when fusing. Per-class NMS, then at most `per_class`
    boxes per class (COCO scores at most 100 per image and class) and at most `total` overall.
    Returns boxes [K, 4], scores [K], labels [K] (1..C)."""
    p = F.softmax(class_logits.float(), dim=-1)
    n, c = boxes.shape[0], boxes.shape[1]
    boxes = box_ops.clip_boxes_to_image(boxes.reshape(-1, 4).float(), image_size).reshape(n, c, 4)
    if fusion == 'none':
        scores = p[:, 1:]
    else:
        pi = image_probs.float().clamp_min(1e-12)
        if fusion == 'mult':
            scores = p[:, 1:] * pi[None] ** gamma
        elif fusion == 'agn':
            scores = (1 - p[:, :1]) * pi[None] ** gamma
        elif fusion == 'hard':
            scores = torch.zeros_like(p[:, 1:])
            k = int(pi.argmax())
            scores[:, k] = 1 - p[:, 0]
        else:
            raise ValueError(fusion)
        if topk < c:
            keep_cls = torch.zeros(c, dtype=torch.bool, device=scores.device)
            keep_cls[pi.topk(topk).indices] = True
            scores = scores * keep_cls[None]
    labels = torch.arange(1, c + 1, device=boxes.device).expand(n, c)
    boxes, scores, labels = boxes.reshape(-1, 4), scores.reshape(-1), labels.reshape(-1)
    keep = torch.where(scores > score_thresh)[0]
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = box_ops.remove_small_boxes(boxes, min_size)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = box_ops.batched_nms(boxes, scores, labels, nms_thresh)  # sorted by score, descending
    if per_class is not None:
        # rank of each kept box within its class, in score order
        lab = labels[keep]
        rank = torch.zeros_like(lab)
        for k in lab.unique():
            idx = torch.where(lab == k)[0]
            rank[idx] = torch.arange(len(idx), device=lab.device)
        keep = keep[rank < per_class]
    if total is not None:
        keep = keep[:total]
    return boxes[keep], scores[keep], labels[keep]
