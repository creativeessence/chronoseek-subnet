import math
from typing import List, Tuple
from chronoseek.schemas import VideoSearchResult

def calculate_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """
    Calculate Intersection over Union (IoU) between two time intervals.
    """
    # Calculate intersection
    start = max(pred_start, gt_start)
    end = min(pred_end, gt_end)
    intersection = max(0.0, end - start)
    
    # Calculate union
    # Union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    pred_len = pred_end - pred_start
    gt_len = gt_end - gt_start
    
    if pred_len <= 0 or gt_len <= 0:
        return 0.0
        
    union = pred_len + gt_len - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union

def score_response(
    predictions: List[VideoSearchResult], 
    ground_truth: Tuple[float, float], 
    latency: float,
    lambda_decay: float = 0.1
) -> float:
    """
    Score a miner's response based on max IoU and latency.
    S_final = max(IoU) * e^(-lambda * latency)
    """
    if not predictions:
        return 0.0
        
    gt_start, gt_end = ground_truth
    
    # Find best matching prediction (max IoU)
    max_iou = 0.0
    for pred in predictions:
        iou = calculate_iou(pred.start, pred.end, gt_start, gt_end)
        if iou > max_iou:
            max_iou = iou
            
    # Apply latency penalty
    # e^(-0.1 * 1.0s) ~= 0.90
    # e^(-0.1 * 5.0s) ~= 0.60
    latency_factor = math.exp(-lambda_decay * latency)
    
    return max_iou * latency_factor
