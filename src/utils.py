import torch
import random
import numpy as np
import logging
import sys

def set_seed(seed=42):
    """
    Sets the seed for reproducibility across Torch, Numpy, and Python random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CuDNN (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to: {seed}")

def setup_logger(name, save_dir=None):
    """
    Creates a logger that prints to console and optionally writes to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (if save_dir provided)
    if save_dir:
        fh = logging.FileHandler(f"{save_dir}/training.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def compute_recall_at_k(similarity_matrix, k_vals=[1, 5, 10]):
    """
    Computes Recall@K given a similarity matrix (Image x Text).
    
    Args:
        similarity_matrix (torch.Tensor): Shape (N, N) where element (i, j) 
                                          is the score between image i and text j.
                                          Assumes (i, i) is the correct match.
        k_vals (list): List of K values to compute (e.g., [1, 5, 10]).
    
    Returns:
        dict: Dictionary containing R@K scores.
    """
    num_samples = similarity_matrix.shape[0]
    results = {}
    
    # Get indices of the top-k scores for each image
    # We only need the top max(k) predictions
    _, indices = torch.topk(similarity_matrix, k=max(k_vals), dim=1)
    
    # The ground truth index for the i-th row is 'i'
    ground_truth = torch.arange(num_samples, device=similarity_matrix.device).view(-1, 1)
    
    for k in k_vals:
        # Check if the ground truth is within the first k columns of prediction indices
        hits = (indices[:, :k] == ground_truth).any(dim=1)
        score = hits.float().mean().item()
        results[f"R@{k}"] = score
        
    return results

def print_trainable_parameters(model):
    """
    Helper to check how many parameters are actually being trained.
    Useful when freezing layers.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )