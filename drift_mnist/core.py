
import torch
import torch.nn.functional as F

def compute_drift(gen, pos, temp=0.5):
    """
    Compute drift field V with attention-based kernel.
    
    Args:
        gen: Generated samples [B, C, H, W] or [B, D]
        pos: Real data samples [B, C, H, W] or [B, D]
        temp: Temperature for softmax kernel
        
    Returns:
        V: Drift vectors [B, C, H, W] or [B, D]
    """
    # Flatten input if necessary
    original_shape = gen.shape
    if len(gen.shape) > 2:
        gen_flat = gen.view(gen.shape[0], -1)
        pos_flat = pos.view(pos.shape[0], -1)
    else:
        gen_flat = gen
        pos_flat = pos
        
    targets = torch.cat([gen_flat, pos_flat], dim=0) # [2B, D]
    G = gen_flat.shape[0]
    
    # Compute pairwise distances
    # dist[i, j] = ||gen[i] - targets[j]||
    dist = torch.cdist(gen_flat, targets) # [B, 2B]
    
    # Mask self-distances for generated curve (first B elements)
    # We don't want the point itself to contribute to its own drift with infinite weight if temp is small
    # But in the formula, k(x, y) counts all y. 
    # The notebook implementation masks self: `dist[:, :G].fill_diagonal_(1e6)`
    # This prevents numerical instability and self-attraction in the attention mechanism.
    dist[:, :G].fill_diagonal_(1e6)
    
    # Softmax kernel
    # kernel(x, y) = exp(-||x - y|| / temp)
    # Using Softmax(d/temp) is equivalent to exp(d/temp) / sum(...)
    # We compute unnormalized kernel values for manual normalization as per paper
    # But for stability we subtract max before exp? 
    # Notebook uses: kernel = (-dist / temp).exp()
    
    kernel = (-dist / temp).exp()
    
    # Normalize
    # Z_p(x) * Z_q(x) normalization
    # Notebook Implementation:
    # normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) ?? No, check notebook
    # Notebook: normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) 
    # Wait, kernel dim is [B, 2B]. Sum dim=-1 is sum over targets. Sum dim=-2 is sum over generated?
    # Actually the paper says Zp * Zq. 
    # Let's align with the notebook code provided.
    
    # Notebook code:
    # normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) 
    # This looks incorrect based on dimensions. dim=-2 of [B, 2B] would be B. 
    # Let's re-read the notebook code carefully.
    
    # Notebook:
    # normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) 
    # If kernel is [G, 2G], sum(-1) -> [G, 1]. sum(-2) -> [1, 2G]? No, kernel is [G, 2G].
    # This line in the notebook seems to imply creating a broadcasting normalizer?
    # It might be `kernel.sum(dim=1, keepdim=True)` which is Z(x).
    # The notebook code: `kernel.sum(dim=-1, keepdim=True)` is [B, 1].
    # `kernel.sum(dim=-2, keepdim=True)` would be [1, 2B].
    # Their product would be [B, 2B].
    # This effectively normalizes by (Sum_y k(x,y)) * (Sum_x k(x,y))? 
    # The paper says 1 / (Zp * Zq). Zp and Zq are integrals over y. They depend on x.
    # So normalizer should be Z(x)^2 or Zp(x)Zq(x).
    # The notebook implementation might be doing something specific. 
    # "normalize along both dimensions, which we found to slightly improve performance"
    
    normalizer = kernel.sum(dim=-1, keepdim=True) * kernel.sum(dim=-2, keepdim=True) # [B, 2B]
    normalizer = normalizer.clamp_min(1e-12).sqrt()
    normalized_kernel = kernel / normalizer
    
    # Compute Drift Vectors
    # V = V+ - V-
    # V+ ~ E_p [k(x, y) (y - x)] 
    # V- ~ E_q [k(x, y) (y - x)]
    
    # Notebook:
    # pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    # This looks like a specific estimator. I will copy the notebook logic exactly.
    
    pos_k = normalized_kernel[:, G:] # [B, B] (Interaction with Real)
    neg_k = normalized_kernel[:, :G] # [B, B] (Interaction with Generated)
    
    # Re-reading notebook logic carefully:
    # pos_coeff = normalized_kernel[:, G:] * normalized_kernel[:, :G].sum(dim=-1, keepdim=True)
    # neg_coeff = normalized_kernel[:, :G] * normalized_kernel[:, G:].sum(dim=-1, keepdim=True)
    
    pos_coeff = pos_k * neg_k.sum(dim=-1, keepdim=True) # Weight for real samples
    neg_coeff = neg_k * pos_k.sum(dim=-1, keepdim=True) # Weight for gen samples
    
    # Calculate weighted sum of targets
    # targets[G:] are real samples, targets[:G] are generated samples
    
    pos_V = pos_coeff @ targets[G:] # [B, D]
    neg_V = neg_coeff @ targets[:G] # [B, D]
    
    # The term (y - x) in the expectation is split: sum(w * y) - sum(w * x)
    # Here we only computed sum(w * y). What about -x?
    # In the paper V_{p,q} = sum w (y^+ - y^-). There is no explicit -x term if combining.
    # The formula is E [ k k (y+ - y-) ]. 
    # So we need weighted sum of y+ minus weighted sum of y-.
    # That is exactly pos_V - neg_V.
    
    drift = pos_V - neg_V
    
    return drift.view(original_shape)

def drifting_loss(gen, pos, temp=0.5):
    """
    Drifting loss: MSE(gen, stopgrad(gen + V)).
    """
    V = compute_drift(gen, pos, temp=temp)
    target = (gen + V).detach()
    return F.mse_loss(gen, target)
