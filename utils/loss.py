# import torch

# def normalize_scale(p, eps=1e-8):
#     """
#     p: (B, N, 3)
#     Normalize so max distance from centroid = 1
#     """
#     centroid = torch.mean(p, dim=1, keepdim=True)
#     p_centered = p - centroid
#     scale = torch.norm(p_centered, dim=2).max(dim=1, keepdim=True)[0]
#     p_normalized = p_centered / (scale.unsqueeze(-1) + eps)
#     return p_normalized

# def get_chamfer_distance(p1, p2):
#     dist_matrix = torch.cdist(p1, p2)
#     min_dist_p1, _ = torch.min(dist_matrix, dim=2)
#     min_dist_p2, _ = torch.min(dist_matrix, dim=1)
#     return torch.mean(min_dist_p1, dim=1) + torch.mean(min_dist_p2, dim=1)

# def to_torch(x, device="cpu", dtype=torch.float32):
#     if not torch.is_tensor(x):
#         x = torch.from_numpy(x)
#     return x.to(device=device, dtype=dtype)


# def align_pca(p):
#     if p.ndim == 2:
#         p = p.unsqueeze(0)
#     centroid = torch.mean(p, dim=1, keepdim=True)
#     p_centered = p - centroid
#     cov = torch.bmm(p_centered.transpose(1, 2), p_centered)
#     e, v = torch.linalg.eigh(cov)
#     p_aligned = torch.bmm(p_centered, v)
#     return p_aligned

# def invariant_chamfer_loss(cloud1, cloud2, return_mean=False):
#     cloud1 = to_torch(cloud1)
#     cloud2 = to_torch(cloud2)
#     cloud1 = normalize_scale(cloud1)
#     cloud2 = normalize_scale(cloud2)
#     if cloud1.ndim == 2: cloud1 = cloud1.unsqueeze(0)
#     if cloud2.ndim == 2: cloud2 = cloud2.unsqueeze(0)
#     c1_aligned = align_pca(cloud1)
#     c2_aligned = align_pca(cloud2)
#     base_loss = get_chamfer_distance(c1_aligned, c2_aligned)
#     mirror_x = torch.tensor([-1.0, 1.0, 1.0], device=cloud1.device)
#     c2_flipped_x = c2_aligned * mirror_x
#     loss_x_flip = get_chamfer_distance(c1_aligned, c2_flipped_x)
#     mirror_y = torch.tensor([1.0, -1.0, 1.0], device=cloud1.device)
#     c2_flipped_y = c2_aligned * mirror_y
#     loss_y_flip = get_chamfer_distance(c1_aligned, c2_flipped_y)
#     c2_flipped_xy = c2_aligned * mirror_x * mirror_y
#     loss_xy_flip = get_chamfer_distance(c1_aligned, c2_flipped_xy)
#     losses = torch.stack([base_loss, loss_x_flip, loss_y_flip, loss_xy_flip], dim=0)
#     min_loss, _ = torch.min(losses, dim=0)
#     if return_mean:
#         return torch.mean(min_loss)
#     else:
#         return min_loss
import torch
import numpy as np


# -------------------------------
# Utilities
# -------------------------------

def to_torch(x, device="cpu", dtype=torch.float32):
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


def ensure_batch(p):
    if p.ndim == 2:
        p = p.unsqueeze(0)
    return p


def normalize_center(p):
    """ Remove translation only """
    centroid = p.mean(dim=1, keepdim=True)
    return p - centroid

def normalize_0_1(p):
    min = torch.min(p)
    max = torch.max(p)
    for i in range(p.shape[0]):
        min = torch.min(p[i])
        max = torch.max(p[i])
        p[i] = (p[i] - min) / (max - min)
    return p


def chamfer_distance(p1, p2):
    dist = torch.cdist(p1, p2)
    d1 = dist.min(dim=2)[0].mean(dim=1)
    d2 = dist.min(dim=1)[0].mean(dim=1)
    return d1 + d2

def chamfer_with_scale_search(
    cloud1,
    cloud2,
    scale_min=0.1,
    scale_max=3,
    scale_step=0.1,
    return_scale=True
):

    cloud1 = ensure_batch(to_torch(cloud1))
    cloud2 = ensure_batch(to_torch(cloud2))

    cloud1 = normalize_0_1(cloud1)
    cloud2 = normalize_0_1(cloud2)

    # scales = torch.arange(scale_min, scale_max + 1e-6, scale_step, device=cloud1.device)

    losses = []
    # for s in scales:
    loss = chamfer_distance(cloud1, cloud2)
    losses.append(loss)

    losses = torch.stack(losses, dim=0)   
    min_loss, idx = losses.min(dim=0)

    return (min_loss.mean(), 1) if return_scale else min_loss.mean()
