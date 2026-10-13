# Module: VSLAM-LAB - Baselines - colmap - hybrid_matcher_utilities.py
# SuperPoint (x4 rotations) + LightGlue helpers of the hybrid matcher (hybrid_matcher.py).
#
# Everything below the marker line is a VERBATIM copy of feature_matcher_utilities.py from
# the original colmap-vslamlab fork @ d5f8541 (the original hybrid pipeline). Keep it byte-identical
# so the port provably matches the original implementation; changes to the
# method belong in new functions, not here. Check with:
#   sed '1,/^# ---- verbatim below ----$/d' hybrid_matcher_utilities.py | md5sum   ->  68f514464fd3087bcdf0c20023fd7f2c
# ---- verbatim below ----
import torch
import numpy as np
import cv2
from lightglue import LightGlue
from lightglue.utils import rbd
from lightglue import  SuperPoint, SIFT
from lightglue.utils import load_image


def unrotate_kps_W(kps_rot, k, H, W):
    # Ensure inputs are Numpy
    if hasattr(kps_rot, 'cpu'): kps_rot = kps_rot.cpu().numpy()
    if hasattr(k, 'cpu'): k = k.cpu().numpy()
    
    # Squeeze if necessary
    if k.ndim > 1: k = k.squeeze()
    if kps_rot.ndim > 2: kps_rot = kps_rot.squeeze()

    x_r = kps_rot[:, 0]
    y_r = kps_rot[:, 1]
    
    x = np.zeros_like(x_r)
    y = np.zeros_like(y_r)
    
    mask0 = (k == 0)
    x[mask0], y[mask0] = x_r[mask0], y_r[mask0]
    
    mask1 = (k == 1)
    x[mask1], y[mask1] = (W - 1) - y_r[mask1], x_r[mask1]
    
    mask2 = (k == 2)
    x[mask2], y[mask2] = (W - 1) - x_r[mask2], (H - 1) - y_r[mask2]
    
    mask3 = (k == 3)
    x[mask3], y[mask3] = y_r[mask3], (H - 1) - x_r[mask3]
    
    return np.stack([x, y], axis=-1)

def extract_keypoints(path_to_image0, features='superpoint', rotations = [0,1,2,3]):
    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    timg = load_image(path_to_image0).to(device)
    _, h, w = timg.shape

    if features == 'sift':
        extractor = SIFT(max_num_keypoints=2048).eval().to(device)
        feats = extractor.extract(timg)
        return feats , h, w
    
    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)

    # --- Extract local features ---
    feats = {}
    for k in (rotations):
        timg_rotated = torch.rot90(timg, k, dims=(1, 2))
        feats[k] = extractor.extract(timg_rotated)
        #print(f"Extracted {feats[k]['keypoints'].shape[1]} keypoints for rotation {k}")

    # --- Merge features back to original coordinate system ---
    all_keypoints = []
    all_scores = []
    all_descriptors = []
    all_rotations = []
    for k, feat in feats.items():
        kpts = feat['keypoints']  # Shape (1, N, 2)
        num_kpts = kpts.shape[1]
        # if k == 0:
        #     kpts_corrected = kpts
        # elif k == 1:
        #     kpts_corrected = torch.stack(
        #         [w - 1 - kpts[..., 1], kpts[..., 0]], dim=-1
        #     )
        # elif k == 2:
        #     kpts_corrected = torch.stack(
        #         [w - 1 - kpts[..., 0], h - 1 - kpts[..., 1]], dim=-1
        #     )
        # elif k == 3:
        #     kpts_corrected = torch.stack(
        #         [kpts[..., 1], h - 1 - kpts[..., 0]], dim=-1
        #     )

        rot_indices = torch.full((1, num_kpts), k, dtype=torch.long, device=device)
        all_keypoints.append(feat['keypoints'])
        all_scores.append(feat['keypoint_scores'])
        all_descriptors.append(feat['descriptors'])
        all_rotations.append(rot_indices)

    # Concatenate all features along the keypoint dimension (dim=1)
    feats_merged = {
        'keypoints': torch.cat(all_keypoints, dim=1),
        'keypoint_scores': torch.cat(all_scores, dim=1),
        'descriptors': torch.cat(all_descriptors, dim=1),
        'rotations': torch.cat(all_rotations, dim=1)
    }
    
    num_kpts = feats_merged['keypoints'].shape[1]
    # perm = torch.randperm(num_kpts, device=device)

    # feats_merged['keypoints'] = feats_merged['keypoints'][:, perm, :]
    # feats_merged['keypoint_scores'] = feats_merged['keypoint_scores'][:, perm]
    # feats_merged['descriptors'] = feats_merged['descriptors'][:, perm, :]

    # Optional: If you want to retain other keys like 'shape' or 'image_size'
    #feats_merged['image_size'] = torch.tensor([w, h], device=device).unsqueeze(0)
    #feats_merged['scales'] = torch.tensor([w, h], device=device).unsqueeze(0)

    # for f in feats_merged:
    #     if 'scales' not in f:
    #         f['scales'] = torch.ones(all_keypoints.shape[:-1], device=device)
    #     if 'oris' not in f:
    #         f['oris'] = torch.zeros(all_keypoints.shape[:-1], device=device)

    return feats_merged , feats, h, w

def lightglue_matching(feats0, feats1, matcher = None):
    if matcher is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        matcher = LightGlue(features='superpoint').eval().to(device)
    
    out_k = matcher({'image0': feats0, 'image1': feats1})
    _, _, out_k = [rbd(x) for x in [feats0, feats1, out_k]]   # remove batch dim
    return out_k['matches'] 

def feature_matching(feats0, feats1, matcher = None, exhaustive = True):
    best_rot = 0
    best_num_matches = 0
    matches_tensor = None
 
    # Find the best rotation alignment
    for rot in [0,1,2,3]:
        matches_tensor_rot = lightglue_matching(feats0[0], feats1[rot], matcher = matcher)
        if (len(matches_tensor_rot) > best_num_matches):
            best_num_matches = len(matches_tensor_rot)
            best_rot = rot
            matches_tensor = matches_tensor_rot

    if matches_tensor is not None and len(matches_tensor) > 0:
        matches_np = matches_tensor.cpu().numpy().astype(np.uint32)
    else:
        return None

    # Adjust matches to account for rotations
    for k in range(best_rot):
        matches_np[:,1] += feats1[k]['keypoints'].shape[1]
    all_matches = [matches_np]  

    if not exhaustive:
        return matches_np
    
    # Find the other rotation combinations
    rots = []
    for rot in [1, 2, 3]:
        rot_i = best_rot + rot
        if rot_i >=4:
            rot_i = rot_i -4
        rots.append(rot_i)

    # Compute matches for the other rotation combinations
    for rot_i in [1,2,3]:
        rot_j = rots[rot_i-1]

        matches_tensor_rot = lightglue_matching(feats0[rot_i], feats1[rot_j], matcher = matcher)
        matches_np_i = matches_tensor_rot.cpu().numpy().astype(np.uint32)
        if rot_i > 0:
            for k in range(rot_i):
                matches_np_i[:,0] += feats0[k]['keypoints'].shape[1]
        if rot_j > 0:
            for k in range(rot_j):
                matches_np_i[:,1] += feats1[k]['keypoints'].shape[1]

        all_matches.append(matches_np_i)
        # print(f"Rotation {rot_i} vs {rot_j}: {len(matches_tensor_rot)} matches")

    # Stack all matches together
    matches_stacked = (
        np.vstack(all_matches) if len(all_matches) and all_matches[0].size else
        np.empty((0, 2), dtype=np.uint32)
    )
    
    # if best_rot > 0:
    #     for k in range(best_rot):
    #         print(f"Adjusting for rotation {k}")
    #         matches_np[:,1] += feats1[k]['keypoints'].shape[1]

    # return matches_np
    return matches_stacked
           

              