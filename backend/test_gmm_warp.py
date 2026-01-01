#!/usr/bin/env python
"""
Quick test to see if GMM is actually warping the cloth
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
sys.path.append('cp-vton')
from networks import GMM

# Load GMM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Opt:
    def __init__(self):
        self.fine_height = 256
        self.fine_width = 192
        self.grid_size = 5

opt = Opt()
GMM_model = GMM(opt, cloth_channels=1).to(device)
GMM_checkpoint = torch.load("models/GMM.pth", map_location=device)
GMM_model.load_state_dict(GMM_checkpoint)
GMM_model.eval()

print("[*] GMM loaded successfully")

# Load test images
person_img = Image.open("uploads/person.jpg").convert("RGB").resize((192, 256), Image.BILINEAR)
cloth_img = Image.open("uploads/cloth.jpg").convert("RGB").resize((192, 256), Image.BILINEAR)

# Preprocess
def preprocess(img):
    img = np.array(img) / 255.0
    img = (img - 0.5) / 0.5
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return img

person_tensor = preprocess(person_img).to(device)
cloth_tensor = preprocess(cloth_img).to(device)

print(f"Person shape: {person_tensor.shape}")
print(f"Cloth shape: {cloth_tensor.shape}")

# Create a REAL agnostic representation with actual pose information
# This is the key - we need to give GMM real pose data to work with

# Create pose heatmaps with actual keypoints
pose_maps = torch.zeros(18, 256, 192, device=device, dtype=torch.float32)

# Add some pose keypoints to guide the warping
# Shoulders
pose_maps[5, 64, 64] = 1.0   # Left shoulder
pose_maps[6, 64, 128] = 1.0  # Right shoulder

# Elbows
pose_maps[7, 128, 32] = 1.0   # Left elbow
pose_maps[8, 128, 160] = 1.0  # Right elbow

# Hips
pose_maps[11, 192, 64] = 1.0  # Left hip
pose_maps[12, 192, 128] = 1.0 # Right hip

# Create shape tensor (person silhouette)
shape_tensor = torch.ones(1, 256, 192, device=device, dtype=torch.float32)

# Create head tensor (preserve head)
head_tensor = torch.zeros(3, 256, 192, device=device, dtype=torch.float32)
head_tensor[:, 0:64, 48:144] = 1.0  # Head region

# Combine to 22 channels
agnostic = torch.cat([shape_tensor, head_tensor, pose_maps], dim=0).unsqueeze(0)

print(f"Agnostic shape: {agnostic.shape}")

# Test GMM
with torch.no_grad():
    cloth_1ch = cloth_tensor[:, 0:1, :, :]  # 1 channel
    
    print("[*] Testing GMM warping...")
    grid, theta = GMM_model(agnostic, cloth_1ch)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Grid range: [{grid.min().item():.3f}, {grid.max().item():.3f}]")
    print(f"Theta shape: {theta.shape}")
    
    # Check if grid is actually different from identity
    identity_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).to(device), (1, 1, 256, 192), align_corners=False)
    grid_diff = torch.abs(grid - identity_grid).mean()
    print(f"Grid difference from identity: {grid_diff.item():.6f}")
    
    if grid_diff.item() < 0.001:
        print("[ERROR] Grid is identity - no warping happening!")
        print("This means the pose information is not being used properly")
    else:
        print("[OK] Grid shows warping transformation")
    
    # Warp cloth
    warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
    
    # Check if warping actually changed the cloth
    diff = torch.abs(cloth_tensor - warped_cloth).mean()
    print(f"Cloth difference after warping: {diff.item():.6f}")
    
    if diff.item() < 0.001:
        print("[ERROR] No warping detected!")
    else:
        print("[OK] Cloth was warped")
    
    # Save results
    warped_img = warped_cloth.squeeze(0).cpu().numpy()
    warped_img = ((warped_img * 0.5 + 0.5) * 255).astype(np.uint8)
    warped_img = warped_img.transpose(1, 2, 0)
    Image.fromarray(warped_img).save("test_warped.jpg")
    
    # Save original cloth for comparison
    orig_img = cloth_tensor.squeeze(0).cpu().numpy()
    orig_img = ((orig_img * 0.5 + 0.5) * 255).astype(np.uint8)
    orig_img = orig_img.transpose(1, 2, 0)
    Image.fromarray(orig_img).save("test_original.jpg")
    
    print("Saved test_warped.jpg and test_original.jpg")
    print("Compare these images to see if warping is working")
