#!/usr/bin/env python
"""
Debug CP-VTON models to find the exact issue
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[*] Loading models for debugging...")

# Load GMM
try:
    import sys
    sys.path.append('cp-vton')
    from networks import GMM, UnetGenerator
    
    class Opt:
        def __init__(self):
            self.fine_height = 256
            self.fine_width = 192
            self.grid_size = 5
    
    opt = Opt()
    GMM_model = GMM(opt, cloth_channels=1).to(device)  # Use 1 channel for cloth
    GMM_checkpoint = torch.load("models/GMM.pth", map_location=device)
    
    # Check what keys are available
    print(f"  GMM checkpoint keys: {list(GMM_checkpoint.keys())}")
    
    # Direct loading
    GMM_model.load_state_dict(GMM_checkpoint)
    
    GMM_model.eval()
    print("[OK] GMM loaded")
except Exception as e:
    print(f"[ERROR] GMM loading failed: {e}")
    GMM_model = None

# Load TOM
try:
    # TOM has architecture issues, skip for now
    TOM_model = None
    print("[WARN] TOM skipped due to architecture mismatch")
except Exception as e:
    print(f"[ERROR] TOM loading failed: {e}")
    TOM_model = None

def debug_pipeline(person_path, cloth_path):
    """Debug the full pipeline step by step"""
    
    print(f"\n[*] Debugging pipeline with:")
    print(f"  Person: {person_path}")
    print(f"  Cloth: {cloth_path}")
    
    # Load and preprocess images
    person_img = Image.open(person_path).convert("RGB").resize((192, 256), Image.BILINEAR)
    cloth_img = Image.open(cloth_path).convert("RGB").resize((192, 256), Image.BILINEAR)
    
    # Convert to tensors
    def preprocess(img):
        img = np.array(img) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)  # Ensure float32
        return img
    
    person_tensor = preprocess(person_img).to(device)
    cloth_tensor = preprocess(cloth_img).to(device)
    
    print(f"  Person tensor shape: {person_tensor.shape}")
    print(f"  Cloth tensor shape: {cloth_tensor.shape}")
    
    if GMM_model is None:
        print("[ERROR] GMM model not loaded!")
        return
    
    with torch.no_grad():
        # Create simple pose
        pose_data = {
            "people": [{
                "pose_keypoints": [0]*54  # Simple pose
            }]
        }
        
        # Create simple parsing
        parse_mask = np.zeros((256, 192), dtype=np.uint8)
        # Upper body
        parse_mask[64:166, 48:144] = 5  # Upper clothes
        # Head
        parse_mask[0:64, 48:144] = 12  # Face
        
        # Create agnostic representation (22 channels total)
        # Shape tensor (1 channel)
        shape_tensor = torch.ones(1, 256, 192, device=device, dtype=torch.float32)
        
        # Head extraction (3 channels for RGB)
        head_tensor = torch.zeros(3, 256, 192, device=device, dtype=torch.float32)
        head_tensor[:, 0:64, 48:144] = 1.0
        
        # Pose maps (18 channels)
        pose_maps = torch.zeros(18, 256, 192, device=device, dtype=torch.float32)
        
        # Combine: 1 + 3 + 18 = 22 channels
        agnostic = torch.cat([
            shape_tensor,
            head_tensor,
            pose_maps
        ], dim=0).unsqueeze(0)  # [1, 22, 256, 192]
        
        print(f"  Agnostic shape: {agnostic.shape}")
        
        # GMM step
        print("\n[*] Testing GMM...")
        cloth_tensor_1ch = cloth_tensor[:, 0:1, :, :]  # 1 channel for GMM
        
        try:
            grid, theta = GMM_model(agnostic, cloth_tensor_1ch)
            print(f"  Grid shape: {grid.shape}")
            print(f"  Grid range: [{grid.min().item():.3f}, {grid.max().item():.3f}]")
            print(f"  Theta shape: {theta.shape}")
            
            # Check if grid is valid
            invalid = torch.sum(torch.abs(grid) > 2.0).item()
            if invalid > 0:
                print(f"  [WARN] {invalid} invalid grid points")
            
            # Warp cloth
            warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
            print(f"  Warped cloth shape: {warped_cloth.shape}")
            
            # Check warping
            diff = torch.abs(cloth_tensor - warped_cloth).mean()
            print(f"  Warping difference: {diff.item():.6f}")
            
            if diff.item() < 0.001:
                print("  [ERROR] No warping detected!")
                return
            
            # Save intermediate results
            warped_img = warped_cloth.squeeze(0).cpu().numpy()
            warped_img = ((warped_img * 0.5 + 0.5) * 255).astype(np.uint8)
            warped_img = warped_img.transpose(1, 2, 0)
            Image.fromarray(warped_img).save("debug_warped.jpg")
            print("  [OK] Saved warped cloth to debug_warped.jpg")
            
            # Create simple final result using warped cloth
            # Load original person and blend
            person_orig = Image.open(person_path).convert("RGB")
            warped_orig = Image.fromarray(warped_img).resize(person_orig.size, Image.LANCZOS)
            
            # Simple blending: preserve head, use warped cloth for body
            w_orig, h_orig = person_orig.size
            head_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
            head_mask[:int(h_orig * 0.25), :] = 1.0  # Top 25% is head
            
            # Convert to numpy
            person_array = np.array(person_orig).astype(np.float32)
            warped_array = np.array(warped_orig).astype(np.float32)
            head_mask_3d = np.stack([head_mask] * 3, axis=2)
            
            # Blend
            final_array = warped_array * (1 - head_mask_3d) + person_array * head_mask_3d
            final_img = Image.fromarray(final_array.astype(np.uint8))
            final_img.save("debug_final.jpg")
            print("  [OK] Saved final result to debug_final.jpg")
            
        except Exception as e:
            print(f"  [ERROR] GMM failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n[OK] Pipeline debug completed!")
        print("Check debug_warped.jpg and debug_final.jpg")

if __name__ == "__main__":
    debug_pipeline("uploads/person.jpg", "uploads/cloth.jpg")
