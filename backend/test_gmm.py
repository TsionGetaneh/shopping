#!/usr/bin/env python
"""
Simple test to check if GMM warping is working
"""
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from inference import load_models, create_simple_pose, create_simple_parsing, create_agnostic
from utils.image_utils import preprocess_image, tensor_to_image

def create_simple_test():
    """Create very simple test images"""
    # Person: simple rectangle
    person_img = Image.new('RGB', (192, 256), 'white')
    draw = ImageDraw.Draw(person_img)
    draw.rectangle([50, 50, 142, 206], fill='lightblue')  # Body
    draw.rectangle([30, 70, 50, 180], fill='peachpuff')   # Left arm
    draw.rectangle([142, 70, 162, 180], fill='peachpuff') # Right arm
    
    # Cloth: simple rectangle with pattern
    cloth_img = Image.new('RGB', (192, 256), 'red')
    draw = ImageDraw.Draw(cloth_img)
    draw.rectangle([20, 20, 172, 236], fill='red')
    # Add some pattern to see warping
    for i in range(20, 172, 20):
        draw.line([i, 20, i, 236], fill='darkred', width=2)
    for i in range(20, 236, 20):
        draw.line([20, i, 172, i], fill='darkred', width=2)
    
    return person_img, cloth_img

def test_gmm_only():
    """Test only the GMM warping"""
    print("[*] Testing GMM warping only...")
    
    # Create test images
    person_img, cloth_img = create_simple_test()
    
    # Save inputs
    person_img.save('gmm_test_person.jpg')
    cloth_img.save('gmm_test_cloth.jpg')
    
    # Load models
    GMM_model, TOM_model = load_models()
    
    if GMM_model is None:
        print("[ERROR] GMM model not loaded!")
        return False
    
    try:
        # Create tensors
        person_tensor = preprocess_image(person_img)
        cloth_tensor = preprocess_image(cloth_img)
        
        print(f"[*] Person tensor: {person_tensor.shape}")
        print(f"[*] Cloth tensor: {cloth_tensor.shape}")
        
        # Create pose and parsing
        pose_data = create_simple_pose(person_img)
        parse_mask = create_simple_parsing(person_img)
        
        # Create agnostic
        agnostic = create_agnostic(person_img, pose_data, parse_mask, person_tensor)
        print(f"[*] Agnostic: {agnostic.shape}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agnostic = agnostic.to(device)
        cloth_tensor = cloth_tensor.to(device)
        
        # Test GMM
        with torch.no_grad():
            grid, theta = GMM_model(agnostic, cloth_tensor)
            print(f"[*] Grid shape: {grid.shape}")
            print(f"[*] Grid range: [{grid.min().item():.3f}, {grid.max().item():.3f}]")
            print(f"[*] Theta shape: {theta.shape}")
            print(f"[*] Theta range: [{theta.min().item():.3f}, {theta.max().item():.3f}]")
            
            # Check if grid is valid
            if torch.any(torch.abs(grid) > 2.0):
                print("[WARN] Grid values seem too large, clamping...")
                grid = torch.clamp(grid, -1, 1)
            
            # Warp cloth
            warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
            print(f"[*] Warped cloth: {warped_cloth.shape}")
            
            # Save results
            original_cloth_img = tensor_to_image(cloth_tensor)
            warped_cloth_img = tensor_to_image(warped_cloth)
            
            original_cloth_img.save('gmm_original.jpg')
            warped_cloth_img.save('gmm_warped.jpg')
            
            # Also save grid visualization
            # Convert grid to image for visualization
            grid_viz = grid[0].permute(1, 2, 0)  # [H, W, 2]
            grid_viz = (grid_viz + 1) * 127.5  # Convert from [-1,1] to [0,255]
            grid_viz = grid_viz.cpu().numpy().astype(np.uint8)
            grid_img = Image.fromarray(grid_viz)
            grid_img.save('gmm_grid.jpg')
            
            print("[*] Saved:")
            print("  - gmm_test_person.jpg (input person)")
            print("  - gmm_test_cloth.jpg (input cloth)")
            print("  - gmm_original.jpg (cloth before warping)")
            print("  - gmm_warped.jpg (cloth after warping)")
            print("  - gmm_grid.jpg (warping grid visualization)")
            
            # Check if warping actually changed anything
            diff = torch.abs(cloth_tensor - warped_cloth).mean()
            print(f"[*] Average difference after warping: {diff.item():.6f}")
            
            if diff.item() < 0.001:
                print("[WARN] Warping didn't change the cloth much!")
            else:
                print("[OK] Warping produced visible changes")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] GMM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_gmm_only()
    if success:
        print("\n[SUCCESS] GMM test completed!")
    else:
        print("\n[FAILED] GMM test failed!")
