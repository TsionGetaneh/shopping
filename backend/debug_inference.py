#!/usr/bin/env python
"""
Debug test script to see each stage of the CP-VTON pipeline
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

def create_test_images():
    """Create better test images for person and cloth"""
    # Create person image (more realistic proportions)
    person_img = Image.new('RGB', (512, 768), 'white')
    draw = ImageDraw.Draw(person_img)
    
    # Draw person with better proportions
    # Head
    draw.ellipse([200, 50, 312, 162], fill='peachpuff', outline='black')
    # Neck
    draw.rectangle([240, 160, 272, 180], fill='peachpuff')
    # Body/torso
    draw.rectangle([220, 180, 292, 380], fill='lightblue', outline='black')
    # Arms
    draw.rectangle([200, 200, 220, 360], fill='peachpuff', outline='black')  # Left arm
    draw.rectangle([292, 200, 312, 360], fill='peachpuff', outline='black')  # Right arm
    # Legs
    draw.rectangle([235, 380, 265, 600], fill='darkblue', outline='black')  # Left leg
    draw.rectangle([247, 380, 277, 600], fill='darkblue', outline='black')  # Right leg
    
    # Create cloth image (t-shirt shape)
    cloth_img = Image.new('RGB', (512, 768), 'white')
    draw = ImageDraw.Draw(cloth_img)
    
    # Draw t-shirt shape
    draw.rectangle([180, 180, 332, 400], fill='red', outline='darkred')  # Main body
    draw.rectangle([150, 180, 180, 280], fill='red', outline='darkred')  # Left sleeve
    draw.rectangle([332, 180, 362, 280], fill='red', outline='darkred')  # Right sleeve
    # Neck hole
    draw.ellipse([235, 180, 265, 210], fill='white')
    
    return person_img, cloth_img

def debug_inference():
    """Debug the inference pipeline step by step"""
    print("[*] Debug CP-VTON inference pipeline...")
    
    # Create test images
    person_img, cloth_img = create_test_images()
    
    # Save test images
    person_path = 'debug_person.jpg'
    cloth_path = 'debug_cloth.jpg'
    
    person_img.save(person_path)
    cloth_img.save(cloth_path)
    
    print(f"[*] Created test images: {person_path}, {cloth_path}")
    
    # Load models
    print("[*] Loading models...")
    GMM_model, TOM_model = load_models()
    
    if GMM_model is None or TOM_model is None:
        print("[ERROR] Models not loaded properly!")
        return False
    
    print("[OK] Models loaded successfully")
    
    try:
        # Resize to CP-VTON standard
        person_resized = person_img.resize((192, 256), Image.BILINEAR)
        cloth_resized = cloth_img.resize((192, 256), Image.BILINEAR)
        
        print(f"[*] Resized images to: {person_resized.size}")
        
        # Create pose and parsing
        pose_data = create_simple_pose(person_resized)
        parse_mask = create_simple_parsing(person_resized)
        
        # Convert to tensors
        person_tensor = preprocess_image(person_resized)
        cloth_tensor = preprocess_image(cloth_resized)
        
        print(f"[*] Person tensor shape: {person_tensor.shape}")
        print(f"[*] Cloth tensor shape: {cloth_tensor.shape}")
        
        # Create agnostic representation
        agnostic = create_agnostic(person_resized, pose_data, parse_mask, person_tensor)
        print(f"[*] Agnostic tensor shape: {agnostic.shape}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agnostic = agnostic.to(device)
        cloth_tensor_gpu = cloth_tensor.to(device)
        
        # ===== GMM STEP =====
        print("[*] Running GMM...")
        with torch.no_grad():
            grid, theta = GMM_model(agnostic, cloth_tensor_gpu)
            print(f"[*] GMM grid shape: {grid.shape}")
            print(f"[*] GMM theta shape: {theta.shape}")
            
            # Clamp grid to valid range
            grid = torch.clamp(grid, -1, 1)
            
            # Warp the cloth
            warped_cloth = F.grid_sample(cloth_tensor_gpu, grid, padding_mode='border', align_corners=False)
            print(f"[*] Warped cloth shape: {warped_cloth.shape}")
            
            # Save warped cloth for inspection
            warped_image = tensor_to_image(warped_cloth)
            warped_image.save('debug_warped_cloth.jpg')
            print("[*] Saved warped cloth to debug_warped_cloth.jpg")
        
        # ===== TOM STEP =====
        print("[*] Running TOM...")
        with torch.no_grad():
            # Combine agnostic + warped cloth
            tom_input = torch.cat([agnostic, warped_cloth], 1)
            print(f"[*] TOM input shape: {tom_input.shape}")
            
            # Add dummy channel if needed
            if tom_input.shape[1] == 25:
                dummy_channel = torch.zeros_like(tom_input[:, 0:1, :, :])
                tom_input = torch.cat([tom_input, dummy_channel], 1)
                print(f"[*] TOM input shape after adding dummy: {tom_input.shape}")
            
            # Run TOM
            outputs = TOM_model(tom_input)
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            
            print(f"[*] TOM p_rendered shape: {p_rendered.shape}")
            print(f"[*] TOM m_composite shape: {m_composite.shape}")
            
            # Apply activations
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            
            # Final composition
            result_tensor = warped_cloth * m_composite + p_rendered * (1 - m_composite)
            print(f"[*] Final result shape: {result_tensor.shape}")
            
            # Convert and save result
            result_image = tensor_to_image(result_tensor)
            result_image = result_image.resize(person_img.size, Image.LANCZOS)
            result_image.save('debug_result.jpg')
            print("[*] Saved final result to debug_result.jpg")
            
            # Also save intermediate results
            agnostic_vis = tensor_to_image(agnostic[:, 0:3, :, :])  # First 3 channels
            agnostic_vis = agnostic_vis.resize(person_img.size, Image.LANCZOS)
            agnostic_vis.save('debug_agnostic.jpg')
            print("[*] Saved agnostic representation to debug_agnostic.jpg")
            
            # Save mask
            mask_vis = tensor_to_image(m_composite.repeat(1, 3, 1, 1))
            mask_vis = mask_vis.resize(person_img.size, Image.LANCZOS)
            mask_vis.save('debug_mask.jpg')
            print("[*] Save composite mask to debug_mask.jpg")
        
        print("[SUCCESS] Debug completed! Check the debug_*.jpg files:")
        print("  - debug_person.jpg (input person)")
        print("  - debug_cloth.jpg (input cloth)")
        print("  - debug_warped_cloth.jpg (after GMM)")
        print("  - debug_agnostic.jpg (agnostic representation)")
        print("  - debug_mask.jpg (composite mask)")
        print("  - debug_result.jpg (final result)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup input files
        for path in [person_path, cloth_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    success = debug_inference()
    if success:
        print("\n[SUCCESS] Debug completed successfully!")
    else:
        print("\n[FAILED] Debug failed!")
