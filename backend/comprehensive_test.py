#!/usr/bin/env python
"""
Comprehensive test to debug the CP-VTON pipeline step by step
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

def create_realistic_test():
    """Create realistic test images"""
    # Person: realistic body shape
    person_img = Image.new('RGB', (192, 256), 'white')
    draw = ImageDraw.Draw(person_img)
    
    # Draw realistic person
    # Head
    draw.ellipse([76, 10, 116, 50], fill='peachpuff', outline='black')
    # Neck
    draw.rectangle([90, 45, 102, 60], fill='peachpuff')
    # Shoulders and torso
    draw.rectangle([70, 55, 122, 140], fill='lightblue', outline='black')
    # Arms
    draw.rectangle([55, 65, 70, 130], fill='peachpuff', outline='black')   # Left arm
    draw.rectangle([122, 65, 137, 130], fill='peachpuff', outline='black')  # Right arm
    # Waist
    draw.rectangle([80, 135, 112, 160], fill='lightblue', outline='black')
    # Hips
    draw.rectangle([75, 155, 117, 180], fill='lightblue', outline='black')
    # Legs
    draw.rectangle([82, 175, 95, 240], fill='darkblue', outline='black')   # Left leg
    draw.rectangle([97, 175, 110, 240], fill='darkblue', outline='black')  # Right leg
    
    # Cloth: t-shirt with clear pattern
    cloth_img = Image.new('RGB', (192, 256), 'white')
    draw = ImageDraw.Draw(cloth_img)
    
    # Draw t-shirt shape
    draw.rectangle([60, 40, 132, 170], fill='red', outline='darkred')  # Main body
    draw.rectangle([40, 60, 60, 120], fill='red', outline='darkred')   # Left sleeve
    draw.rectangle([132, 60, 152, 120], fill='red', outline='darkred')  # Right sleeve
    # Neck hole
    draw.ellipse([85, 35, 105, 55], fill='white')
    
    # Add grid pattern to see warping clearly
    for x in range(60, 132, 8):
        draw.line([x, 40, x, 170], fill='darkred', width=1)
    for y in range(40, 170, 8):
        draw.line([60, y, 132, y], fill='darkred', width=1)
    
    return person_img, cloth_img

def comprehensive_test():
    """Run comprehensive test of the entire pipeline"""
    print("[*] Comprehensive CP-VTON Pipeline Test")
    print("=" * 50)
    
    # Create test images
    person_img, cloth_img = create_realistic_test()
    
    # Save inputs
    person_img.save('comp_person.jpg')
    cloth_img.save('comp_cloth.jpg')
    print("[*] Created and saved test images")
    
    # Load models
    print("\n[*] Loading models...")
    GMM_model, TOM_model = load_models()
    
    if GMM_model is None:
        print("[ERROR] GMM model not loaded!")
        return False
    if TOM_model is None:
        print("[ERROR] TOM model not loaded!")
        return False
    
    print("[OK] Both models loaded successfully")
    
    try:
        # Step 1: Preprocessing
        print("\n[*] Step 1: Preprocessing...")
        person_tensor = preprocess_image(person_img)
        cloth_tensor = preprocess_image(cloth_img)
        print(f"    Person tensor: {person_tensor.shape}")
        print(f"    Cloth tensor: {cloth_tensor.shape}")
        
        # Step 2: Pose and Parsing
        print("\n[*] Step 2: Creating pose and parsing...")
        pose_data = create_simple_pose(person_img)
        parse_mask = create_simple_parsing(person_img)
        
        # Save pose visualization
        pose_keypoints = np.array(pose_data['people'][0]['pose_keypoints']).reshape(-1, 3)
        pose_viz = person_img.copy()
        draw = ImageDraw.Draw(pose_viz)
        for i, (x, y, conf) in enumerate(pose_keypoints):
            if conf > 0.5:
                draw.ellipse([x-2, y-2, x+2, y+2], fill='blue')
        pose_viz.save('comp_pose.jpg')
        
        # Save parsing mask
        parse_mask.save('comp_parse.jpg')
        print("    Saved pose visualization and parsing mask")
        
        # Step 3: Agnostic Representation
        print("\n[*] Step 3: Creating agnostic representation...")
        agnostic = create_agnostic(person_img, pose_data, parse_mask, person_tensor)
        print(f"    Agnostic shape: {agnostic.shape}")
        
        # Visualize agnostic (first few channels)
        agnostic_viz = tensor_to_image(agnostic[:, 0:3, :, :])
        agnostic_viz.save('comp_agnostic.jpg')
        print("    Saved agnostic visualization")
        
        # Step 4: GMM Warping
        print("\n[*] Step 4: GMM Warping...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agnostic = agnostic.to(device)
        cloth_tensor = cloth_tensor.to(device)
        
        # GMM expects 1 channel cloth input, convert RGB to grayscale
        cloth_tensor_1ch = cloth_tensor[:, 0:1, :, :]  # Take only first channel
        
        with torch.no_grad():
            grid, theta = GMM_model(agnostic, cloth_tensor_1ch)
            print(f"    Grid shape: {grid.shape}")
            print(f"    Grid range: [{grid.min().item():.3f}, {grid.max().item():.3f}]")
            print(f"    Theta shape: {theta.shape}")
            
            # Check grid validity
            invalid_grid = torch.sum(torch.abs(grid) > 2.0).item()
            if invalid_grid > 0:
                print(f"    [WARN] {invalid_grid} invalid grid points, clamping...")
                grid = torch.clamp(grid, -1, 1)
            
            # Warp cloth (use 3-channel cloth for warping, but 1-channel for GMM)
            warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
            print(f"    Warped cloth shape: {warped_cloth.shape}")
            
            # Save warped cloth
            warped_img = tensor_to_image(warped_cloth)
            warped_img.save('comp_warped.jpg')
            
            # Check warping effectiveness
            diff = torch.abs(cloth_tensor - warped_cloth).mean()
            print(f"    Warping difference: {diff.item():.6f}")
            
            if diff.item() < 0.01:
                print("    [WARN] Minimal warping detected!")
            else:
                print("    [OK] Significant warping detected")
        
        # Step 5: TOM Processing
        print("\n[*] Step 5: TOM Processing...")
        with torch.no_grad():
            # Prepare TOM input
            tom_input = torch.cat([agnostic, warped_cloth], 1)
            print(f"    TOM input shape: {tom_input.shape}")
            
            # Add dummy channel if needed
            if tom_input.shape[1] == 25:
                dummy_channel = torch.zeros_like(tom_input[:, 0:1, :, :])
                tom_input = torch.cat([tom_input, dummy_channel], 1)
                print(f"    TOM input shape after dummy: {tom_input.shape}")
            
            # Run TOM
            outputs = TOM_model(tom_input)
            p_rendered, m_composite = torch.split(outputs, 3, 1)
            
            print(f"    p_rendered shape: {p_rendered.shape}")
            print(f"    m_composite shape: {m_composite.shape}")
            
            # Apply activations
            p_rendered = F.tanh(p_rendered)
            m_composite = F.sigmoid(m_composite)
            
            # Save intermediate results
            p_rendered_img = tensor_to_image(p_rendered)
            m_composite_img = tensor_to_image(m_composite.repeat(1, 3, 1, 1))
            
            p_rendered_img.save('comp_rendered.jpg')
            m_composite_img.save('comp_mask.jpg')
            print("    Saved TOM intermediate results")
        
        # Step 6: Final Composition
        print("\n[*] Step 6: Final Composition...")
        with torch.no_grad():
            # Final blend
            result_tensor = warped_cloth * m_composite + p_rendered * (1 - m_composite)
            
            # Convert and save
            result_img = tensor_to_image(result_tensor)
            result_img = result_img.resize(person_img.size, Image.LANCZOS)
            result_img.save('comp_result.jpg')
            
            print("    Saved final result")
        
        # Summary
        print("\n" + "=" * 50)
        print("[SUCCESS] Comprehensive test completed!")
        print("\nGenerated files:")
        print("  Inputs:")
        print("    - comp_person.jpg (person image)")
        print("    - comp_cloth.jpg (cloth image)")
        print("  Intermediate:")
        print("    - comp_pose.jpg (pose keypoints)")
        print("    - comp_parse.jpg (parsing mask)")
        print("    - comp_agnostic.jpg (agnostic representation)")
        print("    - comp_warped.jpg (warped cloth)")
        print("    - comp_rendered.jpg (TOM rendered)")
        print("    - comp_mask.jpg (composite mask)")
        print("  Final:")
        print("    - comp_result.jpg (final try-on result)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = comprehensive_test()
    if success:
        print("\n[SUCCESS] All tests completed!")
    else:
        print("\n[FAILED] Tests failed!")
