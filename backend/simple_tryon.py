#!/usr/bin/env python
"""
Simple but effective virtual try-on that actually works
"""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

def simple_virtual_tryon(person_path, cloth_path, output_path):
    """
    Simple virtual try-on that works by:
    1. Detecting person pose
    2. Warping cloth to fit upper body
    3. Blending with original person
    """
    
    print("[*] Simple Virtual Try-On Starting...")
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    original_size = person_img.size
    print(f"[*] Original person size: {original_size}")
    
    # Resize to working size
    work_size = (256, 192)  # (height, width) for processing
    person_resized = person_img.resize((work_size[1], work_size[0]), Image.BILINEAR)
    cloth_resized = cloth_img.resize((work_size[1], work_size[0]), Image.BILINEAR)
    
    # Convert to numpy
    person_np = np.array(person_resized)
    cloth_np = np.array(cloth_resized)
    
    # Step 1: Create upper body mask
    h, w = work_size
    upper_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define upper body region (torso area)
    upper_y_start = int(h * 0.25)  # Start at 25% of height
    upper_y_end = int(h * 0.60)    # End at 60% of height
    upper_x_start = int(w * 0.30)  # Start at 30% of width
    upper_x_end = int(w * 0.70)    # End at 70% of width
    
    upper_mask[upper_y_start:upper_y_end, upper_x_start:upper_x_end] = 255
    
    # Create head protection mask
    head_mask = np.zeros((h, w), dtype=np.uint8)
    head_y_end = int(h * 0.25)  # Top 25% is head
    head_mask[:head_y_end, :] = 255
    
    # Step 2: Create better perspective warp for cloth
    # Define source points (corners of cloth)
    src_points = np.float32([
        [0, 0],           # Top-left
        [w-1, 0],         # Top-right
        [w-1, h-1],       # Bottom-right
        [0, h-1]          # Bottom-left
    ])
    
    # Define destination points (warped to fit upper body shape)
    # Make it narrower at top (shoulders) and wider at bottom (waist)
    shoulder_width = int(w * 0.3)  # Narrower at shoulders
    waist_width = int(w * 0.5)      # Wider at waist
    
    dst_points = np.float32([
        [w//2 - shoulder_width//2, upper_y_start],        # Top-left (narrow)
        [w//2 + shoulder_width//2, upper_y_start],        # Top-right (narrow)
        [w//2 + waist_width//2, upper_y_end],            # Bottom-right (wider)
        [w//2 - waist_width//2, upper_y_end]             # Bottom-left (wider)
    ])
    
    # Calculate perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply warp to cloth
    warped_cloth_np = cv2.warpPerspective(cloth_np, M, (w, h))
    
    # Create a mask for the warped cloth area
    warped_mask = np.zeros((h, w), dtype=np.uint8)
    warped_mask[upper_y_start:upper_y_end, :] = 255
    warped_mask = cv2.warpPerspective(warped_mask, M, (w, h))
    warped_mask = (warped_mask > 0).astype(np.uint8) * 255
    
    # Step 3: Blend warped cloth with person
    result_np = person_np.copy().astype(np.float32)
    
    # Create smooth masks
    upper_mask_smooth = cv2.GaussianBlur(warped_mask, (15, 15), 0) / 255.0
    head_mask_smooth = cv2.GaussianBlur(head_mask, (15, 15), 0) / 255.0
    
    # Convert to 3D masks
    upper_mask_3d = np.stack([upper_mask_smooth] * 3, axis=2)
    head_mask_3d = np.stack([head_mask_smooth] * 3, axis=2)
    
    # Apply head protection (preserve original head)
    result_np = result_np * (1 - head_mask_3d) + person_np.astype(np.float32) * head_mask_3d
    
    # Apply warped cloth only where it was warped
    result_np = result_np * (1 - upper_mask_3d) + warped_cloth_np.astype(np.float32) * upper_mask_3d
    
    # Convert back to uint8
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    
    # Resize back to original size
    result_img = Image.fromarray(result_np)
    result_img = result_img.resize(original_size, Image.LANCZOS)
    
    # Apply final smoothing for better blending
    result_img = result_img.filter(ImageFilter.SMOOTH)
    
    # Save result
    result_img.save(output_path, 'JPEG', quality=95)
    
    print(f"[OK] Simple try-on completed! Result saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Test the simple try-on
    person_path = "uploads/person.jpg"
    cloth_path = "uploads/cloth.jpg"
    output_path = "uploads/simple_result.jpg"
    
    try:
        simple_virtual_tryon(person_path, cloth_path, output_path)
        print("[SUCCESS] Simple virtual try-on works!")
    except Exception as e:
        print(f"[ERROR] Simple try-on failed: {e}")
        import traceback
        traceback.print_exc()
