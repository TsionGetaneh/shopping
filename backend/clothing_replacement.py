#!/usr/bin/env python
"""
Clothing Replacement - Replace existing clothing with new clothing
"""
import torch 
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2

def clothing_replacement(person_path, cloth_path, output_path):
    """
    Replace existing clothing with new clothing
    """
    
    print("[*] Clothing Replacement Starting...")
    
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
    h, w = work_size
    
    # Step 1: Detect existing clothing area (upper body)
    # Use color analysis to find clothing regions
    clothing_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define upper body region where clothing typically is
    upper_y_start = int(h * 0.25)  # Start below head
    upper_y_end = int(h * 0.65)    # End at waist
    upper_x_start = int(w * 0.25)  # Start at 25% width
    upper_x_end = int(w * 0.75)    # End at 75% width
    
    # Create basic upper body mask
    upper_body_mask = np.zeros((h, w), dtype=np.uint8)
    upper_body_mask[upper_y_start:upper_y_end, upper_x_start:upper_x_end] = 255
    
    # Detect existing clothing by analyzing colors in upper body
    upper_body_region = person_np[upper_y_start:upper_y_end, upper_x_start:upper_x_end]
    
    # Simple clothing detection: look for non-skin colors in upper body
    # Skin typically has certain color ranges
    skin_lower = np.array([0, 20, 70], dtype=np.uint8)     # Lower HSV skin bound
    skin_upper = np.array([20, 255, 255], dtype=np.uint8)   # Upper HSV skin bound
    
    # Convert to HSV for better skin detection
    upper_body_hsv = cv2.cvtColor(upper_body_region, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(upper_body_hsv, skin_lower, skin_upper)
    
    # Invert to get clothing (non-skin) areas
    clothing_in_upper = cv2.bitwise_not(skin_mask)
    
    # Apply some morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    clothing_in_upper = cv2.morphologyEx(clothing_in_upper, cv2.MORPH_CLOSE, kernel)
    clothing_in_upper = cv2.morphologyEx(clothing_in_upper, cv2.MORPH_OPEN, kernel)
    
    # Place clothing mask in full image
    clothing_mask[upper_y_start:upper_y_end, upper_x_start:upper_x_end] = clothing_in_upper
    
    # Step 2: Create head protection mask
    head_mask = np.zeros((h, w), dtype=np.uint8)
    head_mask[:int(h * 0.25), :] = 255  # Top 25% is head
    
    # Step 3: Prepare new clothing
    # Extract the main pattern/color from new cloth
    # Take center region of cloth to avoid edges
    cloth_center_y = h // 2
    cloth_center_x = w // 2
    cloth_region_size = min(h, w) // 2
    
    cloth_pattern = cloth_np[
        cloth_center_y - cloth_region_size//2:cloth_center_y + cloth_region_size//2,
        cloth_center_x - cloth_region_size//2:cloth_center_x + cloth_region_size//2
    ]
    
    # Resize pattern to fit upper body area
    upper_body_height = upper_y_end - upper_y_start
    upper_body_width = upper_x_end - upper_x_start
    
    new_cloth_resized = cv2.resize(cloth_pattern, (upper_body_width, upper_body_height))
    
    # Step 4: Replace clothing
    result_np = person_np.copy().astype(np.float32)
    
    # Create smooth masks
    clothing_mask_smooth = cv2.GaussianBlur(clothing_mask, (5, 5), 0) / 255.0
    head_mask_smooth = cv2.GaussianBlur(head_mask, (15, 15), 0) / 255.0
    
    # Convert to 3D masks
    clothing_mask_3d = np.stack([clothing_mask_smooth] * 3, axis=2)
    head_mask_3d = np.stack([head_mask_smooth] * 3, axis=2)
    
    # Protect head area completely
    result_np = result_np * (1 - head_mask_3d) + person_np.astype(np.float32) * head_mask_3d
    
    # Replace clothing in detected areas
    # Place new clothing in the upper body region
    new_cloth_placement = np.zeros_like(result_np)
    new_cloth_placement[upper_y_start:upper_y_end, upper_x_start:upper_x_end] = new_cloth_resized
    
    # Apply new clothing only where old clothing was detected
    result_np = result_np * (1 - clothing_mask_3d) + new_cloth_placement.astype(np.float32) * clothing_mask_3d
    
    # Convert back to uint8
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    
    # Resize back to original size
    result_img = Image.fromarray(result_np)
    result_img = result_img.resize(original_size, Image.LANCZOS)
    
    # Apply final smoothing for better blending
    result_img = result_img.filter(ImageFilter.SMOOTH)
    
    # Save result
    result_img.save(output_path, 'JPEG', quality=95)
    
    print(f"[OK] Clothing replacement completed! Result saved to {output_path}")
    return output_path

if __name__ == "__main__":
    # Test the clothing replacement
    person_path = "uploads/person.jpg"
    cloth_path = "uploads/cloth.jpg"
    output_path = "uploads/replacement_result.jpg"
    
    try:
        clothing_replacement(person_path, cloth_path, output_path)
        print("[SUCCESS] Clothing replacement works!")
    except Exception as e:
        print(f"[ERROR] Clothing replacement failed: {e}")
        import traceback
        traceback.print_exc()
