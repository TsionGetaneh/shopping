import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import sys

def create_body_conforming_warp(person_img, cloth_img):
    """
    Create realistic cloth fitting that actually conforms to body shape
    """
    w, h = person_img.size
    
    # Convert to arrays
    person_array = np.array(person_img)
    cloth_array = np.array(cloth_img.resize((w, h), Image.LANCZOS))
    
    # Create the warped cloth using perspective transformation
    # This will make the cloth actually fit the body shape
    
    # Define source points (corners of cloth)
    src_points = np.float32([
        [0, 0],           # Top-left
        [w, 0],           # Top-right
        [w, h],           # Bottom-right
        [0, h]            # Bottom-left
    ])
    
    # Define destination points that create realistic body shape
    # This is the key - make cloth narrow at top, wider at bottom
    center_x = w // 2
    
    # Calculate realistic body proportions
    neck_width = w // 6
    shoulder_width = w // 2
    chest_width = w // 2 + w // 8
    waist_width = w // 3
    hip_width = w // 2 + w // 6
    
    # Create destination points for realistic clothing fit
    dst_points = np.float32([
        # Top corners - narrow at neck/shoulders
        [center_x - shoulder_width//2, h//4],      # Top-left (shoulder level)
        [center_x + shoulder_width//2, h//4],      # Top-right (shoulder level)
        
        # Bottom corners - wider at hips
        [center_x - hip_width//2, 3*h//4],        # Bottom-left (hip level)
        [center_x + hip_width//2, 3*h//4],        # Bottom-right (hip level)
    ])
    
    print(f"[*] Creating body-conforming warp: shoulders={shoulder_width}, hips={hip_width}")
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply perspective warp to cloth
    warped_cloth = cv2.warpPerspective(cloth_array, M, (w, h))
    
    # Create mask for warped cloth
    mask = np.zeros((h, w), dtype=np.uint8)
    mask.fill(255)
    warped_mask = cv2.warpPerspective(mask, M, (w, h))
    
    return warped_cloth, warped_mask

def create_advanced_body_fit(person_img, cloth_img):
    """
    Advanced method that creates realistic clothing fit
    """
    w, h = person_img.size
    
    # First try perspective warp
    warped_cloth, mask = create_body_conforming_warp(person_img, cloth_img)
    
    # Enhance the warp with additional processing
    # Add subtle curves for more realistic fit
    
    # Create vertical gradient for additional shaping
    person_array = np.array(person_img)
    
    # Find the person's silhouette (simple approach)
    gray = cv2.cvtColor(person_array, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours to get person outline
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (person)
        person_contour = max(contours, key=cv2.contourArea)
        x, y, w_rect, h_rect = cv2.boundingRect(person_contour)
        
        # Create additional warping based on detected person
        center_x = x + w_rect // 2
        
        # Apply additional refinement to make cloth fit better
        refined_cloth = warped_cloth.copy()
        
        # Create horizontal compression at waist level
        waist_y = y + h_rect // 2
        compression_factor = 0.8  # Compress waist by 20%
        
        for y_pos in range(max(0, waist_y - 20), min(h, waist_y + 20)):
            # Calculate compression amount based on distance from waist
            distance = abs(y_pos - waist_y)
            if distance < 20:
                local_compression = 1 - (1 - compression_factor) * (1 - distance / 20)
                
                # Apply horizontal compression
                left_bound = max(0, int(center_x - w_rect // 2 * local_compression))
                right_bound = min(w, int(center_x + w_rect // 2 * local_compression))
                
                if right_bound > left_bound:
                    # Resize this row to be narrower
                    original_section = warped_cloth[y_pos, left_bound:right_bound]
                    if original_section.size > 0:
                        # Stretch to fit original width but compressed
                        target_width = right_bound - left_bound
                        # Handle RGB channels properly
                        if len(original_section.shape) == 2 and original_section.shape[1] == 3:
                            # RGB format - resize to (target_width, 3)
                            resized_section = cv2.resize(original_section, (target_width, 3))
                            refined_cloth[y_pos, left_bound:right_bound] = resized_section
                        elif len(original_section.shape) == 1:
                            # Single channel - convert to RGB
                            rgb_section = np.stack([original_section] * 3, axis=1)
                            resized_section = cv2.resize(rgb_section, (target_width, 3))
                            refined_cloth[y_pos, left_bound:right_bound] = resized_section
                        else:
                            # Handle other formats - ensure correct shape
                            if original_section.shape[0] == 3 and len(original_section.shape) == 1:
                                # Shape (3,) - reshape to (width, 1) then resize
                                reshaped = original_section.reshape(-1, 1)
                                resized_section = cv2.resize(reshaped, (target_width, 3))
                                refined_cloth[y_pos, left_bound:right_bound] = resized_section
                            else:
                                # Fallback - simple approach
                                resized_section = cv2.resize(original_section, (target_width, original_section.shape[-1]))
                                if resized_section.shape == (target_width, 3):
                                    refined_cloth[y_pos, left_bound:right_bound] = resized_section
        
        print("[*] Applied waist compression for better fit")
        return refined_cloth, mask
    
    return warped_cloth, mask

def blend_realistically(person_img, warped_cloth, mask):
    """
    Realistic blending that looks natural
    """
    w, h = person_img.size
    
    # Convert to arrays
    person_array = np.array(person_img).astype(np.float32)
    cloth_array = warped_cloth.astype(np.float32)
    mask_array = mask.astype(np.float32) / 255.0
    
    # Create precise masks
    head_mask = np.zeros((h, w), dtype=np.float32)
    head_mask[:h//5, :] = 1.0  # Top 20% is head
    
    # Upper body mask for clothing
    upper_mask = np.zeros((h, w), dtype=np.float32)
    upper_mask[h//5:4*h//5, :] = 1.0  # Upper 60% for clothing
    
    # Smooth the masks for natural blending
    kernel = np.ones((5, 5), np.float32) / 25
    upper_mask = cv2.filter2D(upper_mask, -1, kernel)
    
    # Convert to 3D
    head_mask_3d = np.stack([head_mask] * 3, axis=2)
    upper_mask_3d = np.stack([upper_mask] * 3, axis=2)
    cloth_mask_3d = np.stack([mask_array] * 3, axis=2)
    
    # Realistic blending with proper masking
    final_array = (
        person_array * head_mask_3d +  # Keep original head
        cloth_array * (1 - head_mask_3d) * upper_mask_3d * cloth_mask_3d +  # Apply cloth to upper body
        person_array * (1 - head_mask_3d) * (1 - upper_mask_3d)  # Keep original lower body
    )
    
    # Add subtle blur for realism
    final_array = cv2.GaussianBlur(final_array, (3, 3), 0)
    
    return Image.fromarray(final_array.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """
    Main virtual try-on function with realistic cloth fitting
    """
    print(f"[*] Virtual try-on: {person_path} + {cloth_path} -> {output_path}")
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    print(f"[*] Processing: person={person_img.size}, cloth={cloth_img.size}")
    
    # Resize cloth to match person
    cloth_resized = cloth_img.resize(person_img.size, Image.LANCZOS)
    
    # Create advanced body fit
    warped_cloth, mask = create_advanced_body_fit(person_img, cloth_resized)
    
    if warped_cloth is not None:
        # Blend realistically
        result_image = blend_realistically(person_img, warped_cloth, mask)
        print("[OK] Realistic clothing fit created successfully")
    else:
        print("[ERROR] Failed to create clothing fit")
        result_image = person_img
    
    # Save result
    if result_image.mode != 'RGB':
        result_image = result_image.convert('RGB')
    
    result_image.save(output_path, 'JPEG', quality=95)
    print(f"[OK] Result saved to {output_path}")
    return output_path
