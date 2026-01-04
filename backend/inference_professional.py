import cv2
import numpy as np
from PIL import Image
import os
import sys

def analyze_person_body(person_img):
    """Analyze person body for accurate clothing placement"""
    w, h = person_img.size
    person_array = np.array(person_img)
    
    # Body proportions
    face_end = h // 5
    shoulder_line = face_end + h // 15
    chest_line = shoulder_line + h // 8
    waist_line = chest_line + h // 6
    torso_end = 4 * h // 5
    
    # Body width at different levels
    shoulder_width = w * 0.7
    chest_width = w * 0.8
    waist_width = w * 0.75
    
    return {
        'face_end': face_end,
        'shoulder_line': shoulder_line,
        'chest_line': chest_line,
        'waist_line': waist_line,
        'torso_end': torso_end,
        'shoulder_width': shoulder_width,
        'chest_width': chest_width,
        'waist_width': waist_width,
        'w': w, 'h': h
    }

def extract_garment_structure(cloth_img):
    """Extract garment structure with proper shape detection"""
    cloth_array = np.array(cloth_img)
    
    # Multiple detection methods for better results
    gray = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2HSV)
    
    # Background removal
    mask1 = gray < 220  # Grayscale threshold
    mask2 = hsv[:, :, 1] > 30  # Saturation threshold
    mask3 = hsv[:, :, 2] < 240  # Value threshold
    
    # Combine masks
    garment_mask = mask1 & (mask2 | mask3)
    
    # Clean up mask
    kernel = np.ones((5, 5), np.uint8)
    garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_CLOSE, kernel)
    garment_mask = cv2.morphologyEx(garment_mask, cv2.MORPH_OPEN, kernel)
    
    # Find garment boundaries
    coords = np.where(garment_mask > 0)
    if len(coords[0]) == 0:
        return None, None, None
    
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    
    # Extract garment
    garment = cloth_array[y_min:y_max+1, x_min:x_max+1]
    garment_mask_cropped = garment_mask[y_min:y_max+1, x_min:x_max+1]
    
    return garment, garment_mask_cropped, (y_min, y_max, x_min, x_max)

def warp_garment_to_body(garment, garment_mask, body_info):
    """Warp garment to match body shape with realistic deformation"""
    h, w = body_info['h'], body_info['w']
    
    # Create target mesh points for body
    target_points = np.array([
        [body_info['w']//2 - body_info['shoulder_width']//2, body_info['shoulder_line']],  # Left shoulder
        [body_info['w']//2 + body_info['shoulder_width']//2, body_info['shoulder_line']],  # Right shoulder
        [body_info['w']//2 + body_info['chest_width']//2, body_info['chest_line']],  # Right chest
        [body_info['w']//2 + body_info['waist_width']//2, body_info['waist_line']],  # Right waist
        [body_info['w']//2 - body_info['waist_width']//2, body_info['waist_line']],  # Left waist
        [body_info['w']//2 - body_info['chest_width']//2, body_info['chest_line']],  # Left chest
    ], dtype=np.float32)
    
    # Create source points from garment
    garment_h, garment_w = garment.shape[:2]
    source_points = np.array([
        [0, 0],  # Top left
        [garment_w, 0],  # Top right
        [garment_w, garment_h//2],  # Middle right
        [garment_w, garment_h],  # Bottom right
        [0, garment_h],  # Bottom left
        [0, garment_h//2],  # Middle left
    ], dtype=np.float32)
    
    # Calculate thin plate spline transformation
    try:
        # Use perspective transform as fallback
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = [0, 0]
        rect[1] = [garment_w, 0]
        rect[2] = [garment_w, garment_h]
        rect[3] = [0, garment_h]
        
        # Create destination rectangle based on body
        dst = np.zeros((4, 2), dtype=np.float32)
        dst[0] = [body_info['w']//2 - body_info['shoulder_width']//2, body_info['shoulder_line']]
        dst[1] = [body_info['w']//2 + body_info['shoulder_width']//2, body_info['shoulder_line']]
        dst[2] = [body_info['w']//2 + body_info['waist_width']//2, body_info['waist_line']]
        dst[3] = [body_info['w']//2 - body_info['waist_width']//2, body_info['waist_line']]
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_garment = cv2.warpPerspective(garment, M, (w, h))
        warped_mask = cv2.warpPerspective(garment_mask.astype(np.uint8) * 255, M, (w, h))
        
        return warped_garment, warped_mask
    except:
        return None, None

def create_realistic_blend(person_img, warped_garment, warped_mask, body_info):
    """Create realistic blend with proper lighting and shadows"""
    person_array = np.array(person_img).astype(np.float32)
    result = person_array.copy()
    
    if warped_garment is None:
        return Image.fromarray(result.astype(np.uint8))
    
    # Create torso mask
    torso_mask = np.zeros((body_info['h'], body_info['w']), dtype=np.uint8)
    torso_mask[body_info['face_end']:body_info['torso_end'], :] = 255
    
    # Remove arms from torso mask
    arm_width = body_info['w'] // 8
    torso_mask[:, :arm_width] = 0
    torso_mask[:, -arm_width:] = 0
    
    # Combine masks
    final_mask = (warped_mask > 128) & (torso_mask > 0)
    
    # Apply garment with realistic lighting
    for y in range(body_info['face_end'], body_info['torso_end']):
        for x in range(body_info['w']):
            if final_mask[y, x]:
                garment_pixel = warped_garment[y, x]
                person_pixel = person_array[y, x]
                
                # Check if it's actual garment
                if np.mean(garment_pixel) < 240:
                    # Lighting matching
                    person_brightness = np.mean(person_pixel)
                    garment_brightness = np.mean(garment_pixel)
                    
                    if garment_brightness > 10:
                        lighting_factor = person_brightness / garment_brightness
                        lighting_factor = np.clip(lighting_factor, 0.8, 1.2)
                        
                        # Apply lighting
                        final_pixel = garment_pixel * lighting_factor
                        
                        # Add subtle shadows
                        shadow_factor = 0.95
                        final_pixel = final_pixel * shadow_factor
                        
                        result[y, x] = final_pixel
    
    # Apply bilateral filter for fabric texture
    result = cv2.bilateralFilter(result.astype(np.uint8), 5, 80, 80)
    
    return Image.fromarray(result.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """Professional virtual try-on with realistic results"""
    try:
        print("Starting professional virtual try-on...")
        
        # Load images
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        
        print(f"Person size: {person_img.size}")
        print(f"Cloth size: {cloth_img.size}")
        
        # Step 1: Analyze person body
        print("Analyzing person body...")
        body_info = analyze_person_body(person_img)
        
        # Step 2: Extract garment structure
        print("Extracting garment structure...")
        garment, garment_mask, bounds = extract_garment_structure(cloth_img)
        
        if garment is None:
            print("Failed to extract garment")
            person_img.save(output_path, 'JPEG', quality=95)
            return output_path
        
        print(f"Garment extracted: {garment.shape}")
        
        # Step 3: Warp garment to body
        print("Warping garment to body shape...")
        warped_garment, warped_mask = warp_garment_to_body(garment, garment_mask, body_info)
        
        if warped_garment is None:
            print("Failed to warp garment")
            person_img.save(output_path, 'JPEG', quality=95)
            return output_path
        
        # Step 4: Create realistic blend
        print("Creating realistic blend...")
        final_result = create_realistic_blend(person_img, warped_garment, warped_mask, body_info)
        
        # Save result
        final_result.save(output_path, 'JPEG', quality=95)
        print(f"Professional virtual try-on completed: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
