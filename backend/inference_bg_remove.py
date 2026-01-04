import cv2
import numpy as np
from PIL import Image
import os
import sys

def remove_white_background(cloth_img):
    """Remove white background from cloth image"""
    cloth_array = np.array(cloth_img)
    
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2HSV)
    
    # Create mask for non-white pixels
    # Method 1: Grayscale threshold
    mask1 = gray < 240
    
    # Method 2: HSV threshold
    mask2 = (hsv[:, :, 1] > 20) | (hsv[:, :, 2] < 240)
    
    # Method 3: RGB threshold
    mask3 = (cloth_array[:, :, 0] < 240) | (cloth_array[:, :, 1] < 240) | (cloth_array[:, :, 2] < 240)
    
    # Combine all methods
    final_mask = mask1 & (mask2 | mask3)
    
    # Apply mask to cloth
    cloth_no_bg = cloth_array.copy()
    cloth_no_bg[~final_mask] = [0, 0, 0]  # Set background to black
    
    return cloth_no_bg, final_mask

def apply_clothing_to_person(person_path, cloth_path, output_path):
    """Apply clothing to person with proper background removal"""
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    w, h = person_img.size
    
    # Convert to arrays
    person = np.array(person_img)
    cloth = np.array(cloth_img)
    
    # Remove white background from cloth
    cloth_no_bg, cloth_mask = remove_white_background(cloth_img)
    
    print(f"Cloth mask pixels: {np.sum(cloth_mask)}")
    
    # Create result
    result = person.copy()
    
    # Define clothing area (torso)
    face_end = h // 5  # Protect face
    torso_start = face_end + 5
    torso_end = 4 * h // 5
    
    # Resize cloth to fit torso area
    torso_height = torso_end - torso_start
    cloth_resized = cv2.resize(cloth_no_bg, (w, torso_height))
    mask_resized = cv2.resize(cloth_mask.astype(np.uint8), (w, torso_height))
    
    # Apply clothing with proper masking
    pixels_applied = 0
    for y in range(torso_start, torso_end):
        cloth_y = y - torso_start
        if cloth_y < cloth_resized.shape[0]:
            for x in range(w):
                cloth_x = x
                if cloth_x < cloth_resized.shape[1]:
                    # Check if this pixel has clothing (not background)
                    if mask_resized[cloth_y, cloth_x] > 0:
                        cloth_pixel = cloth_resized[cloth_y, cloth_x]
                        
                        # Only apply if it's not black (background)
                        if np.mean(cloth_pixel) > 10:
                            result[y, x] = cloth_pixel
                            pixels_applied += 1
    
    print(f"Applied {pixels_applied} clothing pixels")
    
    # Protect face
    result[:face_end, :] = person[:face_end, :]
    
    # Save result
    result_img = Image.fromarray(result)
    result_img.save(output_path, 'JPEG', quality=95)
    
    return output_path

def generate_tryon(person_path, cloth_path, output_path):
    """Apply clothing with proper background removal"""
    try:
        print("=== APPLY CLOTHING WITH BG REMOVAL ===")
        print(f"Person: {person_path}")
        print(f"Cloth: {cloth_path}")
        print(f"Output: {output_path}")
        
        # Check files
        if not os.path.exists(person_path):
            print(f"ERROR: Person file not found: {person_path}")
            return person_path
        if not os.path.exists(cloth_path):
            print(f"ERROR: Cloth file not found: {cloth_path}")
            return person_path
        
        # Load images
        person_img = Image.open(person_path)
        cloth_img = Image.open(cloth_img)
        
        print(f"Person size: {person_img.size}")
        print(f"Cloth size: {cloth_img.size}")
        
        # Apply clothing
        result = apply_clothing_to_person(person_path, cloth_path, output_path)
        
        print(f"Result: {result}")
        print("=== CLOTHING APPLICATION COMPLETED ===")
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
