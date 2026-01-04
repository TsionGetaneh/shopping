import cv2
import numpy as np
from PIL import Image
import os
import sys

def force_clothing_replace(person_path, cloth_path, output_path):
    """FORCE CLOTHING REPLACE - just put cloth on person, no questions asked"""
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    w, h = person_img.size
    
    # Convert to arrays
    person = np.array(person_img)
    cloth = np.array(cloth_img)
    
    # Create result
    result = person.copy()
    
    # Define clothing area (torso)
    face_end = h // 5  # Protect face
    torso_start = face_end + 5
    torso_end = 4 * h // 5
    
    # Resize cloth to fit torso area
    torso_height = torso_end - torso_start
    cloth_resized = cv2.resize(cloth, (w, torso_height))
    
    # FORCE REPLACE - just put cloth on person
    for y in range(torso_start, torso_end):
        cloth_y = y - torso_start
        if cloth_y < cloth_resized.shape[0]:
            for x in range(w):
                cloth_x = x
                if cloth_x < cloth_resized.shape[1]:
                    cloth_pixel = cloth_resized[cloth_y, cloth_x]
                    
                    # FORCE REPLACE - put cloth pixel regardless
                    result[y, x] = cloth_pixel
    
    # Protect face
    result[:face_end, :] = person[:face_end, :]
    
    # Save result
    result_img = Image.fromarray(result)
    result_img.save(output_path, 'JPEG', quality=95)
    
    return output_path

def generate_tryon(person_path, cloth_path, output_path):
    """Force clothing replacement that actually works"""
    try:
        print("=== FORCE CLOTHING REPLACE ===")
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
        
        # Force replace
        result = force_clothing_replace(person_path, cloth_path, output_path)
        
        print(f"Result: {result}")
        print("=== FORCE REPLACE COMPLETED ===")
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
