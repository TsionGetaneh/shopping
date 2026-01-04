import cv2
import numpy as np
from PIL import Image
import os
import sys

def force_clothing_replacement(person_img, cloth_img):
    """FORCE CLOTHING REPLACEMENT - aggressive approach that works"""
    w, h = person_img.size
    
    # Convert to arrays
    person_array = np.array(person_img)
    cloth_array = np.array(cloth_img)
    
    # Start with person
    result = person_array.copy().astype(np.float32)
    
    # PROTECT FACE - top 20% never changes
    face_end = h // 5
    result[:face_end, :] = person_array[:face_end, :]
    
    # Define torso area - be very aggressive
    torso_start = face_end + 5
    torso_end = 4 * h // 5
    
    # Resize cloth to fit torso - make it even bigger
    torso_height = torso_end - torso_start
    cloth_resized = cv2.resize(cloth_array, (w, torso_height))
    
    # Put cloth on person - center it
    start_y = torso_start
    
    # VERY AGGRESSIVE REPLACEMENT - replace everything in torso
    for y in range(torso_start, torso_end):
        cloth_y = y - start_y
        if cloth_y < cloth_resized.shape[0]:
            for x in range(w):
                cloth_x = x
                if cloth_x < cloth_resized.shape[1]:
                    cloth_pixel = cloth_resized[cloth_y, cloth_x]
                    
                    # VERY AGGRESSIVE - replace almost everything
                    # Only skip pure white (255, 255, 255)
                    if not (cloth_pixel[0] == 255 and cloth_pixel[1] == 255 and cloth_pixel[2] == 255):
                        result[y, x] = cloth_pixel
    
    # PROTECT FACE AGAIN - double protection
    result[:face_end, :] = person_array[:face_end, :]
    
    return Image.fromarray(result.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """Force clothing replacement that actually works"""
    try:
        print(f"Starting FORCE clothing replacement...")
        
        # Load images
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        
        print(f"Person size: {person_img.size}")
        print(f"Cloth size: {cloth_img.size}")
        
        # Force replacement
        result = force_clothing_replacement(person_img, cloth_img)
        
        # Save result
        result.save(output_path, 'JPEG', quality=95)
        print(f"Result saved to: {output_path}")
        print("FORCE clothing replacement completed!")
        return output_path
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original person
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
