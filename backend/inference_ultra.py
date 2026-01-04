import cv2
import numpy as np
from PIL import Image
import os
import sys

def ultra_aggressive_replacement(person_path, cloth_path, output_path):
    """ULTRA AGGRESSIVE - replace everything in torso area"""
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    w, h = person_img.size
    
    # Convert to arrays
    person = np.array(person_img)
    cloth = np.array(cloth_img)
    
    # Create result
    result = person.copy()
    
    # Protect face - top 20%
    face_end = h // 5
    result[:face_end, :] = person[:face_end, :]
    
    # Define torso area
    torso_start = face_end + 10
    torso_end = 4 * h // 5
    
    # Resize cloth to cover entire torso
    torso_height = torso_end - torso_start
    cloth_resized = cv2.resize(cloth, (w, torso_height))
    
    # REPLACE EVERYTHING in torso area
    for y in range(torso_start, min(torso_end, h)):
        cloth_y = y - torso_start
        if cloth_y < cloth_resized.shape[0]:
            for x in range(w):
                cloth_x = x
                if cloth_x < cloth_resized.shape[1]:
                    # Get cloth pixel
                    cloth_pixel = cloth_resized[cloth_y, cloth_x]
                    
                    # REPLACE unless pure white
                    if not (cloth_pixel[0] > 250 and cloth_pixel[1] > 250 and cloth_pixel[2] > 250):
                        result[y, x] = cloth_pixel
    
    # Double protect face
    result[:face_end, :] = person[:face_end, :]
    
    # Save result
    result_img = Image.fromarray(result)
    result_img.save(output_path, 'JPEG', quality=95)
    
    return output_path

def generate_tryon(person_path, cloth_path, output_path):
    """Ultra aggressive clothing replacement"""
    try:
        return ultra_aggressive_replacement(person_path, cloth_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
