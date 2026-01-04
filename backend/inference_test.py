import cv2
import numpy as np
from PIL import Image
import os
import sys

def create_test_clothing():
    """Create a test clothing pattern to ensure something appears"""
    # Create a simple colored rectangle as test clothing
    clothing = np.zeros((200, 300, 3), dtype=np.uint8)
    clothing[:, :] = [50, 100, 200]  # Blue color
    
    # Add some pattern
    for i in range(0, 200, 20):
        clothing[i:i+10, :] = [70, 120, 220]  # Lighter blue stripes
    
    return clothing

def force_test_clothing(person_path, cloth_path, output_path):
    """Force test clothing to appear on person"""
    
    # Load person image
    person_img = Image.open(person_path).convert("RGB")
    w, h = person_img.size
    
    person = np.array(person_img)
    result = person.copy()
    
    # Try to load cloth image
    try:
        cloth_img = Image.open(cloth_path).convert("RGB")
        cloth = np.array(cloth_img)
        print(f"Loaded cloth: {cloth.shape}")
        
        # Try to extract non-white pixels
        gray = cv2.cvtColor(cloth, cv2.COLOR_RGB2GRAY)
        mask = gray < 240
        print(f"Cloth mask pixels: {np.sum(mask)}")
        
        if np.sum(mask) > 100:
            # Use actual cloth
            cloth_to_use = cloth
        else:
            print("Using test clothing - cloth has too many white pixels")
            cloth_to_use = create_test_clothing()
            
    except Exception as e:
        print(f"Error loading cloth: {e}")
        print("Using test clothing")
        cloth_to_use = create_test_clothing()
    
    # Define torso area
    face_end = h // 5
    torso_start = face_end + 10
    torso_end = 4 * h // 5
    
    # Resize clothing to fit torso
    torso_height = torso_end - torso_start
    cloth_resized = cv2.resize(cloth_to_use, (w, torso_height))
    
    # FORCE APPLY - just put it on person
    pixels_changed = 0
    for y in range(torso_start, torso_end):
        cloth_y = y - torso_start
        if cloth_y < cloth_resized.shape[0]:
            for x in range(w):
                cloth_x = x
                if cloth_x < cloth_resized.shape[1]:
                    cloth_pixel = cloth_resized[cloth_y, cloth_x]
                    
                    # Check if pixel has some color (not pure white/black)
                    if np.mean(cloth_pixel) > 10 and np.mean(cloth_pixel) < 245:
                        result[y, x] = cloth_pixel
                        pixels_changed += 1
    
    print(f"FORCE APPLIED: {pixels_changed} pixels changed")
    
    # Protect face
    result[:face_end, :] = person[:face_end, :]
    
    # Save result
    result_img = Image.fromarray(result)
    result_img.save(output_path, 'JPEG', quality=95)
    
    return output_path

def generate_tryon(person_path, cloth_path, output_path):
    """Force test clothing to ensure something appears"""
    try:
        print("=== FORCE TEST CLOTHING ===")
        print(f"Person: {person_path}")
        print(f"Cloth: {cloth_path}")
        print(f"Output: {output_path}")
        
        # Check files
        if not os.path.exists(person_path):
            print(f"ERROR: Person file not found: {person_path}")
            return person_path
        
        # Force test clothing
        result = force_test_clothing(person_path, cloth_path, output_path)
        
        print(f"Result: {result}")
        print("=== FORCE TEST COMPLETED ===")
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
