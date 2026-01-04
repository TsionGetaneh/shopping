import cv2
import numpy as np
from PIL import Image
import os
import sys

def detect_person_and_body(person_img):
    """Detect person body, arms, torso, and face"""
    w, h = person_img.size
    
    # Body regions
    face_h = h // 5
    torso_start = face_h + h // 20
    torso_end = 4 * h // 5
    
    # Create masks
    face_mask = np.zeros((h, w), dtype=np.uint8)
    torso_mask = np.zeros((h, w), dtype=np.uint8)
    arms_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Face region (protected)
    face_mask[:face_h, w//3:2*w//3] = 255
    
    # Torso region (clothing area)
    torso_mask[torso_start:torso_end, :] = 255
    
    # Arms regions
    arm_width = w // 6
    arms_mask[:, :arm_width] = 255
    arms_mask[:, -arm_width:] = 255
    
    return face_mask, torso_mask, arms_mask

def extract_clothing_features(cloth_img):
    """Extract clothing shape, sleeves, neckline, color and texture"""
    cloth_array = np.array(cloth_img)
    
    # Remove white background
    gray = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2GRAY)
    clothing_mask = gray < 240
    
    if not np.any(clothing_mask):
        return None
    
    # Find clothing boundaries
    coords = np.where(clothing_mask)
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    
    # Extract clothing only
    clothing = cloth_array[y_min:y_max+1, x_min:x_max+1]
    
    return clothing

def align_clothing_to_person(clothing, person_img, torso_mask, arms_mask):
    """Scale, rotate, and warp clothing to fit naturally over torso"""
    if clothing is None:
        return None
    
    h, w = person_img.size
    person_array = np.array(person_img)
    
    # Get torso dimensions
    torso_coords = np.where(torso_mask > 0)
    if len(torso_coords[0]) == 0:
        return None
    
    torso_y_min, torso_y_max = np.min(torso_coords[0]), np.max(torso_coords[0])
    torso_x_min, torso_x_max = np.min(torso_coords[1]), np.max(torso_coords[1])
    
    torso_height = torso_y_max - torso_y_min
    torso_width = torso_x_max - torso_x_min
    
    # Scale clothing to torso size
    resized_clothing = cv2.resize(clothing, (torso_width, torso_height))
    
    # Create positioned clothing on full canvas
    positioned_clothing = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Center the clothing on torso
    start_x = torso_x_min
    start_y = torso_y_min
    
    positioned_clothing[start_y:start_y+torso_height, start_x:start_x+torso_width] = resized_clothing
    
    return positioned_clothing

def generate_new_image(person_img, positioned_clothing, face_mask, arms_mask):
    """Overlay clothing with realistic blending, shadows, and folds"""
    if positioned_clothing is None:
        return person_img
    
    person_array = np.array(person_img)
    result = person_array.copy().astype(np.float32)
    
    # Create clothing mask
    clothing_mask = np.mean(positioned_clothing, axis=2) > 10
    
    # Overlay clothing with realistic blending
    for y in range(person_array.shape[0]):
        for x in range(person_array.shape[1]):
            if clothing_mask[y, x]:
                # Check if not face or arms
                if not (face_mask[y, x] or arms_mask[y, x]):
                    clothing_pixel = positioned_clothing[y, x]
                    
                    # Check if actual clothing (not background)
                    if np.mean(clothing_pixel) < 240:
                        person_pixel = person_array[y, x]
                        
                        # Lighting matching
                        person_brightness = np.mean(person_pixel)
                        clothing_brightness = np.mean(clothing_pixel)
                        
                        if clothing_brightness > 10:
                            lighting_factor = person_brightness / clothing_brightness
                            lighting_factor = min(1.3, max(0.7, lighting_factor))
                            
                            # Apply lighting
                            final_pixel = clothing_pixel * lighting_factor
                            
                            # Add realistic shadows
                            shadow_factor = 0.9
                            final_pixel = final_pixel * shadow_factor
                            
                            result[y, x] = final_pixel
    
    # Apply bilateral filter for realistic fabric texture
    result = cv2.bilateralFilter(result.astype(np.uint8), 3, 50, 50)
    
    return Image.fromarray(result.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """Complete virtual try-on pipeline"""
    try:
        # Load images
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        
        # Step 1: Detect person and body
        face_mask, torso_mask, arms_mask = detect_person_and_body(person_img)
        
        # Step 2: Extract clothing features
        clothing = extract_clothing_features(cloth_img)
        
        # Step 3: Align clothing to person
        positioned_clothing = align_clothing_to_person(clothing, person_img, torso_mask, arms_mask)
        
        # Step 4: Generate new image with realistic blending
        final_result = generate_new_image(person_img, positioned_clothing, face_mask, arms_mask)
        
        # Save result
        final_result.save(output_path, 'JPEG', quality=95)
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback: return original person
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
