import cv2
import numpy as np
from PIL import Image
import os
import sys

def create_body_mask(person_img):
    """Create a proper body mask for clothing fitting"""
    w, h = person_img.size
    person_array = np.array(person_img)
    
    # Create body mask
    body_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Face protection - top 20%
    face_end = h // 5
    body_mask[face_end:, :] = 255
    
    # Remove arms - create arm cutouts
    arm_width = w // 8
    body_mask[:, :arm_width] = 0
    body_mask[:, -arm_width:] = 0
    
    return body_mask

def extract_clothing_shape(cloth_img):
    """Extract clothing shape with proper background removal"""
    cloth_array = np.array(cloth_img)
    
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(cloth_array, cv2.COLOR_RGB2LAB)
    
    # Multiple detection methods
    mask1 = gray < 230  # Grayscale threshold
    mask2 = lab[:, :, 1] > 130  # A-channel threshold (green-red)
    mask3 = lab[:, :, 2] < 130  # B-channel threshold (blue-yellow)
    
    # Combine masks
    clothing_mask = mask1 & (mask2 | mask3)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_CLOSE, kernel)
    clothing_mask = cv2.morphologyEx(clothing_mask, cv2.MORPH_OPEN, kernel)
    
    return clothing_mask, cloth_array

def warp_clothing_to_body(cloth_array, clothing_mask, person_img, body_mask):
    """Warp clothing to fit body shape using perspective transform"""
    h, w = person_img.size
    person_array = np.array(person_img)
    
    # Find clothing boundaries
    coords = np.where(clothing_mask > 0)
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = np.min(coords[0]), np.max(coords[0])
    x_min, x_max = np.min(coords[1]), np.max(coords[1])
    
    # Extract clothing
    clothing = cloth_array[y_min:y_max+1, x_min:x_max+1]
    clothing_mask_cropped = clothing_mask[y_min:y_max+1, x_min:x_max+1]
    
    # Find body boundaries
    body_coords = np.where(body_mask > 0)
    if len(body_coords[0]) == 0:
        return None
    
    body_y_min, body_y_max = np.min(body_coords[0]), np.max(body_coords[0])
    body_x_min, body_x_max = np.min(body_coords[1]), np.max(body_coords[1])
    
    # Create source points (clothing corners)
    src_points = np.float32([
        [0, 0],
        [clothing.shape[1], 0],
        [clothing.shape[1], clothing.shape[0]],
        [0, clothing.shape[0]]
    ])
    
    # Create destination points (body shape - narrower at top, wider at bottom)
    top_width = int((body_x_max - body_x_min) * 0.8)
    bottom_width = int((body_x_max - body_x_min) * 1.0)
    
    dst_points = np.float32([
        [body_x_min + (body_x_max - body_x_min - top_width) // 2, body_y_min],
        [body_x_min + (body_x_max - body_x_min + top_width) // 2, body_y_min],
        [body_x_max, body_y_max],
        [body_x_min, body_y_max]
    ])
    
    # Calculate perspective transform
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Warp clothing
    warped_clothing = cv2.warpPerspective(clothing, M, (w, h))
    warped_mask = cv2.warpPerspective(clothing_mask_cropped.astype(np.uint8) * 255, M, (w, h))
    
    return warped_clothing, warped_mask

def blend_clothing_on_person(person_img, warped_clothing, warped_mask, body_mask):
    """Blend warped clothing onto person"""
    person_array = np.array(person_img).astype(np.float32)
    result = person_array.copy()
    
    if warped_clothing is None:
        return Image.fromarray(result.astype(np.uint8))
    
    # Create final mask (clothing + body area)
    final_mask = (warped_mask > 128) & (body_mask > 0)
    
    # Apply clothing with proper blending
    for y in range(person_array.shape[0]):
        for x in range(person_array.shape[1]):
            if final_mask[y, x]:
                clothing_pixel = warped_clothing[y, x]
                person_pixel = person_array[y, x]
                
                # Check if it's actual clothing (not background)
                if np.mean(clothing_pixel) < 240:
                    # Simple lighting matching
                    person_brightness = np.mean(person_pixel)
                    clothing_brightness = np.mean(clothing_pixel)
                    
                    if clothing_brightness > 10:
                        lighting_factor = person_brightness / clothing_brightness
                        lighting_factor = np.clip(lighting_factor, 0.7, 1.3)
                        
                        # Apply lighting and blend
                        final_pixel = clothing_pixel * lighting_factor
                        result[y, x] = final_pixel
    
    return Image.fromarray(result.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """Complete virtual try-on with proper clothing fitting"""
    try:
        print("Starting proper clothing fitting...")
        
        # Load images
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        
        print(f"Person size: {person_img.size}")
        print(f"Cloth size: {cloth_img.size}")
        
        # Step 1: Create body mask
        body_mask = create_body_mask(person_img)
        print(f"Body mask created: {np.sum(body_mask > 0)} pixels")
        
        # Step 2: Extract clothing shape
        clothing_mask, cloth_array = extract_clothing_shape(cloth_img)
        print(f"Clothing mask created: {np.sum(clothing_mask > 0)} pixels")
        
        # Step 3: Warp clothing to body
        warped_clothing, warped_mask = warp_clothing_to_body(cloth_array, clothing_mask, person_img, body_mask)
        if warped_clothing is not None:
            print(f"Clothing warped successfully")
        else:
            print("Clothing warping failed")
        
        # Step 4: Blend clothing on person
        final_result = blend_clothing_on_person(person_img, warped_clothing, warped_mask, body_mask)
        
        # Save result
        final_result.save(output_path, 'JPEG', quality=95)
        print(f"Result saved to: {output_path}")
        print("Proper clothing fitting completed!")
        return output_path
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path
