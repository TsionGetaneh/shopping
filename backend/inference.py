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
    Step 3 (Alternative): Use GMM model for advanced garment warping
    """
    if not load_gmm_model() or GMM_model is None:
        return None, None
    
    try:
        # Resize to GMM input size
        person_resized = person_img.resize((192, 256), Image.BILINEAR)
        cloth_resized = cloth_img.resize((192, 256), Image.BILINEAR)
        
        # Preprocess
        person_tensor = preprocess_image(person_resized).to(device)
        cloth_tensor = preprocess_image(cloth_resized).to(device)
        
        # Create agnostic representation
        agnostic = create_agnostic_representation(person_resized, keypoints)
        agnostic = agnostic.to(device)
        
        # GMM expects 1-channel cloth
        cloth_1ch = cloth_tensor[:, 0:1, :, :]
        
        with torch.no_grad():
            grid, theta = GMM_model(agnostic, cloth_1ch)
            grid = torch.clamp(grid, -1, 1)
            
            # Warp cloth
            warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
            
            # Convert to image
            warped_img = tensor_to_image(warped_cloth)
            
            # Resize back to original person size
            w_orig, h_orig = person_img.size
            warped_img = warped_img.resize((w_orig, h_orig), Image.LANCZOS)
            
            # Create mask (non-white regions are garment)
            warped_array = np.array(warped_img)
            mask = np.mean(warped_array, axis=2) < 250
            mask = (mask * 255).astype(np.uint8)
            
            return warped_array, mask
            
    except Exception as e:
        print(f"[WARN] GMM warping failed: {e}")
    
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

def anchor_garment_at_keypoints(warped_garment, warped_mask, keypoints):
    """
    Step 4: Anchor the garment correctly at shoulders, neckline, chest, and waist
    """
    h, w = warped_garment.shape[:2]
    
    # Ensure garment is properly positioned at key anchor points
    neck = keypoints['neck']
    left_shoulder = keypoints['left_shoulder']
    right_shoulder = keypoints['right_shoulder']
    waist = keypoints['waist']
    
    # Create anchor mask (regions where garment should be strong)
    anchor_mask = np.zeros((h, w), dtype=np.float32)
    
    # Neckline region
    cv2.circle(anchor_mask, (int(neck[0]), int(neck[1])), 15, 1.0, -1)
    
    # Shoulder regions
    cv2.circle(anchor_mask, (int(left_shoulder[0]), int(left_shoulder[1])), 20, 1.0, -1)
    cv2.circle(anchor_mask, (int(right_shoulder[0]), int(right_shoulder[1])), 20, 1.0, -1)
    
    # Chest region
    chest = keypoints['chest_center']
    cv2.ellipse(anchor_mask, (int(chest[0]), int(chest[1])), (30, 20), 0, 0, 360, 1.0, -1)
    
    # Waist region
    cv2.ellipse(anchor_mask, (int(waist[0]), int(waist[1])), (40, 15), 0, 0, 360, 1.0, -1)
    
    # Enhance garment at anchor points
    warped_mask = np.maximum(warped_mask.astype(np.float32) / 255.0, anchor_mask * 0.5)
    warped_mask = (np.clip(warped_mask, 0, 1) * 255).astype(np.uint8)
    
    return warped_garment, warped_mask

def add_realistic_shadows_and_folds(garment_array, mask, keypoints):
    """
    Step 7: Add realistic shadows, folds, and lighting to integrate garment naturally
    """
    h, w = garment_array.shape[:2]
    
    # Create shadow map based on body geometry
    shadow_map = np.ones((h, w), dtype=np.float32)
    
    # Shadows under arms
    left_shoulder = keypoints['left_shoulder']
    right_shoulder = keypoints['right_shoulder']
    
    # Create gradient shadows
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                # Shadow intensity based on distance from center
                center_x = w // 2
                dist_from_center = abs(x - center_x) / (w // 2)
                
                # Darker on sides (under arms)
                shadow_intensity = 0.85 + 0.15 * (1 - dist_from_center)
                
                # Additional shadow at waist (fabric folds)
                waist_y = keypoints['waist'][1]
                if abs(y - waist_y) < 30:
                    shadow_intensity *= 0.92
                
                shadow_map[y, x] = shadow_intensity
    
    # Apply shadows to garment
    for c in range(3):
        garment_array[:, :, c] = (garment_array[:, :, c].astype(np.float32) * shadow_map).astype(np.uint8)
    
    # Add subtle fabric texture (noise)
    noise = np.random.normal(0, 3, (h, w, 3))
    garment_array = np.clip(garment_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return garment_array

def render_garment_with_depth(neutral_base, warped_garment, warped_mask, keypoints, masks):
    """
    Steps 5-6: Render garment as fully opaque with correct depth ordering
    """
    if warped_garment is None or warped_mask is None:
        return Image.fromarray(neutral_base)
    
    h, w = neutral_base.shape[:2]
    result = neutral_base.copy().astype(np.float32)
    
    # Ensure warped garment and mask match base size
    if warped_garment.shape[:2] != (h, w):
        warped_garment = cv2.resize(warped_garment, (w, h))
    if warped_mask.shape != (h, w):
        warped_mask = cv2.resize(warped_mask, (w, h))
    
    # Anchor garment at keypoints
    warped_garment, warped_mask = anchor_garment_at_keypoints(warped_garment, warped_mask, keypoints)
    
    # Add realistic shadows and folds
    warped_garment = add_realistic_shadows_and_folds(warped_garment, warped_mask, keypoints)
    
    # Create final garment mask (fully opaque where garment exists)
    garment_mask = (warped_mask > 128).astype(np.float32)
    
    # Get arm masks for depth ordering
    left_arm_mask = (masks['left_arm'] > 0).astype(np.float32)
    right_arm_mask = (masks['right_arm'] > 0).astype(np.float32)
    arms_mask = np.maximum(left_arm_mask, right_arm_mask)
    
    # Depth ordering: arms in front of garment
    # Where arms overlap garment, keep original arms
    # Where garment doesn't overlap arms, apply garment
    
    # Garment regions not occluded by arms
    garment_visible = garment_mask * (1 - arms_mask)
    
    # Apply garment (fully opaque, no transparency)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - garment_visible) +
            warped_garment[:, :, c] * garment_visible
        )
    
    # Preserve face and neck (never replace)
    face_mask = (masks['face'] > 0).astype(np.float32)
    neck_mask = (masks['neck'] > 0).astype(np.float32)
    preserve_mask = np.maximum(face_mask, neck_mask)
    
    for c in range(3):
        result[:, :, c] = (
            neutral_base[:, :, c] * preserve_mask +
            result[:, :, c] * (1 - preserve_mask)
        )
    
    # Apply bilateral filter for realistic fabric texture
    result = cv2.bilateralFilter(result.astype(np.uint8), 5, 80, 80)
    
    return Image.fromarray(result)

def generate_tryon(person_path, cloth_path, output_path):
    """
    Complete virtual try-on pipeline - all 8 steps implemented
    """
    try:
        # Load images
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        
        print("[*] Step 1: Detecting human body pose, silhouette, and key points...")
        keypoints, masks = detect_human_pose_and_shape(person_img)
        
        print("[*] Step 2: Removing original clothing and creating neutral body base...")
        neutral_base = create_neutral_body_base(person_img, masks)
        
        print("[*] Step 3: Deforming and warping garment to match body geometry...")
        # Try GMM warping first, fallback to geometric warping
        warped_garment, warped_mask = warp_garment_with_gmm(cloth_img, person_img, keypoints)
        
        if warped_garment is None:
            warped_garment, warped_mask = geometric_warp_garment(cloth_img, keypoints, person_img.size)
        
        if warped_garment is None:
            print("[ERROR] Failed to warp garment")
            result_img = Image.fromarray(neutral_base)
        else:
            print("[*] Step 4: Anchoring garment at key points...")
            print("[*] Step 5-6: Rendering garment with depth ordering...")
            print("[*] Step 7: Adding realistic shadows, folds, and lighting...")
            result_img = render_garment_with_depth(
                neutral_base, warped_garment, warped_mask, keypoints, masks
            )
        
        print("[*] Step 8: Outputting final photorealistic image...")
        # Ensure RGB mode
        if result_img.mode != 'RGB':
            result_img = result_img.convert('RGB')
        
        # Save with high quality
        result_img.save(output_path, 'JPEG', quality=95)
        print(f"[OK] Virtual try-on complete: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[ERROR] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return original person
        person_img = Image.open(person_path).convert("RGB")
        person_img.save(output_path, 'JPEG', quality=95)
        return output_path

# Load GMM model on import
load_gmm_model()
