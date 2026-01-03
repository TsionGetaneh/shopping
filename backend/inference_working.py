import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
import os
import sys

# Add cp-vton to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cp-vton'))
from networks import GMM
from utils.image_utils import preprocess_image, tensor_to_image

# Model paths
GMM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "GMM.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GMM_model = None

def load_gmm():
    """Load GMM model"""
    global GMM_model
    if GMM_model is not None:
        return True
    
    if not os.path.exists(GMM_MODEL_PATH):
        print(f"[ERROR] GMM model not found at {GMM_MODEL_PATH}")
        return False
    
    try:
        class Opt:
            def __init__(self):
                self.fine_height = 256
                self.fine_width = 192
                self.grid_size = 5
        
        opt = Opt()
        GMM_model = GMM(opt, cloth_channels=1).to(device)
        state_dict = torch.load(GMM_MODEL_PATH, map_location=device)
        GMM_model.load_state_dict(state_dict)
        GMM_model.eval()
        print("[OK] GMM loaded")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load GMM: {e}")
        return False

def extract_body_shape(person_img):
    """Extract actual body shape from person image"""
    # Convert to numpy
    img_array = np.array(person_img.convert('RGB'))
    h, w = img_array.shape[:2]
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (person)
        person_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w_rect, h_rect = cv2.boundingRect(person_contour)
        
        return {
            'x': x, 'y': y, 'width': w_rect, 'height': h_rect,
            'center_x': x + w_rect // 2,
            'shoulder_y': y + h_rect // 4,
            'waist_y': y + h_rect // 2,
            'hip_y': y + 3 * h_rect // 4
        }
    
    # Fallback to default proportions
    return {
        'x': w // 4, 'y': h // 8, 'width': w // 2, 'height': 3 * h // 4,
        'center_x': w // 2,
        'shoulder_y': h // 4,
        'waist_y': h // 2,
        'hip_y': 3 * h // 4
    }

def create_body_conforming_warp(person_img, cloth_img):
    """Create a warp that makes cloth conform to actual body shape"""
    # Extract body shape
    body_shape = extract_body_shape(person_img)
    
    # Get dimensions
    w, h = person_img.size
    
    # Define control points for cloth warping based on body shape
    # These points will make the cloth fit the person's actual shape
    
    # Source points (cloth corners and key points)
    src_points = np.float32([
        [0, 0],                    # Top-left
        [w, 0],                    # Top-right  
        [w, h],                    # Bottom-right
        [0, h],                    # Bottom-left
        [w//2, 0],                 # Top-center
        [w//2, h],                 # Bottom-center
        [0, h//2],                 # Left-center
        [w, h//2],                 # Right-center
    ])
    
    # Destination points (warped to fit body shape)
    dst_points = np.float32([
        # Top corners - narrow at shoulders
        [body_shape['center_x'] - body_shape['width']//3, body_shape['shoulder_y']],
        [body_shape['center_x'] + body_shape['width']//3, body_shape['shoulder_y']],
        # Bottom corners - wider at hips
        [body_shape['center_x'] - body_shape['width']//2, body_shape['hip_y']],
        [body_shape['center_x'] + body_shape['width']//2, body_shape['hip_y']],
        # Top center - neck area
        [body_shape['center_x'], body_shape['shoulder_y'] - 20],
        # Bottom center - waist area
        [body_shape['center_x'], body_shape['waist_y']],
        # Left center - arm area
        [body_shape['center_x'] - body_shape['width']//2, body_shape['waist_y']],
        # Right center - arm area
        [body_shape['center_x'] + body_shape['width']//2, body_shape['waist_y']],
    ])
    
    # Calculate homography transformation
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if M is not None:
        # Apply the transformation
        cloth_array = np.array(cloth_img)
        warped_cloth = cv2.warpPerspective(cloth_array, M, (w, h))
        
        # Create a mask for the warped cloth
        mask = np.zeros((h, w), dtype=np.uint8)
        mask.fill(255)
        warped_mask = cv2.warpPerspective(mask, M, (w, h))
        
        return warped_cloth, warped_mask
    else:
        return None, None

def create_realistic_clothing_fit(person_img, cloth_img):
    """Create realistic clothing fit using multiple techniques"""
    w, h = person_img.size
    
    # Resize cloth to match person
    cloth_resized = cloth_img.resize((w, h), Image.LANCZOS)
    
    # Try body-conforming warp
    warped_cloth, warped_mask = create_body_conforming_warp(person_img, cloth_resized)
    
    if warped_cloth is not None:
        print("[OK] Body-conforming warp successful")
        return warped_cloth, warped_mask
    else:
        print("[WARN] Body-conforming warp failed, using fallback")
        # Fallback: simple perspective warp
        return create_fallback_warp(person_img, cloth_resized)

def create_fallback_warp(person_img, cloth_img):
    """Fallback warp method"""
    w, h = person_img.size
    
    # Simple perspective warp that makes cloth look like it's fitting
    src_points = np.float32([
        [0, 0], [w, 0], [w, h], [0, h]
    ])
    
    dst_points = np.float32([
        [w//3, h//4],      # Narrow at shoulders
        [2*w//3, h//4],    # Narrow at shoulders  
        [3*w//4, 3*h//4],  # Wider at waist/hips
        [w//4, 3*h//4]     # Wider at waist/hips
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    cloth_array = np.array(cloth_img)
    warped_cloth = cv2.warpPerspective(cloth_array, M, (w, h))
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask.fill(255)
    warped_mask = cv2.warpPerspective(mask, M, (w, h))
    
    return warped_cloth, warped_mask

def blend_cloth_with_person(person_img, warped_cloth, warped_mask):
    """Blend warped cloth with person preserving head and natural look"""
    w, h = person_img.size
    
    # Convert to arrays
    person_array = np.array(person_img).astype(np.float32)
    cloth_array = warped_cloth.astype(np.float32)
    mask_array = warped_mask.astype(np.float32) / 255.0
    
    # Create masks for different body regions
    head_mask = np.zeros((h, w), dtype=np.float32)
    head_mask[:int(h * 0.25), :] = 1.0  # Top 25% is head
    
    # Upper body mask for clothing
    upper_mask = np.zeros((h, w), dtype=np.float32)
    upper_mask[int(h * 0.25):int(h * 0.7), :] = 1.0
    
    # Create smooth transitions
    kernel = np.ones((5, 5), np.float32) / 25
    upper_mask = cv2.filter2D(upper_mask, -1, kernel)
    
    # Convert masks to 3D
    head_mask_3d = np.stack([head_mask] * 3, axis=2)
    upper_mask_3d = np.stack([upper_mask] * 3, axis=2)
    cloth_mask_3d = np.stack([mask_array] * 3, axis=2)
    
    # Blend with multiple layers
    final_array = (
        person_array * head_mask_3d +  # Keep original head
        cloth_array * (1 - head_mask_3d) * upper_mask_3d * cloth_mask_3d +  # Apply cloth to upper body
        person_array * (1 - head_mask_3d) * (1 - upper_mask_3d)  # Keep original lower body
    )
    
    # Apply slight blur for more natural blending
    final_array = cv2.GaussianBlur(final_array, (3, 3), 0)
    
    return Image.fromarray(final_array.astype(np.uint8))

def generate_tryon(person_path, cloth_path, output_path):
    """
    Main virtual try-on function that actually makes cloth fit the person
    """
    print(f"[*] Virtual try-on: {person_path} + {cloth_path} -> {output_path}")
    
    # Load images
    person_img = Image.open(person_path).convert("RGB")
    cloth_img = Image.open(cloth_path).convert("RGB")
    
    print(f"[*] Processing: person={person_img.size}, cloth={cloth_img.size}")
    
    # Create realistic clothing fit
    warped_cloth, warped_mask = create_realistic_clothing_fit(person_img, cloth_img)
    
    if warped_cloth is not None:
        # Blend with person
        result_image = blend_cloth_with_person(person_img, warped_cloth, warped_mask)
        print("[OK] Clothing fit created successfully")
    else:
        print("[ERROR] Failed to create clothing fit")
        result_image = person_img  # Return original if failed
    
    # Save result
    if result_image.mode != 'RGB':
        result_image = result_image.convert('RGB')
    
    result_image.save(output_path, 'JPEG', quality=95)
    print(f"[OK] Result saved to {output_path}")
    return output_path

# Load GMM on import (though we may not use it)
load_gmm()
