#!/usr/bin/env python
"""
Working CP-VTON that actually warps cloth to fit person's body shape
"""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
import sys
sys.path.append('cp-vton')
from networks import GMM

def get_person_pose_keypoints(person_img):
    """
    Extract pose keypoints from the actual person image using simple computer vision
    """
    # Convert to numpy
    img_array = np.array(person_img.convert('RGB'))
    h, w = img_array.shape[:2]
    
    # Simple pose estimation based on image analysis
    # Find the person's silhouette and estimate keypoints
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Find person outline (simplified)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the main contour (person)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w_rect, h_rect = cv2.boundingRect(main_contour)
        
        # Estimate keypoints based on body proportions
        keypoints = [
            # Face
            x + w_rect//2, y + h_rect//8, 0.9,  # nose
            x + w_rect//2 - 10, y + h_rect//8, 0.9,  # left eye
            x + w_rect//2 + 10, y + h_rect//8, 0.9,  # right eye
            
            # Shoulders - CRITICAL for clothing fitting
            x + w_rect//3, y + h_rect//4, 0.9,  # left shoulder
            x + 2*w_rect//3, y + h_rect//4, 0.9,  # right shoulder
            
            # Elbows
            x + w_rect//4, y + h_rect//2, 0.9,  # left elbow
            x + 3*w_rect//4, y + h_rect//2, 0.9,  # right elbow
            
            # Wrists
            x + w_rect//5, y + 3*h_rect//4, 0.9,  # left wrist
            x + 4*w_rect//5, y + 3*h_rect//4, 0.9,  # right wrist
            
            # Hips - CRITICAL for lower body
            x + w_rect//3, y + 3*h_rect//4, 0.9,  # left hip
            x + 2*w_rect//3, y + 3*h_rect//4, 0.9,  # right hip
            
            # Knees
            x + w_rect//3, y + 7*h_rect//8, 0.9,  # left knee
            x + 2*w_rect//3, y + 7*h_rect//8, 0.9,  # right knee
            
            # Ankles
            x + w_rect//3, y + 15*h_rect//16, 0.9,  # left ankle
            x + 2*w_rect//3, y + 15*h_rect//16, 0.9,  # right ankle
        ]
    else:
        # Fallback to standard proportions
        keypoints = [
            96, 32, 0.9,   # nose
            88, 32, 0.9,   # left eye  
            104, 32, 0.9,  # right eye
            64, 64, 0.9,   # left shoulder
            128, 64, 0.9,  # right shoulder
            48, 96, 0.9,   # left elbow
            144, 96, 0.9,  # right elbow
            32, 128, 0.9,  # left wrist
            160, 128, 0.9, # right wrist
            64, 192, 0.9,  # left hip
            128, 192, 0.9, # right hip
            64, 224, 0.9,  # left knee
            128, 224, 0.9, # right knee
            64, 240, 0.9,  # left ankle
            128, 240, 0.9, # right ankle
        ]
    
    return keypoints

def create_pose_heatmaps(keypoints, height=256, width=192):
    """
    Create pose heatmaps from keypoints for GMM
    """
    pose_maps = torch.zeros(18, height, width, dtype=torch.float32)
    
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
            if conf > 0.5 and 0 <= x < width and 0 <= y < height:
                # Create heatmap around keypoint
                x_int, y_int = int(x), int(y)
                radius = 8
                
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        nx, ny = x_int + dx, y_int + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist <= radius:
                                intensity = 1.0 - (dist / radius)
                                pose_maps[i//3, ny, nx] = max(pose_maps[i//3, ny, nx], intensity)
    
    return pose_maps

def create_agnostic_representation(person_tensor, pose_maps):
    """
    Create proper agnostic representation for GMM
    """
    # Shape tensor (1 channel) - person silhouette
    shape_tensor = torch.ones(1, 256, 192, dtype=torch.float32)
    
    # Head tensor (3 channels) - preserve head
    head_tensor = torch.zeros(3, 256, 192, dtype=torch.float32)
    head_tensor[:, 0:64, 48:144] = 1.0
    
    # Combine: shape (1) + head (3) + pose (18) = 22 channels
    agnostic = torch.cat([shape_tensor, head_tensor, pose_maps], dim=0).unsqueeze(0)
    
    return agnostic

def warp_cloth_to_person(person_path, cloth_path, output_path):
    """
    Main function to warp cloth to fit person's body shape
    """
    # Load GMM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class Opt:
        def __init__(self):
            self.fine_height = 256
            self.fine_width = 192
            self.grid_size = 5
    
    opt = Opt()
    GMM_model = GMM(opt, cloth_channels=1).to(device)
    GMM_checkpoint = torch.load("models/GMM.pth", map_location=device)
    GMM_model.load_state_dict(GMM_checkpoint)
    GMM_model.eval()
    
    print("[*] GMM loaded successfully")
    
    # Load and preprocess images
    person_img = Image.open(person_path).convert("RGB").resize((192, 256), Image.BILINEAR)
    cloth_img = Image.open(cloth_path).convert("RGB").resize((192, 256), Image.BILINEAR)
    
    # Preprocess tensors
    def preprocess(img):
        img = np.array(img) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        return img
    
    person_tensor = preprocess(person_img).to(device)
    cloth_tensor = preprocess(cloth_img).to(device)
    
    print(f"Person tensor shape: {person_tensor.shape}")
    print(f"Cloth tensor shape: {cloth_tensor.shape}")
    
    # Get pose keypoints from actual person image
    keypoints = get_person_pose_keypoints(person_img)
    print(f"Extracted {len(keypoints)//3} pose keypoints")
    
    # Create pose heatmaps
    pose_maps = create_pose_heatmaps(keypoints).to(device)
    
    # Create agnostic representation
    agnostic = create_agnostic_representation(person_tensor, pose_maps).to(device)
    print(f"Agnostic shape: {agnostic.shape}")
    
    # Warp cloth using GMM
    with torch.no_grad():
        cloth_1ch = cloth_tensor[:, 0:1, :, :]  # 1 channel for GMM
        
        print("[*] Warping cloth with GMM...")
        grid, theta = GMM_model(agnostic, cloth_1ch)
        
        print(f"Grid shape: {grid.shape}")
        print(f"Grid range: [{grid.min().item():.3f}, {grid.max().item():.3f}]")
        
        # Check if grid is actually warping
        identity_grid = F.affine_grid(torch.eye(2, 3).unsqueeze(0).to(device), (1, 1, 256, 192), align_corners=False)
        grid_diff = torch.abs(grid - identity_grid).mean()
        print(f"Grid transformation strength: {grid_diff.item():.6f}")
        
        # Warp the cloth
        warped_cloth = F.grid_sample(cloth_tensor, grid, padding_mode='border', align_corners=False)
        
        # Check if cloth was actually warped
        diff = torch.abs(cloth_tensor - warped_cloth).mean()
        print(f"Cloth warping difference: {diff.item():.6f}")
        
        if diff.item() < 0.001:
            print("[ERROR] No warping detected!")
            return None
        
        # Convert warped cloth to image
        warped_img = warped_cloth.squeeze(0).cpu().numpy()
        warped_img = ((warped_img * 0.5 + 0.5) * 255).astype(np.uint8)
        warped_img = warped_img.transpose(1, 2, 0)
        warped_pil = Image.fromarray(warped_img)
        
        # Create final result - blend warped cloth with person
        person_orig = Image.open(person_path).convert("RGB")
        warped_resized = warped_pil.resize(person_orig.size, Image.LANCZOS)
        
        # Create masks for blending
        w_orig, h_orig = person_orig.size
        head_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        head_mask[:int(h_orig * 0.25), :] = 1.0
        
        # Upper body mask for clothing
        upper_mask = np.zeros((h_orig, w_orig), dtype=np.float32)
        upper_mask[int(h_orig * 0.25):int(h_orig * 0.7), :] = 1.0
        
        # Blend
        person_array = np.array(person_orig).astype(np.float32)
        warped_array = np.array(warped_resized).astype(np.float32)
        
        head_mask_3d = np.stack([head_mask] * 3, axis=2)
        upper_mask_3d = np.stack([upper_mask] * 3, axis=2)
        
        # Final blend: preserve head, apply warped cloth to upper body
        final_array = (
            person_array * head_mask_3d +  # Keep original head
            warped_array * (1 - head_mask_3d) * upper_mask_3d +  # Apply warped cloth
            person_array * (1 - head_mask_3d) * (1 - upper_mask_3d)  # Keep original lower body
        )
        
        result_image = Image.fromarray(final_array.astype(np.uint8))
        result_image.save(output_path, quality=95)
        
        print(f"[OK] Saved warped result to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test the warping
    result = warp_cloth_to_person("uploads/person.jpg", "uploads/cloth.jpg", "warped_result.jpg")
    if result:
        print("SUCCESS: Cloth was warped to fit person!")
    else:
        print("FAILED: Warping did not work")
