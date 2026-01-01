#!/usr/bin/env python
"""
Test script to verify the CP-VTON inference fixes
"""
import os
import sys
import torch
from PIL import Image, ImageDraw
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from inference import load_models, generate_tryon, create_simple_pose, create_simple_parsing

def create_test_images():
    """Create simple test images for person and cloth"""
    # Create person image (simple stick figure)
    person_img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(person_img)
    
    # Draw simple person
    draw.ellipse([200, 50, 312, 162], fill='peachpuff')  # Head
    draw.rectangle([236, 162, 276, 322], fill='lightblue')  # Body
    draw.rectangle([216, 322, 236, 482], fill='darkblue')  # Left leg
    draw.rectangle([276, 322, 296, 482], fill='darkblue')  # Right leg
    draw.rectangle([196, 162, 216, 322], fill='peachpuff')  # Left arm
    draw.rectangle([296, 162, 316, 322], fill='peachpuff')  # Right arm
    
    # Create cloth image (simple t-shirt)
    cloth_img = Image.new('RGB', (512, 512), 'white')
    draw = ImageDraw.Draw(cloth_img)
    draw.rectangle([100, 100, 412, 412], fill='red')  # Red t-shirt
    draw.rectangle([150, 150, 200, 250], fill='white')  # Neck hole
    
    return person_img, cloth_img

def test_inference():
    """Test the inference pipeline"""
    print("[*] Testing CP-VTON inference pipeline...")
    
    # Create test images
    person_img, cloth_img = create_test_images()
    
    # Save test images
    person_path = 'test_person.jpg'
    cloth_path = 'test_cloth.jpg'
    output_path = 'test_result.jpg'
    
    person_img.save(person_path)
    cloth_img.save(cloth_path)
    
    print(f"[*] Created test images: {person_path}, {cloth_path}")
    
    # Test model loading
    print("[*] Loading models...")
    GMM_model, TOM_model = load_models()
    
    if GMM_model is not None and TOM_model is not None:
        print("[OK] Models loaded successfully")
    else:
        print("[WARN] Models not loaded, will use fallback")
    
    # Test pose and parsing creation
    person_resized = person_img.resize((192, 256), Image.BILINEAR)
    pose_data = create_simple_pose(person_resized)
    parse_mask = create_simple_parsing(person_resized)
    
    print("[*] Created pose and parsing data")
    
    # Test full inference
    try:
        print("[*] Running full inference pipeline...")
        result_path = generate_tryon(person_path, cloth_path, output_path)
        
        if os.path.exists(output_path):
            print(f"[OK] Inference completed! Result saved to {output_path}")
            
            # Check result image
            result_img = Image.open(output_path)
            print(f"[*] Result image size: {result_img.size}")
            return True
        else:
            print("[ERROR] No result image generated")
            return False
            
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for path in [person_path, cloth_path, output_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    success = test_inference()
    if success:
        print("\n[SUCCESS] CP-VTON inference test completed successfully!")
    else:
        print("\n[FAILED] CP-VTON inference test failed!")
    
    print("\n[*] Test completed. You can now run the Flask app with: python app.py")
