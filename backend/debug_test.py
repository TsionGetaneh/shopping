import sys
import os
sys.path.append(os.path.dirname(__file__))

from inference_new import generate_tryon
from PIL import Image
import numpy as np

# Test the virtual try-on
person_path = "uploads/person.jpg"
cloth_path = "uploads/cloth.jpg"
output_path = "uploads/debug_result.jpg"

print("Testing virtual try-on...")
print(f"Person: {person_path}")
print(f"Cloth: {cloth_path}")
print(f"Output: {output_path}")

try:
    # Check if files exist
    if not os.path.exists(person_path):
        print(f"ERROR: Person file not found: {person_path}")
    if not os.path.exists(cloth_path):
        print(f"ERROR: Cloth file not found: {cloth_path}")
    
    # Load images to check
    person_img = Image.open(person_path)
    cloth_img = Image.open(cloth_path)
    
    print(f"Person size: {person_img.size}")
    print(f"Cloth size: {cloth_img.size}")
    
    # Run virtual try-on
    result = generate_tryon(person_path, cloth_path, output_path)
    print(f"Result: {result}")
    
    # Check if result was created
    if os.path.exists(output_path):
        result_img = Image.open(output_path)
        print(f"Result size: {result_img.size}")
        print("SUCCESS: Virtual try-on completed!")
    else:
        print("ERROR: No result file created")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
