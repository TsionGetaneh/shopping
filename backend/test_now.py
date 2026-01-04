import sys
import os
sys.path.append(os.path.dirname(__file__))

from inference_new import generate_tryon
from PIL import Image
import numpy as np

# Test the virtual try-on
person_path = "uploads/person.jpg"
cloth_path = "uploads/cloth.jpg"
output_path = "uploads/test_result.jpg"

print("=== TESTING VIRTUAL TRY-ON ===")
print(f"Person: {person_path}")
print(f"Cloth: {cloth_path}")
print(f"Output: {output_path}")

try:
    # Check if files exist
    if not os.path.exists(person_path):
        print(f"ERROR: Person file not found: {person_path}")
        exit(1)
    if not os.path.exists(cloth_path):
        print(f"ERROR: Cloth file not found: {cloth_path}")
        exit(1)
    
    # Load images to check
    person_img = Image.open(person_path)
    cloth_img = Image.open(cloth_path)
    
    print(f"Person size: {person_img.size}")
    print(f"Cloth size: {cloth_img.size}")
    
    # Check cloth pixels
    cloth_array = np.array(cloth_img)
    print(f"Cloth array shape: {cloth_array.shape}")
    print(f"Cloth sample pixel: {cloth_array[100, 100] if cloth_array.shape[0] > 100 and cloth_array.shape[1] > 100 else 'N/A'}")
    
    # Run virtual try-on
    print("\n=== RUNNING VIRTUAL TRY-ON ===")
    result = generate_tryon(person_path, cloth_path, output_path)
    print(f"Result: {result}")
    
    # Check if result was created and is different
    if os.path.exists(output_path):
        result_img = Image.open(output_path)
        print(f"Result size: {result_img.size}")
        
        # Compare with original
        original_img = Image.open(person_path)
        if np.array_equal(np.array(result_img), np.array(original_img)):
            print("ERROR: Result is identical to original!")
        else:
            print("SUCCESS: Result is different from original!")
            
        # Save a comparison
        comparison = np.concatenate([np.array(original_img), np.array(result_img)], axis=1)
        Image.fromarray(comparison).save("uploads/comparison.jpg")
        print("Comparison saved to: uploads/comparison.jpg")
    else:
        print("ERROR: No result file created")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
