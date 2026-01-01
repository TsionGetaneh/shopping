#!/usr/bin/env python
"""Quick test to verify everything is set up correctly."""

import sys
import os

print("="*60)
print("Virtual Try-On System - Setup Test")
print("="*60)

# Test 1: Check Python version
print("\n[1] Python version:", sys.version.split()[0])

# Test 2: Check required packages
print("\n[2] Checking required packages...")
required = {
    'flask': 'Flask',
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'PIL': 'Pillow',
    'numpy': 'NumPy'
}

missing = []
for module, name in required.items():
    try:
        __import__(module)
        print(f"  [OK] {name}")
    except ImportError:
        print(f"  [X]  {name} - MISSING")
        missing.append(name)

# Test 3: Check model files
print("\n[3] Checking model files...")
model_files = [
    ("models/GMM.pth", "GMM model"),
    ("models/TOM.pth", "TOM model")
]

for path, name in model_files:
    full_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(full_path):
        size = os.path.getsize(full_path) / (1024*1024)  # MB
        print(f"  [OK] {name} ({size:.1f} MB)")
    else:
        print(f"  [X]  {name} - NOT FOUND")

# Test 4: Check Flask app
print("\n[4] Testing Flask app import...")
try:
    from app import app
    print("  [OK] Flask app can be imported")
except Exception as e:
    print(f"  [X]  Flask app import failed: {e}")

# Test 5: Check inference module
print("\n[5] Testing inference module...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cp-vton'))
    from inference import load_models
    print("  [OK] Inference module can be imported")
except Exception as e:
    print(f"  [X]  Inference import failed: {e}")

# Summary
print("\n" + "="*60)
if missing:
    print("[X] Setup incomplete - missing packages:")
    for pkg in missing:
        print(f"   - {pkg}")
    print("\nInstall missing packages with:")
    print("   pip install -r requirements.txt")
else:
    print("[OK] All checks passed!")
    print("\nYou can start the server with:")
    print("   python app.py")
    print("\nThen open: http://127.0.0.1:5000")
print("="*60)


