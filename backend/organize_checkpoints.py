#!/usr/bin/env python
"""
Organize downloaded checkpoints to the correct location.
"""
import os
import shutil
from pathlib import Path

def organize_models():
    """Copy models from checkpoints/ to models/ folder."""
    print("="*70)
    print("Organizing CP-VTON Models")
    print("="*70)
    
    # Source paths (where you downloaded)
    source_gmm = Path("checkpoints/GMM/gmm_final.pth")
    source_tom = Path("checkpoints/TOM/tom_final.pth")
    
    # Alternative source paths
    alt_sources = [
        ("checkpoints/gmm_final.pth", "checkpoints/tom_final.pth"),
        ("checkpoints/GMM.pth", "checkpoints/TOM.pth"),
        ("checkpoints/gmm_train/gmm_final.pth", "checkpoints/tom_train/tom_final.pth"),
    ]
    
    # Destination
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    dest_gmm = models_dir / "GMM.pth"
    dest_tom = models_dir / "TOM.pth"
    
    # Find GMM model
    gmm_found = None
    if source_gmm.exists():
        gmm_found = source_gmm
        print(f"[OK] Found GMM: {source_gmm}")
    else:
        for alt_gmm, _ in alt_sources:
            if Path(alt_gmm).exists():
                gmm_found = Path(alt_gmm)
                print(f"[OK] Found GMM: {alt_gmm}")
                break
    
    # Find TOM model
    tom_found = None
    if source_tom.exists():
        tom_found = source_tom
        print(f"[OK] Found TOM: {source_tom}")
    else:
        for _, alt_tom in alt_sources:
            if Path(alt_tom).exists():
                tom_found = Path(alt_tom)
                print(f"[OK] Found TOM: {alt_tom}")
                break
    
    # Check file sizes
    if gmm_found:
        size_mb = gmm_found.stat().st_size / (1024*1024)
        print(f"    Size: {size_mb:.1f} MB")
        if size_mb < 1:
            print(f"    [WARN] File seems too small - might be empty or corrupted")
    
    if tom_found:
        size_mb = tom_found.stat().st_size / (1024*1024)
        print(f"    Size: {size_mb:.1f} MB")
        if size_mb < 1:
            print(f"    [WARN] File seems too small - might be empty or corrupted")
    
    # Copy files
    print("\n" + "="*70)
    print("Copying models to models/ folder...")
    print("="*70)
    
    if gmm_found:
        try:
            shutil.copy2(str(gmm_found), str(dest_gmm))
            print(f"[OK] Copied GMM to {dest_gmm}")
        except Exception as e:
            print(f"[X] Failed to copy GMM: {e}")
    else:
        print("[X] GMM model not found in checkpoints/")
        print("    Searched in:")
        print(f"    - {source_gmm}")
        for alt, _ in alt_sources:
            print(f"    - {alt}")
    
    if tom_found:
        try:
            shutil.copy2(str(tom_found), str(dest_tom))
            print(f"[OK] Copied TOM to {dest_tom}")
        except Exception as e:
            print(f"[X] Failed to copy TOM: {e}")
    else:
        print("[X] TOM model not found in checkpoints/")
        print("    Searched in:")
        print(f"    - {source_tom}")
        for _, alt in alt_sources:
            print(f"    - {alt}")
    
    # Verify
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    if dest_gmm.exists():
        size = dest_gmm.stat().st_size / (1024*1024)
        print(f"[OK] GMM.pth exists: {size:.1f} MB")
        if size < 1:
            print("    [WARN] File is very small - might not be valid")
    else:
        print("[X] GMM.pth not found in models/")
    
    if dest_tom.exists():
        size = dest_tom.stat().st_size / (1024*1024)
        print(f"[OK] TOM.pth exists: {size:.1f} MB")
        if size < 1:
            print("    [WARN] File is very small - might not be valid")
    else:
        print("[X] TOM.pth not found in models/")
    
    print("\n" + "="*70)
    if dest_gmm.exists() and dest_tom.exists():
        gmm_size = dest_gmm.stat().st_size
        tom_size = dest_tom.stat().st_size
        if gmm_size > 1000 and tom_size > 1000:
            print("[OK] Models organized successfully!")
            print("="*70)
            print("\nNext step: Restart your server")
            print("  python app.py")
            print("\nYou should see:")
            print("  [OK] Loaded GMM model (XXX.X MB)")
            print("  [OK] Loaded TOM model (XXX.X MB)")
        else:
            print("[WARN] Models found but seem too small")
            print("="*70)
            print("\nThe files might be empty or corrupted.")
            print("Please re-download the models.")
    else:
        print("[X] Models not properly organized")
        print("="*70)
        print("\nPlease check:")
        print("1. Models are in backend/checkpoints/ folder")
        print("2. File names are correct (gmm_final.pth, tom_final.pth)")
        print("3. Files are not empty")

if __name__ == "__main__":
    organize_models()

