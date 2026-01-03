#!/usr/bin/env python
"""
Quick script to train GMM and TOM models.
This will take several hours, but will create the models you need.
"""
import os
import sys
import subprocess

def check_dataset():
    """Check if dataset is ready."""
    train_pairs = "cp-vton/data/train/train_pairs.txt"
    if not os.path.exists(train_pairs):
        print("[X] train_pairs.txt not found!")
        return False
    
    with open(train_pairs, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    
    if len(lines) == 0:
        print("[X] train_pairs.txt is empty!")
        return False
    
    print(f"[OK] Found {len(lines)} training pairs")
    return True

def train_gmm():
    """Train GMM model."""
    print("\n" + "="*70)
    print("TRAINING GMM (Geometric Matching Module)")
    print("="*70)
    print("This will take 1-3 hours depending on your CPU/GPU...")
    print("You can stop anytime with CTRL+C and resume later.")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable,
        "cp-vton/train.py",
        "--dataroot", "cp-vton/data",
        "--data_list", "train/train_pairs.txt",
        "--stage", "GMM",
        "--name", "gmm_train",
        "--batch-size", "4",
        "--workers", "2",
        "--shuffle",
        "--keep_step", "50000",  # Reduced for faster training
        "--decay_step", "50000",
        "--save_count", "5000"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=".")

def train_tom():
    """Train TOM model."""
    print("\n" + "="*70)
    print("TRAINING TOM (Try-On Module)")
    print("="*70)
    print("This will take 1-3 hours depending on your CPU/GPU...")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable,
        "cp-vton/train.py",
        "--dataroot", "cp-vton/data",
        "--data_list", "train/train_pairs.txt",
        "--stage", "TOM",
        "--name", "tom_train",
        "--batch-size", "4",
        "--workers", "2",
        "--shuffle",
        "--keep_step", "50000",
        "--decay_step", "50000",
        "--save_count", "5000"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, cwd=".")

def copy_models():
    """Copy trained models to models/ directory."""
    import shutil
    
    gmm_source = "cp-vton/checkpoints/gmm_train/gmm_final.pth"
    tom_source = "cp-vton/checkpoints/tom_train/tom_final.pth"
    
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(gmm_source):
        shutil.copy(gmm_source, "models/GMM.pth")
        print(f"[OK] Copied GMM model to models/GMM.pth")
    else:
        print(f"[X] GMM model not found at {gmm_source}")
    
    if os.path.exists(tom_source):
        shutil.copy(tom_source, "models/TOM.pth")
        print(f"[OK] Copied TOM model to models/TOM.pth")
    else:
        print(f"[X] TOM model not found at {tom_source}")

def main():
    print("="*70)
    print("CP-VTON Model Training Script")
    print("="*70)
    
    # Check dataset
    if not check_dataset():
        print("\n[X] Dataset not ready. Please ensure:")
        print("1. Dataset is downloaded and extracted")
        print("2. train_pairs.txt has valid pairs")
        print("3. All required folders exist (cloth, image, pose, etc.)")
        return
    
    print("\n[INFO] Training will take several hours.")
    print("You can stop and resume later - checkpoints are saved.")
    
    response = input("\nStart training GMM now? (y/N): ").strip().lower()
    
    if response == 'y':
        # Train GMM
        result = train_gmm()
        if result.returncode != 0:
            print("\n[X] GMM training failed or was interrupted")
            return
        
        # Copy GMM model
        copy_models()
        
        # Ask about TOM
        print("\n" + "="*70)
        response = input("GMM training complete! Train TOM now? (y/N): ").strip().lower()
        
        if response == 'y':
            # Train TOM
            result = train_tom()
            if result.returncode != 0:
                print("\n[X] TOM training failed or was interrupted")
                return
            
            # Copy TOM model
            copy_models()
            
            print("\n" + "="*70)
            print("[OK] Both models trained and copied!")
            print("="*70)
            print("\nYou can now restart your server:")
            print("  python app.py")
            print("\nThe models will be automatically loaded!")
        else:
            print("\n[INFO] TOM training skipped. You can train it later.")
    else:
        print("\n[INFO] Training cancelled.")
        print("\nTo train manually, run:")
        print("  cd cp-vton")
        print("  python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Checkpoints are saved - you can resume later.")
        sys.exit(1)



