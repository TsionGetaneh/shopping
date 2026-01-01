#!/usr/bin/env python
"""
Alternative methods to get CP-VTON models.
Since OneDrive links are unreliable, this script helps find alternatives.
"""
import os
import sys
import requests
from pathlib import Path

def try_onedrive_download(share_url):
    """Try to download from OneDrive share link."""
    print(f"\n[*] Attempting OneDrive download...")
    print(f"    URL: {share_url}")
    
    # OneDrive share links need special handling
    # Try to convert to direct download
    try:
        # First, get the page to extract download links
        response = requests.get(share_url, allow_redirects=True, timeout=30)
        print(f"    Status: {response.status_code}")
        
        # OneDrive usually redirects to a page with download buttons
        # We need to parse the HTML or use a different method
        print("[WARN] OneDrive share links require manual download from browser")
        print("       The link opens a page where you need to click 'Download'")
        return False
    except Exception as e:
        print(f"[X] Failed: {e}")
        return False

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CP-VTON Models - Alternative Download Methods")
    print("="*70)
    
    onedrive_url = "https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP"
    
    print("\n[INFO] OneDrive links often require manual browser download.")
    print("Here are alternative methods:\n")
    
    print("="*70)
    print("METHOD 1: Manual OneDrive Download (Try This First)")
    print("="*70)
    print("\n1. Open this link in your browser:")
    print(f"   {onedrive_url}")
    print("\n2. If it asks you to sign in, you may need a Microsoft account")
    print("3. Look for a 'Download' button or right-click on files")
    print("4. Download both .pth files")
    print("5. Place in backend/models/ as GMM.pth and TOM.pth")
    
    print("\n" + "="*70)
    print("METHOD 2: GitHub Repositories (Check These)")
    print("="*70)
    print("\nTry these repositories for model downloads:")
    print("\nA. CP-VTON+ Repository:")
    print("   https://github.com/minar09/cp-vton-plus")
    print("   - Check 'Releases' tab")
    print("   - Check 'Issues' for download links")
    print("   - Look for 'checkpoint' or 'pretrained' folders")
    
    print("\nB. Original CP-VTON:")
    print("   https://github.com/sergeywong/cp-vton")
    print("   - Check 'Releases' tab")
    print("   - Check 'Issues' for shared links")
    
    print("\nC. Search GitHub:")
    print("   https://github.com/search?q=cp-vton+pretrained")
    print("   - Look for repositories with model files")
    
    print("\n" + "="*70)
    print("METHOD 3: Train Your Own Models (You Have Dataset!)")
    print("="*70)
    print("\nSince you have the dataset (14,221 pairs), you can train:")
    print("\n  cd backend")
    print("  python train_models_now.py")
    print("\nThis will:")
    print("  - Train GMM model (~1-3 hours)")
    print("  - Train TOM model (~1-3 hours)")
    print("  - Automatically copy to models/ folder")
    print("\nNote: Training takes time but creates models for your setup!")
    
    print("\n" + "="*70)
    print("METHOD 4: Use gdown for Google Drive Links")
    print("="*70)
    print("\nIf you find a Google Drive link, you can use:")
    print("  pip install gdown")
    print("  gdown <google_drive_file_id>")
    
    print("\n" + "="*70)
    print("RECOMMENDED ACTION")
    print("="*70)
    print("\n1. First, try METHOD 1 (manual OneDrive download)")
    print("   - Open link in different browser (Chrome, Edge, Firefox)")
    print("   - Try incognito/private mode")
    print("   - Check if link requires login")
    
    print("\n2. If that fails, try METHOD 2 (GitHub repositories)")
    print("   - Check the repositories above")
    print("   - Look in Releases and Issues")
    
    print("\n3. If still no luck, use METHOD 3 (train your own)")
    print("   - You have the dataset ready!")
    print("   - Takes time but guaranteed to work")
    
    print("\n" + "="*70)
    print("QUICK CHECK: Do You Want to Train?")
    print("="*70)
    
    response = input("\nWould you like to start training your own models? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n[INFO] Starting training process...")
        print("This will take several hours but will create the models you need.")
        print("\nRun this command:")
        print("  python train_models_now.py")
        print("\nOr train manually:")
        print("  cd cp-vton")
        print("  python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train")
    else:
        print("\n[INFO] Continue trying to download pre-trained models.")
        print("Check the GitHub repositories listed above for download links.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

