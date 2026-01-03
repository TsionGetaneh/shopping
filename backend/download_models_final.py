#!/usr/bin/env python
"""
Final script to download CP-VTON models from various sources.
"""
import os
import sys
import requests
from pathlib import Path

def download_file(url, dest_path, description=""):
    """Download file with progress."""
    print(f"\n[*] Downloading {description or os.path.basename(dest_path)}...")
    print(f"    From: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        percent = 100 * downloaded / total
                        size_mb = downloaded / (1024*1024)
                        total_mb = total / (1024*1024)
                        print(f"\r    Progress: {percent:.1f}% ({size_mb:.1f}/{total_mb:.1f} MB)", end='', flush=True)
        
        file_size = os.path.getsize(dest_path)
        if file_size > 1000:  # At least 1 KB
            print(f"\n[OK] Downloaded: {os.path.basename(dest_path)} ({file_size/(1024*1024):.1f} MB)")
            return True
        else:
            print(f"\n[X] File too small ({file_size} bytes) - download may have failed")
            os.remove(dest_path)
            return False
    except Exception as e:
        print(f"\n[X] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CP-VTON Models Downloader - Finding Working Links")
    print("="*70)
    
    # Known model sources (will try these)
    model_sources = [
        {
            'name': 'OneDrive (CP-VTON+)',
            'gmm': 'https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP',
            'tom': 'https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP',
            'note': 'OneDrive share - requires manual download'
        }
    ]
    
    print("\n[INFO] Searching for model download sources...")
    print("\nKnown sources:")
    print("1. OneDrive: https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP")
    print("2. GitHub: https://github.com/minar09/cp-vton-plus")
    print("3. GitHub: https://github.com/sergeywong/cp-vton")
    
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD METHOD (MOST RELIABLE)")
    print("="*70)
    print("\nSince automatic downloads are unreliable, here's the best method:\n")
    
    print("STEP 1: Open OneDrive Link")
    print("  https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP")
    print("\n  Or try these alternatives:")
    print("  - https://github.com/minar09/cp-vton-plus (check releases)")
    print("  - https://github.com/sergeywong/cp-vton (check issues/releases)")
    
    print("\nSTEP 2: Download Files")
    print("  Look for:")
    print("  - gmm_final.pth (or GMM.pth) - ~100-200 MB")
    print("  - tom_final.pth (or TOM.pth) - ~100-200 MB")
    
    print("\nSTEP 3: Place Files")
    print(f"  Copy to: {models_dir.absolute()}/")
    print("  Rename to: GMM.pth and TOM.pth")
    
    print("\nSTEP 4: Verify")
    print("  Run: python -c \"from inference import load_models; gmm, tom = load_models()\"")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Direct URLs (if you have them)")
    print("="*70)
    
    # Check if user has URLs
    print("\nIf you found direct download URLs, enter them below:")
    print("(Press Enter to skip and use manual method)")
    
    gmm_url = input("\nGMM model URL (or press Enter to skip): ").strip()
    tom_url = input("TOM model URL (or press Enter to skip): ").strip()
    
    success = False
    
    if gmm_url:
        gmm_path = models_dir / "GMM.pth"
        if download_file(gmm_url, str(gmm_path), "GMM model"):
            success = True
    
    if tom_url:
        tom_path = models_dir / "TOM.pth"
        if download_file(tom_url, str(tom_path), "TOM model"):
            success = True
    
    if success:
        print("\n" + "="*70)
        print("[OK] Models downloaded!")
        print("="*70)
        print("\nRestart your server:")
        print("  python app.py")
        print("\nYou should see:")
        print("  [OK] Loaded GMM model (XXX.X MB)")
        print("  [OK] Loaded TOM model (XXX.X MB)")
    else:
        print("\n" + "="*70)
        print("Use Manual Download Method Above")
        print("="*70)
        print("\nOr train your own models:")
        print("  python train_models_now.py")
        print("\n(Requires dataset - takes several hours)")

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



