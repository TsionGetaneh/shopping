#!/usr/bin/env python
"""Download CP-VTON pre-trained models."""
import os
import sys
import requests
from pathlib import Path

def download_file(url, dest_path, description=""):
    """Download file with progress."""
    print(f"\n[*] Downloading {description or os.path.basename(dest_path)}...")
    print(f"    URL: {url}")
    print(f"    To: {dest_path}")
    
    try:
        response = requests.get(url, stream=True, timeout=60)
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
        
        print(f"\n[OK] Downloaded: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"\n[X] Download failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CP-VTON Pre-trained Models Downloader")
    print("="*70)
    print("\nThis script will help you download the pre-trained GMM and TOM models.")
    print("These models are required for realistic virtual try-on results.")
    print("\n" + "="*70)
    
    # Common CP-VTON model URLs (these may need to be updated)
    # Try to find models from various sources
    
    print("\n[INFO] Searching for model download sources...")
    print("\nCommon sources for CP-VTON models:")
    print("1. GitHub releases: https://github.com/sergeywong/cp-vton")
    print("2. Google Drive links shared in GitHub issues")
    print("3. Hugging Face model hub")
    print("4. Other CP-VTON implementations")
    
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD REQUIRED")
    print("="*70)
    print("\nDue to Google Drive restrictions, automatic download is difficult.")
    print("Please follow these steps:\n")
    
    print("STEP 1: Find the models")
    print("  - Check: https://github.com/sergeywong/cp-vton (releases/issues)")
    print("  - Search: 'CP-VTON pretrained models download'")
    print("  - Look for files: gmm_final.pth and tom_final.pth")
    print("  - Or check: https://github.com/minar09/cp-vton")
    
    print("\nSTEP 2: Download the files")
    print("  - Download gmm_final.pth (~100-200 MB)")
    print("  - Download tom_final.pth (~100-200 MB)")
    
    print("\nSTEP 3: Place them here")
    print(f"  {models_dir.absolute()}/GMM.pth")
    print(f"  {models_dir.absolute()}/TOM.pth")
    
    print("\nSTEP 4: Verify")
    print("  Run: python -c \"from inference import load_models; gmm, tom = load_models(); print('GMM:', gmm is not None, 'TOM:', tom is not None)\"")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Try direct download")
    print("="*70)
    
    # Try some known URLs (user can update these)
    known_urls = {
        'gmm': None,  # Add URL if found
        'tom': None   # Add URL if found
    }
    
    # Check if user wants to try direct download
    response = input("\nDo you have direct download URLs? (y/N): ").strip().lower()
    
    if response == 'y':
        gmm_url = input("Enter GMM model URL: ").strip()
        tom_url = input("Enter TOM model URL: ").strip()
        
        if gmm_url:
            gmm_path = models_dir / "GMM.pth"
            if download_file(gmm_url, str(gmm_path), "GMM model"):
                print("[OK] GMM model downloaded successfully!")
        
        if tom_url:
            tom_path = models_dir / "TOM.pth"
            if download_file(tom_url, str(tom_path), "TOM model"):
                print("[OK] TOM model downloaded successfully!")
        
        # Verify
        gmm_exists = (models_dir / "GMM.pth").exists() and (models_dir / "GMM.pth").stat().st_size > 1000
        tom_exists = (models_dir / "TOM.pth").exists() and (models_dir / "TOM.pth").stat().st_size > 1000
        
        if gmm_exists and tom_exists:
            print("\n" + "="*70)
            print("[OK] Both models downloaded successfully!")
            print("="*70)
            print("\nYou can now restart the server and use the full GMM→TOM pipeline!")
            return True
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Search for CP-VTON models online")
    print("2. Download gmm_final.pth and tom_final.pth")
    print(f"3. Place them in: {models_dir.absolute()}")
    print("4. Restart your server")
    print("\nThe models are typically 100-200 MB each.")
    print("Once downloaded, the system will automatically use them!")
    
    return False

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
