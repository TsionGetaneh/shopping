#!/usr/bin/env python
"""
Download CP-VTON models from OneDrive link.
Based on search results, models are available at:
https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
"""
import os
import sys
import requests
from pathlib import Path

def download_from_onedrive_share(share_url, dest_path):
    """Download from OneDrive share link."""
    print(f"\n[*] Downloading from OneDrive...")
    print(f"    URL: {share_url}")
    print(f"    To: {dest_path}")
    
    # Convert OneDrive share link to direct download
    # Format: https://1drv.ms/u/s!... -> https://onedrive.live.com/download?...
    if "1drv.ms" in share_url:
        # Try to get direct download link
        share_id = share_url.split("!")[-1].split("?")[0]
        direct_url = f"https://onedrive.live.com/download?cid={share_id}&resid={share_id}&authkey=!{share_id}"
    else:
        direct_url = share_url
    
    try:
        # First, try to get the actual download link
        response = requests.get(share_url, allow_redirects=True, timeout=30)
        final_url = response.url
        
        print(f"    Following redirects...")
        
        # Download the file
        response = requests.get(final_url, stream=True, timeout=60)
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
    print("CP-VTON Models Downloader (OneDrive)")
    print("="*70)
    
    # OneDrive share link from search results
    onedrive_base = "https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP"
    
    print("\n[INFO] Found OneDrive link for CP-VTON+ models")
    print(f"Link: {onedrive_base}")
    print("\n[WARN] OneDrive links often require manual download.")
    print("The script will try, but you may need to download manually.\n")
    
    print("="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS (RECOMMENDED)")
    print("="*70)
    print("\n1. Open this link in your browser:")
    print(f"   {onedrive_base}")
    print("\n2. You should see a folder with model files")
    print("3. Download these files:")
    print("   - gmm_final.pth (or GMM.pth)")
    print("   - tom_final.pth (or TOM.pth)")
    print("\n4. Place them in:")
    print(f"   {models_dir.absolute()}/GMM.pth")
    print(f"   {models_dir.absolute()}/TOM.pth")
    print("\n5. Restart your server!")
    print("="*70)
    
    # Try automatic download
    response = input("\nTry automatic download? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n[INFO] Attempting to download from OneDrive...")
        print("[WARN] This may fail due to OneDrive restrictions.")
        print("If it fails, use manual download above.\n")
        
        # Note: OneDrive share links are tricky - this might not work
        # User should download manually
        print("[X] Automatic download from OneDrive share links is not reliable.")
        print("Please use manual download method above.")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Train Your Own Models")
    print("="*70)
    print("\nSince you have the dataset, you can also train models:")
    print("  python train_models_now.py")
    print("\nThis takes several hours but creates the models you need.")
    print("="*70)

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




