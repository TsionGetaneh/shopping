# 🔧 OneDrive Download Troubleshooting

## ❌ Problem: Can't Download from OneDrive Link

The link `https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP` might not work because:
- Requires Microsoft account login
- Link might be expired or restricted
- Browser compatibility issues
- Network/firewall restrictions

## ✅ Solutions

### Solution 1: Try Different Browser
1. **Try Microsoft Edge** (best for OneDrive)
2. **Try Chrome** in incognito mode
3. **Try Firefox**

### Solution 2: Check if Login Required
1. Open the link
2. If it asks to sign in, create/login to Microsoft account
3. Then try downloading

### Solution 3: Use Alternative Sources

**GitHub Repository 1:**
- https://github.com/minar09/cp-vton-plus
- Check **Releases** tab
- Check **Issues** for download links

**GitHub Repository 2:**
- https://github.com/sergeywong/cp-vton
- Check **Releases** tab
- Check **Issues** for shared Google Drive/OneDrive links

**GitHub Search:**
- Go to: https://github.com/search
- Search: `cp-vton pretrained model`
- Search: `cp-vton checkpoint pth`
- Look for repositories with model files

### Solution 4: Train Your Own Models (RECOMMENDED)

Since you **already have the dataset** (14,221 pairs in train_pairs.txt), you can train:

```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python train_models_now.py
```

**This will:**
- Train GMM model (~1-3 hours on CPU, faster on GPU)
- Train TOM model (~1-3 hours on CPU, faster on GPU)
- Automatically save to `models/` folder

**Advantages:**
- ✅ Guaranteed to work
- ✅ Models trained for your exact setup
- ✅ No download issues

**Disadvantages:**
- ⏰ Takes several hours
- 💻 Better with GPU (CPU is slower)

## 🚀 Quick Decision Guide

**If OneDrive doesn't work:**
1. ✅ **First:** Check GitHub repositories (5 minutes)
2. ✅ **Then:** Train your own models (several hours but guaranteed)

**To start training:**
```bash
cd backend
python train_models_now.py
```

The script will guide you through the process!



