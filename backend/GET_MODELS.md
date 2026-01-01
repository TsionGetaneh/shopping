# 🔥 CRITICAL: Get Pre-trained Models for Full GMM→TOM Pipeline

## Current Status

Your `models/GMM.pth` and `models/TOM.pth` files are **empty (0 bytes)**. 

**Without real model weights, the system falls back to simple blending** - which is why you're seeing images just "aligned together" instead of properly fitted like in your diagrams.

## ✅ What You Need

To get **real virtual try-on** (like your diagrams), you need:

1. **GMM model** (~100-200 MB): Warps clothing to fit person's pose
2. **TOM model** (~100-200 MB): Blends and refines for realistic result

## 📥 How to Get Models

### Option 1: Download Pre-trained Models (FASTEST)

**Search for CP-VTON pre-trained models:**

1. **GitHub Releases**: Check the original CP-VTON repository releases
   - https://github.com/sergeywong/cp-vton
   - Look for releases with `gmm_final.pth` and `tom_final.pth`

2. **Google Drive / OneDrive**: Many implementations share models
   - Search: "CP-VTON pretrained models download"
   - Look for `.pth` files

3. **Hugging Face**: Some models are hosted there
   - Search: "cp-vton" on huggingface.co

**Once downloaded:**

```bash
# Place files here:
backend/models/GMM.pth
backend/models/TOM.pth
```

**Then restart the server** - it will automatically use the real models!

### Option 2: Train Your Own Models (Takes Time)

If you want to train from scratch:

1. **Download dataset** (you already tried this):
   ```bash
   cd backend/cp-vton
   python data_download.py
   # Or manually download viton_resize.tar.gz
   ```

2. **Extract and setup**:
   ```bash
   python extract_and_setup.py
   python fix_pairs.py
   ```

3. **Train GMM**:
   ```bash
   python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train --batch-size 4 --workers 2 --shuffle
   ```

4. **Train TOM** (after GMM):
   ```bash
   python train.py --dataroot data --data_list train/train_pairs.txt --stage TOM --name tom_train --batch-size 4 --workers 2 --shuffle
   ```

5. **Copy trained models**:
   ```bash
   cp checkpoints/gmm_train/gmm_final.pth ../models/GMM.pth
   cp checkpoints/tom_train/tom_final.pth ../models/TOM.pth
   ```

## 🎯 What Happens When Models Are Loaded

**With real models:**
- ✅ **GMM**: Warps clothing to fit person's body shape and pose
- ✅ **TOM**: Blends warped clothing seamlessly with person
- ✅ **Result**: Realistic try-on like your diagrams

**Without models (current):**
- ⚠️ Simple blending (just overlays images)
- ⚠️ No geometric warping
- ⚠️ No realistic refinement

## 🚀 Quick Test

After placing real models:

```bash
cd backend
python app.py
```

You should see:
```
[OK] Loaded GMM model (XXX.X MB)
[OK] Loaded TOM model (XXX.X MB)
```

Instead of:
```
[WARN] GMM model file exists but is too small
```

## 📝 Next Steps

1. **Get models** (Option 1 is fastest)
2. **Place in `backend/models/`**
3. **Restart server**
4. **Test with real images**

The UI is already beautiful and ready - it just needs real models to work perfectly! 🎉


