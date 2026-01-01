# Quick Start Guide - Virtual Try-On System

## ✅ What's Already Working

1. **Browser UI**: Open `http://127.0.0.1:5000` - you can upload person + cloth images
2. **Backend API**: Flask server at `/tryon` endpoint
3. **Pre-trained Models**: You have `models/GMM.pth` and `models/TOM.pth`

## 🔧 What I Just Fixed

### 1. Updated `inference.py`
- Now loads your pre-trained GMM and TOM models
- Implements the full pipeline: **GMM (warp cloth) → TOM (blend & refine) → Result**
- Falls back to simple blending if models can't load

### 2. Created `extract_and_setup.py`
- Automatically extracts `viton_resize.tar.gz` when you download it
- Organizes files into correct structure

## 📋 Next Steps

### Option A: Use Pre-trained Models (Browser Demo Works NOW)

1. **Start the server**:
   ```bash
   cd C:\Users\getan\Documents\virtual_tryon_project\backend
   python app.py
   ```

2. **Open browser**: `http://127.0.0.1:5000`

3. **Upload images** and click "Try On"

   The system will:
   - Load GMM model → warp clothing to fit person
   - Load TOM model → blend and refine
   - Return realistic try-on result

   **Note**: If models fail to load, it falls back to simple blending (still works, just less realistic)

### Option B: Download Dataset for Training (Optional)

If you want to **train** your own models later:

1. **Download** `viton_resize.tar.gz` from:
   ```
   https://drive.google.com/file/d/1MxCUvKxejnwWnoZ-KoCyMCXo3TLhRuTo/view
   ```
   Save it as: `backend/cp-vton/data/viton_resize.tar.gz`

2. **Extract and setup**:
   ```bash
   cd backend/cp-vton
   python extract_and_setup.py
   ```

3. **Clean pairs file**:
   ```bash
   python fix_pairs.py
   ```
   You should see: `Cleaned train_pairs.txt -> N valid pairs` (N > 0)

4. **Train models** (optional):
   ```bash
   # Train GMM
   python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train
   
   # Then train TOM
   python train.py --dataroot data --data_list train/train_pairs.txt --stage TOM --name tom_train
   ```

## 🎯 About `train_pairs.txt`

**Important**: `train_pairs.txt` is ONLY needed for **training** models. 

- ❌ **NOT needed** for browser demo (uses pre-trained models)
- ✅ **Needed** only if you want to train your own models

The file was empty because:
- It lists which person images match which cloth images
- Your dataset folder doesn't have the full VITON dataset yet
- Once you download and extract the dataset, `fix_pairs.py` will automatically fill it

## 🚀 Current Status

- ✅ **Browser demo**: Ready to use (uses pre-trained models)
- ✅ **Inference code**: Updated to use GMM + TOM pipeline
- ⏳ **Training**: Needs dataset download (optional)

## Test It Now!

```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python app.py
```

Then open `http://127.0.0.1:5000` and try uploading images!


