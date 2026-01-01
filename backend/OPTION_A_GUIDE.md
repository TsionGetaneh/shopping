# Option A: Download Pre-trained Models - Step by Step Guide

## 🎯 Goal
Download pre-trained GMM and TOM models so your virtual try-on system works perfectly!

## 📋 Step-by-Step Instructions

### STEP 1: Find the Models

**Option 1: GitHub (Recommended)**
1. Go to: https://github.com/sergeywong/cp-vton
2. Check the **Releases** section
3. Look for files named:
   - `gmm_final.pth` or `GMM.pth`
   - `tom_final.pth` or `TOM.pth`

**Option 2: GitHub Issues**
1. Go to: https://github.com/sergeywong/cp-vton/issues
2. Search for "pretrained" or "models" or "checkpoint"
3. Look for Google Drive or download links shared by users

**Option 3: Alternative Repositories**
- https://github.com/minar09/cp-vton
- Search GitHub for "cp-vton pretrained"

**Option 4: Google Search**
- Search: "CP-VTON pretrained models download"
- Look for Google Drive links or direct download URLs

### STEP 2: Download the Files

You need **TWO files**:
1. **GMM model** (`gmm_final.pth` or `GMM.pth`) - ~100-200 MB
2. **TOM model** (`tom_final.pth` or `TOM.pth`) - ~100-200 MB

**If you find a Google Drive link:**
- Click the link
- Click "Download" (may need to confirm virus scan warning)
- Save the files

**If you find direct download URLs:**
- Right-click → Save As
- Or use: `python download_models.py` and enter the URLs when prompted

### STEP 3: Place Files in Correct Location

**Copy or move the downloaded files to:**

```
C:\Users\getan\Documents\virtual_tryon_project\backend\models\
```

**Rename them to:**
- `GMM.pth` (even if original was `gmm_final.pth`)
- `TOM.pth` (even if original was `tom_final.pth`)

**Final structure should be:**
```
backend/
  models/
    GMM.pth  ← Place here
    TOM.pth  ← Place here
```

### STEP 4: Verify Models Are Loaded

**Run this command:**
```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python -c "from inference import load_models; gmm, tom = load_models(); print('GMM:', gmm is not None, 'TOM:', tom is not None)"
```

**You should see:**
```
[OK] Loaded GMM model (XXX.X MB)
[OK] Loaded TOM model (XXX.X MB)
GMM: True TOM: True
```

**If you see warnings:**
- Check file sizes (should be > 1 MB each)
- Check file names are exactly `GMM.pth` and `TOM.pth`
- Check files are in `backend/models/` folder

### STEP 5: Restart Server and Test

1. **Stop your current server** (CTRL+C if running)

2. **Start server:**
   ```bash
   python app.py
   ```

3. **Check console** - you should see:
   ```
   [OK] Loaded GMM model (XXX.X MB)
   [OK] Loaded TOM model (XXX.X MB)
   ```
   Instead of warnings about empty files!

4. **Open browser:** http://127.0.0.1:5000

5. **Upload images and test!**

## 🎉 What Happens Next

Once models are loaded:
- ✅ **GMM** will warp clothing to fit person's pose
- ✅ **TOM** will blend and refine for realistic results
- ✅ **Output** will look like your diagrams!

## ❓ Troubleshooting

**"Models not found"**
- Check file paths are correct
- Check file names are exactly `GMM.pth` and `TOM.pth`

**"Models too small"**
- Files might be corrupted or incomplete
- Re-download them

**"Failed to load model"**
- Models might be from different CP-VTON version
- Try downloading from the same source as your code

**"Still using fallback blending"**
- Models might not be compatible
- Check console for specific error messages

## 📝 Quick Checklist

- [ ] Found model download links
- [ ] Downloaded `gmm_final.pth` or `GMM.pth`
- [ ] Downloaded `tom_final.pth` or `TOM.pth`
- [ ] Placed files in `backend/models/`
- [ ] Renamed to `GMM.pth` and `TOM.pth`
- [ ] Verified file sizes (> 1 MB each)
- [ ] Restarted server
- [ ] Saw "[OK] Loaded" messages in console
- [ ] Tested in browser - works!

## 🚀 Ready?

Start with **STEP 1** above - find the models!

Once you have the files, place them in `backend/models/` and restart your server. The system will automatically detect and use them!


