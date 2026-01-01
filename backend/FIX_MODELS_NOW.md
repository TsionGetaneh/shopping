# 🔥 FIX GMM and TOM Models - Quick Solution

## ✅ Good News!

I found a **OneDrive link** with pre-trained models! Here's how to get them:

## 📥 Method 1: Download from OneDrive (FASTEST - 10 minutes)

### Step 1: Open the Link
Go to this link in your browser:
```
https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
```

### Step 2: Download Files
You should see a folder with model files. Download:
- `gmm_final.pth` (or `GMM.pth`)
- `tom_final.pth` (or `TOM.pth`)

### Step 3: Place Files
Copy the downloaded files to:
```
C:\Users\getan\Documents\virtual_tryon_project\backend\models\GMM.pth
C:\Users\getan\Documents\virtual_tryon_project\backend\models\TOM.pth
```

**Important:** Rename them to exactly `GMM.pth` and `TOM.pth` if needed.

### Step 4: Verify
Run this to check:
```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python -c "from inference import load_models; gmm, tom = load_models(); print('GMM:', gmm is not None, 'TOM:', tom is not None)"
```

You should see:
```
[OK] Loaded GMM model (XXX.X MB)
[OK] Loaded TOM model (XXX.X MB)
GMM: True TOM: True
```

### Step 5: Restart Server
```bash
python app.py
```

Now you should see:
```
[OK] Loaded GMM model (XXX.X MB)
[OK] Loaded TOM model (XXX.X MB)
```

Instead of warnings! 🎉

---

## 🚀 Method 2: Train Your Own (Takes Hours)

Since you have the dataset (14,221 pairs!), you can train:

```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python train_models_now.py
```

**This will:**
- Train GMM model (~1-3 hours)
- Train TOM model (~1-3 hours)
- Copy models to `models/` folder automatically

**Note:** Training takes time but creates models specifically for your setup.

---

## 🎯 Quick Action Plan

**RIGHT NOW:**
1. Open: https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
2. Download the two `.pth` files
3. Place in `backend/models/` as `GMM.pth` and `TOM.pth`
4. Restart server
5. Done! ✅

**Total time: ~10-15 minutes**

---

## ❓ Troubleshooting

**"OneDrive link doesn't work"**
- Try opening in different browser
- Check if link requires login
- Look for alternative download links in GitHub issues

**"Files downloaded but still showing warnings"**
- Check file sizes (should be > 1 MB each)
- Check file names are exactly `GMM.pth` and `TOM.pth`
- Check files are in `backend/models/` (not `cp-vton/models/`)

**"Models load but results still look bad"**
- Models might need pose/parsing data (we're using simplified versions)
- Try different person/cloth images
- Check console for any errors

---

## ✅ Once Models Are Loaded

Your system will:
- ✅ Use **real GMM** to warp clothing
- ✅ Use **real TOM** to blend and refine
- ✅ Generate **realistic try-on results** like your diagrams!

**Start with Method 1 (OneDrive download) - it's fastest!**

