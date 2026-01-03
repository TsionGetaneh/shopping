# 🔥 DOWNLOAD GMM AND TOM MODELS - EXACT LINKS

## ✅ CONFIRMED DOWNLOAD LINK

**OneDrive Link (CP-VTON+ Models):**
```
https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
```

## 📥 STEP-BY-STEP DOWNLOAD

### Step 1: Open the Link
1. **Copy this link:** https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
2. **Paste in your browser** and press Enter
3. You should see a OneDrive folder with model files

### Step 2: Download Files
Download these **TWO files**:
- `gmm_final.pth` (or any file with "gmm" in name) - ~100-200 MB
- `tom_final.pth` (or any file with "tom" in name) - ~100-200 MB

### Step 3: Place Files

**Option A: In models/ folder (RECOMMENDED)**
```
C:\Users\getan\Documents\virtual_tryon_project\backend\models\GMM.pth
C:\Users\getan\Documents\virtual_tryon_project\backend\models\TOM.pth
```

**Option B: In checkpoints/ folder (also works)**
```
C:\Users\getan\Documents\virtual_tryon_project\backend\cp-vton\checkpoints\GMM\gmm_final.pth
C:\Users\getan\Documents\virtual_tryon_project\backend\cp-vton\checkpoints\TOM\tom_final.pth
```

**The code will automatically find them in either location!**

### Step 4: Verify
Run this command:
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

**Instead of warnings!** ✅

---

## 🔄 ALTERNATIVE SOURCES (if OneDrive doesn't work)

### GitHub Repository 1:
**https://github.com/minar09/cp-vton-plus**
- Check **Releases** section
- Check **Issues** for download links

### GitHub Repository 2:
**https://github.com/sergeywong/cp-vton**
- Check **Releases** section  
- Check **Issues** for model links

---

## ✅ QUICK CHECKLIST

- [ ] Opened OneDrive link
- [ ] Downloaded `gmm_final.pth`
- [ ] Downloaded `tom_final.pth`
- [ ] Placed files in `backend/models/` OR `backend/cp-vton/checkpoints/`
- [ ] Verified file sizes (> 1 MB each)
- [ ] Ran verification command
- [ ] Restarted server
- [ ] Saw "[OK] Loaded" messages

---

## 🚀 START HERE

**Click this link:** https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP

Download the two `.pth` files and place them in `backend/models/` as `GMM.pth` and `TOM.pth`.

**That's it!** The code will automatically find and load them.



