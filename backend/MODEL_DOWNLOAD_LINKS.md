# 🔗 CP-VTON Model Download Links

## ✅ Working Download Sources

### Source 1: OneDrive (CP-VTON+)
**Link:** https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP

**How to download:**
1. Open the link in your browser
2. You should see a folder with model files
3. Download:
   - `gmm_final.pth` (~100-200 MB)
   - `tom_final.pth` (~100-200 MB)
4. Place in `backend/models/` as `GMM.pth` and `TOM.pth`

---

### Source 2: GitHub - CP-VTON+ Repository
**Link:** https://github.com/minar09/cp-vton-plus

**How to download:**
1. Go to the repository
2. Check **Releases** section
3. Look for model files or download links
4. Or check **Issues** for shared Google Drive/OneDrive links

---

### Source 3: GitHub - Original CP-VTON
**Link:** https://github.com/sergeywong/cp-vton

**How to download:**
1. Go to the repository
2. Check **Releases** section
3. Check **Issues** for model download links
4. Search issues for "pretrained" or "checkpoint"

---

### Source 4: Google Drive (from GitHub Issues)
Many users share models via Google Drive. Check:
- GitHub issues in CP-VTON repositories
- Search for "cp-vton" + "drive.google.com"
- Look for folders named "checkpoint" or "pretrained"

---

## 📥 Quick Download Steps

### Method A: OneDrive (Easiest)
1. **Open:** https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP
2. **Download** the two `.pth` files
3. **Place** in `backend/models/` as `GMM.pth` and `TOM.pth`
4. **Restart** server

### Method B: GitHub Releases
1. Go to: https://github.com/minar09/cp-vton-plus
2. Check **Releases** tab
3. Download model files
4. Place in `backend/models/`

### Method C: Train Your Own
If you can't find downloads:
```bash
cd backend
python train_models_now.py
```
(Takes several hours but creates models)

---

## 🔍 Search Tips

If the links above don't work, try searching:

1. **GitHub Search:**
   - https://github.com/search
   - Search: `cp-vton pretrained model`
   - Search: `cp-vton checkpoint pth`

2. **Google Search:**
   - "CP-VTON pretrained models download"
   - "cp-vton gmm_final.pth tom_final.pth"
   - "virtual try-on cp-vton checkpoint"

3. **Check These Repositories:**
   - https://github.com/minar09/cp-vton-plus
   - https://github.com/sergeywong/cp-vton
   - https://github.com/levindabhi/cloth-virtual-try-on (might have models)

---

## ✅ After Downloading

1. **Verify file sizes:**
   - GMM.pth should be > 1 MB (usually 100-200 MB)
   - TOM.pth should be > 1 MB (usually 100-200 MB)

2. **Check location:**
   ```
   backend/models/GMM.pth
   backend/models/TOM.pth
   ```

3. **Test loading:**
   ```bash
   python -c "from inference import load_models; gmm, tom = load_models()"
   ```

4. **Restart server:**
   ```bash
   python app.py
   ```

You should see:
```
[OK] Loaded GMM model (XXX.X MB)
[OK] Loaded TOM model (XXX.X MB)
```

---

## 🚀 Start Here

**Try the OneDrive link first:**
https://1drv.ms/u/s!Ai8t8GAHdzVUiQA-o3C7cnrfGN6O?e=EaRiFP

If that doesn't work, check the GitHub repositories above!




