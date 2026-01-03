# Alternative Sources for CP-VTON Pre-trained Models

Since the main repository doesn't have direct downloads, try these:

## 🔍 Search Strategies

### 1. GitHub Search
- Go to: https://github.com/search
- Search: `cp-vton pretrained model`
- Search: `cp-vton checkpoint pth`
- Look in repositories' releases or issues

### 2. Specific Repositories to Check

**CP-VTON+ (Enhanced version):**
- https://github.com/minar09/cp-vton
- Check releases and issues

**Other implementations:**
- Search GitHub for "virtual try-on" + "cp-vton"
- Check their releases/checkpoints folders

### 3. Google Drive / OneDrive Links
- Check GitHub issues for shared Drive links
- Search: "cp-vton" + "drive.google.com"
- Look for folders with "checkpoint" or "pretrained"

### 4. Academic Paper Repositories
- Papers with Code: https://paperswithcode.com
- Search for "CP-VTON" or "Characteristic-Preserving Virtual Try-On"

### 5. Hugging Face
- https://huggingface.co/models
- Search: "cp-vton" or "virtual try-on"

## 📥 If You Find Models

**File names might be:**
- `gmm_final.pth` or `GMM.pth`
- `tom_final.pth` or `TOM.pth`
- `checkpoint.pth` (might need to check which stage)

**Place them in:**
```
backend/models/GMM.pth
backend/models/TOM.pth
```

## 🚀 Alternative: Train Your Own

Since you have the dataset (14,221 pairs!), you can train:

```bash
cd backend
python train_models_now.py
```

This will take several hours but will create the models you need.



