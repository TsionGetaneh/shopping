# 🚀 Train Your Own Models - Simple Guide

## ✅ Why This Works

You **already have the dataset** (14,221 pairs in train_pairs.txt), so you can train models yourself!

**Advantages:**
- ✅ **Guaranteed to work** - no download issues
- ✅ **Models trained for your exact setup**
- ✅ **Full control**

**Time:** ~2-6 hours (depending on CPU/GPU)

## 📋 Quick Start

### Step 1: Start Training GMM
```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend\cp-vton
python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train --batch-size 4 --workers 2 --shuffle --keep_step 50000 --decay_step 50000
```

**This will:**
- Train GMM model
- Save checkpoints every 5000 steps
- Take ~1-3 hours

**You can stop anytime with CTRL+C** - checkpoints are saved!

### Step 2: After GMM is Done, Train TOM
```bash
python train.py --dataroot data --data_list train/train_pairs.txt --stage TOM --name tom_train --batch-size 4 --workers 2 --shuffle --keep_step 50000 --decay_step 50000
```

### Step 3: Copy Models
```bash
cd ..
copy cp-vton\checkpoints\gmm_train\gmm_final.pth models\GMM.pth
copy cp-vton\checkpoints\tom_train\tom_final.pth models\TOM.pth
```

### Step 4: Restart Server
```bash
python app.py
```

Done! ✅

## 🎯 Or Use the Automated Script

```bash
cd C:\Users\getan\Documents\virtual_tryon_project\backend
python train_models_now.py
```

This will guide you through everything!

## ⚠️ Notes

- **Training takes time** - be patient
- **Better with GPU** - CPU is slower but works
- **You can stop/resume** - checkpoints are saved
- **First training** - models will be created in `cp-vton/checkpoints/`

## ✅ This is the Most Reliable Method!

Since OneDrive downloads are failing, training is your best option.

