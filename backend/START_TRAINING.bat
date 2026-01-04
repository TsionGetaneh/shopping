@echo off
echo ======================================================================
echo CP-VTON Model Training - Quick Start
echo ======================================================================
echo.
echo This will train GMM and TOM models using your dataset.
echo Training will take several hours but will create the models you need.
echo.
echo Press any key to start GMM training, or CTRL+C to cancel...
pause

cd cp-vton
python train.py --dataroot data --data_list train/train_pairs.txt --stage GMM --name gmm_train --batch-size 4 --workers 2 --shuffle --keep_step 50000 --decay_step 50000

echo.
echo ======================================================================
echo GMM Training Complete!
echo ======================================================================
echo.
echo Press any key to start TOM training, or CTRL+C to skip...
pause

python train.py --dataroot data --data_list train/train_pairs.txt --stage TOM --name tom_train --batch-size 4 --workers 2 --shuffle --keep_step 50000 --decay_step 50000

echo.
echo ======================================================================
echo Copying models to models/ folder...
echo ======================================================================

cd ..
if exist cp-vton\checkpoints\gmm_train\gmm_final.pth (
    copy cp-vton\checkpoints\gmm_train\gmm_final.pth models\GMM.pth
    echo [OK] Copied GMM model
) else (
    echo [X] GMM model not found
)

if exist cp-vton\checkpoints\tom_train\tom_final.pth (
    copy cp-vton\checkpoints\tom_train\tom_final.pth models\TOM.pth
    echo [OK] Copied TOM model
) else (
    echo [X] TOM model not found
)

echo.
echo ======================================================================
echo Training Complete!
echo ======================================================================
echo.
echo You can now restart your server:
echo   python app.py
echo.
pause




