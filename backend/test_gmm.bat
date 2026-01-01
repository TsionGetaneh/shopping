@echo off
cd /d "c:\Users\getan\Documents\virtual_tryon_project\backend"
echo Testing GMM warping only...
python test_gmm.py
echo.
echo Check the generated images:
echo   - gmm_test_person.jpg
echo   - gmm_test_cloth.jpg  
echo   - gmm_original.jpg
echo   - gmm_warped.jpg
echo   - gmm_grid.jpg
pause
