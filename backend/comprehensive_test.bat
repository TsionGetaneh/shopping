@echo off
cd /d "c:\Users\getan\Documents\virtual_tryon_project\backend"
echo Running Comprehensive CP-VTON Test...
echo.
echo This will create multiple debug images to show each step:
echo   - comp_person.jpg (input person)
echo   - comp_cloth.jpg (input cloth)
echo   - comp_pose.jpg (pose keypoints)
echo   - comp_parse.jpg (parsing mask)
echo   - comp_agnostic.jpg (agnostic representation)
echo   - comp_warped.jpg (warped cloth)
echo   - comp_rendered.jpg (TOM rendered)
echo   - comp_mask.jpg (composite mask)
echo   - comp_result.jpg (final result)
echo.
python comprehensive_test.py
echo.
echo Test completed! Check the generated images.
pause
