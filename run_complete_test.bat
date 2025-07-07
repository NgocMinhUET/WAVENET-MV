@echo off
echo Setting environment variables...
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1

echo Running Complete WAVENET-MV Evaluation...
python complete_real_test.py

echo.
echo Complete evaluation finished!
echo Check the generated files and visualizations.
pause 