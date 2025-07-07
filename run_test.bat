@echo off
echo Setting environment variables...
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1

echo Running WAVENET-MV test...
python simple_real_test.py

echo.
echo Test completed!
pause 