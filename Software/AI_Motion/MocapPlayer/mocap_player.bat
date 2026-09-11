set CONDA_PATH=C:\Users\%USERNAME%\anaconda3
set ENV_NAME=premiere
set PYTHON_VERSION=3.10

call %CONDA_PATH%\Scripts\activate.bat
call conda activate %ENV_NAME%
python mocap_player.py
pause