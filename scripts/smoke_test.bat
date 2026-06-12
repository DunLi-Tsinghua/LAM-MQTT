@echo off
setlocal
python -c "import pandas, numpy, sklearn, pyarrow; print('LAM-MQTT smoke test OK')"
endlocal
