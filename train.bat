@echo off
cd /d "%USERPROFILE%\project\poultry_paper"
python train_pipeline.py > training_output.txt 2>&1
