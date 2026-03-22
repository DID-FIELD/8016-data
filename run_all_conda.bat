@echo off
chcp 65001 >nul 2>&1  :: Fix Chinese character encoding issues
echo ==============================================
echo          Face Processing Pipeline (Conda)
echo ==============================================
echo.

:: Step 0: Navigate to project root
echo [Step 0] Navigating to project root directory...
E:
cd E:\8016project
if errorlevel 1 (
    echo ❌ Failed to access project directory: E:\8016project
    pause
    exit /b 1
)
echo ✅ Successfully navigated to: E:\8016project
echo.

:: Step 1: Activate Conda environment (face_restoration)
echo [Step 1] Activating Conda environment: face_restoration...
call conda activate face_restoration
if errorlevel 1 (
    echo ❌ Failed to activate Conda environment: face_restoration
    pause
    exit /b 1
)
echo ✅ Conda environment activated: face_restoration
echo.

:: Step 2: Run core preprocessing (MTCNN alignment + mild crop)
echo [Step 2] Running preprocessing script (preprocess.py)...
echo ----------------------------------------------
python scripts\preprocess.py
if errorlevel 1 (
    echo ❌ preprocess.py failed! Check error logs above.
    pause
    exit /b 1
)
echo ----------------------------------------------
echo ✅ preprocess.py completed
echo.

:: Step 3: Run custom crop script for 04646.jpg (fix recognition failure)
echo [Step 3] Running custom crop script for 04646.jpg (fix_04646.py)...
echo ----------------------------------------------
python scripts\fix_04646.py
if errorlevel 1 (
    echo ⚠️ fix_04646.py failed (non-critical) - continue pipeline
) else (
    echo ----------------------------------------------
    echo ✅ fix_04646.py completed (custom crop for 04646.jpg)
)
echo.

:: Step 4: Run GFPGAN-style degradation
echo [Step 4] Running degradation script (degrade.py)...
echo ----------------------------------------------
python scripts\degrade.py
if errorlevel 1 (
    echo ❌ degrade.py failed! Check error logs above.
    pause
    exit /b 1
)
echo ----------------------------------------------
echo ✅ degrade.py completed
echo.

:: Step 5: Validate dataset pairing
echo [Step 5] Running dataset validation (dataset.py)...
echo ----------------------------------------------
python scripts\dataset.py
if errorlevel 1 (
    echo ⚠️ dataset.py failed (non-critical) - continue pipeline
) else (
    echo ----------------------------------------------
    echo ✅ dataset.py completed
)
echo.

:: Step 6: Generate visual comparison examples
echo [Step 6] Generating comparison examples (generate_examples.py)...
echo ----------------------------------------------
python scripts\generate_examples.py
if errorlevel 1 (
    echo ⚠️ generate_examples.py failed (non-critical) - continue pipeline
) else (
    echo ----------------------------------------------
    echo ✅ generate_examples.py completed
)
echo.

:: Final summary
echo ==============================================
echo          Pipeline Execution Complete!
echo ==============================================
echo 📂 Preprocessed output: E:\8016project\data\processed\256x256
echo 📂 Degraded output: E:\8016project\data\degraded\256x256
echo 📂 Comparison examples: E:\8016project\data\compare
echo 📌 04646.jpg fixed via custom crop (fix_04646.py)
echo.
pause