@echo off
REM Script d'entraînement complet pour EvaDentalAI (Windows)
REM Usage: scripts\run_training.bat [options]

setlocal enabledelayedexpansion

REM Configuration par défaut
set NUM_IMAGES=200
set EPOCHS=100
set BATCH_SIZE=16
set MODEL_SIZE=yolov8n.pt
set DEVICE=auto

REM Parser les arguments
:parse_args
if "%~1"=="" goto :start_training
if "%~1"=="--num-images" (
    set NUM_IMAGES=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set MODEL_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: %0 [options]
    echo Options:
    echo   --num-images N    Nombre d'images a generer (defaut: 200)
    echo   --epochs N        Nombre d'epoques (defaut: 100)
    echo   --batch-size N    Taille du batch (defaut: 16)
    echo   --model MODEL     Modele de base (defaut: yolov8n.pt)
    echo   --device DEVICE   Device (cpu/cuda/auto, defaut: auto)
    echo   --help            Afficher cette aide
    exit /b 0
)
echo Option inconnue: %~1
exit /b 1

:start_training
echo.
echo 🦷 Entrainement EvaDentalAI
echo ==================================
echo Configuration:
echo   Images: %NUM_IMAGES%
echo   Epoques: %EPOCHS%
echo   Batch size: %BATCH_SIZE%
echo   Modele: %MODEL_SIZE%
echo   Device: %DEVICE%
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installe
    exit /b 1
)

REM Vérifier les dépendances
echo ℹ️  Verification des dependances...
python -c "import ultralytics, torch, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Installation des dependances...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Erreur lors de l'installation des dependances
        exit /b 1
    )
)

REM Étape 1: Préparation du dataset
echo.
echo ℹ️  Etape 1/3: Preparation du dataset
python scripts\prepare_dataset.py --num-images %NUM_IMAGES%
if errorlevel 1 (
    echo ❌ Erreur lors de la preparation du dataset
    exit /b 1
)
echo ✅ Dataset prepare

REM Étape 2: Entraînement
echo.
echo ℹ️  Etape 2/3: Entrainement du modele
python scripts\train_model.py --config config\data.yaml --model %MODEL_SIZE% --epochs %EPOCHS% --batch-size %BATCH_SIZE% --device %DEVICE% --export --validate
if errorlevel 1 (
    echo ❌ Erreur lors de l'entrainement
    exit /b 1
)
echo ✅ Modele entraine

REM Étape 3: Export et optimisation
echo.
echo ℹ️  Etape 3/3: Export et optimisation

REM Trouver le meilleur modèle
for /r models %%i in (best.pt) do set BEST_MODEL=%%i

if "%BEST_MODEL%"=="" (
    echo ❌ Modele best.pt non trouve
    exit /b 1
)

echo ℹ️  Export du modele: %BEST_MODEL%
python scripts\export_model.py --model "%BEST_MODEL%" --format all
if errorlevel 1 (
    echo ⚠️  Erreur lors de l'export, mais l'entrainement est termine
)

echo ✅ Export termine

REM Résumé final
echo.
echo ✅ 🎉 Entrainement complet termine!
echo.
echo ℹ️  Fichiers generes:
echo   - Modele: %BEST_MODEL%
echo   - ONNX: models\model.onnx
echo   - ONNX optimise: models\model_optimized.onnx
echo   - TorchScript: models\model.pt
echo.
echo ℹ️  Prochaines etapes:
echo   1. Tester le modele: python scripts\predict.py --model "%BEST_MODEL%" --image path\to\image.jpg
echo   2. Lancer l'API: python api\main.py --model "%BEST_MODEL%"
echo   3. Deployer avec Docker: docker-compose up
echo.

REM Test rapide si une image de test existe
if exist "data\processed\test\images\0000.jpg" (
    echo ℹ️  Test rapide du modele...
    python scripts\predict.py --model "%BEST_MODEL%" --image "data\processed\test\images\0000.jpg" --save
    echo ✅ Test termine, verifiez le dossier output\
)

echo.
echo Appuyez sur une touche pour continuer...
pause >nul
