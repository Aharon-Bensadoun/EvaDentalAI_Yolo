#!/usr/bin/env python3
"""
Script de test pour vérifier l'installation d'EvaDentalAI
Vérifie que tous les composants sont fonctionnels
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib

def test_imports():
    """Test des imports des dépendances"""
    print("🔍 Test des imports...")
    
    required_modules = [
        'ultralytics',
        'torch',
        'torchvision', 
        'cv2',
        'numpy',
        'matplotlib',
        'PIL',
        'fastapi',
        'uvicorn',
        'yaml'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Modules manquants: {', '.join(failed_imports)}")
        print("💡 Installez avec: pip install -r requirements.txt")
        return False
    
    print("✅ Tous les imports réussis!")
    return True

def test_cuda():
    """Test de la disponibilité CUDA"""
    print("\n🖥️  Test CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"  📊 Nombre de GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("  ⚠️  CUDA non disponible, utilisation du CPU")
            return False
    except Exception as e:
        print(f"  ❌ Erreur CUDA: {e}")
        return False

def test_yolo():
    """Test de YOLO"""
    print("\n🤖 Test YOLO...")
    
    try:
        from ultralytics import YOLO
        
        # Tester le chargement d'un modèle pré-entraîné
        model = YOLO('yolov8n.pt')
        print("  ✅ Modèle YOLO chargé")
        
        # Test de prédiction sur une image factice
        import numpy as np
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image)
        print("  ✅ Prédiction YOLO fonctionnelle")
        
        return True
    except Exception as e:
        print(f"  ❌ Erreur YOLO: {e}")
        return False

def test_dataset_generation():
    """Test de génération du dataset"""
    print("\n📊 Test génération dataset...")
    
    try:
        # Vérifier que le script existe
        script_path = Path("scripts/prepare_dataset.py")
        if not script_path.exists():
            print("  ❌ Script prepare_dataset.py non trouvé")
            return False
        
        # Test avec un petit dataset
        result = subprocess.run([
            sys.executable, "scripts/prepare_dataset.py", 
            "--num-images", "5"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ Génération dataset réussie")
            
            # Vérifier que les fichiers ont été créés
            data_dir = Path("data/processed")
            if data_dir.exists():
                train_images = list((data_dir / "train/images").glob("*.jpg"))
                if train_images:
                    print(f"  ✅ {len(train_images)} images générées")
                    return True
            
        print(f"  ❌ Erreur génération: {result.stderr}")
        return False
        
    except subprocess.TimeoutExpired:
        print("  ❌ Timeout lors de la génération")
        return False
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_training():
    """Test d'entraînement rapide"""
    print("\n🏋️  Test entraînement...")
    
    try:
        # Vérifier que le script existe
        script_path = Path("scripts/train_model.py")
        if not script_path.exists():
            print("  ❌ Script train_model.py non trouvé")
            return False
        
        # Test avec un entraînement très court
        result = subprocess.run([
            sys.executable, "scripts/train_model.py",
            "--config", "config/data.yaml",
            "--epochs", "2",
            "--batch-size", "4",
            "--device", "cpu"  # Force CPU pour éviter les problèmes GPU
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("  ✅ Entraînement réussi")
            
            # Vérifier que le modèle a été créé
            model_path = Path("models/best.pt")
            if model_path.exists():
                print("  ✅ Modèle sauvegardé")
                return True
            else:
                print("  ⚠️  Modèle non trouvé, mais entraînement OK")
                return True
        
        print(f"  ❌ Erreur entraînement: {result.stderr}")
        return False
        
    except subprocess.TimeoutExpired:
        print("  ❌ Timeout lors de l'entraînement")
        return False
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_prediction():
    """Test de prédiction"""
    print("\n🔍 Test prédiction...")
    
    try:
        # Vérifier qu'il y a une image de test
        test_images = list(Path("data/processed/test/images").glob("*.jpg"))
        if not test_images:
            print("  ❌ Aucune image de test trouvée")
            return False
        
        test_image = test_images[0]
        
        # Vérifier que le script existe
        script_path = Path("scripts/predict.py")
        if not script_path.exists():
            print("  ❌ Script predict.py non trouvé")
            return False
        
        # Test de prédiction
        result = subprocess.run([
            sys.executable, "scripts/predict.py",
            "--model", "models/best.pt",
            "--image", str(test_image)
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("  ✅ Prédiction réussie")
            return True
        
        print(f"  ❌ Erreur prédiction: {result.stderr}")
        return False
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        return False

def test_api():
    """Test de l'API"""
    print("\n🌐 Test API...")
    
    try:
        # Vérifier que le script existe
        script_path = Path("api/main.py")
        if not script_path.exists():
            print("  ❌ Script main.py non trouvé")
            return False
        
        # Test d'import de l'API
        sys.path.append(str(Path("api").absolute()))
        import main
        print("  ✅ API importée avec succès")
        
        # Vérifier que l'API peut être instanciée
        from fastapi import FastAPI
        app = FastAPI()
        print("  ✅ FastAPI fonctionnel")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erreur API: {e}")
        return False

def test_file_structure():
    """Test de la structure des fichiers"""
    print("\n📁 Test structure fichiers...")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
        "config/data.yaml",
        "scripts/prepare_dataset.py",
        "scripts/train_model.py", 
        "scripts/predict.py",
        "scripts/export_model.py",
        "api/main.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    print("✅ Structure des fichiers OK!")
    return True

def main():
    """Fonction principale de test"""
    print("🦷 Test d'installation EvaDentalAI")
    print("=" * 50)
    
    tests = [
        ("Structure fichiers", test_file_structure),
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("YOLO", test_yolo),
        ("Génération dataset", test_dataset_generation),
        ("Entraînement", test_training),
        ("Prédiction", test_prediction),
        ("API", test_api)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erreur inattendue dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nRésultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés!")
        print("✅ EvaDentalAI est prêt à être utilisé!")
        print("\n🚀 Prochaines étapes:")
        print("  1. python scripts/prepare_dataset.py --num-images 200")
        print("  2. python scripts/train_model.py --epochs 50")
        print("  3. python api/main.py --model models/best.pt")
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué")
        print("💡 Consultez les messages d'erreur ci-dessus")
        print("📚 Voir docs/INSTALLATION.md pour plus d'aide")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
