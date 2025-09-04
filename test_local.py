#!/usr/bin/env python3
"""
Script de test complet pour EvaDentalAI en local
"""

import os
import sys
import subprocess
from pathlib import Path

def test_installation():
    """Test de l'installation"""
    print("🔍 Test de l'installation...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU non disponible")
    except ImportError:
        print("❌ PyTorch non installé")
        return False
    
    try:
        import ultralytics
        print("✅ Ultralytics: OK")
    except ImportError:
        print("❌ Ultralytics non installé")
        return False
    
    try:
        import datasets
        print("✅ Datasets: OK")
    except ImportError:
        print("❌ Datasets non installé")
        return False
    
    return True

def test_dataset():
    """Test du dataset"""
    print("\n📊 Test du dataset...")
    
    # Test avec dataset simulé
    if not Path("data/processed").exists():
        print("📥 Création du dataset simulé...")
        result = subprocess.run([
            sys.executable, "scripts/prepare_dataset.py", 
            "--num-images", "50"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset simulé créé")
        else:
            print(f"❌ Erreur: {result.stderr}")
            return False
    
    # Vérifier la structure
    if Path("data/processed/train/images").exists():
        train_images = list(Path("data/processed/train/images").glob("*.jpg"))
        print(f"✅ Images d'entraînement: {len(train_images)}")
    else:
        print("❌ Pas d'images d'entraînement")
        return False
    
    return True

def test_training():
    """Test d'entraînement rapide"""
    print("\n🏋️ Test d'entraînement...")
    
    # Entraînement rapide
    result = subprocess.run([
        sys.executable, "scripts/train_model.py",
        "--config", "config/data.yaml",
        "--model", "yolov8n.pt",
        "--epochs", "2",
        "--batch-size", "4",
        "--device", "cpu"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Entraînement réussi")
        return True
    else:
        print(f"❌ Erreur d'entraînement: {result.stderr}")
        return False

def test_prediction():
    """Test de prédiction"""
    print("\n🔍 Test de prédiction...")
    
    if not Path("models/best.pt").exists():
        print("❌ Modèle non trouvé")
        return False
    
    # Test de prédiction
    result = subprocess.run([
        sys.executable, "scripts/predict.py",
        "--model", "models/best.pt",
        "--image", "data/processed/test/images/0000.jpg",
        "--save"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Prédiction réussie")
        return True
    else:
        print(f"❌ Erreur de prédiction: {result.stderr}")
        return False

def test_api():
    """Test de l'API"""
    print("\n🌐 Test de l'API...")
    
    if not Path("models/best.pt").exists():
        print("❌ Modèle non trouvé")
        return False
    
    # Lancer l'API en arrière-plan
    import threading
    import time
    import requests
    
    def run_api():
        subprocess.run([
            sys.executable, "api/main.py",
            "--model", "models/best.pt",
            "--host", "127.0.0.1",
            "--port", "8000"
        ])
    
    # Lancer l'API
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Attendre que l'API démarre
    time.sleep(5)
    
    try:
        # Test de santé
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API fonctionne")
            return True
        else:
            print("❌ API ne répond pas")
            return False
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        return False

def main():
    """Test principal"""
    print("🧪 Test Local EvaDentalAI")
    print("=" * 40)
    
    tests = [
        ("Installation", test_installation),
        ("Dataset", test_dataset),
        ("Entraînement", test_training),
        ("Prédiction", test_prediction),
        ("API", test_api)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "=" * 40)
    print("📊 Résumé des Tests")
    print("=" * 40)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n🎯 Résultat: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! Prêt pour Colab.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez les erreurs.")

if __name__ == "__main__":
    main()
