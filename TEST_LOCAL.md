# 🧪 Test Local EvaDentalAI + DENTEX

## 🎯 Objectif

Tester le projet EvaDentalAI avec le dataset DENTEX en local avant d'utiliser Google Colab.

## 🚀 Installation Locale

### 1. Prérequis
- Python 3.8+
- Git
- 8GB+ RAM recommandé
- GPU optionnel (mais recommandé pour l'entraînement)

### 2. Installation des Dépendances

```bash
# Cloner le projet
git clone https://github.com/Aharon-Bensadoun/EvaDentalAI_Yolo.git
cd EvaDentalAI_Yolo

# Créer un environnement virtuel (recommandé)
python -m venv evadental_env
source evadental_env/bin/activate  # Linux/Mac
# ou
evadental_env\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Vérification de l'Installation

```bash
# Vérifier que tout est installé
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('Ultralytics: OK')"
python -c "import datasets; print('Datasets: OK')"
```

## 📊 Test du Dataset DENTEX

### Option 1: Dataset DENTEX (Recommandé)

```bash
# Télécharger le dataset DENTEX
python scripts/download_dentex_simple.py

# Vérifier la structure
ls -la data/dentex/
cat data/dentex/data.yaml
```

### Option 2: Dataset Simulé (Test Rapide)

```bash
# Créer un dataset simulé pour test rapide
python scripts/prepare_dataset.py --num-images 100

# Vérifier la structure
ls -la data/processed/
cat config/data.yaml
```

## 🏋️ Test d'Entraînement

### Test Rapide (5-10 minutes)

```bash
# Entraînement rapide avec dataset simulé
python scripts/train_model.py \
    --config config/data.yaml \
    --model yolov8n.pt \
    --epochs 5 \
    --batch-size 8 \
    --device cpu

# Ou avec GPU si disponible
python scripts/train_model.py \
    --config config/data.yaml \
    --model yolov8n.pt \
    --epochs 5 \
    --batch-size 16 \
    --device cuda
```

### Test Complet (30-60 minutes)

```bash
# Entraînement complet avec DENTEX
python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8s.pt \
    --epochs 20 \
    --batch-size 16 \
    --device cuda
```

## 🔍 Test de Prédiction

```bash
# Test sur une image du dataset
python scripts/predict.py \
    --model models/best.pt \
    --image data/processed/test/images/0000.jpg \
    --save \
    --report

# Ou avec DENTEX
python scripts/predict.py \
    --model models/best.pt \
    --image data/dentex/test/images/test_0000.jpg \
    --save \
    --report
```

## 🌐 Test de l'API

```bash
# Lancer l'API
python api/main.py --model models/best.pt

# Dans un autre terminal, tester l'API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/processed/test/images/0000.jpg"
```

## 📤 Test d'Export

```bash
# Exporter le modèle
python scripts/export_model.py --model models/best.pt --format all

# Vérifier les exports
ls -la models/*.onnx models/*.pt
```

## 🧪 Script de Test Complet

Créez un fichier `test_local.py` :

```python
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
```

## 🚀 Exécution du Test

```bash
# Exécuter le test complet
python test_local.py
```

## 📊 Résultats Attendus

### Test Réussi
```
🧪 Test Local EvaDentalAI
========================================
🔍 Test de l'installation...
✅ PyTorch: 2.1.0
✅ GPU: NVIDIA GeForce RTX 3080
✅ Ultralytics: OK
✅ Datasets: OK

📊 Test du dataset...
📥 Création du dataset simulé...
✅ Dataset simulé créé
✅ Images d'entraînement: 50

🏋️ Test d'entraînement...
✅ Entraînement réussi

🔍 Test de prédiction...
✅ Prédiction réussie

🌐 Test de l'API...
✅ API fonctionne

========================================
📊 Résumé des Tests
========================================
Installation: ✅ PASS
Dataset: ✅ PASS
Entraînement: ✅ PASS
Prédiction: ✅ PASS
API: ✅ PASS

🎯 Résultat: 5/5 tests réussis
🎉 Tous les tests sont passés! Prêt pour Colab.
```

## 🔧 Résolution de Problèmes

### 1. Erreur de Dépendances
```bash
# Réinstaller les dépendances
pip install --upgrade -r requirements.txt
```

### 2. Erreur GPU
```bash
# Vérifier CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Installer PyTorch avec CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Erreur de Mémoire
```bash
# Réduire la taille du batch
python scripts/train_model.py --batch-size 4 --device cpu
```

### 4. Erreur de Permissions
```bash
# Donner les permissions (Linux/Mac)
chmod +x scripts/*.py
```

## 🎯 Prochaines Étapes

Une fois les tests locaux réussis :

1. **Commit Git** : Sauvegardez vos changements
2. **Push GitHub** : Mettez à jour le repository
3. **Test Colab** : Utilisez le script corrigé sur Colab
4. **Déploiement** : Déployez en production si nécessaire

---

**🎉 Testez d'abord en local, puis passez à Colab !**
