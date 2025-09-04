# Guide d'Installation - EvaDentalAI

Ce guide vous accompagne dans l'installation et la configuration d'EvaDentalAI pour la détection d'anomalies dentaires.

## 📋 Prérequis

### Système d'exploitation
- **Windows 10/11** (recommandé)
- **Linux Ubuntu 18.04+**
- **macOS 10.15+**

### Matériel recommandé
- **CPU**: Intel i5/AMD Ryzen 5 ou supérieur
- **RAM**: 8GB minimum, 16GB recommandé
- **GPU**: NVIDIA GTX 1060 ou supérieur (optionnel mais recommandé)
- **Stockage**: 5GB d'espace libre

### Logiciels requis
- **Python 3.8-3.11** ([télécharger](https://www.python.org/downloads/))
- **Git** ([télécharger](https://git-scm.com/downloads))
- **CUDA 11.8+** (si GPU NVIDIA) ([télécharger](https://developer.nvidia.com/cuda-downloads))

## 🚀 Installation Rapide

### 1. Cloner le projet
```bash
git clone <votre-repo-url>
cd EvaDentalAI_Yolo
```

### 2. Créer un environnement virtuel
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Vérifier l'installation
```bash
python -c "import ultralytics, torch, cv2; print('✅ Installation réussie!')"
```

## 🔧 Installation Détaillée

### Windows

#### Étape 1: Python
1. Téléchargez Python depuis [python.org](https://www.python.org/downloads/)
2. **Important**: Cochez "Add Python to PATH" lors de l'installation
3. Vérifiez l'installation:
   ```cmd
   python --version
   pip --version
   ```

#### Étape 2: CUDA (optionnel)
1. Téléchargez CUDA Toolkit depuis [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Installez CUDA 11.8 ou supérieur
3. Vérifiez l'installation:
   ```cmd
   nvidia-smi
   ```

#### Étape 3: PyTorch avec CUDA
```bash
# Pour CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Pour CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Linux (Ubuntu)

#### Étape 1: Mise à jour du système
```bash
sudo apt update
sudo apt upgrade -y
```

#### Étape 2: Installation de Python
```bash
sudo apt install python3 python3-pip python3-venv -y
```

#### Étape 3: Dépendances système
```bash
sudo apt install libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 -y
sudo apt install libglib2.0-0 libgtk-3-0 libavcodec-dev libavformat-dev -y
sudo apt install libswscale-dev libv4l-dev libxvidcore-dev libx264-dev -y
sudo apt install libjpeg-dev libpng-dev libtiff-dev libatlas-base-dev -y
```

#### Étape 4: CUDA (optionnel)
```bash
# Ajouter le repository NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repository-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repository-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### macOS

#### Étape 1: Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Étape 2: Python et dépendances
```bash
brew install python@3.11
brew install opencv
```

## 🐳 Installation avec Docker

### Prérequis Docker
1. Installez [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Vérifiez l'installation:
   ```bash
   docker --version
   docker-compose --version
   ```

### Construction de l'image
```bash
# Construire l'image
docker build -f docker/Dockerfile -t evadental-ai .

# Ou utiliser docker-compose
docker-compose build
```

### Lancement avec Docker
```bash
# Lancer l'API
docker-compose up -d

# Vérifier les logs
docker-compose logs -f
```

## 🔍 Vérification de l'Installation

### Test complet
```bash
# Générer un dataset de test
python scripts/prepare_dataset.py --num-images 10

# Entraîner un modèle rapide
python scripts/train_model.py --epochs 5 --batch-size 4

# Tester la prédiction
python scripts/predict.py --model models/best.pt --image data/processed/test/images/0000.jpg

# Lancer l'API
python api/main.py --model models/best.pt
```

### Vérification GPU
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nombre de GPUs: {torch.cuda.device_count()}")
    print(f"GPU actuel: {torch.cuda.get_device_name(0)}")
```

## 🚨 Résolution de Problèmes

### Erreur "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Erreur CUDA
```bash
# Vérifier la version de CUDA
nvidia-smi

# Réinstaller PyTorch avec la bonne version CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erreur de mémoire GPU
```bash
# Réduire la taille du batch
python scripts/train_model.py --batch-size 8

# Ou utiliser CPU
python scripts/train_model.py --device cpu
```

### Erreur de permissions (Linux/macOS)
```bash
# Donner les permissions d'exécution
chmod +x scripts/run_training.sh
```

### Problème de dépendances OpenCV
```bash
# Linux
sudo apt install python3-opencv

# macOS
brew install opencv

# Windows
pip install opencv-python
```

## 📞 Support

Si vous rencontrez des problèmes:

1. **Vérifiez les logs** dans le terminal
2. **Consultez la documentation** YOLO: [docs.ultralytics.com](https://docs.ultralytics.com)
3. **Vérifiez les issues** sur GitHub
4. **Contactez l'équipe** EvaDentalAI

## ✅ Checklist d'Installation

- [ ] Python 3.8+ installé
- [ ] Environnement virtuel créé et activé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] CUDA installé (optionnel)
- [ ] PyTorch avec CUDA (si GPU)
- [ ] Test d'installation réussi
- [ ] Dataset de test généré
- [ ] Modèle d'exemple entraîné
- [ ] API fonctionnelle

🎉 **Félicitations!** Votre installation EvaDentalAI est prête!
