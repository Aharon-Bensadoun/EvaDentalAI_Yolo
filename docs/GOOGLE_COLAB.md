# Guide Google Colab - EvaDentalAI

Ce guide vous explique comment utiliser EvaDentalAI sur Google Colab pour l'entraînement et l'inférence.

## 🚀 Configuration Initiale

### 1. Ouvrir Google Colab
1. Allez sur [colab.research.google.com](https://colab.research.google.com)
2. Créez un nouveau notebook
3. Activez le GPU: `Runtime > Change runtime type > GPU`

### 2. Installation des Dépendances

```python
# Cellule 1: Installation des dépendances
!pip install ultralytics fastapi uvicorn python-multipart
!pip install opencv-python pillow matplotlib seaborn
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Vérifier l'installation
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### 3. Cloner le Projet

```python
# Cellule 2: Cloner le repository
!git clone https://github.com/votre-username/EvaDentalAI_Yolo.git
%cd EvaDentalAI_Yolo

# Vérifier la structure
!ls -la
```

## 📊 Préparation du Dataset

### Génération du Dataset Simulé

```python
# Cellule 3: Génération du dataset
!python scripts/prepare_dataset.py --num-images 200

# Vérifier la structure du dataset
!ls -la data/processed/
!ls -la data/processed/train/images/ | head -5
```

### Upload d'un Dataset Réel (Optionnel)

```python
# Cellule 4: Upload de votre dataset
from google.colab import files
import zipfile
import os

# Upload d'un fichier ZIP contenant vos images
uploaded = files.upload()

# Extraire le ZIP
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('data/raw/')
        print(f"Dataset extrait: {filename}")

# Organiser les fichiers selon la structure YOLO
!python scripts/organize_dataset.py --input data/raw --output data/processed
```

## 🏋️ Entraînement du Modèle

### Entraînement Rapide (Test)

```python
# Cellule 5: Entraînement rapide pour test
!python scripts/train_model.py \
    --config config/data.yaml \
    --model yolov8n.pt \
    --epochs 10 \
    --batch-size 16 \
    --device cuda

# Vérifier les résultats
!ls -la models/
```

### Entraînement Complet

```python
# Cellule 6: Entraînement complet
!python scripts/train_model.py \
    --config config/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch-size 32 \
    --img-size 640 \
    --device cuda \
    --patience 20

# Afficher les métriques
import pandas as pd
import matplotlib.pyplot as plt

# Lire les résultats
results_path = "models/dental_yolo_*/results.csv"
!find models -name "results.csv" -exec cat {} \;
```

### Visualisation des Résultats

```python
# Cellule 7: Visualisation des métriques
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Trouver le fichier de résultats
results_files = glob.glob("models/*/results.csv")
if results_files:
    df = pd.read_csv(results_files[0])
    
    # Graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
    axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
    axes[0, 0].set_title('Box Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # mAP
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
    axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
    axes[0, 1].set_title('Mean Average Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision/Recall
    axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Meilleur mAP@0.5: {df['metrics/mAP50(B)'].max():.3f}")
    print(f"Meilleur mAP@0.5:0.95: {df['metrics/mAP50-95(B)'].max():.3f}")
```

## 🔍 Test et Prédiction

### Test sur une Image

```python
# Cellule 8: Test du modèle
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Charger le modèle
model_path = "models/best.pt"
model = YOLO(model_path)

# Tester sur une image du dataset
test_image = "data/processed/test/images/0000.jpg"
results = model(test_image)

# Afficher les résultats
for r in results:
    im_array = r.plot()  # Image avec détections
    plt.figure(figsize=(12, 8))
    plt.imshow(im_array)
    plt.axis('off')
    plt.title('Détections sur image de test')
    plt.show()
    
    # Afficher les détections
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        
        class_names = {0: "tooth", 1: "cavity", 2: "implant", 3: "lesion", 4: "filling"}
        
        print("Détections trouvées:")
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"  {i+1}. {class_name}: {conf:.3f}")
```

### Upload et Test d'une Image Personnelle

```python
# Cellule 9: Test sur votre image
from google.colab import files
import cv2
import matplotlib.pyplot as plt

# Upload d'une image
uploaded = files.upload()
for filename in uploaded.keys():
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Prédiction
        results = model(filename)
        
        # Affichage
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(im_array)
            plt.axis('off')
            plt.title(f'Détections sur {filename}')
            plt.show()
            
            # Détails des détections
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                print(f"\nDétections sur {filename}:")
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    print(f"  {i+1}. {class_name}: {conf:.3f}")
```

## 📤 Export et Sauvegarde

### Export du Modèle

```python
# Cellule 10: Export du modèle
!python scripts/export_model.py --model models/best.pt --format all

# Vérifier les exports
!ls -la models/*.onnx models/*.pt
```

### Sauvegarde sur Google Drive

```python
# Cellule 11: Sauvegarde sur Google Drive
from google.colab import drive
import shutil

# Monter Google Drive
drive.mount('/content/drive')

# Créer un dossier pour le projet
drive_path = '/content/drive/MyDrive/EvaDentalAI'
!mkdir -p {drive_path}

# Sauvegarder le modèle et les résultats
!cp -r models/ {drive_path}/
!cp -r data/processed/ {drive_path}/
!cp config/data.yaml {drive_path}/

print("Modèle sauvegardé sur Google Drive!")
```

### Téléchargement Local

```python
# Cellule 12: Téléchargement des fichiers
from google.colab import files

# Télécharger le modèle
files.download('models/best.pt')

# Télécharger le modèle ONNX
files.download('models/model.onnx')

# Créer un ZIP avec tous les résultats
!zip -r evadental_results.zip models/ config/ data/processed/
files.download('evadental_results.zip')
```

## 🌐 Déploiement de l'API

### Lancement de l'API sur Colab

```python
# Cellule 13: Lancement de l'API
import subprocess
import threading
import time

def run_api():
    subprocess.run([
        'python', 'api/main.py', 
        '--model', 'models/best.pt',
        '--host', '0.0.0.0',
        '--port', '8000'
    ])

# Lancer l'API en arrière-plan
api_thread = threading.Thread(target=run_api)
api_thread.daemon = True
api_thread.start()

# Attendre que l'API démarre
time.sleep(5)

print("API démarrée sur http://localhost:8000")
print("Documentation: http://localhost:8000/docs")
```

### Test de l'API

```python
# Cellule 14: Test de l'API
import requests
import json

# Test de santé
response = requests.get('http://localhost:8000/health')
print("Status API:", response.json())

# Test de prédiction
with open('data/processed/test/images/0000.jpg', 'rb') as f:
    files = {'file': f}
    data = {'confidence': 0.25}
    
    response = requests.post(
        'http://localhost:8000/predict',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"\nDétections: {result['total_detections']}")
    for detection in result['detections']:
        print(f"- {detection['class_name']}: {detection['confidence']:.3f}")
```

## 🚀 Script Complet pour Colab

```python
# Cellule 15: Script complet automatisé
def run_complete_training():
    """Script complet d'entraînement sur Colab"""
    
    print("🚀 Démarrage de l'entraînement EvaDentalAI sur Colab")
    print("=" * 60)
    
    # 1. Vérification GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU non disponible, utilisation du CPU")
    
    # 2. Génération du dataset
    print("\n📊 Génération du dataset...")
    !python scripts/prepare_dataset.py --num-images 300
    
    # 3. Entraînement
    print("\n🏋️ Entraînement du modèle...")
    !python scripts/train_model.py \
        --config config/data.yaml \
        --model yolov8s.pt \
        --epochs 50 \
        --batch-size 32 \
        --device cuda \
        --patience 15
    
    # 4. Export
    print("\n📤 Export du modèle...")
    !python scripts/export_model.py --model models/best.pt --format all
    
    # 5. Test
    print("\n🔍 Test du modèle...")
    !python scripts/predict.py \
        --model models/best.pt \
        --image data/processed/test/images/0000.jpg \
        --save
    
    print("\n✅ Entraînement terminé!")
    print("📁 Modèles disponibles:")
    !ls -la models/*.pt models/*.onnx

# Lancer le script complet
run_complete_training()
```

## 💡 Conseils pour Colab

### Optimisation des Performances
1. **Utilisez toujours le GPU** pour l'entraînement
2. **Augmentez la taille du batch** selon votre GPU
3. **Sauvegardez régulièrement** sur Google Drive
4. **Utilisez des sessions longues** pour éviter les timeouts

### Gestion de la Mémoire
```python
# Nettoyer la mémoire si nécessaire
import gc
import torch

# Nettoyer le cache GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Nettoyer la mémoire Python
gc.collect()
```

### Monitoring des Ressources
```python
# Vérifier l'utilisation GPU
!nvidia-smi

# Vérifier l'utilisation mémoire
!free -h
```

## 🔧 Résolution de Problèmes Colab

### Problèmes Courants

#### 1. Session Timeout
- Sauvegardez régulièrement sur Google Drive
- Utilisez des checkpoints pendant l'entraînement
- Réduisez le nombre d'épochs si nécessaire

#### 2. Mémoire GPU Insuffisante
```python
# Réduire la taille du batch
!python scripts/train_model.py --batch-size 16

# Ou utiliser un modèle plus petit
!python scripts/train_model.py --model yolov8n.pt
```

#### 3. Erreur de Dépendances
```python
# Réinstaller les dépendances
!pip install --upgrade ultralytics torch torchvision
```

#### 4. Problème de Permissions
```python
# Donner les permissions
!chmod +x scripts/*.py
```

## 📚 Ressources Supplémentaires

- **Colab Pro**: Pour plus de GPU et de temps
- **Colab Pro+**: Pour des sessions plus longues
- **Google Drive**: Pour sauvegarder vos modèles
- **GitHub**: Pour versionner votre code

---

🎉 **Vous êtes maintenant prêt à utiliser EvaDentalAI sur Google Colab!**

Cette configuration vous permet d'entraîner des modèles puissants sans avoir besoin d'un GPU local.
