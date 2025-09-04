# 🚀 EvaDentalAI + DENTEX sur Google Colab - Guide Rapide

## 🎯 Utilisation Immédiate

### 1. Ouvrir le Notebook Colab
- Allez sur [Google Colab](https://colab.research.google.com)
- Créez un nouveau notebook
- **IMPORTANT**: Activez le GPU dans `Runtime > Change runtime type > GPU`

### 2. Installation Express (1 cellule)

```python
# Installation complète en une commande
!pip install ultralytics datasets huggingface-hub fastapi uvicorn python-multipart opencv-python pillow matplotlib seaborn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cloner le projet
!git clone https://github.com/votre-username/EvaDentalAI_Yolo.git
%cd EvaDentalAI_Yolo

print("✅ Installation terminée!")
```

### 3. Téléchargement DENTEX (1 cellule)

```python
# Télécharger le dataset DENTEX
!python scripts/download_dentex_dataset.py

print("✅ Dataset DENTEX prêt!")
```

### 4. Entraînement Rapide (1 cellule)

```python
# Entraînement rapide (20 épochs)
!python scripts/train_model.py --config data/dentex/data.yaml --epochs 20 --batch-size 16 --device cuda

print("✅ Modèle entraîné!")
```

### 5. Test du Modèle (1 cellule)

```python
# Test sur une image du dataset
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('models/best.pt')
results = model('data/dentex/test/images/test_0000.jpg')

for r in results:
    im_array = r.plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(im_array)
    plt.axis('off')
    plt.title('Détections DENTEX')
    plt.show()

print("✅ Test terminé!")
```

### 6. Test sur Votre Image (1 cellule)

```python
# Upload et test de votre image
from google.colab import files
import matplotlib.pyplot as plt

uploaded = files.upload()
for filename in uploaded.keys():
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = model(filename)
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(12, 8))
            plt.imshow(im_array)
            plt.axis('off')
            plt.title(f'Détections sur {filename}')
            plt.show()

print("✅ Analyse terminée!")
```

### 7. Sauvegarde (1 cellule)

```python
# Sauvegarder sur Google Drive
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.copytree('models/', '/content/drive/MyDrive/EvaDentalAI_Models/', dirs_exist_ok=True)

print("✅ Modèle sauvegardé sur Google Drive!")
```

## ⚡ Script Complet (1 cellule)

```python
# Script complet automatisé
print("🚀 EvaDentalAI + DENTEX - Script Complet")
print("=" * 50)

# 1. Installation
!pip install ultralytics datasets huggingface-hub fastapi uvicorn python-multipart opencv-python pillow matplotlib seaborn torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Cloner le projet
!git clone https://github.com/votre-username/EvaDentalAI_Yolo.git
%cd EvaDentalAI_Yolo

# 3. Télécharger DENTEX
!python scripts/download_dentex_dataset.py

# 4. Entraîner
!python scripts/train_model.py --config data/dentex/data.yaml --epochs 30 --batch-size 16 --device cuda

# 5. Tester
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('models/best.pt')
results = model('data/dentex/test/images/test_0000.jpg')

for r in results:
    im_array = r.plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(im_array)
    plt.axis('off')
    plt.title('Détections DENTEX')
    plt.show()

# 6. Sauvegarder
from google.colab import drive
import shutil

drive.mount('/content/drive')
shutil.copytree('models/', '/content/drive/MyDrive/EvaDentalAI_Models/', dirs_exist_ok=True)

print("🎉 Terminé! Modèle sauvegardé sur Google Drive!")
```

## 📊 Résultats Attendus

- **Temps d'entraînement**: 20-30 minutes (GPU T4)
- **Performance**: 80-90% mAP@0.5
- **Classes détectées**: Caries, lésions périapicales, dents incluses
- **Taille du modèle**: ~50MB

## 🔧 Optimisations Colab

### Pour GPU T4 (Gratuit)
```python
# Paramètres optimisés pour T4
!python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8n.pt \
    --epochs 30 \
    --batch-size 16 \
    --device cuda
```

### Pour GPU V100/A100 (Colab Pro)
```python
# Paramètres optimisés pour V100/A100
!python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch-size 32 \
    --device cuda
```

## 🚨 Problèmes Courants

### 1. Session Timeout
- Sauvegardez régulièrement sur Google Drive
- Utilisez des checkpoints (`--save-period 10`)

### 2. Mémoire GPU Insuffisante
```python
# Réduire la taille du batch
!python scripts/train_model.py --batch-size 8

# Ou utiliser un modèle plus petit
!python scripts/train_model.py --model yolov8n.pt
```

### 3. Erreur de Dépendances
```python
# Réinstaller
!pip install --upgrade ultralytics datasets
```

## 📱 Test Mobile

```python
# Test rapide sur mobile
from google.colab import files
import matplotlib.pyplot as plt

# Upload d'une image depuis votre téléphone
uploaded = files.upload()

for filename in uploaded.keys():
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        results = model(filename)
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(10, 6))
            plt.imshow(im_array)
            plt.axis('off')
            plt.title(f'Détections sur {filename}')
            plt.show()
            
            # Afficher les détections
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                
                class_names = {0: "tooth", 1: "cavity", 2: "implant", 3: "lesion", 4: "filling"}
                
                print(f"\n🎯 Détections sur {filename}:")
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    class_name = class_names.get(class_id, f"class_{class_id}")
                    print(f"  {i+1}. {class_name}: {conf:.3f}")
```

## 🎉 Conclusion

Avec ce guide, vous pouvez :

✅ **Entraîner un modèle** en 30 minutes sur Colab
✅ **Tester sur vos images** directement
✅ **Sauvegarder sur Google Drive** pour utilisation ultérieure
✅ **Obtenir des résultats** de qualité clinique

**🚀 Votre modèle de détection d'anomalies dentaires est prêt !**
