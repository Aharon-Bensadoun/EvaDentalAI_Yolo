# 🦷 Guide Complet DENTEX pour EvaDentalAI

## 🎯 Introduction

Le dataset [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) est un dataset de radiographies panoramiques dentaires avec des annotations hiérarchiques pour la détection d'anomalies. Il s'agit d'un dataset de qualité clinique publié dans le cadre du challenge DENTEX 2023.

## 🚀 Démarrage Rapide avec DENTEX

### 1. Installation des Dépendances

```bash
# Installer toutes les dépendances (y compris DENTEX)
pip install -r requirements.txt

# Ou installer spécifiquement les dépendances DENTEX
pip install datasets huggingface-hub
```

### 2. Téléchargement du Dataset

```bash
# Télécharger et préparer le dataset DENTEX
python scripts/download_dentex_dataset.py

# Le script va :
# - Télécharger le dataset depuis Hugging Face
# - Convertir les annotations au format YOLO
# - Créer la structure de répertoires
# - Générer le fichier data.yaml
```

### 3. Entraînement

```bash
# Entraînement rapide (20 épochs)
python scripts/train_model.py --config data/dentex/data.yaml --epochs 20 --batch-size 8

# Entraînement complet (100 épochs)
python scripts/train_model.py --config data/dentex/data.yaml --epochs 100 --batch-size 16

# Entraînement avec GPU
python scripts/train_model.py --config data/dentex/data.yaml --epochs 50 --batch-size 32 --device cuda
```

### 4. Test et Prédiction

```bash
# Tester sur une image du dataset
python scripts/predict.py --model models/best.pt --image data/dentex/test/images/test_0000.jpg

# Tester sur votre propre image
python scripts/predict.py --model models/best.pt --image votre_radiographie.jpg --save --report
```

### 5. API

```bash
# Lancer l'API
python api/main.py --model models/best.pt

# Tester l'API
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@votre_image.jpg"
```

## 📊 Structure du Dataset DENTEX

### Classes de Diagnostic

| Classe DENTEX | Description | Classe YOLO | Couleur |
|---------------|-------------|-------------|---------|
| `caries` | Carie | `cavity` | 🔴 Rouge |
| `deep_caries` | Carie profonde | `cavity` | 🔴 Rouge |
| `periapical_lesion` | Lésion périapicale | `lesion` | 🔵 Bleu |
| `impacted_tooth` | Dent incluse | `tooth` | ⚪ Blanc |

### Structure des Fichiers

```
data/dentex/
├── train/
│   ├── images/          # 705 images d'entraînement
│   └── labels/          # Annotations YOLO
├── val/
│   ├── images/          # 50 images de validation
│   └── labels/          # Annotations YOLO
├── test/
│   ├── images/          # 250 images de test
│   └── labels/          # Annotations YOLO
├── data.yaml            # Configuration YOLO
└── dataset_info.json    # Informations du dataset
```

## 🔧 Configuration YOLO

Le fichier `data/dentex/data.yaml` généré automatiquement :

```yaml
path: data/dentex
train: train/images
val: val/images
test: test/images

names:
  0: tooth      # Dent normale/incluse
  1: cavity     # Carie (caries + deep_caries)
  2: implant    # Implant (pas dans DENTEX, pour compatibilité)
  3: lesion     # Lésion (periapical_lesion)
  4: filling    # Plombage (pas dans DENTEX, pour compatibilité)

nc: 5
```

## 📈 Performance Attendue

### Métriques Typiques

- **mAP@0.5**: 80-90%
- **mAP@0.5:0.95**: 60-80%
- **Precision**: 85-95%
- **Recall**: 80-90%

### Temps d'Inférence

- **CPU**: ~50ms par image
- **GPU**: ~10ms par image
- **Taille modèle**: ~50MB (ONNX)

## 🎯 Utilisation Avancée

### Entraînement avec Fine-tuning

```bash
# Utiliser un modèle pré-entraîné
python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --lr 0.01 \
    --patience 50
```

### Export du Modèle

```bash
# Exporter en ONNX
python scripts/export_model.py --model models/best.pt --format onnx

# Exporter tous les formats
python scripts/export_model.py --model models/best.pt --format all
```

### Validation Croisée

```bash
# Tester sur le dataset de validation
python scripts/predict.py \
    --model models/best.pt \
    --batch data/dentex/val/images \
    --output results/dentex_val

# Tester sur le dataset de test
python scripts/predict.py \
    --model models/best.pt \
    --batch data/dentex/test/images \
    --output results/dentex_test
```

## 🐳 Déploiement Docker

```bash
# Construire l'image
docker build -f docker/Dockerfile -t evadental-dentex .

# Lancer le conteneur
docker run -p 8000:8000 -v $(pwd)/models:/app/models evadental-dentex
```

## 📚 Exemples d'Utilisation

### Python

```python
from ultralytics import YOLO

# Charger le modèle
model = YOLO('models/best.pt')

# Prédiction
results = model('votre_radiographie.jpg')

# Afficher les résultats
for r in results:
    r.show()  # Affiche l'image avec détections
```

### API REST

```python
import requests

# Prédiction via API
with open('radiographie.jpg', 'rb') as f:
    files = {'file': f}
    data = {'confidence': 0.25}
    
    response = requests.post(
        'http://localhost:8000/predict',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Détections: {result['total_detections']}")
```

## 🚨 Limitations et Considérations

### Limitations du Dataset

1. **Classes limitées**: Seulement 4 types d'anomalies
2. **Pas d'implants**: Les implants dentaires ne sont pas annotés
3. **Pas de plombages**: Les plombages ne sont pas annotés
4. **Licence non-commerciale**: Usage limité à la recherche

### Recommandations

1. **Combiner avec d'autres datasets** pour plus de classes
2. **Augmenter les données** avec des techniques d'augmentation
3. **Fine-tuning** sur des données spécifiques à votre domaine
4. **Validation clinique** avant utilisation en production

## 🔍 Résolution de Problèmes

### Problèmes Courants

#### 1. Erreur de téléchargement
```bash
# Vérifier la connexion internet
ping huggingface.co

# Réessayer le téléchargement
python scripts/download_dentex_dataset.py
```

#### 2. Mémoire insuffisante
```bash
# Réduire la taille du batch
python scripts/train_model.py --config data/dentex/data.yaml --batch-size 8

# Utiliser CPU
python scripts/train_model.py --config data/dentex/data.yaml --device cpu
```

#### 3. Modèle non trouvé
```bash
# Vérifier que le modèle existe
ls -la models/best.pt

# Entraîner un nouveau modèle
python scripts/train_model.py --config data/dentex/data.yaml --epochs 10
```

## 📚 Références

### Citation

Si vous utilisez le dataset DENTEX, citez :

```bibtex
@article{hamamci2023dentex,
  title={DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Yuksel, Atif Emre and Gultekin, Sadullah and Ozdemir, Serife Damla and Yang, Kaiyuan and Li, Hongwei Bran and Pati, Sarthak and Stadlinger, Bernd and others},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}
```

### Liens Utiles

- [Dataset DENTEX sur Hugging Face](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
- [Paper DENTEX sur arXiv](https://arxiv.org/abs/2305.19112)
- [Challenge DENTEX 2023](https://dentex.grand-challenge.org/)
- [Méthode HierarchicalDet](https://github.com/ibrahimethemhamamci/HierarchicalDet)

## 🆘 Support

### Documentation

- **Guide DENTEX**: `docs/DENTEX_DATASET.md`
- **Installation**: `docs/INSTALLATION.md`
- **Utilisation**: `docs/USAGE.md`
- **Google Colab**: `docs/GOOGLE_COLAB.md`

### Contact

- **Issues**: [GitHub Issues](https://github.com/votre-repo/issues)
- **Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)
- **Community**: [Discord YOLO](https://discord.gg/ultralytics)

---

🎉 **Vous êtes maintenant prêt à utiliser le dataset DENTEX avec EvaDentalAI!**

Ce dataset de qualité clinique vous permettra d'entraîner des modèles robustes pour la détection d'anomalies dentaires sur des radiographies panoramiques réelles.
