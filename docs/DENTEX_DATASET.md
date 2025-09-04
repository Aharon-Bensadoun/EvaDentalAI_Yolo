# Dataset DENTEX - Guide d'Utilisation

## 📚 À propos du Dataset DENTEX

Le dataset [DENTEX](https://huggingface.co/datasets/ibrahimhamamci/DENTEX) est un dataset de radiographies panoramiques dentaires avec des annotations hiérarchiques pour la détection d'anomalies. Il a été publié dans le cadre du challenge DENTEX 2023 organisé en conjonction avec MICCAI.

### 🏥 Caractéristiques du Dataset

- **Source**: Radiographies panoramiques de 3 institutions différentes
- **Patients**: Âgés de 12 ans et plus
- **Qualité**: Images hétérogènes reflétant la pratique clinique réelle
- **Licence**: CC-BY-NC-SA-4.0 (Creative Commons Attribution-NonCommercial-ShareAlike)

### 📊 Structure du Dataset

Le dataset DENTEX est organisé hiérarchiquement en 3 types de données :

1. **Quadrant Detection** (693 X-rays): Détection de quadrants uniquement
2. **Tooth Detection** (634 X-rays): Détection de dents avec énumération
3. **Abnormal Tooth Detection** (1005 X-rays): Détection complète avec diagnostic

### 🦷 Classes de Diagnostic

Le dataset DENTEX contient 4 catégories de diagnostic :

| Classe DENTEX | Description | Classe YOLO Mappée |
|---------------|-------------|-------------------|
| `caries` | Carie | `cavity` |
| `deep_caries` | Carie profonde | `cavity` |
| `periapical_lesion` | Lésion périapicale | `lesion` |
| `impacted_tooth` | Dent incluse | `tooth` |

## 🚀 Installation et Utilisation

### 1. Installation des Dépendances

```bash
# Installer les dépendances pour DENTEX
pip install datasets huggingface-hub

# Ou installer toutes les dépendances
pip install -r requirements.txt
```

### 2. Téléchargement du Dataset

```bash
# Télécharger et préparer le dataset DENTEX
python scripts/download_dentex_dataset.py

# Avec options
python scripts/download_dentex_dataset.py \
    --output-dir data \
    --use-full-annotations \
    --skip-download  # Si déjà téléchargé
```

### 3. Structure Générée

```
data/dentex/
├── train/
│   ├── images/          # Images d'entraînement
│   └── labels/          # Annotations YOLO
├── val/
│   ├── images/          # Images de validation
│   └── labels/          # Annotations YOLO
├── test/
│   ├── images/          # Images de test
│   └── labels/          # Annotations YOLO
├── data.yaml            # Configuration YOLO
└── dataset_info.json    # Informations du dataset
```

### 4. Entraînement avec DENTEX

```bash
# Entraîner avec le dataset DENTEX
python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8s.pt \
    --epochs 100 \
    --batch-size 16

# Entraînement rapide pour test
python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --epochs 10 \
    --batch-size 8
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

Avec le dataset DENTEX, vous pouvez vous attendre à :

- **Précision**: 80-90% mAP@0.5
- **Classes bien détectées**: caries, lésions périapicales
- **Classes limitées**: implants et plombages (pas dans DENTEX)
- **Robustesse**: Bonne généralisation sur images cliniques réelles

## 🎯 Utilisation Avancée

### Entraînement avec Fine-tuning

```bash
# Utiliser un modèle pré-entraîné sur COCO
python scripts/train_model.py \
    --config data/dentex/data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --lr 0.01 \
    --patience 50
```

### Validation et Test

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

### Export du Modèle

```bash
# Exporter le modèle entraîné
python scripts/export_model.py \
    --model models/best.pt \
    --format all
```

## 🔍 Analyse des Résultats

### Métriques Importantes

- **mAP@0.5**: Précision moyenne à IoU 0.5
- **mAP@0.5:0.95**: Précision moyenne sur différents IoU
- **Precision/Recall**: Par classe de diagnostic

### Visualisation

```python
# Analyser les résultats
import matplotlib.pyplot as plt
import pandas as pd

# Charger les métriques d'entraînement
results = pd.read_csv('models/dental_yolo_*/results.csv')

# Afficher l'évolution de la précision
plt.plot(results['epoch'], results['metrics/mAP50(B)'])
plt.title('Évolution de la précision mAP@0.5')
plt.xlabel('Époque')
plt.ylabel('mAP@0.5')
plt.show()
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

## 🆘 Support et Aide

### Problèmes Courants

1. **Erreur de téléchargement**: Vérifiez votre connexion internet
2. **Mémoire insuffisante**: Réduisez la taille du batch
3. **Annotations manquantes**: Vérifiez le format des annotations

### Contact

- **Issues**: [GitHub Issues](https://github.com/votre-repo/issues)
- **Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com)
- **Community**: [Discord YOLO](https://discord.gg/ultralytics)

---

🎉 **Vous êtes maintenant prêt à utiliser le dataset DENTEX avec EvaDentalAI!**

Ce dataset de qualité clinique vous permettra d'entraîner des modèles robustes pour la détection d'anomalies dentaires.
