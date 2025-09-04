# Guide d'Utilisation - EvaDentalAI

Ce guide vous explique comment utiliser EvaDentalAI pour détecter des anomalies dentaires sur des radiographies.

## 🚀 Démarrage Rapide

### 1. Préparation du Dataset
```bash
# Générer un dataset simulé (100 images)
python scripts/prepare_dataset.py --num-images 100

# Ou avec plus d'images pour de meilleures performances
python scripts/prepare_dataset.py --num-images 500
```

### 2. Entraînement du Modèle
```bash
# Entraînement rapide (5 épochs)
python scripts/train_model.py --epochs 5 --batch-size 8

# Entraînement complet (100 épochs)
python scripts/train_model.py --epochs 100 --batch-size 16
```

### 3. Prédiction sur une Image
```bash
# Analyser une radiographie
python scripts/predict.py --model models/best.pt --image path/to/radiographie.jpg

# Avec sauvegarde des résultats
python scripts/predict.py --model models/best.pt --image path/to/radiographie.jpg --save --report
```

### 4. Lancer l'API
```bash
# Démarrer le serveur API
python api/main.py --model models/best.pt

# L'API sera disponible sur http://localhost:8000
```

## 📊 Classes Détectées

EvaDentalAI peut détecter 5 types d'anomalies dentaires:

| Classe | Description | Couleur |
|--------|-------------|---------|
| `tooth` | Dent normale | Blanc |
| `cavity` | Carie | Rouge |
| `implant` | Implant dentaire | Vert |
| `lesion` | Lésion | Bleu |
| `filling` | Plombage | Jaune |

## 🔧 Utilisation Avancée

### Entraînement Personnalisé

#### Configuration du Dataset
Modifiez `config/data.yaml` pour personnaliser:
```yaml
# Chemins des données
path: ./data
train: processed/train/images
val: processed/val/images
test: processed/test/images

# Classes personnalisées
names:
  0: tooth
  1: cavity
  2: implant
  3: lesion
  4: filling
  5: crown      # Nouvelle classe
  6: bridge     # Nouvelle classe

nc: 7  # Nombre de classes
```

#### Paramètres d'Entraînement
```bash
python scripts/train_model.py \
    --config config/data.yaml \
    --model yolov8s.pt \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --lr 0.01 \
    --device cuda \
    --patience 50
```

**Paramètres disponibles:**
- `--epochs`: Nombre d'épochs (défaut: 100)
- `--batch-size`: Taille du batch (défaut: 16)
- `--img-size`: Taille des images (défaut: 640)
- `--lr`: Learning rate (défaut: 0.01)
- `--device`: Device (cpu/cuda/auto)
- `--patience`: Patience pour early stopping (défaut: 50)

### Prédiction Avancée

#### Prédiction sur une Image
```bash
python scripts/predict.py \
    --model models/best.pt \
    --image radiographie.jpg \
    --conf 0.3 \
    --iou 0.5 \
    --save \
    --report
```

#### Prédiction Batch
```bash
python scripts/predict.py \
    --model models/best.pt \
    --batch data/test_images/ \
    --output results/ \
    --conf 0.25
```

**Paramètres de prédiction:**
- `--conf`: Seuil de confiance (0.0-1.0)
- `--iou`: Seuil IoU pour NMS (0.0-1.0)
- `--save`: Sauvegarder les résultats
- `--report`: Générer un rapport détaillé

### Export et Optimisation

#### Export Multi-Format
```bash
# Export dans tous les formats
python scripts/export_model.py --model models/best.pt --format all

# Export ONNX optimisé
python scripts/export_model.py --model models/best.pt --format optimized
```

#### Formats Supportés
- **ONNX**: Pour déploiement cross-platform
- **TorchScript**: Pour PyTorch mobile
- **TensorFlow Lite**: Pour mobile/edge
- **ONNX optimisé**: Pour inférence rapide

## 🌐 Utilisation de l'API

### Endpoints Disponibles

#### 1. Prédiction sur une Image
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@radiographie.jpg" \
     -F "confidence=0.25" \
     -F "iou=0.45"
```

**Réponse:**
```json
{
  "success": true,
  "image_name": "radiographie.jpg",
  "inference_time": 0.045,
  "total_detections": 3,
  "detections": [
    {
      "class_id": 1,
      "class_name": "cavity",
      "confidence": 0.87,
      "bbox": {
        "x1": 120.5,
        "y1": 200.3,
        "x2": 180.7,
        "y2": 250.1,
        "width": 60.2,
        "height": 49.8
      }
    }
  ]
}
```

#### 2. Prédiction Batch
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "confidence=0.25"
```

#### 3. Informations sur le Modèle
```bash
curl "http://localhost:8000/model/info"
```

#### 4. Vérification de Santé
```bash
curl "http://localhost:8000/health"
```

### Utilisation avec Python

```python
import requests

# Prédiction sur une image
with open('radiographie.jpg', 'rb') as f:
    files = {'file': f}
    data = {'confidence': 0.25, 'iou': 0.45}
    
    response = requests.post(
        'http://localhost:8000/predict',
        files=files,
        data=data
    )
    
    result = response.json()
    print(f"Détections: {result['total_detections']}")
    for detection in result['detections']:
        print(f"- {detection['class_name']}: {detection['confidence']:.2f}")
```

## 🐳 Déploiement avec Docker

### Construction et Lancement
```bash
# Construire l'image
docker build -f docker/Dockerfile -t evadental-ai .

# Lancer le conteneur
docker run -p 8000:8000 -v $(pwd)/models:/app/models evadental-ai

# Ou utiliser docker-compose
docker-compose up -d
```

### Configuration Docker
Modifiez `docker/docker-compose.yml`:
```yaml
services:
  evadental-api:
    environment:
      - MODEL_PATH=/app/models/best.pt
      - CONFIDENCE_THRESHOLD=0.25
      - IOU_THRESHOLD=0.45
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
```

## 📈 Optimisation des Performances

### Pour l'Entraînement
1. **Utilisez un GPU** si disponible
2. **Augmentez la taille du batch** (selon votre GPU)
3. **Utilisez des images plus grandes** (640x640 ou 1024x1024)
4. **Augmentez le nombre d'épochs** pour de meilleures performances

### Pour l'Inférence
1. **Exportez en ONNX optimisé** pour la vitesse
2. **Utilisez FP16** pour réduire la mémoire
3. **Optimisez la taille des images** selon vos besoins
4. **Utilisez le batching** pour traiter plusieurs images

### Exemple d'Optimisation
```bash
# Entraînement optimisé
python scripts/train_model.py \
    --model yolov8s.pt \
    --epochs 200 \
    --batch-size 32 \
    --img-size 640 \
    --device cuda

# Export optimisé
python scripts/export_model.py \
    --model models/best.pt \
    --format optimized
```

## 🔍 Monitoring et Logs

### Logs d'Entraînement
Les logs sont sauvegardés dans:
- `models/dental_yolo_YYYYMMDD_HHMMSS/`
- `results.csv`: Métriques d'entraînement
- `training_plots.png`: Graphiques de performance

### Logs de l'API
```bash
# Voir les logs en temps réel
docker-compose logs -f

# Logs avec niveau de détail
python api/main.py --reload
```

### Métriques Importantes
- **mAP@0.5**: Précision moyenne à IoU 0.5
- **mAP@0.5:0.95**: Précision moyenne sur différents IoU
- **Precision**: Précision des détections
- **Recall**: Rappel des détections
- **Inference Time**: Temps d'inférence par image

## 🚨 Résolution de Problèmes

### Problèmes Courants

#### 1. Modèle non trouvé
```bash
# Vérifier que le modèle existe
ls -la models/best.pt

# Entraîner un nouveau modèle
python scripts/train_model.py --epochs 10
```

#### 2. Erreur de mémoire GPU
```bash
# Réduire la taille du batch
python scripts/train_model.py --batch-size 8

# Utiliser CPU
python scripts/train_model.py --device cpu
```

#### 3. API ne démarre pas
```bash
# Vérifier les ports
netstat -an | grep 8000

# Changer le port
python api/main.py --port 8001
```

#### 4. Prédictions incorrectes
- Vérifiez le seuil de confiance
- Entraînez plus longtemps
- Augmentez la taille du dataset
- Vérifiez la qualité des annotations

### Debug Mode
```bash
# Entraînement avec debug
python scripts/train_model.py --epochs 5 --verbose

# API avec reload automatique
python api/main.py --reload
```

## 📚 Ressources Supplémentaires

- **Documentation YOLO**: [docs.ultralytics.com](https://docs.ultralytics.com)
- **FastAPI**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- **PyTorch**: [pytorch.org/docs](https://pytorch.org/docs)
- **OpenCV**: [docs.opencv.org](https://docs.opencv.org)

## 🎯 Prochaines Étapes

1. **Améliorer le Dataset**: Ajoutez plus d'images réelles
2. **Fine-tuning**: Ajustez les hyperparamètres
3. **Validation**: Testez sur des cas cliniques
4. **Déploiement**: Intégrez dans votre application
5. **Monitoring**: Surveillez les performances en production

---

🎉 **Vous êtes maintenant prêt à utiliser EvaDentalAI!** 

Pour toute question, consultez la documentation ou contactez l'équipe de support.
