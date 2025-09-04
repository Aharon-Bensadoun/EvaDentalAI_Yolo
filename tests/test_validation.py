#!/usr/bin/env python3
"""
Tests de validation pour EvaDentalAI
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

def test_dataset_structure():
    """Test de la structure du dataset"""
    
    data_dir = Path("data/processed")
    if not data_dir.exists():
        pytest.skip("Dataset non généré")
    
    # Vérifier la structure
    required_dirs = ["train", "val", "test"]
    for split in required_dirs:
        split_dir = data_dir / split
        assert split_dir.exists(), f"Répertoire {split} manquant"
        
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        assert images_dir.exists(), f"Répertoire {split}/images manquant"
        assert labels_dir.exists(), f"Répertoire {split}/labels manquant"
        
        # Vérifier qu'il y a des images
        images = list(images_dir.glob("*.jpg"))
        assert len(images) > 0, f"Aucune image dans {split}/images"
        
        # Vérifier qu'il y a des labels correspondants
        labels = list(labels_dir.glob("*.txt"))
        assert len(labels) > 0, f"Aucun label dans {split}/labels"

def test_annotation_format():
    """Test du format des annotations YOLO"""
    
    labels_dir = Path("data/processed/train/labels")
    if not labels_dir.exists():
        pytest.skip("Labels non générés")
    
    label_files = list(labels_dir.glob("*.txt"))
    assert len(label_files) > 0, "Aucun fichier de label trouvé"
    
    for label_file in label_files[:5]:  # Tester les 5 premiers
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:  # Ignorer les lignes vides
                parts = line.split()
                assert len(parts) == 5, f"Format invalide dans {label_file}: {line}"
                
                # Vérifier les types
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:5])
                    
                    assert 0 <= class_id <= 4, f"Class ID invalide: {class_id}"
                    assert 0 <= x <= 1, f"X invalide: {x}"
                    assert 0 <= y <= 1, f"Y invalide: {y}"
                    assert 0 <= w <= 1, f"W invalide: {w}"
                    assert 0 <= h <= 1, f"H invalide: {h}"
                    
                except ValueError:
                    pytest.fail(f"Valeurs non numériques dans {label_file}: {line}")

def test_image_quality():
    """Test de la qualité des images"""
    
    images_dir = Path("data/processed/train/images")
    if not images_dir.exists():
        pytest.skip("Images non générées")
    
    import cv2
    
    image_files = list(images_dir.glob("*.jpg"))
    assert len(image_files) > 0, "Aucune image trouvée"
    
    for image_file in image_files[:5]:  # Tester les 5 premières
        # Charger l'image
        image = cv2.imread(str(image_file))
        assert image is not None, f"Impossible de charger {image_file}"
        
        # Vérifier les dimensions
        height, width = image.shape[:2]
        assert height > 0, f"Hauteur invalide: {height}"
        assert width > 0, f"Largeur invalide: {width}"
        
        # Vérifier que l'image n'est pas vide
        assert np.any(image), f"Image vide: {image_file}"

def test_model_output_format():
    """Test du format de sortie du modèle"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédiction
    results = model(test_image, conf=0.25)
    
    assert len(results) == 1, "Devrait retourner un seul résultat"
    result = results[0]
    
    # Vérifier les attributs
    assert hasattr(result, 'boxes'), "Résultat devrait avoir des boxes"
    assert hasattr(result, 'plot'), "Résultat devrait avoir une méthode plot"
    
    # Vérifier le format des boxes
    if result.boxes is not None:
        boxes = result.boxes
        assert hasattr(boxes, 'xyxy'), "Boxes devrait avoir xyxy"
        assert hasattr(boxes, 'conf'), "Boxes devrait avoir conf"
        assert hasattr(boxes, 'cls'), "Boxes devrait avoir cls"

def test_prediction_consistency():
    """Test de la cohérence des prédictions"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédictions multiples
    results1 = model(test_image, conf=0.25)
    results2 = model(test_image, conf=0.25)
    
    # Vérifier que les résultats sont cohérents
    assert len(results1) == len(results2), "Nombre de résultats incohérent"
    
    if results1[0].boxes is not None and results2[0].boxes is not None:
        boxes1 = results1[0].boxes
        boxes2 = results2[0].boxes
        
        # Vérifier que le nombre de détections est cohérent
        assert len(boxes1) == len(boxes2), "Nombre de détections incohérent"

def test_class_distribution():
    """Test de la distribution des classes dans le dataset"""
    
    labels_dir = Path("data/processed/train/labels")
    if not labels_dir.exists():
        pytest.skip("Labels non générés")
    
    label_files = list(labels_dir.glob("*.txt"))
    assert len(label_files) > 0, "Aucun fichier de label trouvé"
    
    class_counts = {i: 0 for i in range(5)}
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                class_id = int(line.split()[0])
                if class_id in class_counts:
                    class_counts[class_id] += 1
    
    # Vérifier qu'il y a au moins une instance de chaque classe
    for class_id, count in class_counts.items():
        assert count > 0, f"Classe {class_id} non représentée dans le dataset"

def test_bbox_validity():
    """Test de la validité des bounding boxes"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédiction
    results = model(test_image, conf=0.25)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Vérifier que les coordonnées sont valides
            assert x1 < x2, f"X1 ({x1}) devrait être < X2 ({x2})"
            assert y1 < y2, f"Y1 ({y1}) devrait être < Y2 ({y2})"
            
            # Vérifier que les coordonnées sont dans l'image
            assert 0 <= x1 <= 640, f"X1 ({x1}) hors limites"
            assert 0 <= y1 <= 640, f"Y1 ({y1}) hors limites"
            assert 0 <= x2 <= 640, f"X2 ({x2}) hors limites"
            assert 0 <= y2 <= 640, f"Y2 ({y2}) hors limites"

def test_confidence_scores():
    """Test des scores de confiance"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédiction
    results = model(test_image, conf=0.25)
    
    if results[0].boxes is not None:
        confidences = results[0].boxes.conf.cpu().numpy()
        
        for conf in confidences:
            # Vérifier que les scores de confiance sont valides
            assert 0 <= conf <= 1, f"Score de confiance invalide: {conf}"
            
            # Vérifier que les scores respectent le seuil
            assert conf >= 0.25, f"Score de confiance trop bas: {conf}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
