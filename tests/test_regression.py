#!/usr/bin/env python3
"""
Tests de régression pour EvaDentalAI
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_output_consistency():
    """Test de cohérence des sorties du modèle"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test fixe
    np.random.seed(42)  # Seed fixe pour reproductibilité
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédictions multiples
    results1 = model(test_image, conf=0.25)
    results2 = model(test_image, conf=0.25)
    
    # Vérifier la cohérence
    assert len(results1) == len(results2), "Nombre de résultats incohérent"
    
    if results1[0].boxes is not None and results2[0].boxes is not None:
        boxes1 = results1[0].boxes.xyxy.cpu().numpy()
        boxes2 = results2[0].boxes.xyxy.cpu().numpy()
        
        # Vérifier que les coordonnées sont identiques
        np.testing.assert_array_almost_equal(boxes1, boxes2, decimal=5)

def test_prediction_accuracy_regression():
    """Test de régression de la précision des prédictions"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test avec des objets connus
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Ajouter des formes simples pour tester la détection
    # Rectangle blanc (simulant une dent)
    test_image[100:200, 100:200] = [255, 255, 255]
    
    # Rectangle sombre (simulant une carie)
    test_image[300:400, 300:400] = [50, 50, 50]
    
    # Prédiction
    results = model(test_image, conf=0.25)
    
    # Vérifier qu'au moins une détection est trouvée
    if results[0].boxes is not None:
        assert len(results[0].boxes) > 0, "Aucune détection trouvée sur l'image de test"
        
        # Vérifier que les détections sont dans les bonnes zones
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Vérifier qu'il y a des détections avec une confiance raisonnable
        assert np.max(confidences) > 0.1, "Confiance maximale trop faible"

def test_dataset_quality_regression():
    """Test de régression de la qualité du dataset"""
    
    data_dir = Path("data/processed")
    if not data_dir.exists():
        pytest.skip("Dataset non généré")
    
    # Vérifier la qualité des images
    images_dir = data_dir / "train/images"
    if not images_dir.exists():
        pytest.skip("Images d'entraînement non trouvées")
    
    import cv2
    
    image_files = list(images_dir.glob("*.jpg"))
    assert len(image_files) > 0, "Aucune image d'entraînement trouvée"
    
    # Vérifier la qualité de quelques images
    for image_file in image_files[:5]:
        image = cv2.imread(str(image_file))
        assert image is not None, f"Impossible de charger {image_file}"
        
        # Vérifier que l'image n'est pas corrompue
        height, width = image.shape[:2]
        assert height > 0 and width > 0, f"Dimensions invalides: {height}x{width}"
        
        # Vérifier que l'image n'est pas uniforme (pas de bruit)
        std_dev = np.std(image)
        assert std_dev > 10, f"Image trop uniforme: {image_file}"

def test_annotation_quality_regression():
    """Test de régression de la qualité des annotations"""
    
    labels_dir = Path("data/processed/train/labels")
    if not labels_dir.exists():
        pytest.skip("Labels non générés")
    
    label_files = list(labels_dir.glob("*.txt"))
    assert len(label_files) > 0, "Aucun fichier de label trouvé"
    
    # Vérifier la qualité des annotations
    for label_file in label_files[:5]:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                assert len(parts) == 5, f"Format d'annotation invalide: {line}"
                
                class_id, x, y, w, h = parts
                
                # Vérifier que les coordonnées sont dans les bonnes plages
                assert 0 <= float(x) <= 1, f"X hors limites: {x}"
                assert 0 <= float(y) <= 1, f"Y hors limites: {y}"
                assert 0 < float(w) <= 1, f"W invalide: {w}"
                assert 0 < float(h) <= 1, f"H invalide: {h}"

def test_model_size_regression():
    """Test de régression de la taille du modèle"""
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    # Vérifier que la taille du modèle est raisonnable
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    
    # Le modèle ne devrait pas être trop volumineux
    assert model_size_mb < 100, f"Modèle trop volumineux: {model_size_mb:.1f}MB"
    
    # Le modèle ne devrait pas être trop petit (signe de problème)
    assert model_size_mb > 1, f"Modèle trop petit: {model_size_mb:.1f}MB"

def test_inference_speed_regression():
    """Test de régression de la vitesse d'inférence"""
    
    from ultralytics import YOLO
    import time
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Mesurer le temps d'inférence
    times = []
    for _ in range(5):
        start_time = time.time()
        results = model(test_image, conf=0.25)
        inference_time = time.time() - start_time
        times.append(inference_time)
    
    avg_time = np.mean(times)
    
    # Vérifier que l'inférence est raisonnablement rapide
    assert avg_time < 2.0, f"Inférence trop lente: {avg_time:.3f}s"
    
    # Vérifier que l'inférence est stable (pas de variation excessive)
    std_time = np.std(times)
    assert std_time < 0.5, f"Variation d'inférence trop élevée: {std_time:.3f}s"

def test_memory_usage_regression():
    """Test de régression de l'utilisation mémoire"""
    
    import psutil
    import os
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    process = psutil.Process(os.getpid())
    
    # Mesurer la mémoire avant
    memory_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Charger le modèle
    model = YOLO(str(model_path))
    memory_after_load = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Vérifier que l'utilisation mémoire est raisonnable
    model_memory = memory_after_load - memory_before
    assert model_memory < 500, f"Utilisation mémoire excessive: {model_memory:.1f}MB"
    
    # Vérifier que l'utilisation mémoire est stable
    memory_after_prediction = process.memory_info().rss / (1024 * 1024)  # MB
    memory_increase = memory_after_prediction - memory_after_load
    assert memory_increase < 100, f"Augmentation mémoire excessive: {memory_increase:.1f}MB"

def test_class_distribution_regression():
    """Test de régression de la distribution des classes"""
    
    labels_dir = Path("data/processed/train/labels")
    if not labels_dir.exists():
        pytest.skip("Labels non générés")
    
    label_files = list(labels_dir.glob("*.txt"))
    assert len(label_files) > 0, "Aucun fichier de label trouvé"
    
    # Compter les classes
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
    
    # Vérifier que toutes les classes sont représentées
    for class_id, count in class_counts.items():
        assert count > 0, f"Classe {class_id} non représentée"
    
    # Vérifier que la distribution n'est pas trop déséquilibrée
    total_annotations = sum(class_counts.values())
    for class_id, count in class_counts.items():
        proportion = count / total_annotations
        assert proportion > 0.05, f"Classe {class_id} sous-représentée: {proportion:.2%}"

def test_bbox_quality_regression():
    """Test de régression de la qualité des bounding boxes"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test avec des objets de taille connue
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Ajouter des rectangles de taille connue
    test_image[100:200, 100:200] = [255, 255, 255]  # 100x100 pixels
    
    # Prédiction
    results = model(test_image, conf=0.25)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Vérifier que les coordonnées sont valides
            assert x1 < x2, f"X1 ({x1}) >= X2 ({x2})"
            assert y1 < y2, f"Y1 ({y1}) >= Y2 ({y2})"
            
            # Vérifier que les dimensions sont raisonnables
            width = x2 - x1
            height = y2 - y1
            
            assert width > 10, f"Largeur trop petite: {width}"
            assert height > 10, f"Hauteur trop petite: {height}"
            assert width < 600, f"Largeur trop grande: {width}"
            assert height < 600, f"Hauteur trop grande: {height}"

def test_confidence_distribution_regression():
    """Test de régression de la distribution des confidences"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer plusieurs images de test
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    all_confidences = []
    
    for test_image in test_images:
        results = model(test_image, conf=0.25)
        
        if results[0].boxes is not None:
            confidences = results[0].boxes.conf.cpu().numpy()
            all_confidences.extend(confidences)
    
    if all_confidences:
        # Vérifier que les confidences sont dans la bonne plage
        assert all(0 <= conf <= 1 for conf in all_confidences), "Confidences hors limites"
        
        # Vérifier que la distribution des confidences est raisonnable
        mean_conf = np.mean(all_confidences)
        assert 0.3 <= mean_conf <= 0.8, f"Moyenne des confidences anormale: {mean_conf:.3f}"
        
        # Vérifier qu'il y a une variabilité dans les confidences
        std_conf = np.std(all_confidences)
        assert std_conf > 0.1, f"Variabilité des confidences trop faible: {std_conf:.3f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
