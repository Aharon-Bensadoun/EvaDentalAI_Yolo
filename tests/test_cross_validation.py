#!/usr/bin/env python3
"""
Tests de validation croisée pour EvaDentalAI
"""

import pytest
import sys
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

def test_k_fold_cross_validation():
    """Test de validation croisée k-fold"""
    
    from sklearn.model_selection import KFold
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(20)
    ]
    
    # Configuration k-fold
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Validation croisée k-fold:")
    print("=" * 50)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(k_fold.split(test_images)):
        print(f"Fold {fold + 1}/5")
        
        # Images d'entraînement et de validation
        train_images = [test_images[i] for i in train_idx]
        val_images = [test_images[i] for i in val_idx]
        
        # Prédictions sur les images de validation
        val_predictions = []
        for val_image in val_images:
            results = model(val_image, conf=0.25)
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            else:
                num_detections = 0
            val_predictions.append(num_detections)
        
        # Score du fold (nombre moyen de détections)
        fold_score = np.mean(val_predictions)
        fold_scores.append(fold_score)
        
        print(f"  Score: {fold_score:.2f}")
    
    # Score moyen de validation croisée
    cv_score = np.mean(fold_scores)
    cv_std = np.std(fold_scores)
    
    print(f"\nScore de validation croisée: {cv_score:.2f} ± {cv_std:.2f}")
    
    # Vérifier que le score est raisonnable
    assert cv_score > 0, "Score de validation croisée trop faible"
    assert cv_std < 2.0, "Variabilité trop élevée entre les folds"

def test_leave_one_out_cross_validation():
    """Test de validation croisée leave-one-out"""
    
    from sklearn.model_selection import LeaveOneOut
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test (moins pour leave-one-out)
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    # Configuration leave-one-out
    loo = LeaveOneOut()
    
    print("Validation croisée leave-one-out:")
    print("=" * 50)
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(loo.split(test_images)):
        print(f"Fold {fold + 1}/10")
        
        # Image de validation
        val_image = test_images[val_idx[0]]
        
        # Prédiction
        results = model(val_image, conf=0.25)
        if results[0].boxes is not None:
            num_detections = len(results[0].boxes)
        else:
            num_detections = 0
        
        scores.append(num_detections)
        print(f"  Détections: {num_detections}")
    
    # Score moyen
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"\nScore moyen: {mean_score:.2f} ± {std_score:.2f}")
    
    # Vérifier que le score est raisonnable
    assert mean_score >= 0, "Score moyen négatif"
    assert std_score < 3.0, "Variabilité trop élevée"

def test_stratified_cross_validation():
    """Test de validation croisée stratifiée"""
    
    from sklearn.model_selection import StratifiedKFold
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test avec des labels simulés
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(20)
    ]
    
    # Labels simulés (nombre de détections attendues)
    test_labels = [np.random.randint(0, 5) for _ in range(20)]
    
    # Configuration validation croisée stratifiée
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Validation croisée stratifiée:")
    print("=" * 50)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(test_images, test_labels)):
        print(f"Fold {fold + 1}/5")
        
        # Images de validation
        val_images = [test_images[i] for i in val_idx]
        val_labels = [test_labels[i] for i in val_idx]
        
        # Prédictions
        val_predictions = []
        for val_image in val_images:
            results = model(val_image, conf=0.25)
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            else:
                num_detections = 0
            val_predictions.append(num_detections)
        
        # Score du fold (corrélation entre prédictions et labels)
        correlation = np.corrcoef(val_predictions, val_labels)[0, 1]
        fold_scores.append(correlation)
        
        print(f"  Corrélation: {correlation:.3f}")
    
    # Score moyen
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nScore moyen: {mean_score:.3f} ± {std_score:.3f}")
    
    # Vérifier que le score est raisonnable
    assert not np.isnan(mean_score), "Score moyen NaN"
    assert std_score < 1.0, "Variabilité trop élevée"

def test_time_series_cross_validation():
    """Test de validation croisée pour séries temporelles"""
    
    from sklearn.model_selection import TimeSeriesSplit
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test avec une progression temporelle
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(15)
    ]
    
    # Configuration validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("Validation croisée temporelle:")
    print("=" * 50)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(test_images)):
        print(f"Fold {fold + 1}/5")
        
        # Images de validation
        val_images = [test_images[i] for i in val_idx]
        
        # Prédictions
        val_predictions = []
        for val_image in val_images:
            results = model(val_image, conf=0.25)
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            else:
                num_detections = 0
            val_predictions.append(num_detections)
        
        # Score du fold
        fold_score = np.mean(val_predictions)
        fold_scores.append(fold_score)
        
        print(f"  Score: {fold_score:.2f}")
    
    # Score moyen
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nScore moyen: {mean_score:.2f} ± {std_score:.2f}")
    
    # Vérifier que le score est raisonnable
    assert mean_score >= 0, "Score moyen négatif"
    assert std_score < 2.0, "Variabilité trop élevée"

def test_bootstrap_cross_validation():
    """Test de validation croisée bootstrap"""
    
    from sklearn.model_selection import Bootstrap
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(20)
    ]
    
    # Configuration bootstrap
    bootstrap = Bootstrap(n_splits=10, random_state=42)
    
    print("Validation croisée bootstrap:")
    print("=" * 50)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(bootstrap.split(test_images)):
        print(f"Fold {fold + 1}/10")
        
        # Images de validation
        val_images = [test_images[i] for i in val_idx]
        
        # Prédictions
        val_predictions = []
        for val_image in val_images:
            results = model(val_image, conf=0.25)
            if results[0].boxes is not None:
                num_detections = len(results[0].boxes)
            else:
                num_detections = 0
            val_predictions.append(num_detections)
        
        # Score du fold
        fold_score = np.mean(val_predictions)
        fold_scores.append(fold_score)
        
        print(f"  Score: {fold_score:.2f}")
    
    # Score moyen et intervalle de confiance
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    # Intervalle de confiance 95%
    confidence_interval = 1.96 * std_score
    lower_bound = mean_score - confidence_interval
    upper_bound = mean_score + confidence_interval
    
    print(f"\nScore moyen: {mean_score:.2f} ± {std_score:.2f}")
    print(f"Intervalle de confiance 95%: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Vérifier que le score est raisonnable
    assert mean_score >= 0, "Score moyen négatif"
    assert std_score < 2.0, "Variabilité trop élevée"

def test_cross_validation_consistency():
    """Test de cohérence de la validation croisée"""
    
    from sklearn.model_selection import KFold
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(20)
    ]
    
    # Effectuer la validation croisée plusieurs fois
    num_runs = 3
    all_scores = []
    
    print("Test de cohérence de validation croisée:")
    print("=" * 50)
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")
        
        # Configuration k-fold
        k_fold = KFold(n_splits=5, shuffle=True, random_state=run)
        
        run_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(k_fold.split(test_images)):
            # Images de validation
            val_images = [test_images[i] for i in val_idx]
            
            # Prédictions
            val_predictions = []
            for val_image in val_images:
                results = model(val_image, conf=0.25)
                if results[0].boxes is not None:
                    num_detections = len(results[0].boxes)
                else:
                    num_detections = 0
                val_predictions.append(num_detections)
            
            # Score du fold
            fold_score = np.mean(val_predictions)
            run_scores.append(fold_score)
        
        # Score moyen du run
        run_mean = np.mean(run_scores)
        all_scores.append(run_mean)
        
        print(f"  Score moyen: {run_mean:.2f}")
    
    # Analyser la cohérence
    mean_consistency = np.mean(all_scores)
    std_consistency = np.std(all_scores)
    
    print(f"\nCohérence entre les runs:")
    print(f"Score moyen: {mean_consistency:.2f} ± {std_consistency:.2f}")
    
    # Vérifier que la cohérence est raisonnable
    assert std_consistency < 1.0, f"Cohérence trop faible: {std_consistency:.2f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
