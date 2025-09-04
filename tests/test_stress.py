#!/usr/bin/env python3
"""
Tests de stress pour EvaDentalAI
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.slow
def test_high_volume_prediction():
    """Test de prédiction avec un volume élevé d'images"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer un grand nombre d'images de test
    num_images = 100
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(num_images)
    ]
    
    start_time = time.time()
    
    # Prédictions séquentielles
    results = []
    for i, test_image in enumerate(test_images):
        result = model(test_image, conf=0.25)
        results.append(result)
        
        # Afficher le progrès
        if (i + 1) % 10 == 0:
            print(f"Traité {i + 1}/{num_images} images")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_images
    
    print(f"Temps total: {total_time:.2f}s")
    print(f"Temps moyen par image: {avg_time:.3f}s")
    print(f"FPS: {num_images/total_time:.1f}")
    
    # Vérifier que toutes les prédictions ont réussi
    assert len(results) == num_images, "Toutes les prédictions n'ont pas réussi"
    
    # Vérifier que le temps moyen est raisonnable
    assert avg_time < 1.0, f"Temps moyen trop élevé: {avg_time:.3f}s"

@pytest.mark.slow
def test_memory_stress():
    """Test de stress mémoire"""
    
    from ultralytics import YOLO
    import psutil
    import os
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Créer de nombreuses images et les traiter
    for i in range(50):
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image, conf=0.25)
        
        # Vérifier la mémoire toutes les 10 itérations
        if (i + 1) % 10 == 0:
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = current_memory - initial_memory
            
            print(f"Itération {i + 1}: Mémoire utilisée: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Vérifier qu'il n'y a pas de fuite mémoire excessive
            assert memory_increase < 500, f"Fuite mémoire détectée: +{memory_increase:.1f}MB"

@pytest.mark.slow
def test_concurrent_predictions():
    """Test de prédictions concurrentes"""
    
    import threading
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Fonction de prédiction pour les threads
    def predict_image(thread_id, results_list):
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        start_time = time.time()
        
        result = model(test_image, conf=0.25)
        
        end_time = time.time()
        results_list.append({
            'thread_id': thread_id,
            'time': end_time - start_time,
            'success': True
        })
    
    # Lancer plusieurs threads
    num_threads = 5
    threads = []
    results = []
    
    start_time = time.time()
    
    for i in range(num_threads):
        thread = threading.Thread(target=predict_image, args=(i, results))
        threads.append(thread)
        thread.start()
    
    # Attendre que tous les threads se terminent
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    print(f"Temps total avec {num_threads} threads: {total_time:.2f}s")
    print(f"Temps moyen par thread: {total_time/num_threads:.3f}s")
    
    # Vérifier que tous les threads ont réussi
    assert len(results) == num_threads, "Tous les threads n'ont pas réussi"
    
    for result in results:
        assert result['success'], f"Thread {result['thread_id']} a échoué"

@pytest.mark.slow
def test_large_image_processing():
    """Test de traitement d'images de grande taille"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Tester différentes tailles d'images
    image_sizes = [
        (1280, 1280),  # 2x
        (1920, 1920),  # 3x
        (2560, 2560),  # 4x
    ]
    
    for width, height in image_sizes:
        print(f"Test avec image {width}x{height}")
        
        # Créer une image de grande taille
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        start_time = time.time()
        results = model(test_image, conf=0.25)
        processing_time = time.time() - start_time
        
        print(f"Temps de traitement: {processing_time:.3f}s")
        
        # Vérifier que le traitement a réussi
        assert len(results) == 1, "Prédiction échouée"
        
        # Vérifier que le temps de traitement est raisonnable
        assert processing_time < 10.0, f"Traitement trop lent: {processing_time:.3f}s"

@pytest.mark.slow
def test_extreme_confidence_thresholds():
    """Test avec des seuils de confiance extrêmes"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Tester des seuils extrêmes
    extreme_thresholds = [0.01, 0.05, 0.95, 0.99]
    
    for threshold in extreme_thresholds:
        print(f"Test avec seuil de confiance: {threshold}")
        
        start_time = time.time()
        results = model(test_image, conf=threshold)
        processing_time = time.time() - start_time
        
        print(f"Temps de traitement: {processing_time:.3f}s")
        
        # Vérifier que le traitement a réussi
        assert len(results) == 1, f"Prédiction échouée avec seuil {threshold}"
        
        # Vérifier que le temps de traitement est raisonnable
        assert processing_time < 5.0, f"Traitement trop lent avec seuil {threshold}: {processing_time:.3f}s"

@pytest.mark.slow
def test_continuous_operation():
    """Test d'opération continue"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Opération continue pendant 5 minutes
    duration = 300  # 5 minutes
    start_time = time.time()
    processed_count = 0
    
    while time.time() - start_time < duration:
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image, conf=0.25)
        
        processed_count += 1
        
        # Afficher le progrès
        if processed_count % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Traité {processed_count} images en {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    fps = processed_count / total_time
    
    print(f"Opération continue terminée:")
    print(f"Images traitées: {processed_count}")
    print(f"Temps total: {total_time:.1f}s")
    print(f"FPS moyen: {fps:.1f}")
    
    # Vérifier que le système a traité un nombre raisonnable d'images
    assert processed_count > 100, f"Trop peu d'images traitées: {processed_count}"
    
    # Vérifier que le FPS est raisonnable
    assert fps > 1.0, f"FPS trop faible: {fps:.1f}"

@pytest.mark.slow
def test_mixed_workload():
    """Test de charge de travail mixte"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Mélanger différents types de tâches
    tasks = [
        # (taille_image, seuil_confiance, nombre_iterations)
        ((640, 640), 0.25, 10),
        ((1280, 1280), 0.5, 5),
        ((320, 320), 0.1, 15),
        ((640, 640), 0.75, 8),
    ]
    
    total_processed = 0
    start_time = time.time()
    
    for size, threshold, iterations in tasks:
        width, height = size
        print(f"Tâche: {width}x{height}, seuil {threshold}, {iterations} itérations")
        
        for i in range(iterations):
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            results = model(test_image, conf=threshold)
            
            total_processed += 1
            
            # Vérifier que la prédiction a réussi
            assert len(results) == 1, f"Prédiction échouée à l'itération {i}"
    
    total_time = time.time() - start_time
    fps = total_processed / total_time
    
    print(f"Charge de travail mixte terminée:")
    print(f"Images traitées: {total_processed}")
    print(f"Temps total: {total_time:.1f}s")
    print(f"FPS moyen: {fps:.1f}")
    
    # Vérifier que le système a géré la charge mixte
    assert total_processed > 30, f"Trop peu d'images traitées: {total_processed}"
    assert fps > 0.5, f"FPS trop faible: {fps:.1f}"

@pytest.mark.slow
def test_error_recovery():
    """Test de récupération d'erreur"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Tester avec des images problématiques
    problematic_images = [
        # Image vide
        np.zeros((640, 640, 3), dtype=np.uint8),
        
        # Image uniforme
        np.full((640, 640, 3), 128, dtype=np.uint8),
        
        # Image avec des valeurs extrêmes
        np.full((640, 640, 3), 255, dtype=np.uint8),
        np.full((640, 640, 3), 0, dtype=np.uint8),
        
        # Image avec du bruit
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
    ]
    
    for i, test_image in enumerate(problematic_images):
        print(f"Test image problématique {i + 1}")
        
        try:
            results = model(test_image, conf=0.25)
            assert len(results) == 1, f"Prédiction échouée pour l'image {i + 1}"
            print(f"  ✅ Image {i + 1} traitée avec succès")
        except Exception as e:
            print(f"  ❌ Erreur avec l'image {i + 1}: {e}")
            # Vérifier que l'erreur est gérée gracieusement
            assert False, f"Erreur non gérée: {e}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
