#!/usr/bin/env python3
"""
Tests de benchmark pour EvaDentalAI
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.slow
def test_inference_benchmark():
    """Benchmark de l'inférence"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Benchmark avec différents seuils de confiance
    confidence_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    print("Benchmark d'inférence:")
    print("=" * 50)
    
    for conf in confidence_thresholds:
        times = []
        
        # Warmup
        for _ in range(3):
            model(test_image, conf=conf)
        
        # Benchmark
        for _ in range(10):
            start_time = time.time()
            results = model(test_image, conf=conf)
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1 / avg_time
        
        print(f"Confiance {conf:4.2f}: {avg_time:.3f} ± {std_time:.3f}s ({fps:.1f} FPS)")
        
        # Vérifier que l'inférence est raisonnablement rapide
        assert avg_time < 2.0, f"Inférence trop lente avec conf {conf}: {avg_time:.3f}s"

@pytest.mark.slow
def test_batch_size_benchmark():
    """Benchmark avec différentes tailles de batch"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]
    
    print("Benchmark de taille de batch:")
    print("=" * 50)
    
    # Test avec différentes tailles de batch
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        batch_images = test_images[:batch_size]
        
        times = []
        
        # Warmup
        for _ in range(2):
            model(batch_images, conf=0.25)
        
        # Benchmark
        for _ in range(5):
            start_time = time.time()
            results = model(batch_images, conf=0.25)
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = batch_size / avg_time
        
        print(f"Batch {batch_size:2d}: {avg_time:.3f} ± {std_time:.3f}s ({fps:.1f} FPS)")
        
        # Vérifier que l'inférence est raisonnablement rapide
        assert avg_time < 5.0, f"Inférence trop lente avec batch {batch_size}: {avg_time:.3f}s"

@pytest.mark.slow
def test_image_size_benchmark():
    """Benchmark avec différentes tailles d'images"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Tester différentes tailles d'images
    image_sizes = [
        (320, 320),   # 0.5x
        (640, 640),   # 1x (standard)
        (1280, 1280), # 2x
        (1920, 1920), # 3x
    ]
    
    print("Benchmark de taille d'image:")
    print("=" * 50)
    
    for width, height in image_sizes:
        test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        times = []
        
        # Warmup
        for _ in range(2):
            model(test_image, conf=0.25)
        
        # Benchmark
        for _ in range(5):
            start_time = time.time()
            results = model(test_image, conf=0.25)
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1 / avg_time
        
        print(f"{width:4d}x{height:4d}: {avg_time:.3f} ± {std_time:.3f}s ({fps:.1f} FPS)")
        
        # Vérifier que l'inférence est raisonnablement rapide
        assert avg_time < 10.0, f"Inférence trop lente avec {width}x{height}: {avg_time:.3f}s"

@pytest.mark.slow
def test_model_comparison_benchmark():
    """Benchmark de comparaison de modèles"""
    
    from ultralytics import YOLO
    
    # Modèles à comparer
    models_to_test = [
        ("YOLOv8n", "yolov8n.pt"),
        ("YOLOv8s", "yolov8s.pt"),
        ("Entraîné", "models/best.pt")
    ]
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("Benchmark de comparaison de modèles:")
    print("=" * 50)
    
    results = []
    
    for model_name, model_path in models_to_test:
        if not Path(model_path).exists():
            print(f"⚠️  Modèle non trouvé: {model_path}")
            continue
        
        try:
            model = YOLO(model_path)
            
            times = []
            
            # Warmup
            for _ in range(3):
                model(test_image, conf=0.25)
            
            # Benchmark
            for _ in range(10):
                start_time = time.time()
                results_yolo = model(test_image, conf=0.25)
                inference_time = time.time() - start_time
                times.append(inference_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1 / avg_time
            
            # Compter les détections
            num_detections = 0
            if results_yolo[0].boxes is not None:
                num_detections = len(results_yolo[0].boxes)
            
            results.append({
                'name': model_name,
                'time': avg_time,
                'std': std_time,
                'fps': fps,
                'detections': num_detections
            })
            
            print(f"{model_name:12s}: {avg_time:.3f} ± {std_time:.3f}s ({fps:.1f} FPS, {num_detections} détections)")
            
        except Exception as e:
            print(f"❌ Erreur avec {model_name}: {e}")
    
    # Analyser les résultats
    if len(results) > 1:
        print("\nAnalyse comparative:")
        print("-" * 30)
        
        # Trouver le plus rapide
        fastest = min(results, key=lambda x: x['time'])
        print(f"Plus rapide: {fastest['name']} ({fastest['fps']:.1f} FPS)")
        
        # Trouver le plus précis
        most_detections = max(results, key=lambda x: x['detections'])
        print(f"Plus précis: {most_detections['name']} ({most_detections['detections']} détections)")

@pytest.mark.slow
def test_memory_benchmark():
    """Benchmark de l'utilisation mémoire"""
    
    import psutil
    import os
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    process = psutil.Process(os.getpid())
    
    print("Benchmark de mémoire:")
    print("=" * 50)
    
    # Mesurer la mémoire initiale
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Mémoire initiale: {initial_memory:.1f} MB")
    
    # Charger le modèle
    model = YOLO(str(model_path))
    after_load_memory = process.memory_info().rss / (1024 * 1024)  # MB
    model_memory = after_load_memory - initial_memory
    print(f"Mémoire après chargement: {after_load_memory:.1f} MB (+{model_memory:.1f} MB)")
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Mesurer la mémoire après prédiction
    results = model(test_image, conf=0.25)
    after_prediction_memory = process.memory_info().rss / (1024 * 1024)  # MB
    prediction_memory = after_prediction_memory - after_load_memory
    print(f"Mémoire après prédiction: {after_prediction_memory:.1f} MB (+{prediction_memory:.1f} MB)")
    
    # Test de prédictions multiples
    for i in range(10):
        results = model(test_image, conf=0.25)
        
        if (i + 1) % 5 == 0:
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = current_memory - after_prediction_memory
            print(f"Après {i + 1} prédictions: {current_memory:.1f} MB (+{memory_increase:.1f} MB)")
    
    # Vérifier que l'utilisation mémoire est raisonnable
    assert model_memory < 500, f"Modèle trop volumineux: {model_memory:.1f} MB"
    assert prediction_memory < 100, f"Prédiction trop gourmande: {prediction_memory:.1f} MB"

@pytest.mark.slow
def test_accuracy_benchmark():
    """Benchmark de précision"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer des images de test avec des objets connus
    test_images = []
    
    for i in range(10):
        # Créer une image avec des formes simples
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Ajouter des rectangles de différentes tailles
        for j in range(3):
            x = 100 + j * 150
            y = 100 + j * 150
            w = 50 + j * 20
            h = 50 + j * 20
            
            # Couleur différente pour chaque rectangle
            color = [100 + j * 50, 150 + j * 30, 200 + j * 20]
            image[y:y+h, x:x+w] = color
        
        test_images.append(image)
    
    print("Benchmark de précision:")
    print("=" * 50)
    
    # Tester avec différents seuils de confiance
    confidence_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    for conf in confidence_thresholds:
        total_detections = 0
        total_time = 0
        
        for test_image in test_images:
            start_time = time.time()
            results = model(test_image, conf=conf)
            inference_time = time.time() - start_time
            
            total_time += inference_time
            
            if results[0].boxes is not None:
                total_detections += len(results[0].boxes)
        
        avg_detections = total_detections / len(test_images)
        avg_time = total_time / len(test_images)
        fps = 1 / avg_time
        
        print(f"Confiance {conf:4.2f}: {avg_detections:.1f} détections/image, {avg_time:.3f}s/image ({fps:.1f} FPS)")
        
        # Vérifier que la précision est raisonnable
        assert avg_detections > 0, f"Aucune détection avec confiance {conf}"

@pytest.mark.slow
def test_throughput_benchmark():
    """Benchmark de débit"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer de nombreuses images de test
    num_images = 100
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(num_images)
    ]
    
    print("Benchmark de débit:")
    print("=" * 50)
    
    # Test de débit séquentiel
    start_time = time.time()
    for test_image in test_images:
        results = model(test_image, conf=0.25)
    sequential_time = time.time() - start_time
    sequential_fps = num_images / sequential_time
    
    print(f"Débit séquentiel: {sequential_fps:.1f} FPS ({sequential_time:.2f}s total)")
    
    # Test de débit batch (si supporté)
    try:
        start_time = time.time()
        results = model(test_images, conf=0.25)
        batch_time = time.time() - start_time
        batch_fps = num_images / batch_time
        
        print(f"Débit batch: {batch_fps:.1f} FPS ({batch_time:.2f}s total)")
        print(f"Accélération batch: {batch_fps/sequential_fps:.1f}x")
        
        # Vérifier que le batch est plus rapide
        assert batch_fps > sequential_fps, "Batch devrait être plus rapide que séquentiel"
        
    except Exception as e:
        print(f"Batch non supporté: {e}")
    
    # Vérifier que le débit est raisonnable
    assert sequential_fps > 1.0, f"Débit séquentiel trop faible: {sequential_fps:.1f} FPS"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
