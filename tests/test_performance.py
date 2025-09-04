#!/usr/bin/env python3
"""
Tests de performance pour EvaDentalAI
"""

import pytest
import sys
import time
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.slow
def test_inference_speed():
    """Test de la vitesse d'inférence"""
    
    from ultralytics import YOLO
    
    # Charger le modèle
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Mesurer le temps d'inférence
    times = []
    for _ in range(10):  # 10 itérations pour une moyenne stable
        start_time = time.time()
        results = model(test_image, conf=0.25)
        inference_time = time.time() - start_time
        times.append(inference_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"Temps d'inférence moyen: {avg_time:.3f} ± {std_time:.3f} secondes")
    print(f"FPS: {1/avg_time:.1f}")
    
    # Vérifier que l'inférence est raisonnablement rapide
    assert avg_time < 1.0, f"Inférence trop lente: {avg_time:.3f}s"
    assert std_time < 0.1, f"Variance trop élevée: {std_time:.3f}s"

@pytest.mark.gpu
def test_gpu_vs_cpu_performance():
    """Test de performance GPU vs CPU"""
    
    import torch
    from ultralytics import YOLO
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA non disponible")
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test CPU
    model.to('cpu')
    cpu_times = []
    for _ in range(5):
        start_time = time.time()
        results = model(test_image, conf=0.25)
        cpu_times.append(time.time() - start_time)
    
    # Test GPU
    model.to('cuda')
    gpu_times = []
    for _ in range(5):
        start_time = time.time()
        results = model(test_image, conf=0.25)
        gpu_times.append(time.time() - start_time)
    
    cpu_avg = np.mean(cpu_times)
    gpu_avg = np.mean(gpu_times)
    
    print(f"CPU: {cpu_avg:.3f}s")
    print(f"GPU: {gpu_avg:.3f}s")
    print(f"Accélération: {cpu_avg/gpu_avg:.1f}x")
    
    # Vérifier que le GPU est plus rapide
    assert gpu_avg < cpu_avg, "GPU devrait être plus rapide que CPU"

@pytest.mark.slow
def test_memory_usage():
    """Test de l'utilisation mémoire"""
    
    import psutil
    import os
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    # Mesurer la mémoire avant
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Charger le modèle
    model = YOLO(str(model_path))
    memory_after_load = process.memory_info().rss / 1024 / 1024  # MB
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Prédiction
    results = model(test_image, conf=0.25)
    memory_after_prediction = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Mémoire avant: {memory_before:.1f} MB")
    print(f"Mémoire après chargement: {memory_after_load:.1f} MB")
    print(f"Mémoire après prédiction: {memory_after_prediction:.1f} MB")
    print(f"Utilisation modèle: {memory_after_load - memory_before:.1f} MB")
    
    # Vérifier que l'utilisation mémoire est raisonnable
    model_memory = memory_after_load - memory_before
    assert model_memory < 1000, f"Utilisation mémoire trop élevée: {model_memory:.1f} MB"

@pytest.mark.slow
def test_batch_processing_speed():
    """Test de la vitesse de traitement batch"""
    
    from ultralytics import YOLO
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    model = YOLO(str(model_path))
    
    # Créer plusieurs images de test
    batch_size = 5
    test_images = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(batch_size)
    ]
    
    # Traitement séquentiel
    start_time = time.time()
    for image in test_images:
        results = model(image, conf=0.25)
    sequential_time = time.time() - start_time
    
    # Traitement batch (si supporté)
    start_time = time.time()
    results = model(test_images, conf=0.25)
    batch_time = time.time() - start_time
    
    print(f"Traitement séquentiel: {sequential_time:.3f}s")
    print(f"Traitement batch: {batch_time:.3f}s")
    print(f"Accélération batch: {sequential_time/batch_time:.1f}x")
    
    # Vérifier que le batch est plus rapide
    assert batch_time < sequential_time, "Traitement batch devrait être plus rapide"

@pytest.mark.slow
def test_model_size():
    """Test de la taille du modèle"""
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    # Taille du modèle PyTorch
    pytorch_size = model_path.stat().st_size / 1024 / 1024  # MB
    
    # Taille du modèle ONNX (si disponible)
    onnx_path = Path("models/model.onnx")
    if onnx_path.exists():
        onnx_size = onnx_path.stat().st_size / 1024 / 1024  # MB
    else:
        onnx_size = None
    
    print(f"Taille modèle PyTorch: {pytorch_size:.1f} MB")
    if onnx_size:
        print(f"Taille modèle ONNX: {onnx_size:.1f} MB")
        print(f"Compression: {pytorch_size/onnx_size:.1f}x")
    
    # Vérifier que la taille est raisonnable
    assert pytorch_size < 200, f"Modèle trop volumineux: {pytorch_size:.1f} MB"
    if onnx_size:
        assert onnx_size < 100, f"Modèle ONNX trop volumineux: {onnx_size:.1f} MB"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
