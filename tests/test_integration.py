#!/usr/bin/env python3
"""
Tests d'intégration pour EvaDentalAI
"""

import pytest
import sys
import subprocess
import time
import requests
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline():
    """Test du pipeline complet: dataset -> entraînement -> prédiction"""
    
    # 1. Génération du dataset
    result = subprocess.run([
        sys.executable, "scripts/prepare_dataset.py", 
        "--num-images", "10"
    ], capture_output=True, text=True, timeout=120)
    
    assert result.returncode == 0, f"Erreur génération dataset: {result.stderr}"
    
    # 2. Entraînement rapide
    result = subprocess.run([
        sys.executable, "scripts/train_model.py",
        "--config", "config/data.yaml",
        "--epochs", "3",
        "--batch-size", "4",
        "--device", "cpu"
    ], capture_output=True, text=True, timeout=300)
    
    assert result.returncode == 0, f"Erreur entraînement: {result.stderr}"
    
    # 3. Vérifier que le modèle a été créé
    model_path = Path("models/best.pt")
    assert model_path.exists(), "Modèle non créé"
    
    # 4. Test de prédiction
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    assert len(test_images) > 0, "Aucune image de test"
    
    result = subprocess.run([
        sys.executable, "scripts/predict.py",
        "--model", "models/best.pt",
        "--image", str(test_images[0])
    ], capture_output=True, text=True, timeout=60)
    
    assert result.returncode == 0, f"Erreur prédiction: {result.stderr}"

@pytest.mark.api
@pytest.mark.slow
def test_api_integration():
    """Test d'intégration de l'API"""
    
    # Vérifier qu'un modèle existe
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé, exécutez d'abord test_full_pipeline")
    
    # Lancer l'API en arrière-plan
    api_process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8001"  # Port différent pour éviter les conflits
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Test de santé
        response = requests.get("http://localhost:8001/health", timeout=5)
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Test de prédiction
        test_images = list(Path("data/processed/test/images").glob("*.jpg"))
        if test_images:
            with open(test_images[0], 'rb') as f:
                files = {'file': f}
                data = {'confidence': 0.25}
                
                response = requests.post(
                    "http://localhost:8001/predict",
                    files=files,
                    data=data,
                    timeout=30
                )
                
                assert response.status_code == 200
                result = response.json()
                assert "detections" in result
                assert "total_detections" in result
    
    finally:
        # Arrêter l'API
        api_process.terminate()
        api_process.wait()

@pytest.mark.slow
def test_export_integration():
    """Test d'intégration de l'export"""
    
    # Vérifier qu'un modèle existe
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé, exécutez d'abord test_full_pipeline")
    
    # Test d'export ONNX
    result = subprocess.run([
        sys.executable, "scripts/export_model.py",
        "--model", "models/best.pt",
        "--format", "onnx"
    ], capture_output=True, text=True, timeout=120)
    
    assert result.returncode == 0, f"Erreur export: {result.stderr}"
    
    # Vérifier que le fichier ONNX a été créé
    onnx_path = Path("models/model.onnx")
    assert onnx_path.exists(), "Fichier ONNX non créé"

@pytest.mark.slow
def test_batch_prediction():
    """Test de prédiction batch"""
    
    # Vérifier qu'un modèle existe
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé, exécutez d'abord test_full_pipeline")
    
    # Test de prédiction batch
    result = subprocess.run([
        sys.executable, "scripts/predict.py",
        "--model", "models/best.pt",
        "--batch", "data/processed/test/images",
        "--output", "output/test_batch"
    ], capture_output=True, text=True, timeout=120)
    
    assert result.returncode == 0, f"Erreur prédiction batch: {result.stderr}"
    
    # Vérifier que les résultats ont été créés
    output_dir = Path("output/test_batch")
    assert output_dir.exists(), "Répertoire de sortie non créé"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
