#!/usr/bin/env python3
"""
Tests unitaires de base pour EvaDentalAI
"""

import pytest
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test que tous les modules peuvent être importés"""
    import ultralytics
    import torch
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import fastapi
    import uvicorn
    import yaml
    
    assert True  # Si on arrive ici, tous les imports ont réussi

def test_yolo_model_loading():
    """Test du chargement d'un modèle YOLO"""
    from ultralytics import YOLO
    
    # Test avec un modèle pré-entraîné
    model = YOLO('yolov8n.pt')
    assert model is not None

def test_dataset_generator():
    """Test du générateur de dataset"""
    from scripts.prepare_dataset import DentalDatasetGenerator
    
    generator = DentalDatasetGenerator()
    assert generator is not None
    assert len(generator.classes) == 5
    assert generator.class_names[0] == "tooth"

def test_predictor_initialization():
    """Test de l'initialisation du prédicteur"""
    from scripts.predict import DentalPredictor
    
    # Test avec un modèle factice
    predictor = DentalPredictor("yolov8n.pt")
    assert predictor is not None
    assert len(predictor.class_names) == 5

def test_config_file():
    """Test que le fichier de configuration existe et est valide"""
    import yaml
    
    config_path = Path("config/data.yaml")
    assert config_path.exists()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'names' in config
    assert 'nc' in config
    assert config['nc'] == 5
    assert len(config['names']) == 5

def test_file_structure():
    """Test que la structure des fichiers est correcte"""
    required_files = [
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
        "config/data.yaml",
        "scripts/prepare_dataset.py",
        "scripts/train_model.py",
        "scripts/predict.py",
        "scripts/export_model.py",
        "api/main.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml"
    ]
    
    for file_path in required_files:
        assert Path(file_path).exists(), f"Fichier manquant: {file_path}"

@pytest.mark.slow
def test_dataset_generation():
    """Test de génération d'un petit dataset"""
    import subprocess
    import sys
    
    # Générer un dataset très petit
    result = subprocess.run([
        sys.executable, "scripts/prepare_dataset.py", 
        "--num-images", "5"
    ], capture_output=True, text=True, timeout=60)
    
    assert result.returncode == 0, f"Erreur génération dataset: {result.stderr}"
    
    # Vérifier que les fichiers ont été créés
    data_dir = Path("data/processed")
    assert data_dir.exists()
    
    train_images = list((data_dir / "train/images").glob("*.jpg"))
    assert len(train_images) > 0

@pytest.mark.gpu
def test_cuda_availability():
    """Test de la disponibilité CUDA"""
    import torch
    
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
        assert torch.cuda.get_device_name(0) is not None
    else:
        pytest.skip("CUDA non disponible")

def test_api_import():
    """Test que l'API peut être importée"""
    sys.path.append(str(Path("api").absolute()))
    import main
    
    assert main.app is not None
    assert hasattr(main, 'predictor')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
