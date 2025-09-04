#!/usr/bin/env python3
"""
Tests de l'API EvaDentalAI
"""

import pytest
import sys
import subprocess
import time
import requests
import json
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="module")
def api_server():
    """Fixture pour lancer l'API en arrière-plan"""
    
    # Vérifier qu'un modèle existe
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    # Lancer l'API
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8002"  # Port différent pour éviter les conflits
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Attendre que l'API démarre
    time.sleep(10)
    
    yield "http://localhost:8002"
    
    # Arrêter l'API
    process.terminate()
    process.wait()

@pytest.mark.api
def test_health_endpoint(api_server):
    """Test de l'endpoint de santé"""
    
    response = requests.get(f"{api_server}/health", timeout=5)
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_path" in data
    assert "timestamp" in data

@pytest.mark.api
def test_root_endpoint(api_server):
    """Test de l'endpoint racine"""
    
    response = requests.get(f"{api_server}/", timeout=5)
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data

@pytest.mark.api
def test_model_info_endpoint(api_server):
    """Test de l'endpoint d'informations sur le modèle"""
    
    response = requests.get(f"{api_server}/model/info", timeout=5)
    assert response.status_code == 200
    
    data = response.json()
    assert "model_path" in data
    assert "model_exists" in data
    assert "confidence_threshold" in data
    assert "iou_threshold" in data
    assert "classes" in data

@pytest.mark.api
def test_classes_endpoint(api_server):
    """Test de l'endpoint des classes"""
    
    response = requests.get(f"{api_server}/classes", timeout=5)
    assert response.status_code == 200
    
    data = response.json()
    assert "classes" in data
    assert len(data["classes"]) == 5
    assert data["classes"][0] == "tooth"
    assert data["classes"][1] == "cavity"

@pytest.mark.api
def test_predict_endpoint(api_server):
    """Test de l'endpoint de prédiction"""
    
    # Créer une image de test
    import numpy as np
    from PIL import Image
    import io
    
    # Créer une image factice
    test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    
    # Convertir en bytes
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test de prédiction
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'confidence': 0.25, 'iou': 0.45}
    
    response = requests.post(
        f"{api_server}/predict",
        files=files,
        data=data,
        timeout=30
    )
    
    assert response.status_code == 200
    
    result = response.json()
    assert "success" in result
    assert "image_name" in result
    assert "inference_time" in result
    assert "total_detections" in result
    assert "detections" in result
    
    # Vérifier la structure des détections
    for detection in result["detections"]:
        assert "class_id" in detection
        assert "class_name" in detection
        assert "confidence" in detection
        assert "bbox" in detection
        
        bbox = detection["bbox"]
        assert "x1" in bbox
        assert "y1" in bbox
        assert "x2" in bbox
        assert "y2" in bbox
        assert "width" in bbox
        assert "height" in bbox

@pytest.mark.api
def test_predict_with_image_return(api_server):
    """Test de prédiction avec retour d'image"""
    
    import numpy as np
    from PIL import Image
    import io
    
    # Créer une image de test
    test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    
    # Convertir en bytes
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test de prédiction avec retour d'image
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'confidence': 0.25, 'iou': 0.45, 'return_image': True}
    
    response = requests.post(
        f"{api_server}/predict",
        files=files,
        data=data,
        timeout=30
    )
    
    assert response.status_code == 200
    
    result = response.json()
    assert "image_with_detections" in result
    
    # Vérifier que l'image est en base64
    import base64
    try:
        base64.b64decode(result["image_with_detections"])
    except Exception:
        pytest.fail("Image retournée n'est pas en base64 valide")

@pytest.mark.api
def test_predict_batch_endpoint(api_server):
    """Test de l'endpoint de prédiction batch"""
    
    import numpy as np
    from PIL import Image
    import io
    
    # Créer plusieurs images de test
    test_images = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        test_images.append(('file', (f'test{i}.jpg', img_bytes, 'image/jpeg')))
    
    # Test de prédiction batch
    data = {'confidence': 0.25, 'iou': 0.45}
    
    response = requests.post(
        f"{api_server}/predict/batch",
        files=test_images,
        data=data,
        timeout=60
    )
    
    assert response.status_code == 200
    
    result = response.json()
    assert "total_files" in result
    assert "successful" in result
    assert "failed" in result
    assert "results" in result
    
    assert result["total_files"] == 3
    assert result["successful"] == 3
    assert result["failed"] == 0

@pytest.mark.api
def test_predict_invalid_file(api_server):
    """Test de prédiction avec fichier invalide"""
    
    # Créer un fichier texte au lieu d'une image
    files = {'file': ('test.txt', b'This is not an image', 'text/plain')}
    data = {'confidence': 0.25}
    
    response = requests.post(
        f"{api_server}/predict",
        files=files,
        data=data,
        timeout=30
    )
    
    assert response.status_code == 400

@pytest.mark.api
def test_predict_large_file(api_server):
    """Test de prédiction avec fichier trop volumineux"""
    
    # Créer un fichier volumineux (simulé)
    large_data = b'x' * (11 * 1024 * 1024)  # 11MB
    
    files = {'file': ('large.jpg', large_data, 'image/jpeg')}
    data = {'confidence': 0.25}
    
    response = requests.post(
        f"{api_server}/predict",
        files=files,
        data=data,
        timeout=30
    )
    
    assert response.status_code == 413

@pytest.mark.api
def test_predict_invalid_parameters(api_server):
    """Test de prédiction avec paramètres invalides"""
    
    import numpy as np
    from PIL import Image
    import io
    
    # Créer une image de test
    test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    
    # Convertir en bytes
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test avec confidence invalide
    files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
    data = {'confidence': 1.5}  # Invalide (> 1.0)
    
    response = requests.post(
        f"{api_server}/predict",
        files=files,
        data=data,
        timeout=30
    )
    
    # L'API devrait accepter et ajuster automatiquement
    assert response.status_code == 200

@pytest.mark.api
def test_reload_model_endpoint(api_server):
    """Test de l'endpoint de rechargement du modèle"""
    
    response = requests.post(f"{api_server}/reload-model", timeout=30)
    assert response.status_code == 200
    
    data = response.json()
    assert "success" in data
    assert "message" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "api"])
