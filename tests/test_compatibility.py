#!/usr/bin/env python3
"""
Tests de compatibilité pour EvaDentalAI
"""

import pytest
import sys
import platform
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

def test_python_version():
    """Test de la version de Python"""
    
    version = sys.version_info
    assert version.major == 3, f"Python 3 requis, version {version.major}.{version.minor} trouvée"
    assert version.minor >= 8, f"Python 3.8+ requis, version {version.major}.{version.minor} trouvée"

def test_platform_compatibility():
    """Test de compatibilité avec la plateforme"""
    
    system = platform.system()
    assert system in ['Windows', 'Linux', 'Darwin'], f"Plateforme non supportée: {system}"

def test_import_compatibility():
    """Test de compatibilité des imports"""
    
    # Test des imports principaux
    try:
        import ultralytics
        import torch
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        import fastapi
        import uvicorn
        import yaml
    except ImportError as e:
        pytest.fail(f"Import échoué: {e}")

def test_torch_compatibility():
    """Test de compatibilité PyTorch"""
    
    import torch
    
    # Vérifier la version de PyTorch
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 1, f"PyTorch 1.0+ requis, version {version} trouvée"
    assert minor >= 9, f"PyTorch 1.9+ recommandé, version {version} trouvée"
    
    # Vérifier la compatibilité CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # Vérifier que la version CUDA est supportée
        assert cuda_version is not None, "Version CUDA non détectée"

def test_opencv_compatibility():
    """Test de compatibilité OpenCV"""
    
    import cv2
    
    # Vérifier la version d'OpenCV
    version = cv2.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 4, f"OpenCV 4.0+ requis, version {version} trouvée"
    assert minor >= 5, f"OpenCV 4.5+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(cv2, 'imread'), "cv2.imread non disponible"
    assert hasattr(cv2, 'imwrite'), "cv2.imwrite non disponible"
    assert hasattr(cv2, 'rectangle'), "cv2.rectangle non disponible"

def test_numpy_compatibility():
    """Test de compatibilité NumPy"""
    
    import numpy as np
    
    # Vérifier la version de NumPy
    version = np.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 1, f"NumPy 1.0+ requis, version {version} trouvée"
    assert minor >= 20, f"NumPy 1.20+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(np, 'array'), "np.array non disponible"
    assert hasattr(np, 'random'), "np.random non disponible"
    assert hasattr(np, 'mean'), "np.mean non disponible"

def test_pil_compatibility():
    """Test de compatibilité PIL/Pillow"""
    
    from PIL import Image
    
    # Vérifier la version de Pillow
    version = Image.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 8, f"Pillow 8.0+ requis, version {version} trouvée"
    assert minor >= 0, f"Pillow 8.0+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(Image, 'open'), "Image.open non disponible"
    assert hasattr(Image, 'fromarray'), "Image.fromarray non disponible"
    assert hasattr(Image, 'new'), "Image.new non disponible"

def test_fastapi_compatibility():
    """Test de compatibilité FastAPI"""
    
    import fastapi
    
    # Vérifier la version de FastAPI
    version = fastapi.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 0, f"FastAPI 0.100+ requis, version {version} trouvée"
    assert minor >= 100, f"FastAPI 0.100+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(fastapi, 'FastAPI'), "FastAPI non disponible"
    assert hasattr(fastapi, 'File'), "File non disponible"
    assert hasattr(fastapi, 'UploadFile'), "UploadFile non disponible"

def test_ultralytics_compatibility():
    """Test de compatibilité Ultralytics"""
    
    import ultralytics
    
    # Vérifier la version d'Ultralytics
    version = ultralytics.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 8, f"Ultralytics 8.0+ requis, version {version} trouvée"
    assert minor >= 0, f"Ultralytics 8.0+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(ultralytics, 'YOLO'), "YOLO non disponible"

def test_yaml_compatibility():
    """Test de compatibilité PyYAML"""
    
    import yaml
    
    # Vérifier la version de PyYAML
    version = yaml.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 5, f"PyYAML 5.0+ requis, version {version} trouvée"
    assert minor >= 0, f"PyYAML 5.0+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(yaml, 'safe_load'), "yaml.safe_load non disponible"
    assert hasattr(yaml, 'dump'), "yaml.dump non disponible"

def test_matplotlib_compatibility():
    """Test de compatibilité Matplotlib"""
    
    import matplotlib
    
    # Vérifier la version de Matplotlib
    version = matplotlib.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 3, f"Matplotlib 3.0+ requis, version {version} trouvée"
    assert minor >= 5, f"Matplotlib 3.5+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(matplotlib, 'pyplot'), "matplotlib.pyplot non disponible"
    assert hasattr(matplotlib.pyplot, 'plot'), "pyplot.plot non disponible"
    assert hasattr(matplotlib.pyplot, 'show'), "pyplot.show non disponible"

def test_seaborn_compatibility():
    """Test de compatibilité Seaborn"""
    
    import seaborn as sns
    
    # Vérifier la version de Seaborn
    version = sns.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 0, f"Seaborn 0.11+ requis, version {version} trouvée"
    assert minor >= 11, f"Seaborn 0.11+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(sns, 'set_style'), "sns.set_style non disponible"
    assert hasattr(sns, 'color_palette'), "sns.color_palette non disponible"

def test_pandas_compatibility():
    """Test de compatibilité Pandas"""
    
    import pandas as pd
    
    # Vérifier la version de Pandas
    version = pd.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 1, f"Pandas 1.0+ requis, version {version} trouvée"
    assert minor >= 3, f"Pandas 1.3+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(pd, 'DataFrame'), "pd.DataFrame non disponible"
    assert hasattr(pd, 'read_csv'), "pd.read_csv non disponible"

def test_requests_compatibility():
    """Test de compatibilité Requests"""
    
    import requests
    
    # Vérifier la version de Requests
    version = requests.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 2, f"Requests 2.0+ requis, version {version} trouvée"
    assert minor >= 25, f"Requests 2.25+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(requests, 'get'), "requests.get non disponible"
    assert hasattr(requests, 'post'), "requests.post non disponible"

def test_tqdm_compatibility():
    """Test de compatibilité TQDM"""
    
    import tqdm
    
    # Vérifier la version de TQDM
    version = tqdm.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 4, f"TQDM 4.0+ requis, version {version} trouvée"
    assert minor >= 60, f"TQDM 4.60+ recommandé, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(tqdm, 'tqdm'), "tqdm.tqdm non disponible"

def test_uvicorn_compatibility():
    """Test de compatibilité Uvicorn"""
    
    import uvicorn
    
    # Vérifier la version d'Uvicorn
    version = uvicorn.__version__
    major, minor = map(int, version.split('.')[:2])
    
    assert major >= 0, f"Uvicorn 0.20+ requis, version {version} trouvée"
    assert minor >= 20, f"Uvicorn 0.20+ requis, version {version} trouvée"
    
    # Vérifier les fonctionnalités essentielles
    assert hasattr(uvicorn, 'run'), "uvicorn.run non disponible"

def test_cross_platform_paths():
    """Test de compatibilité des chemins cross-platform"""
    
    from pathlib import Path
    
    # Test des chemins Windows
    if platform.system() == 'Windows':
        windows_path = Path("C:\\Users\\test\\file.txt")
        assert windows_path.exists() or not windows_path.exists()  # Peut exister ou non
        
        # Test des chemins avec backslashes
        relative_path = Path("data\\processed\\train\\images")
        assert str(relative_path) == "data\\processed\\train\\images"
    
    # Test des chemins Unix/Linux/macOS
    else:
        unix_path = Path("/home/user/file.txt")
        assert unix_path.exists() or not unix_path.exists()  # Peut exister ou non
        
        # Test des chemins avec slashes
        relative_path = Path("data/processed/train/images")
        assert str(relative_path) == "data/processed/train/images"

def test_encoding_compatibility():
    """Test de compatibilité d'encodage"""
    
    # Test d'encodage UTF-8
    test_string = "🦷 EvaDentalAI - Détection d'anomalies dentaires"
    
    try:
        encoded = test_string.encode('utf-8')
        decoded = encoded.decode('utf-8')
        assert decoded == test_string, "Problème d'encodage UTF-8"
    except UnicodeError:
        pytest.fail("Erreur d'encodage UTF-8")

def test_memory_compatibility():
    """Test de compatibilité mémoire"""
    
    import psutil
    import os
    
    # Vérifier la mémoire disponible
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Vérifier qu'il y a au moins 2GB de mémoire disponible
    assert available_gb >= 2, f"Mémoire insuffisante: {available_gb:.1f}GB disponible"

def test_disk_space_compatibility():
    """Test de compatibilité espace disque"""
    
    import shutil
    
    # Vérifier l'espace disque disponible
    free_space = shutil.disk_usage('.').free
    free_gb = free_space / (1024**3)
    
    # Vérifier qu'il y a au moins 5GB d'espace libre
    assert free_gb >= 5, f"Espace disque insuffisant: {free_gb:.1f}GB disponible"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
