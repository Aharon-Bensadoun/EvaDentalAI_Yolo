#!/usr/bin/env python3
"""
Tests de sécurité pour EvaDentalAI
"""

import pytest
import sys
import subprocess
import requests
import json
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.api
def test_api_security_headers():
    """Test des en-têtes de sécurité de l'API"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8003"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        import time
        time.sleep(10)
        
        # Test des en-têtes de sécurité
        response = requests.get("http://localhost:8003/", timeout=5)
        
        # Vérifier les en-têtes de sécurité
        security_headers = [
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection'
        ]
        
        for header in security_headers:
            assert header in response.headers, f"En-tête de sécurité manquant: {header}"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.api
def test_api_input_validation():
    """Test de validation des entrées de l'API"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8004"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        import time
        time.sleep(10)
        
        # Test avec des paramètres invalides
        invalid_tests = [
            # Confidence invalide
            {'confidence': -0.1},
            {'confidence': 1.1},
            {'confidence': 'invalid'},
            
            # IoU invalide
            {'iou': -0.1},
            {'iou': 1.1},
            {'iou': 'invalid'},
        ]
        
        for test_data in invalid_tests:
            response = requests.post(
                "http://localhost:8004/predict",
                files={'file': ('test.jpg', b'fake_image_data', 'image/jpeg')},
                data=test_data,
                timeout=30
            )
            
            # L'API devrait gérer gracieusement les entrées invalides
            assert response.status_code in [200, 400, 422], f"Code de statut inattendu: {response.status_code}"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.api
def test_api_file_upload_security():
    """Test de sécurité des uploads de fichiers"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8005"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        import time
        time.sleep(10)
        
        # Test avec des fichiers malveillants
        malicious_files = [
            # Script malveillant
            ('malicious.py', b'import os; os.system("rm -rf /")', 'text/plain'),
            
            # Fichier exécutable
            ('malicious.exe', b'MZ\x90\x00', 'application/octet-stream'),
            
            # Fichier avec extension trompeuse
            ('image.jpg.exe', b'MZ\x90\x00', 'image/jpeg'),
        ]
        
        for filename, content, content_type in malicious_files:
            response = requests.post(
                "http://localhost:8005/predict",
                files={'file': (filename, content, content_type)},
                data={'confidence': 0.25},
                timeout=30
            )
            
            # L'API devrait rejeter les fichiers non-image
            assert response.status_code == 400, f"Fichier malveillant accepté: {filename}"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.api
def test_api_rate_limiting():
    """Test de limitation du débit de l'API"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8006"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        import time
        time.sleep(10)
        
        # Envoyer de nombreuses requêtes rapidement
        responses = []
        for i in range(20):
            response = requests.get("http://localhost:8006/health", timeout=5)
            responses.append(response.status_code)
            time.sleep(0.1)  # Petite pause entre les requêtes
        
        # Vérifier que toutes les requêtes ont été traitées
        success_count = sum(1 for status in responses if status == 200)
        assert success_count > 0, "Aucune requête n'a réussi"
        
        # Note: L'API actuelle n'a pas de rate limiting, donc toutes les requêtes devraient réussir
        # Ce test peut être étendu si un rate limiting est ajouté
    
    finally:
        process.terminate()
        process.wait()

def test_model_file_security():
    """Test de sécurité des fichiers de modèle"""
    
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("Modèle non trouvé")
    
    # Vérifier que le fichier de modèle n'est pas exécutable
    import stat
    file_stat = model_path.stat()
    file_mode = stat.filemode(file_stat.st_mode)
    
    # Le fichier ne devrait pas avoir les permissions d'exécution
    assert 'x' not in file_mode, "Fichier de modèle ne devrait pas être exécutable"

def test_config_file_security():
    """Test de sécurité des fichiers de configuration"""
    
    config_path = Path("config/data.yaml")
    assert config_path.exists(), "Fichier de configuration manquant"
    
    # Vérifier que le fichier de configuration ne contient pas d'informations sensibles
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Vérifier qu'il n'y a pas de mots de passe ou de clés secrètes
    sensitive_patterns = [
        'password',
        'secret',
        'key',
        'token',
        'api_key',
        'private'
    ]
    
    for pattern in sensitive_patterns:
        assert pattern.lower() not in content.lower(), f"Information sensible trouvée: {pattern}"

def test_script_permissions():
    """Test des permissions des scripts"""
    
    import stat
    
    script_files = [
        "scripts/prepare_dataset.py",
        "scripts/train_model.py",
        "scripts/predict.py",
        "scripts/export_model.py",
        "api/main.py"
    ]
    
    for script_path in script_files:
        if Path(script_path).exists():
            file_stat = Path(script_path).stat()
            file_mode = stat.filemode(file_stat.st_mode)
            
            # Vérifier que les scripts ont les bonnes permissions
            assert 'r' in file_mode, f"Script non lisible: {script_path}"
            # Note: Sur Windows, les permissions peuvent être différentes

def test_dockerfile_security():
    """Test de sécurité du Dockerfile"""
    
    dockerfile_path = Path("docker/Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile manquant"
    
    with open(dockerfile_path, 'r') as f:
        content = f.read()
    
    # Vérifier que le Dockerfile utilise un utilisateur non-root
    assert 'USER app' in content, "Dockerfile devrait utiliser un utilisateur non-root"
    
    # Vérifier qu'il n'y a pas de commandes dangereuses
    dangerous_commands = [
        'rm -rf /',
        'chmod 777',
        'su -',
        'sudo'
    ]
    
    for command in dangerous_commands:
        assert command not in content, f"Commande dangereuse trouvée: {command}"

def test_dependencies_security():
    """Test de sécurité des dépendances"""
    
    requirements_path = Path("requirements.txt")
    assert requirements_path.exists(), "Fichier requirements.txt manquant"
    
    with open(requirements_path, 'r') as f:
        requirements = f.readlines()
    
    # Vérifier qu'il n'y a pas de dépendances suspectes
    suspicious_packages = [
        'requests[security]',  # Devrait être 'requests'
        'urllib3<1.26',       # Version obsolète
        'pyyaml<5.4',         # Version avec vulnérabilité
    ]
    
    for requirement in requirements:
        requirement = requirement.strip()
        if requirement and not requirement.startswith('#'):
            for suspicious in suspicious_packages:
                assert suspicious not in requirement, f"Dépendance suspecte: {requirement}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "api"])
