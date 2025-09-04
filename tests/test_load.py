#!/usr/bin/env python3
"""
Tests de charge pour EvaDentalAI
"""

import pytest
import sys
import time
import threading
import requests
import subprocess
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

@pytest.mark.slow
def test_api_load():
    """Test de charge de l'API"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8007"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Fonction de requête pour les threads
        def make_request(thread_id, results_list):
            try:
                response = requests.get("http://localhost:8007/health", timeout=5)
                results_list.append({
                    'thread_id': thread_id,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'success': response.status_code == 200
                })
            except Exception as e:
                results_list.append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'success': False
                })
        
        # Lancer de nombreuses requêtes concurrentes
        num_threads = 20
        threads = []
        results = []
        
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=make_request, args=(i, results))
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyser les résultats
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = num_threads - successful_requests
        
        if successful_requests > 0:
            response_times = [r['response_time'] for r in results if r['success']]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0
            max_response_time = 0
        
        print(f"Test de charge terminé:")
        print(f"Requêtes réussies: {successful_requests}/{num_threads}")
        print(f"Requêtes échouées: {failed_requests}")
        print(f"Temps total: {total_time:.2f}s")
        print(f"Temps de réponse moyen: {avg_response_time:.3f}s")
        print(f"Temps de réponse max: {max_response_time:.3f}s")
        
        # Vérifier que la plupart des requêtes ont réussi
        success_rate = successful_requests / num_threads
        assert success_rate > 0.8, f"Taux de succès trop faible: {success_rate:.2%}"
        
        # Vérifier que les temps de réponse sont raisonnables
        assert avg_response_time < 2.0, f"Temps de réponse moyen trop élevé: {avg_response_time:.3f}s"
        assert max_response_time < 10.0, f"Temps de réponse max trop élevé: {max_response_time:.3f}s"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.slow
def test_api_prediction_load():
    """Test de charge de prédiction de l'API"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8008"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Créer une image de test
        import numpy as np
        from PIL import Image
        import io
        
        test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Fonction de prédiction pour les threads
        def make_prediction(thread_id, results_list):
            try:
                img_bytes.seek(0)  # Remettre le pointeur au début
                files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
                data = {'confidence': 0.25}
                
                start_time = time.time()
                response = requests.post(
                    "http://localhost:8008/predict",
                    files=files,
                    data=data,
                    timeout=30
                )
                response_time = time.time() - start_time
                
                results_list.append({
                    'thread_id': thread_id,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'success': response.status_code == 200
                })
            except Exception as e:
                results_list.append({
                    'thread_id': thread_id,
                    'error': str(e),
                    'success': False
                })
        
        # Lancer de nombreuses prédictions concurrentes
        num_threads = 10  # Moins de threads pour les prédictions (plus lourdes)
        threads = []
        results = []
        
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=make_prediction, args=(i, results))
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyser les résultats
        successful_requests = sum(1 for r in results if r['success'])
        failed_requests = num_threads - successful_requests
        
        if successful_requests > 0:
            response_times = [r['response_time'] for r in results if r['success']]
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = 0
            max_response_time = 0
        
        print(f"Test de charge de prédiction terminé:")
        print(f"Prédictions réussies: {successful_requests}/{num_threads}")
        print(f"Prédictions échouées: {failed_requests}")
        print(f"Temps total: {total_time:.2f}s")
        print(f"Temps de réponse moyen: {avg_response_time:.3f}s")
        print(f"Temps de réponse max: {max_response_time:.3f}s")
        
        # Vérifier que la plupart des prédictions ont réussi
        success_rate = successful_requests / num_threads
        assert success_rate > 0.7, f"Taux de succès trop faible: {success_rate:.2%}"
        
        # Vérifier que les temps de réponse sont raisonnables
        assert avg_response_time < 5.0, f"Temps de réponse moyen trop élevé: {avg_response_time:.3f}s"
        assert max_response_time < 20.0, f"Temps de réponse max trop élevé: {max_response_time:.3f}s"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.slow
def test_sustained_load():
    """Test de charge soutenue"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8009"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Charge soutenue pendant 2 minutes
        duration = 120  # 2 minutes
        start_time = time.time()
        request_count = 0
        successful_requests = 0
        response_times = []
        
        while time.time() - start_time < duration:
            try:
                request_start = time.time()
                response = requests.get("http://localhost:8009/health", timeout=5)
                request_time = time.time() - request_start
                
                request_count += 1
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append(request_time)
                
                # Petite pause entre les requêtes
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Erreur de requête: {e}")
                request_count += 1
        
        total_time = time.time() - start_time
        
        # Analyser les résultats
        success_rate = successful_requests / request_count if request_count > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        print(f"Test de charge soutenue terminé:")
        print(f"Durée: {total_time:.1f}s")
        print(f"Requêtes totales: {request_count}")
        print(f"Requêtes réussies: {successful_requests}")
        print(f"Taux de succès: {success_rate:.2%}")
        print(f"Temps de réponse moyen: {avg_response_time:.3f}s")
        print(f"Temps de réponse max: {max_response_time:.3f}s")
        
        # Vérifier que le système a géré la charge soutenue
        assert request_count > 100, f"Trop peu de requêtes: {request_count}"
        assert success_rate > 0.9, f"Taux de succès trop faible: {success_rate:.2%}"
        assert avg_response_time < 1.0, f"Temps de réponse moyen trop élevé: {avg_response_time:.3f}s"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.slow
def test_memory_under_load():
    """Test de mémoire sous charge"""
    
    import psutil
    import os
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8010"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Obtenir le PID du processus API
        api_process = psutil.Process(process.pid)
        
        # Mesurer la mémoire initiale
        initial_memory = api_process.memory_info().rss / (1024 * 1024)  # MB
        
        # Créer une image de test
        import numpy as np
        from PIL import Image
        import io
        
        test_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        test_image.save(img_bytes, format='JPEG')
        
        # Envoyer de nombreuses requêtes
        num_requests = 50
        for i in range(num_requests):
            img_bytes.seek(0)
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            data = {'confidence': 0.25}
            
            try:
                response = requests.post(
                    "http://localhost:8010/predict",
                    files=files,
                    data=data,
                    timeout=30
                )
            except Exception as e:
                print(f"Erreur de requête {i}: {e}")
            
            # Mesurer la mémoire toutes les 10 requêtes
            if (i + 1) % 10 == 0:
                current_memory = api_process.memory_info().rss / (1024 * 1024)  # MB
                memory_increase = current_memory - initial_memory
                
                print(f"Requête {i + 1}: Mémoire utilisée: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
                
                # Vérifier qu'il n'y a pas de fuite mémoire excessive
                assert memory_increase < 200, f"Fuite mémoire détectée: +{memory_increase:.1f}MB"
        
        # Mesurer la mémoire finale
        final_memory = api_process.memory_info().rss / (1024 * 1024)  # MB
        total_memory_increase = final_memory - initial_memory
        
        print(f"Mémoire initiale: {initial_memory:.1f}MB")
        print(f"Mémoire finale: {final_memory:.1f}MB")
        print(f"Augmentation totale: {total_memory_increase:.1f}MB")
        
        # Vérifier qu'il n'y a pas de fuite mémoire excessive
        assert total_memory_increase < 300, f"Fuite mémoire excessive: +{total_memory_increase:.1f}MB"
    
    finally:
        process.terminate()
        process.wait()

@pytest.mark.slow
def test_error_handling_under_load():
    """Test de gestion d'erreur sous charge"""
    
    # Lancer l'API en arrière-plan
    process = subprocess.Popen([
        sys.executable, "api/main.py",
        "--model", "models/best.pt",
        "--port", "8011"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Attendre que l'API démarre
        time.sleep(10)
        
        # Envoyer des requêtes avec des erreurs
        error_requests = [
            # Fichier invalide
            {'files': {'file': ('test.txt', b'not an image', 'text/plain')}},
            
            # Fichier trop volumineux
            {'files': {'file': ('large.jpg', b'x' * (11 * 1024 * 1024), 'image/jpeg')}},
            
            # Paramètres invalides
            {'files': {'file': ('test.jpg', b'fake_image', 'image/jpeg')}, 'data': {'confidence': 1.5}},
            
            # Requête malformée
            {'files': {}, 'data': {}},
        ]
        
        for i, request_data in enumerate(error_requests):
            try:
                response = requests.post(
                    "http://localhost:8011/predict",
                    **request_data,
                    timeout=10
                )
                
                print(f"Requête d'erreur {i + 1}: Status {response.status_code}")
                
                # Vérifier que l'erreur est gérée gracieusement
                assert response.status_code in [200, 400, 413, 422], f"Status inattendu: {response.status_code}"
                
            except Exception as e:
                print(f"Erreur de requête {i + 1}: {e}")
                # Vérifier que l'erreur est gérée gracieusement
                assert False, f"Erreur non gérée: {e}"
        
        # Vérifier que l'API est toujours fonctionnelle après les erreurs
        response = requests.get("http://localhost:8011/health", timeout=5)
        assert response.status_code == 200, "API non fonctionnelle après les erreurs"
    
    finally:
        process.terminate()
        process.wait()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
