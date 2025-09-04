#!/usr/bin/env python3
"""
Exemples d'utilisation d'EvaDentalAI
Démontre les différentes façons d'utiliser le système
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.predict import DentalPredictor
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

def example_basic_prediction():
    """Exemple de prédiction basique"""
    print("🔍 Exemple 1: Prédiction basique")
    print("-" * 40)
    
    # Charger le modèle
    model_path = "models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("💡 Entraînez d'abord un modèle avec: python scripts/train_model.py")
        return
    
    # Initialiser le prédicteur
    predictor = DentalPredictor(model_path)
    
    # Trouver une image de test
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if not test_images:
        print("❌ Aucune image de test trouvée")
        print("💡 Générez d'abord un dataset avec: python scripts/prepare_dataset.py")
        return
    
    test_image = str(test_images[0])
    print(f"📸 Analyse de: {Path(test_image).name}")
    
    # Prédiction
    prediction = predictor.predict_image(test_image, conf_threshold=0.25)
    
    if prediction:
        print(f"✅ {prediction['total_detections']} détection(s) trouvée(s)")
        print(f"⏱️  Temps: {prediction['inference_time']:.3f}s")
        
        # Afficher les détections
        for i, detection in enumerate(prediction['detections'], 1):
            print(f"  {i}. {detection['class_name']}: {detection['confidence']:.3f}")
    
    print()

def example_batch_prediction():
    """Exemple de prédiction batch"""
    print("📦 Exemple 2: Prédiction batch")
    print("-" * 40)
    
    model_path = "models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    predictor = DentalPredictor(model_path)
    
    # Répertoire d'images de test
    test_dir = "data/processed/test/images"
    if not Path(test_dir).exists():
        print(f"❌ Répertoire non trouvé: {test_dir}")
        return
    
    print(f"📁 Analyse du répertoire: {test_dir}")
    
    # Prédiction batch
    results = predictor.batch_predict(test_dir, output_dir="output", conf_threshold=0.25)
    
    if results:
        total_detections = sum(r['total_detections'] for r in results)
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        
        print(f"✅ {len(results)} images analysées")
        print(f"🔍 Total détections: {total_detections}")
        print(f"⏱️  Temps moyen: {avg_time:.3f}s par image")
    
    print()

def example_custom_thresholds():
    """Exemple avec seuils personnalisés"""
    print("🎯 Exemple 3: Seuils personnalisés")
    print("-" * 40)
    
    model_path = "models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    predictor = DentalPredictor(model_path)
    
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if not test_images:
        print("❌ Aucune image de test trouvée")
        return
    
    test_image = str(test_images[0])
    
    # Test avec différents seuils
    thresholds = [0.1, 0.25, 0.5, 0.75]
    
    print(f"📸 Image: {Path(test_image).name}")
    print("Seuils de confiance:")
    
    for conf in thresholds:
        prediction = predictor.predict_image(test_image, conf_threshold=conf)
        if prediction:
            print(f"  Conf={conf:.2f}: {prediction['total_detections']} détections")
    
    print()

def example_visualization():
    """Exemple de visualisation"""
    print("🎨 Exemple 4: Visualisation")
    print("-" * 40)
    
    model_path = "models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    predictor = DentalPredictor(model_path)
    
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if not test_images:
        print("❌ Aucune image de test trouvée")
        return
    
    test_image = str(test_images[0])
    
    # Prédiction
    prediction = predictor.predict_image(test_image)
    
    if prediction and prediction['detections']:
        # Visualisation
        vis_image = predictor.visualize_predictions(test_image, prediction)
        
        # Afficher avec matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f"Détections sur {Path(test_image).name}")
        plt.show()
        
        # Sauvegarder
        output_path = "output/visualization_example.jpg"
        Path("output").mkdir(exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print(f"💾 Visualisation sauvegardée: {output_path}")
    
    print()

def example_model_comparison():
    """Exemple de comparaison de modèles"""
    print("⚖️  Exemple 5: Comparaison de modèles")
    print("-" * 40)
    
    # Modèles à comparer
    models = [
        ("YOLOv8n", "yolov8n.pt"),
        ("YOLOv8s", "yolov8s.pt"),
        ("Entraîné", "models/best.pt")
    ]
    
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if not test_images:
        print("❌ Aucune image de test trouvée")
        return
    
    test_image = str(test_images[0])
    print(f"📸 Image de test: {Path(test_image).name}")
    
    results = []
    
    for model_name, model_path in models:
        if not Path(model_path).exists():
            print(f"⚠️  Modèle non trouvé: {model_path}")
            continue
        
        try:
            # Charger le modèle
            model = YOLO(model_path)
            
            # Prédiction
            import time
            start_time = time.time()
            results_yolo = model(test_image, conf=0.25)
            inference_time = time.time() - start_time
            
            # Compter les détections
            num_detections = 0
            if results_yolo[0].boxes is not None:
                num_detections = len(results_yolo[0].boxes)
            
            results.append({
                'model': model_name,
                'detections': num_detections,
                'time': inference_time
            })
            
            print(f"  {model_name}: {num_detections} détections, {inference_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ Erreur avec {model_name}: {e}")
    
    # Afficher le résumé
    if results:
        print("\n📊 Résumé:")
        for result in results:
            print(f"  {result['model']}: {result['detections']} détections, {result['time']:.3f}s")
    
    print()

def example_api_usage():
    """Exemple d'utilisation de l'API"""
    print("🌐 Exemple 6: Utilisation de l'API")
    print("-" * 40)
    
    import requests
    import json
    
    # URL de l'API (supposée en cours d'exécution)
    api_url = "http://localhost:8000"
    
    try:
        # Test de santé
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API en ligne: {health['status']}")
            print(f"🤖 Modèle chargé: {health['model_loaded']}")
        else:
            print(f"❌ API non disponible: {response.status_code}")
            return
    except requests.exceptions.RequestException:
        print("❌ API non accessible")
        print("💡 Lancez l'API avec: python api/main.py --model models/best.pt")
        return
    
    # Test de prédiction
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if not test_images:
        print("❌ Aucune image de test trouvée")
        return
    
    test_image = test_images[0]
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            data = {'confidence': 0.25, 'iou': 0.45}
            
            response = requests.post(
                f"{api_url}/predict",
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prédiction réussie")
                print(f"🔍 {result['total_detections']} détection(s)")
                print(f"⏱️  Temps: {result['inference_time']:.3f}s")
                
                for detection in result['detections']:
                    print(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
            else:
                print(f"❌ Erreur API: {response.status_code}")
                print(response.text)
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print()

def example_export_models():
    """Exemple d'export de modèles"""
    print("📤 Exemple 7: Export de modèles")
    print("-" * 40)
    
    model_path = "models/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        return
    
    try:
        from scripts.export_model import ModelExporter
        
        exporter = ModelExporter(model_path)
        
        # Export ONNX
        onnx_path = exporter.export_onnx("models")
        if onnx_path:
            print(f"✅ ONNX exporté: {onnx_path}")
        
        # Export optimisé
        opt_path = exporter.optimize_for_inference("models")
        if opt_path:
            print(f"✅ Modèle optimisé: {opt_path}")
        
    except Exception as e:
        print(f"❌ Erreur export: {e}")
    
    print()

def main():
    """Fonction principale"""
    print("🦷 Exemples d'utilisation EvaDentalAI")
    print("=" * 50)
    
    # Créer le répertoire de sortie
    Path("output").mkdir(exist_ok=True)
    
    examples = [
        ("Prédiction basique", example_basic_prediction),
        ("Prédiction batch", example_batch_prediction),
        ("Seuils personnalisés", example_custom_thresholds),
        ("Visualisation", example_visualization),
        ("Comparaison modèles", example_model_comparison),
        ("Utilisation API", example_api_usage),
        ("Export modèles", example_export_models)
    ]
    
    for example_name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n⏹️  Arrêt demandé par l'utilisateur")
            break
        except Exception as e:
            print(f"❌ Erreur dans {example_name}: {e}")
            continue
    
    print("🎉 Exemples terminés!")
    print("📁 Vérifiez le dossier 'output/' pour les résultats")

if __name__ == "__main__":
    main()
