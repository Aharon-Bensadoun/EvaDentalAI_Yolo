#!/usr/bin/env python3
"""
Script Colab Ultra-Simple qui Fonctionne à 100%
Version corrégée avec tous les imports au bon endroit
"""

# Imports globaux en premier
import os
import sys
from pathlib import Path
import yaml
import random

def setup_environment():
    """Configuration de l'environnement"""
    print("🔧 Configuration de l'environnement...")
    
    current_dir = Path.cwd()
    print(f"📍 Répertoire: {current_dir}")
    
    # Créer les répertoires nécessaires
    for dir_name in ['data', 'models', 'runs']:
        dir_path = current_dir / dir_name
        dir_path.mkdir(exist_ok=True)
    
    return current_dir

def create_simple_dataset():
    """Crée un dataset dentaire simple et fonctionnel"""
    print("🦷 Création du dataset dentaire...")
    
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np
    except ImportError:
        print("❌ Erreur d'import PIL/numpy")
        return False
    
    # Structure du dataset
    output_dir = Path("data/dentex")
    splits = {'train': 50, 'val': 15, 'test': 10}
    
    for split, count in splits.items():
        print(f"📁 {split}: {count} images...")
        
        # Créer les répertoires
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            # Créer une image simple
            img = create_simple_xray()
            
            # Sauvegarder l'image
            img_path = output_dir / split / 'images' / f"{split}_{i:04d}.jpg"
            img.save(img_path, 'JPEG')
            
            # Créer des annotations simples
            annotations = create_simple_annotations()
            
            # Sauvegarder les labels
            label_path = output_dir / split / 'labels' / f"{split}_{i:04d}.txt"
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
    
    # Créer la configuration YOLO
    create_config(output_dir, splits)
    
    print("✅ Dataset créé avec succès!")
    return True

def create_simple_xray():
    """Crée une image de radiographie simple"""
    from PIL import Image, ImageDraw
    
    # Image simple en niveaux de gris
    img = Image.new('L', (640, 480), color=40)
    draw = ImageDraw.Draw(img)
    
    # Ajouter quelques formes simples (dents)
    for i in range(10):
        x = 50 + i * 50
        y = 200
        draw.rectangle([x, y, x+30, y+40], fill=120)
        
        # Dents du bas
        draw.rectangle([x, y+100, x+30, y+140], fill=120)
    
    # Convertir en RGB
    return img.convert('RGB')

def create_simple_annotations():
    """Crée des annotations simples"""
    annotations = []
    
    # Ajouter 1-3 objets aléatoires
    for _ in range(random.randint(1, 3)):
        class_id = random.randint(0, 2)  # 3 classes seulement
        x = random.uniform(0.2, 0.8)
        y = random.uniform(0.2, 0.8)
        w = random.uniform(0.05, 0.15)
        h = random.uniform(0.05, 0.15)
        
        annotations.append([class_id, x, y, w, h])
    
    return annotations

def create_config(output_dir, splits):
    """Crée la configuration YOLO"""
    abs_path = Path.cwd() / output_dir
    
    config = {
        'path': str(abs_path),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: "tooth",
            1: "cavity",
            2: "implant"
        },
        'nc': 3
    }
    
    config_path = abs_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"✅ Configuration: {config_path}")

def train_simple_model():
    """Entraîne le modèle avec la syntaxe la plus simple"""
    print("🏋️ Entraînement du modèle...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Vérifier le device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Device: {device}")
        
        # Charger le modèle le plus léger
        model = YOLO('yolov8n.pt')
        
        # Configuration minimale
        config_path = Path('data/dentex/data.yaml')
        
        if not config_path.exists():
            print("❌ Configuration non trouvée")
            return None
        
        print("🚀 Entraînement en cours...")
        
        # Entraînement avec paramètres minimaux
        results = model.train(
            data=str(config_path),
            epochs=10,      # Très court pour test
            batch=8,        # Petit batch
            imgsz=320,      # Image plus petite
            device=device,
            verbose=True
        )
        
        print("✅ Entraînement terminé!")
        return model, results
        
    except Exception as e:
        print(f"❌ Erreur d'entraînement: {e}")
        return None

def test_model(model):
    """Test simple du modèle"""
    print("🔍 Test du modèle...")
    
    try:
        import matplotlib.pyplot as plt
        
        # Trouver une image de test
        test_dir = Path('data/dentex/test/images')
        images = list(test_dir.glob('*.jpg'))
        
        if not images:
            print("❌ Aucune image de test")
            return
        
        # Tester sur la première image
        test_image = str(images[0])
        results = model(test_image)
        
        # Afficher les résultats
        for r in results:
            im_array = r.plot()
            plt.figure(figsize=(10, 6))
            plt.imshow(im_array)
            plt.axis('off')
            plt.title('Test EvaDentalAI')
            plt.show()
            
            if r.boxes is not None:
                print(f"🎯 {len(r.boxes)} détections trouvées")
            else:
                print("💡 Aucune détection")
                
    except Exception as e:
        print(f"❌ Erreur de test: {e}")

def run_simple_pipeline():
    """Pipeline complet simplifié"""
    print("🚀 EvaDentalAI - Version Simple")
    print("=" * 40)
    
    # 1. Setup
    setup_environment()
    
    # 2. Dataset
    if not create_simple_dataset():
        print("❌ Échec dataset")
        return False
    
    # 3. Training
    result = train_simple_model()
    if result is None:
        print("❌ Échec entraînement")
        return False
    
    model, training_results = result
    
    # 4. Test
    test_model(model)
    
    print("🎉 Pipeline terminé!")
    return True

if __name__ == "__main__":
    success = run_simple_pipeline()
    
    if success:
        print("\n✅ EvaDentalAI fonctionne!")
        print("💡 Modèle disponible dans: runs/detect/train/weights/best.pt")
    else:
        print("\n❌ Échec du pipeline")
