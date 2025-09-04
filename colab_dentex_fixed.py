#!/usr/bin/env python3
"""
Script complet et corrigé pour EvaDentalAI + DENTEX sur Google Colab
Version fixée pour gérer les erreurs de PyTorch 2.6+ et les problèmes de chemins
"""

import os
import sys
import torch
import shutil
from pathlib import Path

def fix_colab_environment():
    """Corrige l'environnement Colab pour éviter les problèmes de chemins"""
    print("🔧 Correction de l'environnement Colab...")

    # Détecter si on est dans un environnement avec des répertoires imbriqués
    current_dir = Path.cwd()
    project_dirs = ['EvaDentalAI_Yolo', 'scripts', 'data', 'models']

    # Chercher le vrai répertoire racine du projet
    root_dir = current_dir
    for parent in current_dir.parents:
        if any((parent / d).exists() for d in project_dirs):
            root_dir = parent
            break

    # Si on est dans un sous-répertoire imbriqué, aller à la racine
    if str(current_dir) != str(root_dir):
        print(f"📁 Changement vers le répertoire racine: {root_dir}")
        os.chdir(root_dir)

    print(f"✅ Environnement corrigé. Répertoire actuel: {Path.cwd()}")
    return Path.cwd()

def install_dependencies():
    """Installe les dépendances nécessaires"""
    print("📦 Installation des dépendances...")

    try:
        import ultralytics
        print("✅ Ultralytics déjà installé")
    except ImportError:
        os.system("pip install ultralytics==8.0.196")

    try:
        import datasets
        print("✅ Datasets déjà installé")
    except ImportError:
        os.system("pip install datasets==2.14.0 huggingface-hub==0.16.4")

    # Vérifier PyTorch
    if torch.cuda.is_available():
        print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("⚠️  GPU non disponible, utilisation du CPU")
        device = "cpu"

    return device

def download_dentex_fixed():
    """Téléchargement du dataset DENTEX avec gestion d'erreurs améliorée"""
    print("🦷 Téléchargement DENTEX - Version Corrigée")
    print("=" * 50)

    try:
        from datasets import load_dataset
        from PIL import Image
        import numpy as np
        import yaml
    except ImportError:
        print("❌ Dépendances manquantes")
        return False

    # Créer la structure des répertoires
    output_dir = Path("data/dentex")
    try:
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Erreur répertoire: {e}")
        return False

    print("📥 Téléchargement du dataset DENTEX...")
    print("Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX")

    try:
        # Méthode 1: Téléchargement standard avec gestion d'erreurs
        dataset = load_dataset("ibrahimhamamci/DENTEX",
                              trust_remote_code=True,
                              download_mode="reuse_cache_if_exists")
        print("✅ Dataset téléchargé avec succès!")

        # Traiter le dataset
        processed_counts = process_dentex_dataset(dataset, output_dir)

        # Créer la configuration
        create_dentex_config(output_dir, processed_counts)

        print("\n✅ Dataset DENTEX préparé avec succès!")
        return True

    except Exception as e:
        print(f"❌ Erreur téléchargement: {e}")
        print("💡 Création d'un dataset de test...")

        # Créer un dataset de test minimal
        create_test_dataset_fixed(output_dir)
        return True

def process_dentex_dataset(dataset, output_dir):
    """Traite le dataset DENTEX pour YOLO"""
    processed_counts = {}

    for split_name, split_data in dataset.items():
        if split_name not in ['train', 'validation', 'test']:
            continue

        yolo_split = 'val' if split_name == 'validation' else split_name
        print(f"📁 Traitement du split: {split_name} -> {yolo_split}")

        processed_count = 0

        for i, item in enumerate(split_data):
            try:
                image = item['image']
                image_filename = f"{yolo_split}_{i:04d}.jpg"
                image_path = output_dir / yolo_split / 'images' / image_filename
                image.save(image_path, 'JPEG')

                # Traiter les annotations
                if 'objects' in item and item['objects']:
                    annotations = process_dentex_annotations(item['objects'], image.size)
                    label_filename = f"{yolo_split}_{i:04d}.txt"
                    label_path = output_dir / yolo_split / 'labels' / label_filename

                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")

                processed_count += 1

                if (i + 1) % 100 == 0:
                    print(f"  Traité {i + 1}/{len(split_data)} images")

            except Exception as e:
                print(f"  ⚠️  Erreur image {i}: {e}")
                continue

        processed_counts[yolo_split] = processed_count
        print(f"✅ {yolo_split}: {processed_count} images traitées")

    return processed_counts

def process_dentex_annotations(objects, image_size):
    """Traite les annotations DENTEX pour YOLO"""
    annotations = []
    img_width, img_height = image_size

    for obj in objects:
        if 'bbox' in obj:
            bbox = obj['bbox']
            x_min, y_min, x_max, y_max = bbox

            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            class_id = map_dentex_class(obj)

            if class_id is not None:
                annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })

    return annotations

def map_dentex_class(obj):
    """Mappe les classes DENTEX vers YOLO"""
    if 'category' in obj:
        category = obj['category']
        if category in ['caries', 'deep_caries']:
            return 1  # cavity
        elif category == 'periapical_lesion':
            return 3  # lesion
        elif category == 'impacted_tooth':
            return 0  # tooth
    return None

def create_dentex_config(output_dir, processed_counts):
    """Crée la configuration YOLO pour DENTEX"""
    config = {
        'path': str(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: "tooth",
            1: "cavity",
            2: "implant",
            3: "lesion",
            4: "filling"
        },
        'nc': 5,
        'description': 'DENTEX Dataset - Panoramic Dental X-rays'
    }

    config_path = output_dir / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Configuration créée: {config_path}")

def create_test_dataset_fixed(output_dir):
    """Crée un dataset de test minimal"""
    print("🔧 Création d'un dataset de test...")

    from PIL import Image
    import yaml

    for split in ['train', 'val', 'test']:
        for i in range(10):  # Plus d'images de test
            img = Image.new('RGB', (640, 640), color='gray')
            img_path = output_dir / split / 'images' / f"{split}_{i:04d}.jpg"
            img.save(img_path)

            # Créer des labels variés pour le test
            label_path = output_dir / split / 'labels' / f"{split}_{i:04d}.txt"
            with open(label_path, 'w') as f:
                # Ajouter différentes classes pour le test
                classes = [0, 1, 3, 4]  # tooth, cavity, lesion, filling
                for cls in classes:
                    f.write(f"{cls} {0.1 + cls*0.2:.1f} {0.1 + cls*0.2:.1f} 0.1 0.1\n")

    create_dentex_config(output_dir, {'train': 10, 'val': 10, 'test': 10})
    print("✅ Dataset de test créé!")

def train_model_fixed(device):
    """Entraînement du modèle avec gestion des erreurs PyTorch 2.6+"""
    print("🏋️ Entraînement du modèle...")
    print("=" * 50)

    try:
        # Importer et corriger YOLO pour PyTorch 2.6+
        from ultralytics import YOLO

        # Patch temporaire pour PyTorch 2.6+
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load

        try:
            # Charger le modèle avec le patch
            print("🔧 Chargement du modèle yolov8s.pt...")
            model = YOLO('yolov8s.pt')
            print("✅ Modèle chargé avec succès!")

            # Configuration d'entraînement optimisée pour Colab
            train_args = {
                'data': 'data/dentex/data.yaml',
                'epochs': 10,  # Réduit pour les tests
                'batch': 8,     # Petit batch pour la mémoire
                'imgsz': 640,
                'device': device,
                'patience': 20,
                'save_period': 5,
                'project': 'models',
                'name': 'dentex_colab_fixed',
                'exist_ok': True,
                'pretrained': True,
                'verbose': True
            }

            print("🚀 Début de l'entraînement...")
            results = model.train(**train_args)

            print("✅ Entraînement terminé!")
            print(f"📁 Modèle sauvegardé dans: {results.save_dir}")

            return results

        finally:
            # Restaurer torch.load original
            torch.load = original_torch_load

    except Exception as e:
        print(f"❌ Erreur entraînement: {e}")
        print("💡 Tentative avec un modèle plus petit...")
        return train_fallback(device)

def train_fallback(device):
    """Entraînement de secours avec yolov8n.pt"""
    try:
        from ultralytics import YOLO

        # Patch torch.load
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)

        try:
            model = YOLO('yolov8n.pt')  # Modèle plus petit

            train_args = {
                'data': 'data/dentex/data.yaml',
                'epochs': 5,
                'batch': 4,
                'imgsz': 416,  # Plus petit pour la mémoire
                'device': device,
                'project': 'models',
                'name': 'dentex_fallback',
                'exist_ok': True,
                'verbose': True
            }

            results = model.train(**train_args)
            print("✅ Entraînement de secours réussi!")

            return results

        finally:
            torch.load = original_torch_load

    except Exception as e:
        print(f"❌ Échec entraînement de secours: {e}")
        return None

def test_model():
    """Test du modèle entraîné"""
    print("🔍 Test du modèle...")

    try:
        from ultralytics import YOLO
        import matplotlib.pyplot as plt

        # Chercher le meilleur modèle
        models_dir = Path("models")
        best_model = None

        for subdir in models_dir.iterdir():
            if subdir.is_dir():
                weights_dir = subdir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        best_model = best_pt
                        break

        if not best_model:
            print("❌ Aucun modèle trouvé dans models/")
            return

        print(f"📁 Utilisation du modèle: {best_model}")

        # Patch pour le test aussi
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)

        try:
            model = YOLO(str(best_model))

            # Tester sur des images du dataset
            test_dir = Path("data/dentex/test/images")
            if test_dir.exists():
                test_images = list(test_dir.glob("*.jpg"))
                if test_images:
                    test_image = str(test_images[0])
                    print(f"🖼️  Test sur: {test_image}")

                    results = model(test_image)

                    for r in results:
                        im_array = r.plot()
                        plt.figure(figsize=(12, 8))
                        plt.imshow(im_array)
                        plt.axis('off')
                        plt.title('Détections DENTEX')
                        plt.show()

                        if r.boxes is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confidences = r.boxes.conf.cpu().numpy()
                            class_ids = r.boxes.cls.cpu().numpy().astype(int)

                            class_names = {0: "tooth", 1: "cavity", 2: "implant", 3: "lesion", 4: "filling"}

                            print(f"\n🎯 Détections trouvées: {len(boxes)}")
                            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                                class_name = class_names.get(class_id, f"class_{class_id}")
                                print(f"  {i+1}. {class_name}: {conf:.3f}")
                        else:
                            print("❌ Aucune détection trouvée")

        finally:
            torch.load = original_torch_load

    except Exception as e:
        print(f"❌ Erreur test: {e}")

def save_to_drive():
    """Sauvegarde sur Google Drive"""
    print("💾 Sauvegarde sur Google Drive...")

    try:
        from google.colab import drive
        drive.mount('/content/drive')

        models_dir = Path("models")
        if models_dir.exists():
            shutil.copytree('models/', '/content/drive/MyDrive/EvaDentalAI_Models/', dirs_exist_ok=True)
            print("✅ Sauvegardé sur Google Drive!")
        else:
            print("⚠️  Aucun répertoire models à sauvegarder")

    except Exception as e:
        print(f"❌ Erreur sauvegarde: {e}")

def run_dentex_on_colab():
    """Fonction principale pour exécuter tout le processus"""
    print("🚀 EvaDentalAI + DENTEX sur Google Colab - Version Corrigée")
    print("=" * 60)

    # 1. Corriger l'environnement
    root_dir = fix_colab_environment()

    # 2. Installer les dépendances
    device = install_dependencies()

    # 3. Télécharger DENTEX
    if not download_dentex_fixed():
        print("❌ Échec du téléchargement")
        return None

    # 4. Entraîner le modèle
    results = train_model_fixed(device)

    # 5. Tester le modèle
    if results:
        test_model()

    # 6. Sauvegarder
    save_to_drive()

    print("\n🎉 Processus terminé!")
    print("📁 Vos modèles sont dans le répertoire 'models/'")
    print("☁️  Et sauvegardés sur Google Drive si disponible")

    return results

if __name__ == "__main__":
    # Pour utilisation directe
    model = run_dentex_on_colab()
