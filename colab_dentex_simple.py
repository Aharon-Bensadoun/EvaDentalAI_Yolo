#!/usr/bin/env python3
"""
Script complet et corrige pour EvaDentalAI + DENTEX sur Google Colab
Version fixee pour gerer les erreurs de PyTorch 2.6+ et les problemes de chemins
"""

import os
import sys
import torch
import shutil
from pathlib import Path

def fix_colab_environment():
    """Corrige l'environnement Colab pour eviter les problemes de chemins"""
    print("Correction de l'environnement Colab...")

    # Detecter si on est dans un environnement avec des repertoires imbriques
    current_dir = Path.cwd()
    project_dirs = ['EvaDentalAI_Yolo', 'scripts', 'data', 'models']

    # Chercher le vrai repertoire racine du projet
    root_dir = current_dir
    for parent in current_dir.parents:
        if any((parent / d).exists() for d in project_dirs):
            root_dir = parent
            break

    # Si on est dans un sous-repertoire imbrique, aller a la racine
    if str(current_dir) != str(root_dir):
        print(f"Changement vers le repertoire racine: {root_dir}")
        os.chdir(root_dir)

    # Verifier et creer les repertoires necessaires
    data_dir = Path("data/dentex")
    models_dir = Path("models")

    for split in ['train', 'val', 'test']:
        (data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    models_dir.mkdir(exist_ok=True)

    print(f"Environnement corrige. Repertoire actuel: {Path.cwd()}")
    print(f"Chemins verifies:")
    print(f"  Data: {data_dir.absolute()}")
    print(f"  Models: {models_dir.absolute()}")

    return Path.cwd()

def install_dependencies():
    """Installe les dependances necessaires"""
    print("Installation des dependances...")

    try:
        import ultralytics
        print("Ultralytics deja installe")
    except ImportError:
        os.system("pip install ultralytics==8.0.196")

    try:
        import datasets
        print("Datasets deja installe")
    except ImportError:
        os.system("pip install datasets==2.14.0 huggingface-hub==0.16.4")

    # Verifier PyTorch
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("GPU non disponible, utilisation du CPU")
        device = "cpu"

    return device

def download_dentex_fixed():
    """Telechargement du dataset DENTEX avec gestion d'erreurs amelioree"""
    print("Telechargement DENTEX - Version Corrigee")
    print("=" * 50)

    try:
        from datasets import load_dataset
        from PIL import Image
        import numpy as np
        import yaml
    except ImportError:
        print("Dependances manquantes")
        return False

    # Creer la structure des repertoires
    output_dir = Path("data/dentex")
    try:
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Erreur repertoire: {e}")
        return False

    print("Telechargement du dataset DENTEX...")
    print("Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX")

    try:
        # Methode 1: Telechargement standard avec gestion d'erreurs
        dataset = load_dataset("ibrahimhamamci/DENTEX",
                              trust_remote_code=True,
                              download_mode="reuse_cache_if_exists")
        print("Dataset telecharge avec succes!")

        # Traiter le dataset
        processed_counts = process_dentex_dataset(dataset, output_dir)

        # Creer la configuration
        create_dentex_config(output_dir, processed_counts)

        print("\nDataset DENTEX prepare avec succes!")
        return True

    except Exception as e:
        print(f"Erreur telechargement: {e}")
        print("Creation d'un dataset de test...")

        # Creer un dataset de test minimal
        create_test_dataset_fixed(output_dir)
        return True

def process_dentex_dataset(dataset, output_dir):
    """Traite le dataset DENTEX pour YOLO"""
    processed_counts = {}

    for split_name, split_data in dataset.items():
        if split_name not in ['train', 'validation', 'test']:
            continue

        yolo_split = 'val' if split_name == 'validation' else split_name
        print(f"Traitement du split: {split_name} -> {yolo_split}")

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
                    print(f"  Traite {i + 1}/{len(split_data)} images")

            except Exception as e:
                print(f"  Erreur image {i}: {e}")
                continue

        processed_counts[yolo_split] = processed_count
        print(f"{yolo_split}: {processed_count} images traitees")

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
    """Cree la configuration YOLO pour DENTEX avec chemins absolus"""
    # Utiliser des chemins absolus pour eviter les problemes de repertoires
    abs_path = Path.cwd() / output_dir

    config = {
        'path': str(abs_path),  # Chemin absolu vers le repertoire dataset
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

    config_path = abs_path / 'data.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration creee: {config_path}")
    print(f"Chemins utilises:")
    print(f"  Train: {abs_path}/train/images")
    print(f"  Val: {abs_path}/val/images")
    print(f"  Test: {abs_path}/test/images")

def create_test_dataset_fixed(output_dir):
    """Cree un dataset de test minimal"""
    print("Creation d'un dataset de test...")

    from PIL import Image
    import yaml

    for split in ['train', 'val', 'test']:
        for i in range(10):  # Plus d'images de test
            img = Image.new('RGB', (640, 640), color='gray')
            img_path = output_dir / split / 'images' / f"{split}_{i:04d}.jpg"
            img.save(img_path)

            # Creer des labels varies pour le test
            label_path = output_dir / split / 'labels' / f"{split}_{i:04d}.txt"
            with open(label_path, 'w') as f:
                # Ajouter differentes classes pour le test
                classes = [0, 1, 3, 4]  # tooth, cavity, lesion, filling
                for cls in classes:
                    f.write(f"{cls} {0.1 + cls*0.2:.1f} {0.1 + cls*0.2:.1f} 0.1 0.1\n")

    create_dentex_config(output_dir, {'train': 10, 'val': 10, 'test': 10})
    print("Dataset de test cree!")

def train_model_fixed(device):
    """Entrainement du modele avec gestion des erreurs PyTorch 2.6+"""
    print("Entrainement du modele...")
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
            # Charger le modele avec le patch
            print("Chargement du modele yolov8s.pt...")
            model = YOLO('yolov8s.pt')
            print("Modele charge avec succes!")

            # Configuration d'entrainement optimisee pour Colab
            # Utiliser le chemin absolu pour le fichier de configuration
            config_path = Path.cwd() / 'data' / 'dentex' / 'data.yaml'

            train_args = {
                'data': str(config_path.absolute()),
                'epochs': 10,  # Reduit pour les tests
                'batch': 8,     # Petit batch pour la memoire
                'imgsz': 640,
                'device': device,
                'patience': 20,
                'save_period': 5,
                'project': 'models',
                'name': 'dentex_colab_fixed',
                'exist_ok': True,
                'pretrained': True,
                'verbose': True,
                'plots': True,
                'save': True,
                'val': True,
                'amp': True,
                'workers': 2,
                'seed': 42,
                'deterministic': False,
            }

            print(f"Configuration utilisee: {config_path.absolute()}")
            if not config_path.exists():
                print(f"⚠️  Fichier de configuration non trouve: {config_path}")
                print("Creation automatique...")
                create_test_dataset_fixed(Path('data/dentex'))

            print("Debut de l'entrainement...")
            results = model.train(**train_args)

            print("Entrainement termine!")
            print(f"Modele sauvegarde dans: {results.save_dir}")

            return results

        finally:
            # Restaurer torch.load original
            torch.load = original_torch_load

    except Exception as e:
        print(f"Erreur entrainement: {e}")
        print("Tentative avec un modele plus petit...")
        return train_fallback(device)

def train_fallback(device):
    """Entrainement de secours avec yolov8n.pt"""
    try:
        from ultralytics import YOLO

        # Patch torch.load
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)

        try:
            model = YOLO('yolov8n.pt')  # Modele plus petit

            # Utiliser le chemin absolu pour le fichier de configuration
            config_path = Path.cwd() / 'data' / 'dentex' / 'data.yaml'

            train_args = {
                'data': str(config_path.absolute()),
                'epochs': 5,
                'batch': 4,
                'imgsz': 416,  # Plus petit pour la memoire
                'device': device,
                'project': 'models',
                'name': 'dentex_fallback',
                'exist_ok': True,
                'verbose': True,
                'plots': True,
                'save': True,
                'val': True,
                'amp': True,
                'workers': 1,
                'seed': 42,
                'deterministic': False,
            }

            results = model.train(**train_args)
            print("Entrainement de secours reussi!")

            return results

        finally:
            torch.load = original_torch_load

    except Exception as e:
        print(f"Echec entrainement de secours: {e}")
        return None

def test_model():
    """Test du modele entraine avec gestion d'erreurs amelioree"""
    print("Test du modele...")

    try:
        from ultralytics import YOLO

        # Chercher le meilleur modele
        models_dir = Path("models")
        best_model = None

        if models_dir.exists():
            # Trier par date de modification (le plus recent en premier)
            subdirs = sorted([d for d in models_dir.iterdir() if d.is_dir()],
                           key=lambda x: x.stat().st_mtime, reverse=True)

            for subdir in subdirs:
                weights_dir = subdir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        best_model = best_pt
                        print(f"Modele trouve: {best_model}")
                        break

        if not best_model:
            print("Aucun modele entraine trouve dans models/")
            print("Test avec le modele pre-entraine yolov8n.pt...")

            # Patch pour le test
            original_torch_load = torch.load
            torch.load = lambda *args, **kwargs: original_torch_load(*args, weights_only=False, **kwargs)

            try:
                model = YOLO('yolov8n.pt')
                print("Test avec yolov8n.pt (modele pre-entraine)")

                # Tester sur des images du dataset si disponible
                test_dir = Path("data/dentex/test/images")
                if test_dir.exists():
                    test_images = list(test_dir.glob("*.jpg"))
                    if test_images:
                        test_image = str(test_images[0])
                        print(f"Test sur: {Path(test_image).name}")

                        results = model(test_image, conf=0.25, iou=0.5)

                        if results and results[0].boxes is not None:
                            boxes = results[0].boxes.xyxy.cpu().numpy()
                            confidences = results[0].boxes.conf.cpu().numpy()
                            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                            print(f"Detections trouvees: {len(boxes)}")
                            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                                print(".3f")
                        else:
                            print("Aucune detection trouvee avec yolov8n.pt")
                    else:
                        print("Aucune image de test disponible")
                else:
                    print("Repertoire de test non trouve")

            finally:
                torch.load = original_torch_load

            return

        print(f"Test du modele entraine: {best_model}")

        # Patch pour le test du modele entraine
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
                    print(f"Test sur: {Path(test_image).name}")

                    results = model(test_image, conf=0.25, iou=0.5)

                    if results and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                        class_names = {0: "tooth", 1: "cavity", 2: "implant", 3: "lesion", 4: "filling"}

                        print(f"Detections trouvees: {len(boxes)}")
                        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                            class_name = class_names.get(class_id, f"class_{class_id}")
                            print(".3f")
                    else:
                        print("Aucune detection trouvee avec le modele entraine")
                else:
                    print("Aucune image de test disponible")
            else:
                print("Repertoire de test non trouve")

        finally:
            torch.load = original_torch_load

    except Exception as e:
        print(f"Erreur lors du test: {e}")
        print("Le test a echoue, mais l'entrainement peut avoir reussi")

def save_to_drive():
    """Sauvegarde sur Google Drive"""
    print("Sauvegarde sur Google Drive...")

    try:
        from google.colab import drive
        drive.mount('/content/drive')

        models_dir = Path("models")
        if models_dir.exists():
            shutil.copytree('models/', '/content/drive/MyDrive/EvaDentalAI_Models/', dirs_exist_ok=True)
            print("Sauvegarde sur Google Drive!")
        else:
            print("Aucun repertoire models a sauvegarder")

    except Exception as e:
        print(f"Erreur sauvegarde: {e}")

def run_dentex_on_colab():
    """Fonction principale pour executer tout le processus"""
    print("EvaDentalAI + DENTEX sur Google Colab - Version Corrigee")
    print("=" * 60)

    # 1. Corriger l'environnement
    root_dir = fix_colab_environment()

    # 2. Installer les dependances
    device = install_dependencies()

    # 3. Telecharger DENTEX
    if not download_dentex_fixed():
        print("Echec du telechargement")
        return None

    # 4. Entrainer le modele
    results = train_model_fixed(device)

    # 5. Tester le modele
    if results:
        test_model()

    # 6. Sauvegarder
    save_to_drive()

    print("\nProcessus termine!")
    print("Vos modeles sont dans le repertoire 'models/'")
    print("Et sauvegardes sur Google Drive si disponible")

    return results

if __name__ == "__main__":
    # Pour utilisation directe
    model = run_dentex_on_colab()