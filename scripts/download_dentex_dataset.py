#!/usr/bin/env python3
"""
Script pour télécharger et préparer le dataset DENTEX
Dataset de radiographies panoramiques dentaires avec annotations hiérarchiques
Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX
"""

import os
import json
import shutil
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import yaml

try:
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
except ImportError:
    print("❌ Dépendances manquantes. Installez avec:")
    print("pip install datasets pillow")
    exit(1)

class DENTEXDatasetProcessor:
    """Processeur pour le dataset DENTEX"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.dataset_name = "ibrahimhamamci/DENTEX"
        
        # Classes DENTEX (diagnosis categories)
        self.diagnosis_classes = {
            0: "caries",           # Carie
            1: "deep_caries",      # Carie profonde  
            2: "periapical_lesion", # Lésion périapicale
            3: "impacted_tooth"    # Dent incluse
        }
        
        # Classes pour notre modèle YOLO
        self.yolo_classes = {
            0: "tooth",            # Dent normale
            1: "cavity",           # Carie (caries + deep_caries)
            2: "implant",          # Implant (pas dans DENTEX, on garde pour compatibilité)
            3: "lesion",           # Lésion (periapical_lesion)
            4: "filling"           # Plombage (pas dans DENTEX, on garde pour compatibilité)
        }
        
    def download_dataset(self):
        """Télécharge le dataset DENTEX depuis Hugging Face"""
        print("📥 Téléchargement du dataset DENTEX...")
        print(f"Source: {self.dataset_name}")
        
        try:
            # Charger le dataset avec gestion d'erreur pour les patterns
            dataset = load_dataset(self.dataset_name, trust_remote_code=True)
            print("✅ Dataset téléchargé avec succès!")
            
            # Afficher les informations du dataset
            print(f"📊 Informations du dataset:")
            for split_name, split_data in dataset.items():
                print(f"  {split_name}: {len(split_data)} images")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            print("💡 Tentative de téléchargement alternatif...")
            
            try:
                # Tentative alternative avec streaming
                dataset = load_dataset(self.dataset_name, streaming=True, trust_remote_code=True)
                print("✅ Dataset téléchargé en mode streaming!")
                return dataset
            except Exception as e2:
                print(f"❌ Erreur alternative: {e2}")
                return None
    
    def process_dataset(self, dataset, use_full_annotations: bool = True):
        """Traite le dataset DENTEX pour le format YOLO"""
        print("🔄 Traitement du dataset pour le format YOLO...")
        
        # Créer la structure des répertoires
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.output_dir / 'dentex' / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'dentex' / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Traiter chaque split
        processed_counts = {}
        
        for split_name, split_data in dataset.items():
            if split_name not in ['train', 'validation', 'test']:
                continue
                
            # Mapper les noms de splits
            yolo_split = 'val' if split_name == 'validation' else split_name
            
            print(f"📁 Traitement du split: {split_name} -> {yolo_split}")
            
            processed_count = 0
            
            for i, item in enumerate(split_data):
                try:
                    # Extraire l'image
                    image = item['image']
                    
                    # Sauvegarder l'image
                    image_filename = f"{yolo_split}_{i:04d}.jpg"
                    image_path = self.output_dir / 'dentex' / yolo_split / 'images' / image_filename
                    image.save(image_path, 'JPEG')
                    
                    # Traiter les annotations si disponibles
                    if 'objects' in item and item['objects']:
                        annotations = self._process_annotations(item['objects'], image.size)
                        
                        # Sauvegarder les annotations YOLO
                        label_filename = f"{yolo_split}_{i:04d}.txt"
                        label_path = self.output_dir / 'dentex' / yolo_split / 'labels' / label_filename
                        
                        with open(label_path, 'w') as f:
                            for ann in annotations:
                                f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                    
                    processed_count += 1
                    
                    if (i + 1) % 100 == 0:
                        print(f"  Traité {i + 1}/{len(split_data)} images")
                
                except Exception as e:
                    print(f"  ⚠️  Erreur sur l'image {i}: {e}")
                    continue
            
            processed_counts[yolo_split] = processed_count
            print(f"✅ {yolo_split}: {processed_count} images traitées")
        
        return processed_counts
    
    def _process_annotations(self, objects: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """Traite les annotations DENTEX pour le format YOLO"""
        annotations = []
        img_width, img_height = image_size
        
        for obj in objects:
            # Extraire les informations de l'objet
            if 'bbox' in obj:
                bbox = obj['bbox']
                x_min, y_min, x_max, y_max = bbox
                
                # Convertir en format YOLO (normalisé)
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Mapper la classe de diagnostic
                class_id = self._map_diagnosis_class(obj)
                
                if class_id is not None:
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        return annotations
    
    def _map_diagnosis_class(self, obj: Dict) -> int:
        """Mappe les classes de diagnostic DENTEX vers nos classes YOLO"""
        if 'category' in obj:
            category = obj['category']
            
            # Mapper selon les catégories DENTEX
            if category in ['caries', 'deep_caries']:
                return 1  # cavity
            elif category == 'periapical_lesion':
                return 3  # lesion
            elif category == 'impacted_tooth':
                return 0  # tooth (dent incluse)
        
        return None
    
    def create_yolo_config(self):
        """Crée le fichier de configuration YOLO pour DENTEX"""
        config_path = self.output_dir / 'dentex' / 'data.yaml'
        
        config = {
            'path': str(self.output_dir / 'dentex'),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.yolo_classes,
            'nc': len(self.yolo_classes),
            'description': 'DENTEX Dataset - Panoramic Dental X-rays',
            'source': 'https://huggingface.co/datasets/ibrahimhamamci/DENTEX',
            'license': 'CC-BY-NC-SA-4.0',
            'version': '1.0'
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Configuration YOLO créée: {config_path}")
        return config_path
    
    def create_dataset_info(self, processed_counts: Dict):
        """Crée un fichier d'informations sur le dataset"""
        info_path = self.output_dir / 'dentex' / 'dataset_info.json'
        
        info = {
            'dataset_name': 'DENTEX',
            'source': 'https://huggingface.co/datasets/ibrahimhamamci/DENTEX',
            'description': 'Panoramic dental X-rays with hierarchical annotations',
            'license': 'CC-BY-NC-SA-4.0',
            'processed_counts': processed_counts,
            'classes': {
                'original_dentex': self.diagnosis_classes,
                'yolo_mapped': self.yolo_classes
            },
            'mapping': {
                'caries + deep_caries': 'cavity',
                'periapical_lesion': 'lesion', 
                'impacted_tooth': 'tooth',
                'Note': 'implant and filling classes kept for compatibility but not in DENTEX'
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Informations du dataset sauvegardées: {info_path}")
        return info_path

def main():
    parser = argparse.ArgumentParser(description="Téléchargement et préparation du dataset DENTEX")
    parser.add_argument("--output-dir", type=str, default="data", help="Répertoire de sortie")
    parser.add_argument("--use-full-annotations", action="store_true", help="Utiliser les annotations complètes")
    parser.add_argument("--skip-download", action="store_true", help="Passer le téléchargement (si déjà fait)")
    
    args = parser.parse_args()
    
    print("🦷 Préparation du dataset DENTEX pour EvaDentalAI")
    print("=" * 60)
    print("📚 Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX")
    print("📄 Licence: CC-BY-NC-SA-4.0")
    print("🔬 Dataset de radiographies panoramiques dentaires")
    print()
    
    # Initialiser le processeur
    processor = DENTEXDatasetProcessor(args.output_dir)
    
    # Télécharger le dataset
    if not args.skip_download:
        dataset = processor.download_dataset()
        if dataset is None:
            print("❌ Échec du téléchargement")
            return
    else:
        print("⏭️  Téléchargement ignoré")
        dataset = None
    
    # Traiter le dataset
    processed_counts = processor.process_dataset(dataset, args.use_full_annotations)
    
    # Créer la configuration YOLO
    config_path = processor.create_yolo_config()
    
    # Créer les informations du dataset
    info_path = processor.create_dataset_info(processed_counts)
    
    print("\n✅ Dataset DENTEX préparé avec succès!")
    print("📁 Structure créée:")
    print("   data/dentex/train/")
    print("   data/dentex/val/")
    print("   data/dentex/test/")
    print("   data/dentex/data.yaml")
    print()
    print("🚀 Utilisation:")
    print("   python scripts/train_model.py --config data/dentex/data.yaml")
    print()
    print("📊 Statistiques:")
    for split, count in processed_counts.items():
        print(f"   {split}: {count} images")

if __name__ == "__main__":
    main()
