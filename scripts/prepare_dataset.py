#!/usr/bin/env python3
"""
Script de préparation du dataset pour la détection d'anomalies dentaires
Génère un dataset simulé et prépare les annotations YOLO
"""

import os
import cv2
import numpy as np
import yaml
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil
from typing import List, Tuple, Dict
import argparse

class DentalDatasetGenerator:
    """Générateur de dataset simulé pour radiographies dentaires"""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.classes = {
            0: "tooth",
            1: "cavity", 
            2: "implant",
            3: "lesion",
            4: "filling"
        }
        self.class_colors = {
            0: (255, 255, 255),  # Blanc pour dents normales
            1: (0, 0, 255),      # Rouge pour caries
            2: (0, 255, 0),      # Vert pour implants
            3: (255, 0, 0),      # Bleu pour lésions
            4: (255, 255, 0)     # Jaune pour plombages
        }
        
    def create_dental_xray(self, width: int = 640, height: int = 640) -> np.ndarray:
        """Crée une radiographie dentaire simulée"""
        # Fond sombre (typique des radiographies)
        image = np.random.normal(30, 10, (height, width)).astype(np.uint8)
        
        # Ajouter des dents (formes rectangulaires claires)
        num_teeth = random.randint(4, 8)
        for _ in range(num_teeth):
            x = random.randint(50, width - 100)
            y = random.randint(100, height - 200)
            w = random.randint(40, 80)
            h = random.randint(60, 120)
            
            # Dessiner une dent
            cv2.rectangle(image, (x, y), (x + w, y + h), 180, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), 220, 2)
            
        # Ajouter du bruit pour réalisme
        noise = np.random.normal(0, 5, (height, width))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def add_anomaly(self, image: np.ndarray, class_id: int, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Ajoute une anomalie spécifique à l'image"""
        x, y, w, h = bbox
        color = self.class_colors[class_id]
        
        if class_id == 1:  # Carie - zone sombre
            cv2.rectangle(image, (x, y), (x + w, y + h), 50, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), 30, 2)
            
        elif class_id == 2:  # Implant - forme métallique brillante
            cv2.rectangle(image, (x, y), (x + w, y + h), 200, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 3)
            
        elif class_id == 3:  # Lésion - zone floue
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
            roi = image[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(roi, (15, 15), 0)
            roi[mask > 0] = blurred[mask > 0]
            
        elif class_id == 4:  # Plombage - zone très claire
            cv2.rectangle(image, (x, y), (x + w, y + h), 240, -1)
            cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)
            
        return image
    
    def generate_annotations(self, num_images: int = 100) -> List[Dict]:
        """Génère les annotations pour le dataset"""
        annotations = []
        
        for i in range(num_images):
            # Créer une radiographie
            image = self.create_dental_xray()
            
            # Générer 1-3 anomalies par image
            num_anomalies = random.randint(1, 3)
            image_annotations = []
            
            for _ in range(num_anomalies):
                class_id = random.choice(list(self.classes.keys()))
                x = random.randint(50, 500)
                y = random.randint(100, 400)
                w = random.randint(30, 80)
                h = random.randint(40, 100)
                
                # Normaliser les coordonnées pour YOLO
                x_norm = x / 640
                y_norm = y / 640
                w_norm = w / 640
                h_norm = h / 640
                
                image_annotations.append({
                    'class_id': class_id,
                    'bbox': (x_norm, y_norm, w_norm, h_norm)
                })
                
                # Ajouter l'anomalie à l'image
                image = self.add_anomaly(image, class_id, (x, y, w, h))
            
            annotations.append({
                'image': image,
                'annotations': image_annotations
            })
            
        return annotations
    
    def save_dataset(self, annotations: List[Dict], split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """Sauvegarde le dataset avec split train/val/test"""
        train_ratio, val_ratio, test_ratio = split_ratios
        
        # Créer les dossiers
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'processed' / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'processed' / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Calculer les indices de split
        total = len(annotations)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        
        splits = {
            'train': annotations[:train_end],
            'val': annotations[train_end:val_end],
            'test': annotations[val_end:]
        }
        
        # Sauvegarder chaque split
        for split_name, split_data in splits.items():
            print(f"Génération du split {split_name}: {len(split_data)} images")
            
            for i, data in enumerate(split_data):
                # Sauvegarder l'image
                image_path = self.output_dir / 'processed' / split_name / 'images' / f'{i:04d}.jpg'
                cv2.imwrite(str(image_path), data['image'])
                
                # Sauvegarder les annotations YOLO
                label_path = self.output_dir / 'processed' / split_name / 'labels' / f'{i:04d}.txt'
                with open(label_path, 'w') as f:
                    for ann in data['annotations']:
                        class_id = ann['class_id']
                        x, y, w, h = ann['bbox']
                        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"Dataset généré avec succès dans {self.output_dir}")
        print(f"Train: {len(splits['train'])} images")
        print(f"Val: {len(splits['val'])} images") 
        print(f"Test: {len(splits['test'])} images")

def download_sample_dataset():
    """Télécharge un dataset d'exemple si disponible"""
    print("Tentative de téléchargement d'un dataset d'exemple...")
    # Ici vous pourriez ajouter le téléchargement d'un vrai dataset
    # Pour l'instant, on utilise le dataset simulé
    print("Utilisation du dataset simulé généré localement.")

def main():
    parser = argparse.ArgumentParser(description="Préparation du dataset dentaire")
    parser.add_argument("--num-images", type=int, default=100, help="Nombre d'images à générer")
    parser.add_argument("--output-dir", type=str, default="data", help="Répertoire de sortie")
    parser.add_argument("--download", action="store_true", help="Tenter de télécharger un dataset réel")
    
    args = parser.parse_args()
    
    print("🦷 Préparation du dataset EvaDentalAI")
    print("=" * 50)
    
    if args.download:
        download_sample_dataset()
    
    # Générer le dataset simulé
    generator = DentalDatasetGenerator(args.output_dir)
    annotations = generator.generate_annotations(args.num_images)
    generator.save_dataset(annotations)
    
    print("\n✅ Dataset préparé avec succès!")
    print("📁 Structure créée:")
    print("   data/processed/train/")
    print("   data/processed/val/")
    print("   data/processed/test/")
    print("\n🚀 Prêt pour l'entraînement!")

if __name__ == "__main__":
    main()
