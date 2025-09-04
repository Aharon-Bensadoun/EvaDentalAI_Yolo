#!/usr/bin/env python3
"""
Script de prédiction et visualisation pour la détection d'anomalies dentaires
Utilise un modèle YOLO entraîné pour analyser des radiographies
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
import time
from typing import List, Dict, Tuple, Optional

class DentalPredictor:
    """Classe pour la prédiction et visualisation des anomalies dentaires"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.class_names = {
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
        
    def load_model(self):
        """Charge le modèle YOLO"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Modèle chargé: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle: {e}")
            return False
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict:
        """Prédit les anomalies sur une image"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Impossible de charger l'image: {image_path}")
                return None
            
            # Prédiction
            start_time = time.time()
            results = self.model(image, conf=conf_threshold, iou=iou_threshold)
            inference_time = time.time() - start_time
            
            # Extraire les détections
            detections = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                    x1, y1, x2, y2 = box
                    detections.append({
                        'id': i,
                        'class_id': int(class_id),
                        'class_name': self.class_names.get(class_id, f"class_{class_id}"),
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(x1),
                            'y1': float(y1),
                            'x2': float(x2),
                            'y2': float(y2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        }
                    })
            
            return {
                'image_path': image_path,
                'inference_time': inference_time,
                'detections': detections,
                'total_detections': len(detections)
            }
            
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return None
    
    def visualize_predictions(self, image_path: str, predictions: Dict, save_path: str = None) -> np.ndarray:
        """Visualise les prédictions sur l'image"""
        # Charger l'image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Créer une copie pour le dessin
        vis_image = image_rgb.copy()
        
        # Dessiner les bounding boxes
        for detection in predictions['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Couleur de la classe
            color = self.class_colors.get(class_id, (255, 255, 255))
            
            # Coordonnées
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            
            # Dessiner le rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Texte de la classe et confiance
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fond pour le texte
            cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Texte
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Ajouter les informations générales
        info_text = f"Détections: {predictions['total_detections']} | Temps: {predictions['inference_time']:.3f}s"
        cv2.putText(vis_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Sauvegarder si demandé
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Image sauvegardée: {save_path}")
        
        return vis_image
    
    def create_detection_report(self, predictions: Dict, save_path: str = None) -> str:
        """Crée un rapport détaillé des détections"""
        report = []
        report.append("🦷 RAPPORT DE DÉTECTION D'ANOMALIES DENTAIRES")
        report.append("=" * 50)
        report.append(f"📁 Image: {Path(predictions['image_path']).name}")
        report.append(f"⏱️  Temps d'inférence: {predictions['inference_time']:.3f} secondes")
        report.append(f"🔍 Total détections: {predictions['total_detections']}")
        report.append("")
        
        if predictions['detections']:
            report.append("📊 DÉTAIL DES DÉTECTIONS:")
            report.append("-" * 30)
            
            # Grouper par classe
            class_counts = {}
            for detection in predictions['detections']:
                class_name = detection['class_name']
                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(detection)
            
            for class_name, detections in class_counts.items():
                report.append(f"\n🔸 {class_name.upper()} ({len(detections)} détection(s)):")
                for i, detection in enumerate(detections, 1):
                    bbox = detection['bbox']
                    report.append(f"   {i}. Confiance: {detection['confidence']:.3f}")
                    report.append(f"      Position: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})")
                    report.append(f"      Taille: {bbox['width']:.0f} x {bbox['height']:.0f} pixels")
        else:
            report.append("✅ Aucune anomalie détectée")
        
        report.append("\n" + "=" * 50)
        report.append("📝 Rapport généré par EvaDentalAI")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Rapport sauvegardé: {save_path}")
        
        return report_text
    
    def batch_predict(self, image_dir: str, output_dir: str = None, conf_threshold: float = 0.25) -> List[Dict]:
        """Prédit sur un batch d'images"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"❌ Répertoire non trouvé: {image_dir}")
            return []
        
        # Créer le répertoire de sortie
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trouver toutes les images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ Aucune image trouvée dans: {image_dir}")
            return []
        
        print(f"🔍 Traitement de {len(image_files)} images...")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"📸 Traitement {i}/{len(image_files)}: {image_file.name}")
            
            # Prédiction
            prediction = self.predict_image(str(image_file), conf_threshold)
            if prediction:
                results.append(prediction)
                
                # Visualisation
                vis_image = self.visualize_predictions(str(image_file), prediction)
                
                # Sauvegarder si demandé
                if output_dir:
                    # Image avec détections
                    vis_path = output_dir / f"detected_{image_file.name}"
                    cv2.imwrite(str(vis_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    
                    # Rapport
                    report_path = output_dir / f"report_{image_file.stem}.txt"
                    self.create_detection_report(prediction, str(report_path))
        
        print(f"✅ Traitement terminé: {len(results)} images analysées")
        return results

def main():
    parser = argparse.ArgumentParser(description="Prédiction d'anomalies dentaires")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle YOLO")
    parser.add_argument("--image", type=str, help="Chemin vers une image")
    parser.add_argument("--batch", type=str, help="Répertoire d'images pour traitement batch")
    parser.add_argument("--output", type=str, help="Répertoire de sortie")
    parser.add_argument("--conf", type=float, default=0.25, help="Seuil de confiance")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU")
    parser.add_argument("--save", action="store_true", help="Sauvegarder les résultats")
    parser.add_argument("--report", action="store_true", help="Générer un rapport détaillé")
    
    args = parser.parse_args()
    
    print("🦷 Prédiction EvaDentalAI")
    print("=" * 50)
    
    # Vérifier que le modèle existe
    if not Path(args.model).exists():
        print(f"❌ Modèle non trouvé: {args.model}")
        print("💡 Entraînez d'abord un modèle avec: python scripts/train_model.py")
        return
    
    # Initialiser le prédicteur
    predictor = DentalPredictor(args.model)
    
    if args.image:
        # Prédiction sur une seule image
        print(f"🔍 Analyse de: {args.image}")
        
        prediction = predictor.predict_image(args.image, args.conf, args.iou)
        if prediction:
            print(f"✅ {prediction['total_detections']} détection(s) trouvée(s)")
            print(f"⏱️  Temps: {prediction['inference_time']:.3f}s")
            
            # Visualisation
            vis_image = predictor.visualize_predictions(args.image, prediction)
            
            # Afficher
            plt.figure(figsize=(12, 8))
            plt.imshow(vis_image)
            plt.axis('off')
            plt.title(f"Détections sur {Path(args.image).name}")
            plt.show()
            
            # Sauvegarder si demandé
            if args.save:
                output_dir = Path(args.output) if args.output else Path("output")
                output_dir.mkdir(exist_ok=True)
                
                save_path = output_dir / f"detected_{Path(args.image).name}"
                cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                print(f"💾 Image sauvegardée: {save_path}")
            
            # Rapport
            if args.report:
                report = predictor.create_detection_report(prediction)
                print("\n" + report)
                
                if args.save:
                    report_path = output_dir / f"report_{Path(args.image).stem}.txt"
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(report)
                    print(f"📄 Rapport sauvegardé: {report_path}")
    
    elif args.batch:
        # Prédiction batch
        results = predictor.batch_predict(args.batch, args.output, args.conf)
        
        if results:
            # Statistiques globales
            total_detections = sum(r['total_detections'] for r in results)
            avg_time = sum(r['inference_time'] for r in results) / len(results)
            
            print(f"\n📊 STATISTIQUES GLOBALES:")
            print(f"   Images traitées: {len(results)}")
            print(f"   Total détections: {total_detections}")
            print(f"   Temps moyen: {avg_time:.3f}s par image")
            
            # Sauvegarder les statistiques
            if args.save and args.output:
                stats = {
                    'total_images': len(results),
                    'total_detections': total_detections,
                    'average_inference_time': avg_time,
                    'results': results
                }
                
                stats_path = Path(args.output) / "batch_statistics.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                print(f"📊 Statistiques sauvegardées: {stats_path}")
    
    else:
        print("❌ Spécifiez --image ou --batch")
        parser.print_help()

if __name__ == "__main__":
    main()
