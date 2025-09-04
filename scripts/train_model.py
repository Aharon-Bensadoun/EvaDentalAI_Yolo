#!/usr/bin/env python3
"""
Script d'entraînement YOLO pour la détection d'anomalies dentaires
Utilise YOLOv8 avec fine-tuning à partir d'un modèle pré-entraîné
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Fix for PyTorch 2.6+ serialization issue
def safe_torch_load(file_path, map_location=None):
    """Safe torch.load with PyTorch 2.6+ compatibility"""
    try:
        # Try with weights_only=True (PyTorch 2.6+ default)
        return torch.load(file_path, map_location=map_location, weights_only=True)
    except Exception as e:
        if "Unsupported global" in str(e):
            # Fallback for PyTorch 2.6+ with safe globals
            try:
                import ultralytics.nn.tasks
                with torch.serialization.safe_globals([ultralytics.nn.tasks.DetectionModel]):
                    return torch.load(file_path, map_location=map_location, weights_only=True)
            except Exception:
                # Final fallback - use weights_only=False (less secure but compatible)
                print("⚠️  Using legacy torch.load (weights_only=False) for compatibility")
                return torch.load(file_path, map_location=map_location, weights_only=False)
        else:
            # Re-raise other exceptions
            raise

class DentalYOLOTrainer:
    """Classe pour l'entraînement du modèle YOLO dentaire"""
    
    def __init__(self, config_path: str = "config/data.yaml"):
        self.config_path = config_path
        self.model = None
        self.results = None
        
    def setup_model(self, model_size: str = "yolov8n.pt", pretrained: bool = True):
        """Initialise le modèle YOLO"""
        print(f"🔧 Initialisation du modèle {model_size}")

        try:
            if pretrained:
                # Charger un modèle pré-entraîné avec gestion d'erreurs PyTorch 2.6+
                print("📥 Téléchargement et chargement du modèle pré-entraîné...")
                self.model = YOLO(model_size)
                print(f"✅ Modèle pré-entraîné chargé: {model_size}")
            else:
                # Créer un nouveau modèle
                self.model = YOLO(f"{model_size}.yaml")
                print(f"✅ Nouveau modèle créé: {model_size}")

        except Exception as e:
            if "Weights only load failed" in str(e) or "Unsupported global" in str(e):
                print(f"⚠️  Erreur de sérialisation PyTorch détectée: {e}")
                print("🔄 Tentative de contournement...")

                # Essayer de patcher temporairement torch.load
                original_torch_load = torch.load

                def patched_torch_load(*args, **kwargs):
                    # Forcer weights_only=False pour la compatibilité
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)

                torch.load = patched_torch_load

                try:
                    if pretrained:
                        self.model = YOLO(model_size)
                        print(f"✅ Modèle chargé avec contournement: {model_size}")
                    else:
                        self.model = YOLO(f"{model_size}.yaml")
                        print(f"✅ Nouveau modèle créé avec contournement: {model_size}")
                finally:
                    # Restaurer torch.load original
                    torch.load = original_torch_load
            else:
                raise e

        return self.model
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              learning_rate: float = 0.01,
              device: str = "auto",
              patience: int = 50,
              save_period: int = 10):
        """Entraîne le modèle YOLO"""
        
        print("🚀 Début de l'entraînement")
        print("=" * 50)
        
        # Vérifier la disponibilité du GPU
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🖥️  Device utilisé: {device}")
        print(f"📊 Épochs: {epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🖼️  Image size: {img_size}")
        print(f"📈 Learning rate: {learning_rate}")
        
        # Configuration d'entraînement
        train_args = {
            'data': self.config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': learning_rate,
            'device': device,
            'patience': patience,
            'save_period': save_period,
            'project': 'models',
            'name': f'dental_yolo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'auto_augment': 'randaugment',
            'erasing': 0.4,
            'crop_fraction': 1.0,
            'save_json': True,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'source': None,
            'show': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'vid_stride': 1,
            'line_width': None,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'classes': None,
            'retina_masks': False,
            'boxes': True,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'lr_scheduler': 'cosine',
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_dir': None,
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_dir': None,
            'verbose': True,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False
        }
        
        try:
            # Lancer l'entraînement
            self.results = self.model.train(**train_args)
            
            print("\n✅ Entraînement terminé avec succès!")
            print(f"📁 Modèle sauvegardé dans: {self.results.save_dir}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            raise
    
    def validate(self, model_path: str = None):
        """Valide le modèle entraîné"""
        if model_path:
            model = YOLO(model_path)
        else:
            model = self.model
            
        print("🔍 Validation du modèle...")
        results = model.val()
        
        print(f"📊 Résultats de validation:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        print(f"   Precision: {results.box.mp:.3f}")
        print(f"   Recall: {results.box.mr:.3f}")
        
        return results
    
    def export_model(self, model_path: str, formats: list = ['onnx', 'torchscript']):
        """Exporte le modèle dans différents formats"""
        print(f"📤 Export du modèle: {model_path}")
        
        model = YOLO(model_path)
        
        for format_type in formats:
            try:
                exported_path = model.export(format=format_type)
                print(f"✅ Export {format_type.upper()}: {exported_path}")
            except Exception as e:
                print(f"❌ Erreur export {format_type}: {e}")
    
    def plot_training_results(self, results_dir: str):
        """Génère des graphiques des résultats d'entraînement"""
        results_path = Path(results_dir)
        
        if not results_path.exists():
            print(f"❌ Répertoire de résultats non trouvé: {results_dir}")
            return
        
        # Lire les métriques d'entraînement
        csv_file = results_path / "results.csv"
        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            # Créer les graphiques
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Résultats d\'entraînement EvaDentalAI', fontsize=16)
            
            # Loss
            axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            axes[0, 0].set_title('Box Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # mAP
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
            axes[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            axes[0, 1].set_title('Mean Average Precision')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision/Recall
            axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Learning Rate
            axes[1, 1].plot(df['epoch'], df['lr/pg0'], label='LR')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(results_path / 'training_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"📊 Graphiques sauvegardés: {results_path / 'training_plots.png'}")

def main():
    parser = argparse.ArgumentParser(description="Entraînement YOLO pour détection dentaire")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="Fichier de configuration")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modèle de base")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'épochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille du batch")
    parser.add_argument("--img-size", type=int, default=640, help="Taille des images")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--patience", type=int, default=50, help="Patience pour early stopping")
    parser.add_argument("--export", action="store_true", help="Exporter le modèle après entraînement")
    parser.add_argument("--validate", action="store_true", help="Valider le modèle après entraînement")
    
    args = parser.parse_args()
    
    print("🦷 Entraînement EvaDentalAI YOLO")
    print("=" * 50)
    
    # Vérifier que le dataset existe
    if not Path(args.config).exists():
        print(f"❌ Fichier de configuration non trouvé: {args.config}")
        print("💡 Exécutez d'abord: python scripts/prepare_dataset.py")
        return
    
    # Initialiser le trainer
    trainer = DentalYOLOTrainer(args.config)
    
    # Configurer le modèle
    trainer.setup_model(args.model)
    
    # Entraîner
    results = trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.lr,
        device=args.device,
        patience=args.patience
    )
    
    # Validation
    if args.validate:
        trainer.validate()
    
    # Export
    if args.export:
        best_model = results.save_dir / "weights" / "best.pt"
        if best_model.exists():
            trainer.export_model(str(best_model))
    
    # Graphiques
    if results:
        trainer.plot_training_results(results.save_dir)
    
    print("\n🎉 Entraînement terminé!")
    print(f"📁 Modèle sauvegardé: {results.save_dir}")
    print("🚀 Prêt pour la prédiction!")

if __name__ == "__main__":
    main()
