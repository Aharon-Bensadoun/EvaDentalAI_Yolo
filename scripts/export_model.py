#!/usr/bin/env python3
"""
Script d'export du modèle YOLO vers différents formats
Optimise le modèle pour l'inférence rapide
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch

class ModelExporter:
    """Classe pour l'export et l'optimisation des modèles YOLO"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Charge le modèle YOLO"""
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Modèle chargé: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def export_onnx(self, output_dir: str = "models", optimize: bool = True):
        """Exporte le modèle vers ONNX"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            print("📤 Export vers ONNX...")
            
            # Configuration d'export
            export_args = {
                'format': 'onnx',
                'imgsz': 640,
                'optimize': optimize,
                'half': False,  # FP32 pour compatibilité
                'dynamic': False,
                'simplify': True,
                'opset': 11
            }
            
            # Export
            exported_path = self.model.export(**export_args)
            
            # Déplacer vers le répertoire de sortie
            output_path = Path(output_dir) / "model.onnx"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if exported_path != str(output_path):
                import shutil
                shutil.move(exported_path, output_path)
                exported_path = str(output_path)
            
            print(f"✅ ONNX exporté: {exported_path}")
            
            # Informations sur le modèle
            self._get_model_info(exported_path, "ONNX")
            
            return exported_path
            
        except Exception as e:
            print(f"❌ Erreur export ONNX: {e}")
            return None
    
    def export_torchscript(self, output_dir: str = "models"):
        """Exporte le modèle vers TorchScript"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            print("📤 Export vers TorchScript...")
            
            exported_path = self.model.export(
                format='torchscript',
                imgsz=640,
                optimize=True
            )
            
            # Déplacer vers le répertoire de sortie
            output_path = Path(output_dir) / "model.pt"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if exported_path != str(output_path):
                import shutil
                shutil.move(exported_path, output_path)
                exported_path = str(output_path)
            
            print(f"✅ TorchScript exporté: {exported_path}")
            return exported_path
            
        except Exception as e:
            print(f"❌ Erreur export TorchScript: {e}")
            return None
    
    def export_tflite(self, output_dir: str = "models"):
        """Exporte le modèle vers TensorFlow Lite"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            print("📤 Export vers TensorFlow Lite...")
            
            exported_path = self.model.export(
                format='tflite',
                imgsz=640,
                int8=False  # FP32 pour compatibilité
            )
            
            # Déplacer vers le répertoire de sortie
            output_path = Path(output_dir) / "model.tflite"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if exported_path != str(output_path):
                import shutil
                shutil.move(exported_path, output_path)
                exported_path = str(output_path)
            
            print(f"✅ TensorFlow Lite exporté: {exported_path}")
            return exported_path
            
        except Exception as e:
            print(f"❌ Erreur export TensorFlow Lite: {e}")
            return None
    
    def optimize_for_inference(self, output_dir: str = "models"):
        """Optimise le modèle pour l'inférence rapide"""
        if not self.model:
            if not self.load_model():
                return None
        
        try:
            print("⚡ Optimisation pour l'inférence...")
            
            # Export optimisé
            optimized_path = self.model.export(
                format='onnx',
                imgsz=640,
                optimize=True,
                half=True,  # FP16 pour vitesse
                dynamic=False,
                simplify=True,
                workspace=4
            )
            
            # Déplacer vers le répertoire de sortie
            output_path = Path(output_dir) / "model_optimized.onnx"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if optimized_path != str(output_path):
                import shutil
                shutil.move(optimized_path, output_path)
                optimized_path = str(output_path)
            
            print(f"✅ Modèle optimisé: {optimized_path}")
            
            # Test de performance
            self._benchmark_model(optimized_path)
            
            return optimized_path
            
        except Exception as e:
            print(f"❌ Erreur optimisation: {e}")
            return None
    
    def _get_model_info(self, model_path: str, format_type: str):
        """Affiche les informations sur le modèle exporté"""
        try:
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"📊 {format_type} - Taille: {file_size:.1f} MB")
        except Exception as e:
            print(f"⚠️  Impossible de lire les infos: {e}")
    
    def _benchmark_model(self, model_path: str, num_runs: int = 10):
        """Benchmark du modèle optimisé"""
        try:
            print("🏃 Benchmark du modèle...")
            
            import time
            import numpy as np
            
            # Charger le modèle ONNX
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(model_path)
                
                # Créer une image de test
                dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
                
                # Warmup
                for _ in range(3):
                    session.run(None, {"images": dummy_input})
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    start = time.time()
                    session.run(None, {"images": dummy_input})
                    times.append(time.time() - start)
                
                avg_time = np.mean(times) * 1000  # ms
                std_time = np.std(times) * 1000
                
                print(f"⚡ Performance: {avg_time:.1f} ± {std_time:.1f} ms")
                print(f"📈 FPS: {1000/avg_time:.1f}")
                
            except ImportError:
                print("⚠️  onnxruntime non installé, benchmark ignoré")
                
        except Exception as e:
            print(f"⚠️  Erreur benchmark: {e}")
    
    def export_all(self, output_dir: str = "models"):
        """Exporte le modèle dans tous les formats supportés"""
        print("🚀 Export complet du modèle")
        print("=" * 50)
        
        results = {}
        
        # ONNX standard
        onnx_path = self.export_onnx(output_dir, optimize=False)
        if onnx_path:
            results['onnx'] = onnx_path
        
        # ONNX optimisé
        onnx_opt_path = self.optimize_for_inference(output_dir)
        if onnx_opt_path:
            results['onnx_optimized'] = onnx_opt_path
        
        # TorchScript
        torchscript_path = self.export_torchscript(output_dir)
        if torchscript_path:
            results['torchscript'] = torchscript_path
        
        # TensorFlow Lite
        tflite_path = self.export_tflite(output_dir)
        if tflite_path:
            results['tflite'] = tflite_path
        
        print("\n✅ Export terminé!")
        print("📁 Modèles exportés:")
        for format_type, path in results.items():
            print(f"   {format_type}: {path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Export et optimisation de modèles YOLO")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle YOLO")
    parser.add_argument("--output", type=str, default="models", help="Répertoire de sortie")
    parser.add_argument("--format", type=str, choices=['onnx', 'torchscript', 'tflite', 'optimized', 'all'], 
                       default='all', help="Format d'export")
    parser.add_argument("--optimize", action="store_true", help="Optimiser pour l'inférence")
    
    args = parser.parse_args()
    
    print("📤 Export EvaDentalAI Model")
    print("=" * 50)
    
    # Vérifier que le modèle existe
    if not Path(args.model).exists():
        print(f"❌ Modèle non trouvé: {args.model}")
        return
    
    # Initialiser l'exporteur
    exporter = ModelExporter(args.model)
    
    # Export selon le format demandé
    if args.format == 'onnx':
        exporter.export_onnx(args.output, args.optimize)
    elif args.format == 'torchscript':
        exporter.export_torchscript(args.output)
    elif args.format == 'tflite':
        exporter.export_tflite(args.output)
    elif args.format == 'optimized':
        exporter.optimize_for_inference(args.output)
    elif args.format == 'all':
        exporter.export_all(args.output)
    
    print("\n🎉 Export terminé!")

if __name__ == "__main__":
    main()
