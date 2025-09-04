#!/usr/bin/env python3
"""
Démonstration EvaDentalAI avec le dataset DENTEX
Montre les capacités du système avec des données cliniques réelles
"""

import os
import sys
import time
from pathlib import Path

def print_banner():
    """Affiche la bannière du projet"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                🦷 EvaDentalAI + DENTEX 🦷                   ║
    ║                                                              ║
    ║        Détection d'Anomalies Dentaires avec YOLO            ║
    ║              Dataset DENTEX - Données Cliniques             ║
    ║                                                              ║
    ║  🎯 Classes: caries, lésions, dents incluses                ║
    ║  🏥 Source: Radiographies panoramiques cliniques            ║
    ║  📊 Performance: 80-90% mAP@0.5                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dentex_requirements():
    """Vérifie que les dépendances DENTEX sont installées"""
    print("🔍 Vérification des dépendances DENTEX...")
    
    required_modules = ['ultralytics', 'torch', 'cv2', 'fastapi', 'uvicorn', 'datasets', 'huggingface_hub']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Modules manquants: {', '.join(missing)}")
        print("💡 Installez avec: pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances DENTEX sont installées!")
    return True

def run_dentex_demo():
    """Lance une démonstration avec DENTEX"""
    print("\n🚀 Démonstration EvaDentalAI + DENTEX")
    print("=" * 60)
    
    # 1. Téléchargement du dataset DENTEX
    print("\n📥 Étape 1/4: Téléchargement du dataset DENTEX...")
    print("   Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX")
    print("   Licence: CC-BY-NC-SA-4.0")
    
    os.system("python scripts/download_dentex_dataset.py")
    
    # 2. Entraînement avec DENTEX
    print("\n🏋️ Étape 2/4: Entraînement avec DENTEX...")
    print("   Configuration: data/dentex/data.yaml")
    print("   Classes: caries, lésions, dents incluses")
    
    os.system("python scripts/train_model.py --config data/dentex/data.yaml --epochs 20 --batch-size 8 --device cpu")
    
    # 3. Test de prédiction
    print("\n🔍 Étape 3/4: Test de prédiction...")
    test_images = list(Path("data/dentex/test/images").glob("*.jpg"))
    if test_images:
        os.system(f"python scripts/predict.py --model models/best.pt --image {test_images[0]} --save --report")
    
    # 4. Lancement de l'API
    print("\n🌐 Étape 4/4: Lancement de l'API...")
    print("L'API sera disponible sur http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("\n⏹️  Appuyez sur Ctrl+C pour arrêter l'API")
    
    try:
        os.system("python api/main.py --model models/best.pt")
    except KeyboardInterrupt:
        print("\n✅ Démonstration DENTEX terminée!")

def show_dentex_menu():
    """Affiche le menu interactif pour DENTEX"""
    while True:
        print("\n" + "=" * 60)
        print("🦷 EvaDentalAI + DENTEX - Menu Principal")
        print("=" * 60)
        print("1. 🚀 Démonstration complète DENTEX (10-15 minutes)")
        print("2. 📥 Télécharger le dataset DENTEX")
        print("3. 🏋️ Entraîner avec DENTEX")
        print("4. 🔍 Tester une prédiction DENTEX")
        print("5. 🌐 Lancer l'API avec DENTEX")
        print("6. 📊 Analyser les résultats DENTEX")
        print("7. 📚 Voir la documentation DENTEX")
        print("8. ❌ Quitter")
        print("=" * 60)
        
        choice = input("Choisissez une option (1-8): ").strip()
        
        if choice == "1":
            run_dentex_demo()
        elif choice == "2":
            print("📥 Téléchargement du dataset DENTEX...")
            os.system("python scripts/download_dentex_dataset.py")
        elif choice == "3":
            epochs = input("Nombre d'épochs (défaut: 50): ").strip() or "50"
            batch_size = input("Taille du batch (défaut: 16): ").strip() or "16"
            os.system(f"python scripts/train_model.py --config data/dentex/data.yaml --epochs {epochs} --batch-size {batch_size}")
        elif choice == "4":
            test_images = list(Path("data/dentex/test/images").glob("*.jpg"))
            if test_images:
                print(f"Images de test disponibles: {len(test_images)}")
                image_path = input(f"Chemin vers l'image (défaut: {test_images[0]}): ").strip() or str(test_images[0])
                os.system(f"python scripts/predict.py --model models/best.pt --image {image_path} --save --report")
            else:
                print("❌ Aucune image de test trouvée. Téléchargez d'abord le dataset DENTEX.")
        elif choice == "5":
            print("🌐 Lancement de l'API avec DENTEX...")
            print("URL: http://localhost:8000")
            print("Docs: http://localhost:8000/docs")
            print("⏹️  Ctrl+C pour arrêter")
            try:
                os.system("python api/main.py --model models/best.pt")
            except KeyboardInterrupt:
                print("\n✅ API arrêtée")
        elif choice == "6":
            analyze_dentex_results()
        elif choice == "7":
            show_dentex_documentation()
        elif choice == "8":
            print("👋 Au revoir!")
            break
        else:
            print("❌ Option invalide")

def analyze_dentex_results():
    """Analyse les résultats DENTEX"""
    print("\n📊 Analyse des résultats DENTEX")
    print("=" * 40)
    
    # Vérifier les fichiers de résultats
    results_files = list(Path("models").glob("*/results.csv"))
    if results_files:
        print("📈 Fichiers de résultats trouvés:")
        for result_file in results_files:
            print(f"  - {result_file}")
        
        # Analyser le dernier résultat
        latest_result = max(results_files, key=os.path.getctime)
        print(f"\n📊 Analyse du dernier entraînement: {latest_result}")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            
            df = pd.read_csv(latest_result)
            
            # Afficher les métriques finales
            final_metrics = df.iloc[-1]
            print(f"\n🎯 Métriques finales:")
            print(f"  mAP@0.5: {final_metrics['metrics/mAP50(B)']:.3f}")
            print(f"  mAP@0.5:0.95: {final_metrics['metrics/mAP50-95(B)']:.3f}")
            print(f"  Precision: {final_metrics['metrics/precision(B)']:.3f}")
            print(f"  Recall: {final_metrics['metrics/recall(B)']:.3f}")
            
            # Graphique d'évolution
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
            plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
            plt.title('Box Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
            plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
            plt.title('Mean Average Precision')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
            plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
            plt.title('Precision & Recall')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('dentex_training_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n📊 Graphique sauvegardé: dentex_training_analysis.png")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'analyse: {e}")
    else:
        print("❌ Aucun fichier de résultats trouvé")
        print("💡 Entraînez d'abord un modèle avec DENTEX")

def show_dentex_documentation():
    """Affiche la documentation DENTEX"""
    print("\n📚 Documentation DENTEX")
    print("=" * 40)
    print("📖 Guides disponibles:")
    print("  • docs/DENTEX_DATASET.md - Guide complet DENTEX")
    print("  • docs/INSTALLATION.md - Installation détaillée")
    print("  • docs/USAGE.md - Guide d'utilisation")
    print("  • docs/GOOGLE_COLAB.md - Utilisation sur Colab")
    print("\n🌐 Liens externes:")
    print("  • Dataset DENTEX: https://huggingface.co/datasets/ibrahimhamamci/DENTEX")
    print("  • Paper DENTEX: https://arxiv.org/abs/2305.19112")
    print("  • Challenge DENTEX: https://dentex.grand-challenge.org/")
    print("  • YOLO: https://docs.ultralytics.com")
    print("  • FastAPI: https://fastapi.tiangolo.com")

def show_dentex_status():
    """Affiche le statut du projet DENTEX"""
    print("\n📊 Statut du projet EvaDentalAI + DENTEX")
    print("=" * 50)
    
    # Vérifier les fichiers DENTEX
    dentex_files = [
        ("Dataset DENTEX", "data/dentex/train/images"),
        ("Configuration DENTEX", "data/dentex/data.yaml"),
        ("Modèle", "models/best.pt"),
        ("Script DENTEX", "scripts/download_dentex_dataset.py"),
        ("API", "api/main.py")
    ]
    
    for name, path in dentex_files:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (manquant)")
    
    # Vérifier les dépendances DENTEX
    print("\n🔧 Dépendances DENTEX:")
    try:
        import datasets
        print("  ✅ datasets")
    except ImportError:
        print("  ❌ datasets")
    
    try:
        import huggingface_hub
        print("  ✅ huggingface-hub")
    except ImportError:
        print("  ❌ huggingface-hub")
    
    try:
        import ultralytics
        print("  ✅ ultralytics")
    except ImportError:
        print("  ❌ ultralytics")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifier les dépendances DENTEX
    if not check_dentex_requirements():
        print("\n💡 Pour installer les dépendances DENTEX:")
        print("   pip install -r requirements.txt")
        return
    
    # Afficher le statut
    show_dentex_status()
    
    # Menu interactif
    show_dentex_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("💡 Consultez la documentation DENTEX pour plus d'aide")
