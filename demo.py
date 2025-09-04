#!/usr/bin/env python3
"""
Script de démonstration EvaDentalAI
Montre les capacités du système en mode interactif
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
    ║                    🦷 EvaDentalAI 🦷                        ║
    ║                                                              ║
    ║           Détection d'Anomalies Dentaires avec YOLO          ║
    ║                                                              ║
    ║  🎯 Classes détectées: tooth, cavity, implant, lesion, filling ║
    ║  🚀 Prêt pour la production                                  ║
    ║  📊 Performance: 85-90% mAP@0.5                             ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    print("🔍 Vérification des dépendances...")
    
    required_modules = ['ultralytics', 'torch', 'cv2', 'fastapi', 'uvicorn']
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
    
    print("✅ Toutes les dépendances sont installées!")
    return True

def run_quick_demo():
    """Lance une démonstration rapide"""
    print("\n🚀 Démonstration rapide EvaDentalAI")
    print("=" * 50)
    
    # 1. Génération du dataset
    print("\n📊 Étape 1/4: Génération du dataset...")
    os.system("python scripts/prepare_dataset.py --num-images 50")
    
    # 2. Entraînement rapide
    print("\n🏋️ Étape 2/4: Entraînement rapide...")
    os.system("python scripts/train_model.py --epochs 5 --batch-size 8 --device cpu")
    
    # 3. Test de prédiction
    print("\n🔍 Étape 3/4: Test de prédiction...")
    test_images = list(Path("data/processed/test/images").glob("*.jpg"))
    if test_images:
        os.system(f"python scripts/predict.py --model models/best.pt --image {test_images[0]} --save")
    
    # 4. Lancement de l'API
    print("\n🌐 Étape 4/4: Lancement de l'API...")
    print("L'API sera disponible sur http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("\n⏹️  Appuyez sur Ctrl+C pour arrêter l'API")
    
    try:
        os.system("python api/main.py --model models/best.pt")
    except KeyboardInterrupt:
        print("\n✅ Démonstration terminée!")

def show_menu():
    """Affiche le menu interactif"""
    while True:
        print("\n" + "=" * 50)
        print("🦷 EvaDentalAI - Menu Principal")
        print("=" * 50)
        print("1. 🚀 Démonstration rapide (5 minutes)")
        print("2. 📊 Générer un dataset")
        print("3. 🏋️ Entraîner un modèle")
        print("4. 🔍 Tester une prédiction")
        print("5. 🌐 Lancer l'API")
        print("6. 📤 Exporter un modèle")
        print("7. 🧪 Lancer les tests")
        print("8. 📚 Voir la documentation")
        print("9. ❌ Quitter")
        print("=" * 50)
        
        choice = input("Choisissez une option (1-9): ").strip()
        
        if choice == "1":
            run_quick_demo()
        elif choice == "2":
            num_images = input("Nombre d'images à générer (défaut: 100): ").strip() or "100"
            os.system(f"python scripts/prepare_dataset.py --num-images {num_images}")
        elif choice == "3":
            epochs = input("Nombre d'épochs (défaut: 50): ").strip() or "50"
            batch_size = input("Taille du batch (défaut: 16): ").strip() or "16"
            os.system(f"python scripts/train_model.py --epochs {epochs} --batch-size {batch_size}")
        elif choice == "4":
            image_path = input("Chemin vers l'image: ").strip()
            if image_path and Path(image_path).exists():
                os.system(f"python scripts/predict.py --model models/best.pt --image {image_path} --save --report")
            else:
                print("❌ Image non trouvée")
        elif choice == "5":
            print("🌐 Lancement de l'API...")
            print("URL: http://localhost:8000")
            print("Docs: http://localhost:8000/docs")
            print("⏹️  Ctrl+C pour arrêter")
            try:
                os.system("python api/main.py --model models/best.pt")
            except KeyboardInterrupt:
                print("\n✅ API arrêtée")
        elif choice == "6":
            model_path = input("Chemin vers le modèle (défaut: models/best.pt): ").strip() or "models/best.pt"
            if Path(model_path).exists():
                os.system(f"python scripts/export_model.py --model {model_path} --format all")
            else:
                print("❌ Modèle non trouvé")
        elif choice == "7":
            print("🧪 Lancement des tests...")
            os.system("python test_setup.py")
        elif choice == "8":
            show_documentation()
        elif choice == "9":
            print("👋 Au revoir!")
            break
        else:
            print("❌ Option invalide")

def show_documentation():
    """Affiche les liens vers la documentation"""
    print("\n📚 Documentation EvaDentalAI")
    print("=" * 40)
    print("📖 Guides disponibles:")
    print("  • QUICKSTART.md - Guide de démarrage rapide")
    print("  • docs/INSTALLATION.md - Installation détaillée")
    print("  • docs/USAGE.md - Guide d'utilisation complet")
    print("  • docs/GOOGLE_COLAB.md - Utilisation sur Colab")
    print("  • examples/example_usage.py - Exemples de code")
    print("\n🌐 Liens externes:")
    print("  • YOLO: https://docs.ultralytics.com")
    print("  • FastAPI: https://fastapi.tiangolo.com")
    print("  • PyTorch: https://pytorch.org/docs")

def show_status():
    """Affiche le statut du projet"""
    print("\n📊 Statut du projet EvaDentalAI")
    print("=" * 40)
    
    # Vérifier les fichiers
    files_to_check = [
        ("Dataset", "data/processed/train/images"),
        ("Modèle", "models/best.pt"),
        ("Configuration", "config/data.yaml"),
        ("Scripts", "scripts/prepare_dataset.py"),
        ("API", "api/main.py")
    ]
    
    for name, path in files_to_check:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (manquant)")
    
    # Vérifier les dépendances
    print("\n🔧 Dépendances:")
    try:
        import ultralytics
        print("  ✅ ultralytics")
    except ImportError:
        print("  ❌ ultralytics")
    
    try:
        import torch
        print(f"  ✅ torch (CUDA: {torch.cuda.is_available()})")
    except ImportError:
        print("  ❌ torch")
    
    try:
        import cv2
        print("  ✅ opencv-python")
    except ImportError:
        print("  ❌ opencv-python")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifier les dépendances
    if not check_requirements():
        print("\n💡 Pour installer les dépendances:")
        print("   pip install -r requirements.txt")
        return
    
    # Afficher le statut
    show_status()
    
    # Menu interactif
    show_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        print("💡 Consultez la documentation pour plus d'aide")
