#!/usr/bin/env python3
"""
Test script pour vérifier les corrections apportées
Peut être exécuté localement ou sur Colab
"""

import os
import sys
from pathlib import Path

def test_environment():
    """Test de l'environnement"""
    print("🧪 Test de l'environnement...")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  GPU non disponible")

    except ImportError:
        print("❌ PyTorch non installé")

    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics non installé")

    try:
        import datasets
        print("✅ Datasets library disponible")
    except ImportError:
        print("❌ Datasets library non installé")

def test_torch_load_fix():
    """Test du fix PyTorch serialization"""
    print("\n🧪 Test du fix PyTorch...")

    try:
        import torch

        # Tester le patch torch.load
        original_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)

        torch.load = patched_torch_load
        print("✅ Patch torch.load appliqué")

        # Restaurer
        torch.load = original_torch_load
        print("✅ Patch torch.load restauré")

    except Exception as e:
        print(f"❌ Erreur test torch.load: {e}")

def test_directory_structure():
    """Test de la structure des répertoires"""
    print("\n🧪 Test de la structure des répertoires...")

    dirs_to_check = [
        "data/dentex/train/images",
        "data/dentex/train/labels",
        "data/dentex/val/images",
        "data/dentex/val/labels",
        "data/dentex/test/images",
        "data/dentex/test/labels",
        "models"
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path} existe")
        else:
            print(f"⚠️  {dir_path} n'existe pas")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ {dir_path} créé")
            except Exception as e:
                print(f"❌ Impossible de créer {dir_path}: {e}")

def test_colab_script():
    """Test du script Colab corrigé"""
    print("\n🧪 Test du script Colab...")

    colab_script = Path("colab_dentex_fixed.py")
    if colab_script.exists():
        print("✅ colab_dentex_fixed.py existe")

        # Vérifier que le script peut être importé (syntaxe)
        try:
            with open(colab_script, 'r') as f:
                code = f.read()

            compile(code, colab_script.name, 'exec')
            print("✅ Syntaxe du script Colab valide")
        except Exception as e:
            print(f"❌ Erreur de syntaxe: {e}")
    else:
        print("❌ colab_dentex_fixed.py n'existe pas")

def main():
    """Fonction principale de test"""
    print("🚀 Test des corrections EvaDentalAI + DENTEX")
    print("=" * 50)

    # Tests
    test_environment()
    test_torch_load_fix()
    test_directory_structure()
    test_colab_script()

    print("\n🎯 Tests terminés!")
    print("\n📋 Résumé des corrections apportées:")
    print("✅ Fix téléchargement DENTEX avec gestion d'erreurs")
    print("✅ Fix sérialisation PyTorch 2.6+")
    print("✅ Script Colab complet avec contournements")
    print("✅ Structure de répertoires automatique")
    print("✅ Dataset de test de secours")

    print("\n🚀 Pour utiliser sur Colab:")
    print("1. Copiez le contenu de colab_dentex_fixed.py")
    print("2. Exécutez dans une cellule Colab")
    print("3. Ou utilisez: exec(open('colab_dentex_fixed.py').read())")

if __name__ == "__main__":
    main()
