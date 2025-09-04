#!/usr/bin/env python3
"""
Vérification rapide pour Google Colab
À exécuter pour confirmer que tout fonctionne
"""

import os
import sys
from pathlib import Path

def verify_colab_setup():
    """Vérifie que Colab est correctement configuré"""
    print("🔍 Vérification de la configuration Colab")
    print("=" * 45)

    # 1. Vérifier l'environnement
    print("📍 Environnement :")
    print(f"   Python: {sys.version}")
    print(f"   Répertoire: {os.getcwd()}")

    # 2. Vérifier PyTorch
    torch_available = False
    try:
        import torch
        torch_available = True
        print(f"   ✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  GPU non disponible")
    except ImportError:
        print("   ❌ PyTorch non installé")

    # 3. Vérifier Ultralytics
    try:
        import ultralytics
        print(f"   ✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("   ❌ Ultralytics non installé")

    # 4. Vérifier la structure des répertoires
    print("\n📂 Structure des répertoires :")

    dirs_to_check = [
        "data/dentex/train/images",
        "data/dentex/val/images",
        "data/dentex/test/images",
        "models"
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob("*")) if path.is_dir() else []
            print(f"   ✅ {dir_path} ({len(files)} fichiers)")
        else:
            print(f"   ❌ {dir_path} manquant")

    # 5. Vérifier la configuration dataset
    config_path = Path("data/dentex/data.yaml")
    if config_path.exists():
        print("\n📄 Configuration dataset :")
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            print(f"   ✅ data.yaml trouvé")
            print(f"   📁 Path: {config.get('path', 'N/A')}")
            print(f"   🔢 Classes: {len(config.get('names', {}))}")
            print(f"   📊 Train: {config.get('train', 'N/A')}")
            print(f"   📊 Val: {config.get('val', 'N/A')}")
        except Exception as e:
            print(f"   ❌ Erreur lecture config: {e}")
    else:
        print("\n❌ Configuration dataset manquante")
        print("   💡 Le script principal la créera automatiquement")
    # 6. Vérifier les scripts
    print("\n📜 Scripts disponibles :")

    scripts = [
        "colab_dentex_simple.py",
        "colab_path_test.py",
        "colab_verify.py"
    ]

    for script in scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} manquant")

    print("\n🎯 Status :")
    print("=" * 15)

    # Vérifier si tout est prêt
    ready = True
    issues = []

    if torch_available and not torch.cuda.is_available():
        issues.append("GPU non disponible")
        ready = False
    elif not torch_available:
        issues.append("PyTorch non installé")
        ready = False

    if not config_path.exists():
        issues.append("Configuration dataset à créer")
        # Ce n'est pas critique car le script la crée automatiquement

    if ready:
        print("✅ Configuration Colab valide !")
        print("🚀 Prêt pour l'entraînement !")
    else:
        print("⚠️  Configuration incomplète :")
        for issue in issues:
            print(f"   - {issue}")

    print("\n📋 Prochaines étapes :")
    print("1. Exécuter: exec(open('colab_dentex_simple.py').read())")
    print("2. Lancer: model = run_dentex_on_colab()")
    print("3. Attendre 15-30 minutes pour l'entraînement")

if __name__ == "__main__":
    verify_colab_setup()
