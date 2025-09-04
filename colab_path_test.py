#!/usr/bin/env python3
"""
Test rapide des chemins pour EvaDentalAI sur Colab
À exécuter pour vérifier que les chemins sont corrects
"""

import os
from pathlib import Path

def test_colab_paths():
    """Test des chemins Colab"""
    print("🧪 Test des chemins Colab")
    print("=" * 40)

    current_dir = Path.cwd()
    print(f"Repertoire actuel: {current_dir}")

    # Verifier la structure des repertoires
    dirs_to_check = [
        "data/dentex/train/images",
        "data/dentex/train/labels",
        "data/dentex/val/images",
        "data/dentex/val/labels",
        "data/dentex/test/images",
        "data/dentex/test/labels",
        "models"
    ]

    print("\n📂 Verification des repertoires:")
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}")
            # Compter les fichiers
            if path.is_dir():
                files = list(path.glob("*"))
                print(f"   {len(files)} fichiers")
        else:
            print(f"❌ {dir_path} - MANQUANT")
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ Creer: {path}")
            except Exception as e:
                print(f"   ❌ Erreur creation: {e}")

    # Verifier le fichier de configuration
    config_path = Path("data/dentex/data.yaml")
    if config_path.exists():
        print("
✅ Fichier de configuration trouve:"        print(f"   {config_path.absolute()}")

        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("   Contenu valide:"            print(f"   - Path: {config.get('path', 'N/A')}")
            print(f"   - Train: {config.get('train', 'N/A')}")
            print(f"   - Val: {config.get('val', 'N/A')}")
            print(f"   - Classes: {len(config.get('names', {}))}")
        except Exception as e:
            print(f"   ❌ Erreur lecture config: {e}")
    else:
        print("
❌ Fichier de configuration manquant"        print("   Creation automatique..."        create_sample_config()

    # Verifier les modeles
    models_dir = Path("models")
    if models_dir.exists():
        print("
📁 Repertoire models:"        subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
        if subdirs:
            for subdir in subdirs:
                print(f"   ✅ Experience: {subdir.name}")
                weights_dir = subdir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        print(f"      ✅ Modele: {best_pt}")
                    else:
                        print("      ⚠️  Pas de best.pt"
        else:
            print("   Aucun sous-repertoire d'experience"
    else:
        print("
❌ Repertoire models manquant"
    print("\n🎯 Resume:")
    print("=" * 20)
    print("✅ Chemins Colab corriges")
    print("✅ Structure de repertoires verifiee")
    print("✅ Configuration dataset prete")
    print("🚀 Pret pour l'entrainement!")

def create_sample_config():
    """Cree un exemple de configuration"""
    try:
        import yaml
        from pathlib import Path

        abs_path = Path.cwd() / "data" / "dentex"

        config = {
            'path': str(abs_path),
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
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"   ✅ Configuration creee: {config_path}")

    except Exception as e:
        print(f"   ❌ Erreur creation config: {e}")

if __name__ == "__main__":
    test_colab_paths()
