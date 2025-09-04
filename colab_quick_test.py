#!/usr/bin/env python3
"""
Test rapide pour vérifier le fix des chemins sur Colab
À coller directement dans une cellule Colab
"""

import os
from pathlib import Path

def test_colab_environment():
    """Test rapide de l'environnement Colab"""
    print("🔍 Test rapide Colab")
    print("=" * 30)

    current_dir = Path.cwd()
    print(f"Répertoire actuel: {current_dir}")

    # Analyser la structure du chemin
    path_parts = current_dir.parts
    project_name = 'EvaDentalAI_Yolo'
    nested_count = path_parts.count(project_name)

    print(f"Niveau d'imbrication: {nested_count}")

    if nested_count > 1:
        print("⚠️ Structure imbriquée détectée!")

        # Montrer tous les candidats
        candidates = []
        for i, part in enumerate(path_parts):
            if part == project_name:
                candidate = Path(*path_parts[:i+1])
                candidates.append(candidate)
                print(f"  Candidat {len(candidates)-1}: {candidate}")

        if candidates:
            # Sélectionner le premier (plus externe)
            selected = candidates[0]
            print(f"✅ Sélectionné: {selected}")

            # Vérifier si on doit naviguer
            if str(current_dir) != str(selected):
                print("📁 Navigation nécessaire"                return False  # Indique qu'une navigation est nécessaire
            else:
                print("✅ Déjà au bon endroit")
                return True
        else:
            print("❌ Aucun candidat trouvé")
            return False
    else:
        print("✅ Structure normale")
        return True

def show_project_structure():
    """Afficher la structure du projet"""
    print("\n📂 Structure du projet:")

    dirs_to_check = [
        "scripts",
        "data",
        "models",
        "data/dentex",
        "data/dentex/train",
        "data/dentex/val",
        "data/dentex/test"
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")

    # Vérifier les fichiers importants
    files_to_check = [
        "scripts/download_dentex_simple.py",
        "scripts/train_model.py"
    ]

    print("\n📄 Fichiers importants:")
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")

# Exécuter les tests
if __name__ == "__main__":
    print("🚀 Test rapide pour EvaDentalAI sur Colab")
    print("=" * 45)

    env_ok = test_colab_environment()
    show_project_structure()

    print("\n🎯 Résumé:")
    if env_ok:
        print("  ✅ Environnement OK - prêt pour l'entraînement")
    else:
        print("  ⚠️ Navigation nécessaire - utiliser fix_colab_environment()")

    print("\n📋 Prochaines étapes:")
    print("  1. Corriger l'environnement si nécessaire")
    print("  2. Exécuter: python scripts/download_dentex_simple.py")
    print("  3. Puis: python scripts/train_model.py --config data/dentex/data.yaml")
