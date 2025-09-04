#!/usr/bin/env python3
"""
Script de correction des chemins pour Google Colab
À exécuter si les chemins sont encore imbriqués
"""

import os
from pathlib import Path

def force_fix_colab_paths():
    """Force la correction des chemins Colab"""
    print("🔧 Correction FORCÉE des chemins Colab")
    print("=" * 50)

    current_dir = Path.cwd()
    print(f"Répertoire actuel: {current_dir}")

    # Analyser le chemin actuel
    path_str = str(current_dir)
    project_name = 'EvaDentalAI_Yolo'

    if project_name in path_str:
        # Trouver toutes les occurrences
        parts = path_str.split(project_name)
        print(f"Occurrences de '{project_name}': {len(parts) - 1}")

        if len(parts) > 2:  # Plus d'une occurrence
            print("🔍 Chemins imbriqués détectés")

            # Construire le chemin corrigé (première occurrence seulement)
            corrected_path = parts[0] + project_name
            print(f"Chemin corrigé: {corrected_path}")

            target_dir = Path(corrected_path)

            # Vérifier que le répertoire cible existe et contient les fichiers du projet
            if target_dir.exists():
                print(f"✅ Répertoire cible existe: {target_dir}")

                # Vérifier les fichiers du projet
                has_scripts = (target_dir / 'scripts').exists()
                has_data = (target_dir / 'data').exists()

                print(f"   Scripts: {'✅' if has_scripts else '❌'}")
                print(f"   Data: {'✅' if has_data else '❌'}")

                if has_scripts and has_data:
                    print("🎯 Navigation vers le répertoire corrigé...")
                    try:
                        os.chdir(target_dir)
                        print(f"✅ Navigation réussie vers: {Path.cwd()}")

                        # Vérifier le résultat
                        verify_fix()
                        return True
                    except Exception as e:
                        print(f"❌ Erreur de navigation: {e}")
                        return False
                else:
                    print("⚠️ Le répertoire cible ne contient pas tous les fichiers du projet")
                    return False
            else:
                print(f"❌ Répertoire cible n'existe pas: {target_dir}")
                return False
        else:
            print("✅ Aucun chemin imbriqué détecté")
            verify_fix()
            return True
    else:
        print(f"⚠️ Projet '{project_name}' non trouvé dans le chemin")
        return False

def verify_fix():
    """Vérifie que la correction a fonctionné"""
    print("\n🔍 Vérification de la correction:")

    current = Path.cwd()
    print(f"Répertoire actuel: {current}")

    # Vérifier les fichiers du projet
    dirs_to_check = ['scripts', 'data', 'models']
    for dir_name in dirs_to_check:
        dir_path = current / dir_name
        status = "✅" if dir_path.exists() else "❌"
        print(f"   {status} {dir_name}/")

    # Vérifier si le chemin contient encore des imbrications
    path_str = str(current)
    project_name = 'EvaDentalAI_Yolo'
    occurrences = path_str.count(project_name)

    if occurrences <= 1:
        print("✅ Chemin corrigé avec succès")
        return True
    else:
        print(f"⚠️ Chemin encore imbriqué ({occurrences} occurrences)")
        return False

def emergency_fix():
    """Solution d'urgence si la correction automatique échoue"""
    print("\n🚨 SOLUTION D'URGENCE")
    print("=" * 30)

    print("Si la correction automatique échoue, exécutez manuellement:")
    print()
    print("# Dans Google Colab, exécutez ces commandes:")
    print("import os")
    print("os.chdir('/content/EvaDentalAI_Yolo')  # Adapter selon votre structure")
    print("print('Répertoire actuel:', os.getcwd())")
    print()
    print("# Puis relancer le script:")
    print("exec(open('scripts/download_dentex_simple.py').read())")

if __name__ == "__main__":
    print("🚀 Correction des chemins Colab")
    print("=" * 40)

    success = force_fix_colab_paths()

    if success:
        print("\n🎉 Correction terminée avec succès!")
        print("Vous pouvez maintenant exécuter le script de téléchargement.")
    else:
        print("\n❌ Correction automatique échouée")
        emergency_fix()

    print("\n💡 Prochaines étapes:")
    print("1. Vérifiez que vous êtes dans le bon répertoire")
    print("2. Exécutez: python scripts/download_dentex_simple.py")
    print("3. Les chemins devraient maintenant être corrects")
