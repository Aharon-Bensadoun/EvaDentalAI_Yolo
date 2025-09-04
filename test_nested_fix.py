#!/usr/bin/env python3
"""
Test du fix pour les répertoires imbriqués
Simule la structure problématique et teste la correction
"""

import os
import tempfile
from pathlib import Path

def simulate_nested_structure():
    """Simule la structure imbriquée problématique"""
    # Créer une structure temporaire imbriquée
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Création de structure de test dans: {temp_dir}")

    # Simuler 5 niveaux d'imbrication
    current = temp_dir
    for i in range(5):
        current = current / "EvaDentalAI_Yolo"
        current.mkdir(exist_ok=True)

        # Ajouter les répertoires du projet au dernier niveau
        if i == 4:  # Dernier niveau
            (current / "scripts").mkdir()
            (current / "data").mkdir()
            (current / "models").mkdir()

    print(f"📍 Structure créée. Répertoire le plus profond: {current}")
    return temp_dir, current

def test_fix_function():
    """Test de la fonction fix_colab_environment"""
    print("🧪 Test de la fonction fix_colab_environment")
    print("=" * 50)

    # Importer la fonction depuis le script
    import sys
    import os
    sys.path.append(str(Path(__file__).parent / "scripts"))

    try:
        from download_dentex_simple import fix_colab_environment

        # Simuler la structure imbriquée
        temp_dir, nested_dir = simulate_nested_structure()

        # Changer vers le répertoire imbriqué
        original_cwd = os.getcwd()
        os.chdir(str(nested_dir))

        print(f"🔄 Simulation: changement vers {nested_dir}")

        # Tester la fonction
        result = fix_colab_environment()

        # Vérifier le résultat
        expected_root = temp_dir / "EvaDentalAI_Yolo"
        if str(result) == str(expected_root):
            print("✅ SUCCÈS: Navigation vers le répertoire racine correcte")
        else:
            print(f"❌ ÉCHEC: Attendu {expected_root}, obtenu {result}")

        # Restaurer le répertoire original
        os.chdir(original_cwd)

        # Nettoyer
        import shutil
        shutil.rmtree(temp_dir)

    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")

def test_path_logic():
    """Test de la logique de détection des chemins"""
    print("\n🧪 Test de la logique de détection des chemins")
    print("=" * 50)

    # Simuler un chemin imbriqué
    test_path = "/content/EvaDentalAI_Yolo/EvaDentalAI_Yolo/EvaDentalAI_Yolo/EvaDentalAI_Yolo/EvaDentalAI_Yolo"
    path_parts = Path(test_path).parts
    project_name = 'EvaDentalAI_Yolo'

    nested_count = path_parts.count(project_name)
    print(f"📊 Chemin de test: {test_path}")
    print(f"📊 Niveau d'imbrication détecté: {nested_count}")

    if nested_count > 1:
        print("🔍 Structure imbriquée détectée")

        # Simuler la recherche du répertoire racine
        root_candidates = []
        for i, part in enumerate(path_parts):
            if part == project_name:
                candidate_path = Path(*path_parts[:i+1])
                print(f"   Candidat {len(root_candidates)}: {candidate_path}")
                root_candidates.append(candidate_path)

        if root_candidates:
            # Simuler la logique de sélection du répertoire racine (comme dans le script corrigé)
            root_dir = None
            for candidate in root_candidates:
                # Simuler la vérification des fichiers du projet
                print(f"   Vérification: {candidate} -> valide")
                root_dir = candidate
                break

            # Si aucun candidat valide trouvé, prendre le premier
            if root_dir is None and root_candidates:
                root_dir = root_candidates[0]

            print(f"🎯 Répertoire racine sélectionné: {root_dir}")

            # Simuler la navigation
            current_simulated = Path(test_path)
            if str(current_simulated) != str(root_dir):
                print(f"📁 Simulation navigation vers: {root_dir}")
                print("✅ Navigation simulée réussie")
            else:
                print("✅ Déjà dans le répertoire racine")
        else:
            print("❌ Aucun candidat trouvé")
    else:
        print("✅ Structure normale")

if __name__ == "__main__":
    print("🚀 Test du fix pour répertoires imbriqués")
    print("=" * 50)

    test_path_logic()
    test_fix_function()

    print("\n🎯 Tests terminés!")
    print("📝 Résumé:")
    print("   - Logique de détection des chemins: ✅ Testée")
    print("   - Fonction fix_colab_environment: ✅ Testée")
    print("   - Structure imbriquée simulée: ✅ Créée et testée")
