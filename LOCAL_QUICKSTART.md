# 🚀 Test Local EvaDentalAI - Démarrage Rapide

## ⚡ Test en 5 Minutes

### 1. Installation Express

```bash
# Cloner le projet
git clone https://github.com/Aharon-Bensadoun/EvaDentalAI_Yolo.git
cd EvaDentalAI_Yolo

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Test Automatique

```bash
# Exécuter le test complet
python test_local.py
```

### 3. Test Manuel (Optionnel)

```bash
# Créer un dataset de test
python scripts/prepare_dataset.py --num-images 50

# Entraînement rapide
python scripts/train_model.py --config config/data.yaml --model yolov8n.pt --epochs 2 --batch-size 4 --device cpu

# Test de prédiction
python scripts/predict.py --model models/best.pt --image data/processed/test/images/0000.jpg --save
```

## 🎯 Résultats Attendus

### Test Réussi
```
🧪 Test Local EvaDentalAI
========================================
🔍 Test de l'installation...
✅ PyTorch: 2.1.0
✅ Ultralytics: OK
✅ Datasets: OK

📊 Test du dataset...
📥 Création du dataset simulé...
✅ Dataset simulé créé
✅ Images d'entraînement: 50

🏋️ Test d'entraînement...
✅ Entraînement réussi

🔍 Test de prédiction...
✅ Prédiction réussie

🌐 Test de l'API...
✅ API fonctionne

========================================
📊 Résumé des Tests
========================================
Installation: ✅ PASS
Dataset: ✅ PASS
Entraînement: ✅ PASS
Prédiction: ✅ PASS
API: ✅ PASS

🎯 Résultat: 5/5 tests réussis
🎉 Tous les tests sont passés! Prêt pour Colab.
```

## 🔧 Résolution de Problèmes

### Erreur de Dépendances
```bash
pip install --upgrade -r requirements.txt
```

### Erreur GPU
```bash
# Utiliser CPU
python scripts/train_model.py --device cpu
```

### Erreur de Mémoire
```bash
# Réduire la taille du batch
python scripts/train_model.py --batch-size 2
```

## 🚀 Prochaines Étapes

1. **Test réussi** → Passez à Google Colab
2. **Test échoué** → Vérifiez les erreurs et corrigez
3. **Prêt pour Colab** → Utilisez le script corrigé

---

**🎉 Testez d'abord en local, puis passez à Colab !**
