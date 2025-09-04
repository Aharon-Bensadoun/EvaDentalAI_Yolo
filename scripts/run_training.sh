#!/bin/bash
# Script d'entraînement complet pour EvaDentalAI
# Usage: ./scripts/run_training.sh [options]

set -e  # Arrêter en cas d'erreur

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Configuration par défaut
NUM_IMAGES=200
EPOCHS=100
BATCH_SIZE=16
MODEL_SIZE="yolov8n.pt"
DEVICE="auto"

# Parser les arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-images)
            NUM_IMAGES="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --num-images N    Nombre d'images à générer (défaut: 200)"
            echo "  --epochs N        Nombre d'épochs (défaut: 100)"
            echo "  --batch-size N    Taille du batch (défaut: 16)"
            echo "  --model MODEL     Modèle de base (défaut: yolov8n.pt)"
            echo "  --device DEVICE   Device (cpu/cuda/auto, défaut: auto)"
            echo "  --help            Afficher cette aide"
            exit 0
            ;;
        *)
            print_error "Option inconnue: $1"
            exit 1
            ;;
    esac
done

print_info "🦷 Entraînement EvaDentalAI"
echo "=================================="
print_info "Configuration:"
echo "  Images: $NUM_IMAGES"
echo "  Épochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Modèle: $MODEL_SIZE"
echo "  Device: $DEVICE"
echo ""

# Vérifier Python
if ! command -v python &> /dev/null; then
    print_error "Python n'est pas installé"
    exit 1
fi

# Vérifier les dépendances
print_info "Vérification des dépendances..."
python -c "import ultralytics, torch, cv2, numpy" 2>/dev/null || {
    print_warning "Installation des dépendances..."
    pip install -r requirements.txt
}

# Étape 1: Préparation du dataset
print_info "Étape 1/3: Préparation du dataset"
python scripts/prepare_dataset.py --num-images $NUM_IMAGES

if [ $? -ne 0 ]; then
    print_error "Erreur lors de la préparation du dataset"
    exit 1
fi

print_success "Dataset préparé"

# Étape 2: Entraînement
print_info "Étape 2/3: Entraînement du modèle"
python scripts/train_model.py \
    --config config/data.yaml \
    --model $MODEL_SIZE \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --device $DEVICE \
    --export \
    --validate

if [ $? -ne 0 ]; then
    print_error "Erreur lors de l'entraînement"
    exit 1
fi

print_success "Modèle entraîné"

# Étape 3: Export et optimisation
print_info "Étape 3/3: Export et optimisation"

# Trouver le meilleur modèle
BEST_MODEL=$(find models -name "best.pt" -type f | head -1)

if [ -z "$BEST_MODEL" ]; then
    print_error "Modèle best.pt non trouvé"
    exit 1
fi

print_info "Export du modèle: $BEST_MODEL"
python scripts/export_model.py --model "$BEST_MODEL" --format all

if [ $? -ne 0 ]; then
    print_warning "Erreur lors de l'export, mais l'entraînement est terminé"
fi

print_success "Export terminé"

# Résumé final
echo ""
print_success "🎉 Entraînement complet terminé!"
echo ""
print_info "📁 Fichiers générés:"
echo "  - Modèle: $BEST_MODEL"
echo "  - ONNX: models/model.onnx"
echo "  - ONNX optimisé: models/model_optimized.onnx"
echo "  - TorchScript: models/model.pt"
echo ""
print_info "🚀 Prochaines étapes:"
echo "  1. Tester le modèle: python scripts/predict.py --model $BEST_MODEL --image path/to/image.jpg"
echo "  2. Lancer l'API: python api/main.py --model $BEST_MODEL"
echo "  3. Déployer avec Docker: docker-compose up"
echo ""

# Test rapide si une image de test existe
if [ -f "data/processed/test/images/0000.jpg" ]; then
    print_info "Test rapide du modèle..."
    python scripts/predict.py --model "$BEST_MODEL" --image "data/processed/test/images/0000.jpg" --save
    print_success "Test terminé, vérifiez le dossier output/"
fi
