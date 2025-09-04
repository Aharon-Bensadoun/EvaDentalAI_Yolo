#!/usr/bin/env python3
"""
Serveur API FastAPI pour la détection d'anomalies dentaires
Endpoint pour analyser des radiographies et retourner les détections
"""

import os
import io
import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import uvicorn

# Import du prédicteur
import sys
sys.path.append(str(Path(__file__).parent.parent))
from scripts.predict import DentalPredictor

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Initialisation de l'API
app = FastAPI(
    title="EvaDentalAI API",
    description="API pour la détection d'anomalies dentaires sur radiographies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: Dict[str, float]

class PredictionResponse(BaseModel):
    success: bool
    image_name: str
    inference_time: float
    total_detections: int
    detections: List[Detection]
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    timestamp: float

# Variables globales
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API"""
    global predictor
    
    print("🚀 Démarrage de EvaDentalAI API")
    print("=" * 50)
    
    # Vérifier que le modèle existe
    if not Path(MODEL_PATH).exists():
        print(f"⚠️  Modèle non trouvé: {MODEL_PATH}")
        print("💡 Utilisez un modèle par défaut ou entraînez-en un nouveau")
        # Créer un prédicteur vide pour les tests
        predictor = None
    else:
        # Charger le modèle
        predictor = DentalPredictor(MODEL_PATH)
        if predictor.load_model():
            print(f"✅ Modèle chargé: {MODEL_PATH}")
        else:
            print(f"❌ Erreur lors du chargement du modèle")
            predictor = None
    
    print("🌐 API prête à recevoir des requêtes")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint racine"""
    return {
        "message": "EvaDentalAI API - Détection d'anomalies dentaires",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de l'API"""
    return HealthResponse(
        status="healthy" if predictor is not None else "model_not_loaded",
        model_loaded=predictor is not None,
        model_path=MODEL_PATH,
        timestamp=time.time()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Form(CONFIDENCE_THRESHOLD),
    iou: float = Form(IOU_THRESHOLD),
    return_image: bool = Form(False)
):
    """
    Prédiction d'anomalies dentaires sur une radiographie
    
    Args:
        file: Image à analyser (JPG, PNG, etc.)
        confidence: Seuil de confiance (0.0-1.0)
        iou: Seuil IoU pour NMS (0.0-1.0)
        return_image: Retourner l'image avec détections (base64)
    
    Returns:
        Résultats de détection avec coordonnées et confiances
    """
    
    # Vérifier que le modèle est chargé
    if predictor is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Vérifiez que le modèle existe et est accessible."
        )
    
    # Vérifier la taille du fichier
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux. Taille max: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image"
        )
    
    try:
        # Lire l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convertir en RGB si nécessaire
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Sauvegarder temporairement pour le prédicteur
        temp_path = f"temp_{int(time.time())}_{file.filename}"
        image.save(temp_path)
        
        try:
            # Prédiction
            prediction = predictor.predict_image(
                temp_path, 
                conf_threshold=confidence, 
                iou_threshold=iou
            )
            
            if prediction is None:
                raise HTTPException(
                    status_code=500,
                    detail="Erreur lors de la prédiction"
                )
            
            # Convertir les détections au format API
            detections = []
            for det in prediction['detections']:
                detections.append(Detection(
                    class_id=det['class_id'],
                    class_name=det['class_name'],
                    confidence=det['confidence'],
                    bbox=det['bbox']
                ))
            
            # Préparer la réponse
            response_data = {
                "success": True,
                "image_name": file.filename,
                "inference_time": prediction['inference_time'],
                "total_detections": prediction['total_detections'],
                "detections": detections
            }
            
            # Ajouter l'image avec détections si demandé
            if return_image:
                vis_image = predictor.visualize_predictions(temp_path, prediction)
                
                # Convertir en base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data["image_with_detections"] = image_base64
            
            return PredictionResponse(**response_data)
            
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement: {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(CONFIDENCE_THRESHOLD),
    iou: float = Form(IOU_THRESHOLD)
):
    """
    Prédiction batch sur plusieurs images
    
    Args:
        files: Liste d'images à analyser
        confidence: Seuil de confiance
        iou: Seuil IoU
    
    Returns:
        Résultats pour chaque image
    """
    
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    if len(files) > 10:  # Limite de sécurité
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images par batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Utiliser l'endpoint de prédiction simple
            response = await predict_image(file, confidence, iou, False)
            results.append({
                "filename": file.filename,
                "success": True,
                "data": response.dict()
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total_files": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }

@app.get("/model/info")
async def model_info():
    """Informations sur le modèle chargé"""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé"
        )
    
    return {
        "model_path": MODEL_PATH,
        "model_exists": Path(MODEL_PATH).exists(),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "classes": predictor.class_names,
        "class_colors": predictor.class_colors
    }

@app.get("/classes")
async def get_classes():
    """Liste des classes détectables"""
    if predictor is None:
        return {
            "classes": {
                0: "tooth",
                1: "cavity", 
                2: "implant",
                3: "lesion",
                4: "filling"
            }
        }
    
    return {
        "classes": predictor.class_names,
        "colors": predictor.class_colors
    }

@app.post("/reload-model")
async def reload_model():
    """Recharger le modèle (utile pour les mises à jour)"""
    global predictor
    
    try:
        if Path(MODEL_PATH).exists():
            predictor = DentalPredictor(MODEL_PATH)
            if predictor.load_model():
                return {"success": True, "message": "Modèle rechargé avec succès"}
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Erreur lors du rechargement du modèle"
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Modèle non trouvé: {MODEL_PATH}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur: {str(e)}"
        )

# Gestion des erreurs
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "detail": str(exc)
        }
    )

def main():
    """Point d'entrée pour lancer l'API"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EvaDentalAI API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Chemin vers le modèle")
    parser.add_argument("--reload", action="store_true", help="Mode développement avec rechargement")
    
    args = parser.parse_args()
    
    # Mettre à jour le chemin du modèle
    global MODEL_PATH
    MODEL_PATH = args.model
    
    print(f"🌐 Démarrage du serveur EvaDentalAI")
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📚 Documentation: http://{args.host}:{args.port}/docs")
    print(f"🤖 Modèle: {MODEL_PATH}")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
