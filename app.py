# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 22:41:10 2025

@author: ander
"""

import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace
from functools import wraps
import logging
from datetime import datetime
import requests

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty(): raise RuntimeError("Falha ao carregar Haar Cascade. Verifique instalação do OpenCV.")

logging.basicConfig(
    filename="api_usage.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = "logs"
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-KEY")
        if key and key == API_KEY:
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401
    return decorated

# Função utilitária: maior rosto detectado (assumimos 1 pessoa na foto)
def largest_face_bbox(gray_img):
    """
    Detecta o maior rosto em uma imagem em escala de cinza.
    Ajuste fino: scaleFactor e minNeighbors influenciam recall/precision.
    """
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    # Pega o maior retângulo (pela área w*h)
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)

def decode_image(file_storage):
    """
    Lê bytes de um arquivo e decodifica em arrays OpenCV:
    - BGR (original OpenCV)
    - RGB (útil para bibliotecas de visão)
    - Gray (escala de cinza, útil para detecção)
    """
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        raise ValueError("Não foi possível decodificar a imagem.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    return bgr, rgb, gray

def send_log_to_supabase(log_data):
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            },
            json=log_data,
            timeout=3
        )
        if resp.status_code not in (200, 201):
            app.logger.error(f"Falha ao salvar log no Supabase: {resp.text}")
    except Exception as e:
        app.logger.error(f"Erro ao enviar log para Supabase: {e}")

# -----------------------------------------------------------------------------
# Endpoint de saúde
# -----------------------------------------------------------------------------
@app.route("/status", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# -----------------------------------------------------------------------------
# Endpoint principal
# -----------------------------------------------------------------------------
@app.before_request
def log_request():
    logging.info(f"Request de {request.remote_addr} para {request.path} - Headers: {dict(request.headers)}")


@app.route("/analyze", methods=["POST"])
@require_api_key
def analyze():
    """
    Form-data:
      image: arquivo da foto (jpg/png)
    Retorna:
      {
        "gender": {"label": "man"|"woman", "scores": {"Man": p, "Woman": p}},
        "emotion": {"label": "happy", "scores": {...}},
        "bbox": {"x": int, "y": int, "w": int, "h": int}
      }
    """
    if "image" not in request.files:
        return jsonify({"error": "Envie a imagem no campo 'image' (multipart/form-data)."}), 400

    try:
        bgr, rgb, gray = decode_image(request.files["image"])
    except Exception as e:
        return jsonify({"error": f"Falha ao carregar imagem: {e}"}), 400

    bbox = largest_face_bbox(gray)
    if bbox is None:
        return jsonify({"error": "Nenhum rosto detectado pelo Haar Cascade."}), 422

    x, y, w, h = bbox
    # Garante recorte dentro dos limites
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    face_rgb = rgb[y0:y1, x0:x1]

    if face_rgb.size == 0:
        return jsonify({"error": "Falha ao recortar o rosto."}), 500

    try:
        result = DeepFace.analyze(
            img_path=face_rgb,
            actions=["gender", "emotion"],
            enforce_detection=False
        )
        # DeepFace.analyze pode retornar dict ou lista; normalizamos
        if isinstance(result, list):
            result = result[0]

        dom_gender = result.get("dominant_gender", "")
        gender_scores = result.get("gender", {})

        gender_label = "homem" if dom_gender.lower().startswith("man") else "mulher"

        # Emoções
        dom_emotion = result.get("dominant_emotion", "")
        emotion_scores = result.get("emotion", {})

        payload = {
            "genero": {
                "label": gender_label,
                "scores": gender_scores  # probabilidade por classe
            },
            "emocao": {
                "label": dom_emotion,
                "scores": emotion_scores
            },
            "bbox": {
                "x": int(x0),
                "y": int(y0),
                "w": int(x1 - x0),
                "h": int(y1 - y0)
            }
        }
        return jsonify(payload)

    except Exception as e:
        return jsonify({"error": f"Falha na análise do DeepFace: {e}"}), 500

@app.after_request
def log_response(response):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "ip": request.remote_addr,
        "endpoint": request.path,
        "method": request.method,
        "status_code": response.status_code,
        "success": True,
        "user_agent": request.headers.get("User-Agent", ""),
        "api_key": request.headers.get("X-API-KEY", "")
    }
    # salva local
    logging.info(f"Response {response.status_code} para {request.remote_addr} em {request.path}")
    # envia para supabase
    send_log_to_supabase(log_data)
    return response
# -----------------------------------------------------------------------------
# Execução local
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
