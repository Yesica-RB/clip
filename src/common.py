"""Funciones comunes: cargar el dataset y comparar embeddings."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Carpeta data/, calculada a partir de la ubicación de este archivo
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset():
    """Devuelve la lista de fichas del JSON y la lista de imágenes abiertas."""
    with open(DATA_DIR / "captions.json", encoding="utf-8") as f:
        items = json.load(f)
    images = [Image.open(DATA_DIR / "images" / it["image"]).convert("RGB") for it in items]
    return items, images


def normalize(x):
    """Divide cada vector por su longitud, para que todos midan 1."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine_matrix(a, b):
    """Matriz de similitud coseno: casilla [i, j] = parecido entre a[i] y b[j]."""
    return normalize(a) @ normalize(b).T

def nearest_neighbors(S):
    """Para cada fila, el índice del elemento más parecido (excluyéndose a sí mismo)."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return S.argmax(axis=1)


def upper_triangle(S):
    """Valores por encima de la diagonal: cada pareja (i, j) una sola vez."""
    i, j = np.triu_indices(len(S), k=1)
    return S[i, j]


def spearman(a, b):
    """Correlación de Spearman: compara el ORDEN de los valores, no los valores en sí."""
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def intra_inter(S, categories):
    """Similitud media dentro de la misma categoría y entre categorías distintas."""
    cats = np.array(categories)
    same = cats[:, None] == cats[None, :]
    not_self = ~np.eye(len(cats), dtype=bool)
    return S[same & not_self].mean(), S[~same].mean()

# ---------- CLIP (partes 3 y 4) ----------
import torch

CLIP_NAME = "openai/clip-vit-base-patch32"


def load_clip(name=CLIP_NAME):
    """Carga el modelo CLIP y su procesador (prepara imágenes y textos)."""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(name).eval()
    processor = CLIPProcessor.from_pretrained(name)
    return model, processor


def clip_image_embeddings(model, processor, images):
    """Codificador de imagen + proyección al espacio compartido."""
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        feats = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        return model.visual_projection(feats).numpy()


def clip_text_embeddings(model, processor, texts):
    """Codificador de texto + proyección al MISMO espacio compartido."""
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        feats = model.text_model(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"]).pooler_output
        return model.text_projection(feats).numpy()