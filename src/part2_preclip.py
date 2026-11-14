"""Parte 2: pre-CLIP. ResNet18 para imágenes y distiluse para textos, entrenados por separado."""
import numpy as np
import torch
import torchvision
from sentence_transformers import SentenceTransformer

from common import (cosine_matrix, intra_inter, load_dataset, nearest_neighbors,
                    spearman, upper_triangle)

items, images = load_dataset()
names = [it["image"] for it in items]
captions = [it["caption"] for it in items]
categories = [it["category"] for it in items]

# ---------- 2a. Embeddings de imagen con ResNet18 ----------
weights = torchvision.models.ResNet18_Weights.DEFAULT     # pesos preentrenados en ImageNet
resnet = torchvision.models.resnet18(weights=weights)
resnet.fc = torch.nn.Identity()       # quitamos la cabeza de clasificación (las 1000 clases)
resnet.eval()                         # modo evaluación: no estamos entrenando
preprocess = weights.transforms()     # redimensiona a 224x224 y normaliza como en ImageNet

with torch.no_grad():
    batch = torch.stack([preprocess(img) for img in images])
    img_emb = resnet(batch).numpy()

# ---------- 2b. Embeddings de texto con distiluse ----------
text_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased")
txt_emb = text_model.encode(captions, convert_to_numpy=True)

print("Embeddings de imagen:", img_emb.shape)
print("Embeddings de texto: ", txt_emb.shape)

# Matrices de similitud DENTRO de cada modalidad (20x20)
S_img = cosine_matrix(img_emb, img_emb)
S_txt = cosine_matrix(txt_emb, txt_emb)

# ---------- 2c. Comparación ----------
nn_img = nearest_neighbors(S_img)
nn_txt = nearest_neighbors(S_txt)

print("\n1) Vecino más parecido de cada elemento en cada espacio")
print(f"  {'elemento':14s} {'según ResNet':16s} {'según distiluse':16s} ¿coinciden?")
for i, name in enumerate(names):
    same = "SÍ" if nn_img[i] == nn_txt[i] else "-"
    print(f"  {name:14s} {names[nn_img[i]]:16s} {names[nn_txt[i]]:16s} {same}")

hits_img = sum(categories[i] == categories[nn_img[i]] for i in range(len(names)))
hits_txt = sum(categories[i] == categories[nn_txt[i]] for i in range(len(names)))
agree = int((nn_img == nn_txt).sum())
print(f"\n  Vecino de la misma categoría: ResNet {hits_img}/20 · distiluse {hits_txt}/20")
print(f"  El vecino coincide en los dos espacios: {agree}/20")

print("\n2) Correlación de Spearman entre los dos mapas de parecidos")
rho = spearman(upper_triangle(S_img), upper_triangle(S_txt))
print(f"  rho = {rho:.2f}   (1 = mismo orden, 0 = sin relación)")

print("\n3) Coherencia por categoría (similitud media)")
for label, S in (("ResNet", S_img), ("distiluse", S_txt)):
    intra, inter = intra_inter(S, categories)
    print(f"  {label:10s} misma categoría {intra:.2f} · distinta {inter:.2f} · diferencia {intra - inter:.2f}")

print("\n4) Prueba 'prohibida': imagen contra texto directamente (espacios distintos)")
S_cross = cosine_matrix(img_emb, txt_emb)
correct = int((S_cross.argmax(axis=1) == np.arange(len(names))).sum())
print(f"  Imágenes cuyo caption más parecido es el suyo: {correct}/20 (al azar sería ~1/20)")