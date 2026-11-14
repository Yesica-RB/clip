"""Parte 3: CLIP. Imagen y texto codificados en el MISMO espacio."""
import numpy as np

from common import (clip_image_embeddings, clip_text_embeddings, cosine_matrix, intra_inter,
                    load_clip, load_dataset, nearest_neighbors, spearman, upper_triangle)

items, images = load_dataset()
names = [it["image"] for it in items]
captions = [it["caption"] for it in items]
categories = [it["category"] for it in items]

model, processor = load_clip()
img_emb = clip_image_embeddings(model, processor, images)
txt_emb = clip_text_embeddings(model, processor, captions)
print("Embeddings de imagen:", img_emb.shape)
print("Embeddings de texto: ", txt_emb.shape)

# ---------- 1. Imagen contra caption (ahora SÍ es válido) ----------
S_cross = cosine_matrix(img_emb, txt_emb)     # [i, j] = parecido de la imagen i con el caption j

print("\n1) Cada imagen contra los 20 captions")
print(f"  {'imagen':14s} {'sim. con el suyo':>16s} {'caption más parecido':>22s} {'puesto del suyo':>16s}")
for i, name in enumerate(names):
    best = int(S_cross[i].argmax())
    rank = int((S_cross[i] > S_cross[i, i]).sum()) + 1    # 1 = su caption es el más parecido
    print(f"  {name:14s} {S_cross[i, i]:16.2f} {names[best]:>22s} {rank:16d}")

top1_i2t = int((S_cross.argmax(axis=1) == np.arange(20)).sum())   # imagen -> texto
top1_t2i = int((S_cross.argmax(axis=0) == np.arange(20)).sum())   # texto -> imagen
diag = np.diag(S_cross).mean()
off = S_cross[~np.eye(20, dtype=bool)].mean()
print(f"\n  Imagen -> su caption en 1er puesto: {top1_i2t}/20   (pre-CLIP: 3/20)")
print(f"  Caption -> su imagen en 1er puesto: {top1_t2i}/20")
print(f"  Similitud media con su caption {diag:.2f} · con los demás {off:.2f}")

# ---------- 2. Estructura dentro de cada modalidad (igual que en la parte 2) ----------
S_img = cosine_matrix(img_emb, img_emb)
S_txt = cosine_matrix(txt_emb, txt_emb)
nn_img, nn_txt = nearest_neighbors(S_img), nearest_neighbors(S_txt)
hits_img = sum(categories[i] == categories[nn_img[i]] for i in range(20))
hits_txt = sum(categories[i] == categories[nn_txt[i]] for i in range(20))
print("\n2) Estructura dentro de cada modalidad")
print(f"  Vecino de la misma categoría: imagen {hits_img}/20 · texto {hits_txt}/20"
      f"   (pre-CLIP: ResNet 15/20 · distiluse 19/20)")
print(f"  El vecino coincide en los dos espacios: {int((nn_img == nn_txt).sum())}/20   (pre-CLIP: 7/20)")
rho = spearman(upper_triangle(S_img), upper_triangle(S_txt))
print(f"  Spearman entre los dos mapas: rho = {rho:.2f}   (pre-CLIP: 0.24)")

print("\n3) Coherencia por categoría (similitud media)")
for label, S in (("imagen", S_img), ("texto", S_txt), ("img-texto", S_cross)):
    intra, inter = intra_inter(S, categories)
    print(f"  {label:10s} misma categoría {intra:.2f} · distinta {inter:.2f} · diferencia {intra - inter:.2f}")