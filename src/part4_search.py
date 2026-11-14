"""Parte 4: buscador de imágenes por texto sobre el dataset, usando CLIP."""
import argparse

import numpy as np

from common import (DATA_DIR, clip_image_embeddings, clip_text_embeddings, load_clip,
                    load_dataset, normalize)

CACHE = DATA_DIR / "clip_image_embeddings.npz"


def get_image_embeddings(model, processor, items, images):
    """Calcula los embeddings de imagen una sola vez y los guarda; después los reutiliza."""
    names = [it["image"] for it in items]
    if CACHE.exists():
        data = np.load(CACHE)
        if list(data["names"]) == names:          # la caché corresponde a este mismo dataset
            print("Embeddings de imagen cargados de la caché.")
            return data["emb"]
    print("Calculando embeddings de imagen (solo la primera vez)...")
    emb = clip_image_embeddings(model, processor, images)
    np.savez(CACHE, names=np.array(names), emb=emb)
    return emb


def search(query, model, processor, img_emb_norm, k):
    """Devuelve los k índices de imagen más parecidos a la consulta, con su puntuación."""
    q = normalize(clip_text_embeddings(model, processor, [query]))[0]
    scores = img_emb_norm @ q                     # similitud coseno con cada imagen
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


def main():
    parser = argparse.ArgumentParser(description="Buscador de imágenes por texto con CLIP")
    parser.add_argument("-k", type=int, default=3, help="número de resultados (por defecto 3)")
    args = parser.parse_args()

    items, images = load_dataset()
    model, processor = load_clip()
    img_emb = normalize(get_image_embeddings(model, processor, items, images))
    k = args.k

    print("\nEscribe una consulta en inglés (p. ej. 'a person playing an instrument').")
    print("Comandos:  :k N  -> cambiar el número de resultados   |   salir -> terminar")

    while True:
        try:
            query = input(f"\nConsulta (k={k})> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("salir", "exit", "q"):
            break
        if query.startswith(":k"):
            try:
                k = max(1, min(int(query.split()[1]), len(items)))
                print(f"Ahora se muestran {k} resultados.")
            except (IndexError, ValueError):
                print("Uso: :k 5")
            continue

        for pos, (i, score) in enumerate(search(query, model, processor, img_emb, k), start=1):
            it = items[i]
            print(f"  {pos}. {it['image']:14s} {score:.3f}  [{it['category']}]  {it['caption']}")


if __name__ == "__main__":
    main()