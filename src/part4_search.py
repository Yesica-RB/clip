"""Parte 4: buscador sobre el dataset usando CLIP.

- Búsqueda por texto: el usuario escribe una consulta en inglés.
- Búsqueda por imagen (extra): ':img <foto>' devuelve las fotos más parecidas a otra foto.
Las dos usan el MISMO espacio de embeddings de CLIP y los mismos embeddings de imagen precalculados.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from common import (DATA_DIR, clip_image_embeddings, clip_text_embeddings, load_clip,
                    load_dataset, normalize)

CACHE = DATA_DIR / "clip_image_embeddings.npz"
IMAGES_DIR = DATA_DIR / "images"
QUERIES_DIR = DATA_DIR / "queries"      # fotos de consulta que NO forman parte del dataset


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


def rank(scores, k, exclude=None):
    """Los k índices con mayor puntuación (opcionalmente sin uno de ellos), con su puntuación."""
    order = [int(i) for i in np.argsort(-scores) if i != exclude][:k]
    return [(i, float(scores[i])) for i in order]


def search(query, model, processor, img_emb_norm, k):
    """Búsqueda por texto: compara la consulta con cada imagen (similitud coseno)."""
    q = normalize(clip_text_embeddings(model, processor, [query]))[0]
    return rank(img_emb_norm @ q, k)


def resolve_image(arg):
    """Busca la foto tal cual, en data/images/ o en data/queries/."""
    for path in (Path(arg), IMAGES_DIR / arg, QUERIES_DIR / arg):
        if path.is_file():
            return path
    return None


def search_by_image(path, model, processor, names, img_emb_norm, k):
    """Búsqueda por imagen: compara una foto con cada imagen del dataset."""
    if path.parent.resolve() == IMAGES_DIR.resolve() and path.name in names:
        idx = names.index(path.name)              # ya está en el dataset: reutilizamos su embedding
        return rank(img_emb_norm @ img_emb_norm[idx], k, exclude=idx)   # sin contarse a sí misma
    image = Image.open(path).convert("RGB")       # foto nueva: calculamos su embedding
    q = normalize(clip_image_embeddings(model, processor, [image]))[0]
    return rank(img_emb_norm @ q, k)


def print_results(results, items):
    for pos, (i, score) in enumerate(results, start=1):
        it = items[i]
        print(f"  {pos}. {it['image']:14s} {score:.3f}  [{it['category']}]  {it['caption']}")


def main():
    parser = argparse.ArgumentParser(description="Buscador de imágenes por texto (y por imagen) con CLIP")
    parser.add_argument("-k", type=int, default=3, help="número de resultados (por defecto 3)")
    args = parser.parse_args()

    items, images = load_dataset()
    names = [it["image"] for it in items]
    model, processor = load_clip()
    img_emb = normalize(get_image_embeddings(model, processor, items, images))
    k = max(1, min(args.k, len(items)))

    print("\nEscribe una consulta en inglés (p. ej. 'a person playing an instrument').")
    print("Comandos:  :img FOTO -> buscar fotos parecidas a una foto (de data/images o data/queries)")
    print("           :k N      -> cambiar el número de resultados   |   salir -> terminar")

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
        if query.startswith(":img"):
            path = resolve_image(query[4:].strip())
            if path is None:
                print("No encuentro esa foto. Uso: :img korea_2.jpeg  (o una foto de data/queries/)")
                continue
            print_results(search_by_image(path, model, processor, names, img_emb, k), items)
            continue

        print_results(search(query, model, processor, img_emb, k), items)


if __name__ == "__main__":
    main()
