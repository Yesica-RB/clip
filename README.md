# Práctica 4 — Pre-CLIP, CLIP y buscador multimodal

Sistemas Interactivos Inteligentes · Unidad IV

**Sistema operativo empleado:** macOS (Apple Silicon), ejecución con **Docker** (imagen y `Makefile` proporcionados).
No se necesita GPU: todo se ejecuta en CPU.

## Ejecución

```bash
make build          # construir la imagen (solo la primera vez)
make run            # comprobar el entorno (entorno_test.py)

make part2          # Parte 2: pre-CLIP (ResNet18 + distiluse)
make part3          # Parte 3: CLIP
make search         # Parte 4: buscador (k = 3 por defecto)
make search K=5     # buscador mostrando 5 resultados
```

Equivalente sin los atajos: `make run SCRIPT=src/part2_preclip.py`, `make run SCRIPT=src/part3_clip.py`,
`make run SCRIPT="src/part4_search.py -k 5"`.

La primera ejecución descarga los modelos (ResNet18 ≈ 45 MB, distiluse ≈ 540 MB, CLIP ≈ 600 MB); se guardan
en el volumen de Docker `p4-cache` y no se vuelven a descargar.

**Uso del buscador:** escribir una consulta en inglés y pulsar Enter. `:k N` cambia el número de resultados
durante la ejecución; `salir` (o `Ctrl+C`) termina.

## Estructura

```
data/
  images/            20 imágenes: food_1..5, korea_1..5, music_1..5, sport_1..5 (.jpeg)
  captions.json      una entrada por imagen: {"image", "category", "caption"}
src/
  common.py          carga del dataset, similitud coseno, métricas y funciones de CLIP (compartido)
  part2_preclip.py   Parte 2
  part3_clip.py      Parte 3
  part4_search.py    Parte 4
  entorno_test.py    (proporcionado)
```

## Parte 1 — Dataset

- 20 imágenes en 4 categorías de 5: **food**, **korea**, **music** y **sport**.
- Imágenes **reales**, descargadas de [Unsplash](https://unsplash.com) (licencia de Unsplash, uso libre). Ninguna generada por IA.
- Captions en inglés escritos a mano, describiendo lo que se ve en cada foto.
- Las categorías se eligieron para no solaparse: *food* no incluye comida coreana y *korea* son calles y lugares de Corea.
- Dos captions de *korea* (korea_3 y korea_5) no contienen ninguna palabra "coreana", a propósito, para ver si los modelos las asocian igualmente.

## Parte 2 — Pre-CLIP

- **Imagen:** ResNet18 preentrenada en ImageNet (`torchvision`). Se elimina la cabeza de clasificación
  (`resnet.fc = Identity()`) para obtener el vector de 512 características.
- **Texto:** `sentence-transformers/distiluse-base-multilingual-cased` (512 dimensiones).
- Aunque las dos dan 512 dimensiones, **no comparten espacio de representación**, por lo que no se comparan
  directamente. Se compara la **estructura** de cada espacio:
  1. vecino más parecido de cada elemento en cada espacio (¿es de su misma categoría? ¿coincide en los dos?),
  2. **correlación de Spearman** entre las dos matrices de similitud (orden de las parejas),
  3. similitud media dentro y entre categorías,
  4. como demostración, la comparación directa imagen-texto ("prueba prohibida").

## Parte 3 — CLIP

- `openai/clip-vit-base-patch32` (`transformers`). Cada embedding = codificador (`vision_model` / `text_model`)
  + proyección al espacio compartido (`visual_projection` / `text_projection`).
- Se mide el puesto del caption correcto para cada imagen (y viceversa) y se repiten las métricas de la parte 2.

## Resultados

| Métrica | Pre-CLIP | CLIP |
|---|---|---|
| Imagen → su caption en 1.er puesto | 3/20 (azar) | **20/20** |
| Caption → su imagen en 1.er puesto | — | **20/20** |
| Vecino de imagen de la misma categoría | 15/20 (ResNet) | **20/20** |
| Vecino de texto de la misma categoría | **19/20** (distiluse) | 17/20 |
| Vecino coincidente en los dos espacios | 7/20 | 10/20 |
| Spearman entre las dos estructuras | 0.24 | **0.46** |

**Interpretación resumida:**
- ResNet agrupa por **apariencia visual** (colores, texturas, iluminación): acierta en *food*, visualmente
  homogénea, pero confunde, por ejemplo, el teclado del estudio con las pesas del gimnasio (fotos oscuras con objetos negros alargados).
- distiluse agrupa por **significado y vocabulario**: casi perfecto por categoría, pero empareja la guitarra con
  las tortitas por las palabras compartidas *wooden* / *dark*.
- Sin espacio compartido, la comparación directa imagen-texto es azar (3/20).
- CLIP, entrenado de forma **contrastiva** con pares imagen-texto, alinea las dos modalidades: 20/20.
  Las similitudes imagen-texto son bajas en valor absoluto (~0.33 con su caption frente a ~0.12 con los demás)
  por la **brecha de modalidad**; lo relevante es el orden.
- Comparando frases entre sí, distiluse (especializado en ello) supera al codificador de texto de CLIP.

## Parte 4 — Buscador

- Consulta libre en inglés → embedding de texto de CLIP → similitud coseno con las 20 imágenes → ranking.
- Se muestran los **k** mejores (`-k` / `K=`, o `:k N` durante la ejecución) con su **puntuación**, categoría y caption.
- Los embeddings de imagen se calculan **una sola vez**: al arrancar se reutilizan para todas las consultas y
  además se guardan en `data/clip_image_embeddings.npz`; en ejecuciones siguientes se cargan de ahí
  (la caché se invalida si cambian las imágenes del dataset).
- **Extra: búsqueda por imagen.** `:img FOTO` devuelve las imágenes más parecidas a una foto, ya sea del dataset
  (`:img korea_2.jpeg`, que se excluye a sí misma y reutiliza su embedding de la caché) o una foto nueva guardada en
  `data/queries/` (se calcula su embedding con CLIP). Justificación: a veces es más fácil enseñar un ejemplo que
  describirlo con palabras ("busca fotos como esta"), y demuestra que texto e imagen comparten el mismo espacio:
  ambas búsquedas usan los mismos embeddings de imagen precalculados.
- Limitación: siempre devuelve k resultados aunque no haya ninguno relevante (p. ej. "a cat" da puntuaciones
  bajas, ~0.20, y casi empatadas).

## Cambios respecto a los archivos proporcionados

- `Makefile`: añadidos los atajos `part2`, `part3` y `search` (con la variable `K`). Las recetas originales no cambian.
- `requirements.txt` y `Dockerfile`: sin cambios.
