#!/usr/bin/env python3
"""
build_aliaug_splits.py — el dataset de Ali-AUG, esta vez CON particiones
=========================================================================
POR QUE HAY QUE REHACERLO
--------------------------
El constructor original (`finalPaper1/data/mvtec_dataset.py`) escribe solo en
`trainA/trainB/trainC`. Nunca genera un `test_*`. Quien monto `tile_dataset`
copio `train_*` a `test_*`, y se comprobo por md5 sobre el dataset que uso el
paper:

    A   train=69  test=69   huella identica   69 de 69 imagenes compartidas
    B   train=69  test=69   huella identica   69 de 69
    C   train=69  test=69   huella identica   27 de 27 unicas
    prompts: el mismo fichero, mismo md5

Es decir: **el experimento de Tile de las Tablas 2-3 no tenia conjunto de prueba
retenido**. Y viola L1 de `RESULTS_SPEC.md` -- "el generador se entrena SOLO con
el pliegue de entrenamiento" -- porque sin particion el generador vio todo.

QUE CONSTRUYE, y en el mismo formato
-------------------------------------
    train_A / test_A   la mascara de ground_truth
    train_B / test_B   la imagen defectuosa real de test/
    train_C / test_C   una imagen limpia de train/good
    train_prompts.json / test_prompts.json

Formato identico al que espera `PairedDataset` de `finalPaper1/utils.py`, que lee
las tres (A = entrada condicionante, C = imagen real, B = objetivo). Los textos
son los EXACTOS del paper, extraidos del constructor original: son constantes por
tipo ("Add a crack.", "Add a bent wire."), asi que el texto ES la etiqueta.

LA PARTICION
------------
Estratificada por (categoria, tipo) y **por imagen de origen**, como pide S1 del
spec. Con K semillas distintas se obtienen las K particiones que el protocolo
exige (K=5), y cada una escribe su propio directorio.

Las imagenes limpias de `train_C` salen SOLO del pliegue correspondiente: para
train, de `train/good` -- que en MVTec no tiene defectos y no se solapa con
`test/` -- y ahi no hay fuga posible. La restriccion que si importa es que una
imagen DEFECTUOSA no aparezca en los dos lados, y eso es lo que garantiza la
particion por mascara de origen.

RESOLUCION
----------
El constructor original reescala a 512, pero `tile_dataset` esta a 840, que es el
tamanio nativo de las imagenes de tile en MVTec. Se conserva el nativo por
defecto (--size 0) para no introducir una diferencia mas respecto a lo publicado.

Uso:
  python build_aliaug_splits.py --categorias tile --k 5
  python build_aliaug_splits.py --categorias todas --k 5 --out D:/DATASET/aliaug_splits
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

MVTEC = Path(r'D:\DATASET\industrial\mvtec_ad')
TEXTOS = Path(r'D:\DATASET\_aliaug_prompts.json')


def guardar(orig: Path, dst: Path, size: int):
    with Image.open(orig) as im:
        if size:
            im = im.resize((size, size))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categorias', nargs='+', default=['tile'],
                    help='"todas" para las 15')
    ap.add_argument('--out', default=r'D:\DATASET\aliaug_splits')
    ap.add_argument('--k', type=int, default=5,
                    help='numero de particiones (S1 del spec pide 5)')
    ap.add_argument('--test-frac', type=float, default=0.30)
    ap.add_argument('--size', type=int, default=0,
                    help='0 = tamanio nativo. El original reescalaba a 512, '
                         'pero tile_dataset esta a 840 (el nativo de tile)')
    ap.add_argument('--seed0', type=int, default=1000)
    a = ap.parse_args()

    textos = json.load(open(TEXTOS, encoding='utf-8'))
    cats = sorted(textos) if a.categorias == ['todas'] else a.categorias

    # ── inventario: una entrada por mascara de ground_truth ─────────────────
    inv = defaultdict(list)
    for cat in cats:
        gd = MVTEC / cat / 'ground_truth'
        if not gd.exists():
            print(f'  {cat}: sin ground_truth, saltada')
            continue
        for ddir in sorted(x for x in gd.iterdir() if x.is_dir()):
            for m in sorted(ddir.glob('*.png')):
                # el defectuoso correspondiente vive en test/<tipo>/<n>.png
                b = (MVTEC / cat / 'test' / ddir.name /
                     m.name.replace('_mask', ''))
                if b.exists():
                    inv[(cat, ddir.name)].append((m, b))

    tot = sum(len(v) for v in inv.values())
    print(f'{tot} muestras defectuosas, {len(inv)} pares (categoria, tipo)\n')

    for k in range(a.k):
        semilla = a.seed0 + k
        rng = random.Random(semilla)
        O = Path(a.out) / f'split_{k}'
        if O.exists():
            shutil.rmtree(O)
        pt, pe = {}, {}
        n_tr = n_te = 0

        for (cat, tipo), pares in sorted(inv.items()):
            idx = list(range(len(pares)))
            rng.shuffle(idx)
            # al menos una a cada lado: con 3-4 muestras por tipo, redondear
            # hacia abajo dejaria tipos sin prueba
            n_test = max(1, round(len(idx) * a.test_frac))
            n_test = min(n_test, len(idx) - 1) if len(idx) > 1 else 0
            test_i = set(idx[:n_test])

            texto = textos.get(cat, {}).get(tipo)
            if not texto:
                print(f'  aviso: sin texto para {cat}/{tipo}, saltado')
                continue

            goods = sorted((MVTEC / cat / 'train' / 'good').glob('*.png'))
            for j, (m, b) in enumerate(pares):
                split = 'test' if j in test_i else 'train'
                nombre = f'{cat}_{tipo}_{j:03d}.png'
                guardar(m, O / f'{split}_A' / nombre, a.size)
                guardar(b, O / f'{split}_B' / nombre, a.size)
                # L3: la limpia sale del pliegue de entrenamiento. `train/good`
                # de MVTec no contiene defectos y es disjunto de `test/`, asi
                # que sirve para los dos lados sin fuga de defectos.
                guardar(rng.choice(goods), O / f'{split}_C' / nombre, a.size)
                (pe if split == 'test' else pt)[nombre] = texto
                if split == 'test':
                    n_te += 1
                else:
                    n_tr += 1

        json.dump(pt, open(O / 'train_prompts.json', 'w', encoding='utf-8'),
                  indent=4, ensure_ascii=False)
        json.dump(pe, open(O / 'test_prompts.json', 'w', encoding='utf-8'),
                  indent=4, ensure_ascii=False)
        comunes = set(pt) & set(pe)
        print(f'split_{k} (semilla {semilla}): train {n_tr}  test {n_te}  '
              f'compartidas {len(comunes)}'
              + ('   *** FUGA ***' if comunes else '   (ninguna, correcto)'))

    print(f'\nescrito en {a.out}')
    print('comprobar con: python verificar_splits.py')


if __name__ == '__main__':
    main()
