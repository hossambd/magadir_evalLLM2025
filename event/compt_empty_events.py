#!/usr/bin/env python3
"""
Compter les documents dont la liste `events` est vide.

Usage :
    python count_empty_events.py \
        data/predictions/*_reconstructed*.json \
        data/predictions/*.jsonl
Si aucun argument n’est fourni, le script regarde *.json[l] dans le dossier
DATA_DIR (voir variable ci-dessous).

Sortie :
- un récapitulatif fichier par fichier
- un total global
- code retour 0 si aucun doc vide, 1 sinon (pratique pour la CI)
"""

import json
import sys
from pathlib import Path
from typing import List

# Dossier par défaut si aucun fichier/motif n’est indiqué
DATA_DIR = Path("event/final_outputs")

def load_docs(path: Path) -> List[dict]:
    """
    Charge un fichier .json (liste) ou .jsonl (1 doc / ligne) et renvoie
    une liste de dictionnaires.
    """
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    else:  # .json (on suppose une liste à la racine)
        return json.loads(path.read_text(encoding="utf-8"))

def count_empty(docs: List[dict]) -> int:
    """
    Compte les docs sans événements (events manquant ou [] vide).
    """
    return sum(1 for d in docs if not d.get("events"))

def main():
    # 1) récupérer la liste de fichiers à analyser
    if len(sys.argv) > 1:
        files = [Path(p) for arg in sys.argv[1:] for p in Path().glob(arg)]
    else:
        files = [*DATA_DIR.glob("*.json"), *DATA_DIR.glob("*.jsonl")]
    if not files:
        print("Aucun fichier trouvé.")
        sys.exit(1)

    total_empty = total_docs = 0
    for fp in files:
        docs = load_docs(fp)
        n_empty = count_empty(docs)
        total_empty += n_empty
        total_docs  += len(docs)
        print(f"{fp.name:40} : {n_empty:5d} / {len(docs)} docs sans events")

    print("\n=== RÉCAP ===")
    print(f"Documents inspectés : {total_docs}")
    print(f"Sans events         : {total_empty}")

    # code sortie ≠ 0 si au moins un doc vide
    sys.exit(0 if total_empty == 0 else 1)

if __name__ == "__main__":
    main()
