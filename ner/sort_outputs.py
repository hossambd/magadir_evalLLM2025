#!/usr/bin/env python3
"""
Trie chaque fichier <strategy>_outputs.jsonl par ordre croissant
de l'identifiant 'id' (ex. prompt_001, prompt_002, …).

- Par défaut, écrit un nouveau fichier *.sorted.jsonl.
- Avec --in-place, réécrit le fichier original (création d'un .bak).
"""

import argparse
import json
import re
from pathlib import Path
import shutil

# ── PARAMÈTRES ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path("ner/batch_results")
ID_RE = re.compile(r"\D*(\d+)$")      # capture la partie numérique de l'id

# ── ARGPARSE ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--in-place", action="store_true",
                    help="remplacer le fichier d'origine au lieu de créer *.sorted.jsonl")
args = parser.parse_args()

# ── FONCTION DE CLÉ DE TRI ──────────────────────────────────────────────────
def id_key(item: dict) -> int:
    """
    Extrait la partie numérique de item['id'] pour un tri correct.
    Les ids sans chiffres sont relégués en fin de liste.
    """
    match = ID_RE.match(item.get("id", ""))
    return int(match.group(1)) if match else float("inf")

# ── TRAITEMENT ──────────────────────────────────────────────────────────────
jsonl_files = sorted(RESULTS_DIR.glob("*_outputs.jsonl"))
if not jsonl_files:
    raise SystemExit(f"❌ Aucun *_outputs.jsonl trouvé dans {RESULTS_DIR}")

for path in jsonl_files:
    print(f"🔄 Tri du fichier {path.name} …")

    # lecture
    entries = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    # tri
    entries.sort(key=id_key)

    # destination
    if args.in_place:
        tmp_path = path.with_suffix(".jsonl.tmp")
        out_path = path
    else:
        tmp_path = path.with_suffix(".sorted.jsonl")
        out_path = tmp_path

    # écriture
    with open(tmp_path, "w", encoding="utf-8") as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write("\n")

    # remplacement éventuel
    if args.in_place:
        bak_path = path.with_suffix(".bak")
        shutil.move(path, bak_path)   # sauvegarde
        shutil.move(tmp_path, path)   # nouvelle version
        print(f"✅ {path.name} trié (ancienne version → {bak_path.name})")
    else:
        print(f"✅ Fichier trié écrit → {out_path.name}")

print("\n🏁 Tous les fichiers sont maintenant ordonnés.")
