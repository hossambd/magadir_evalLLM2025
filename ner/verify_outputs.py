#!/usr/bin/env python3
"""
1. Pour chaque fichier <strategy>_reconstructed.json passé en argument :
   • signale les entités dont l'offset ne pointe pas sur le bon texte ;
   • supprime les doublons (même start et end) dans un même document,
     en gardant la première occurrence.
2. Écrit le fichier « _dedup.json » à côté de l'original.

Usage
-----
python verify_and_dedup_outputs.py ner/reconstructed_outputs/*_reconstructed.json
Retour code 0 si aucune erreur d'offset, 1 sinon.
"""

import json
import sys
from pathlib import Path
from collections import OrderedDict  # pour conserver l'ordre d'origine

def process_file(path: Path):
    with path.open(encoding="utf-8") as f:
        docs = json.load(f)

    offset_errors, total_dupes = 0, 0
    cleaned_docs = []

    for doc in docs:
        text = doc["text"]
        seen = set()
        new_entities = []

        for ent in doc.get("entities", []):
            start, end = ent["start"][0], ent["end"][0]

            # 1️⃣  Vérif de correspondance texte / offsets
            if text[start:end] != ent["text"]:
                offset_errors += 1
                continue                      # on écarte l'entité fautive

            # 2️⃣  Déduplication
            span = (start, end)
            if span in seen:
                total_dupes += 1              # doublon détecté → on saute
                continue
            seen.add(span)
            new_entities.append(ent)

        cleaned_doc = doc.copy()
        cleaned_doc["entities"] = new_entities
        cleaned_docs.append(cleaned_doc)

    # ── Écriture du fichier nettoyé ─────────────────────────────────────────
    out_path = path.with_name(path.stem + "_dedup.json")
    out_path.write_text(json.dumps(cleaned_docs, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    return offset_errors, total_dupes, out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_and_dedup_outputs.py <glob_or_files>")
        sys.exit(1)

    files = [Path(p) for arg in sys.argv[1:] for p in Path().glob(arg)]
    if not files:
        print("Aucun fichier correspondant.")
        sys.exit(1)

    global_errs, global_dupes = 0, 0
    for fp in files:
        err, dup, out_p = process_file(fp)
        global_errs  += err
        global_dupes += dup
        print(f"🗂️  {fp.name}  →  {out_p.name}   "
              f"[offset errors : {err} | doublons supprimés : {dup}]")

    # ── Résumé global ───────────────────────────────────────────────────────
    if global_errs:
        print(f"\n❌ Offsets incorrects détectés : {global_errs}")
        sys.exit(1)
    else:
        print(f"\n✅ Terminé. Doublons supprimés : {global_dupes}")


if __name__ == "__main__":
    main()
