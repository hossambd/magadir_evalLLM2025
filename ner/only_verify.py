#!/usr/bin/env python3
"""
Vérifie deux contraintes dans l’ensemble de vos fichiers de sortie :

1) l’extrait text[start:end] doit être identique à ent["text"]
2) il ne doit pas exister de doublon (mêmes start & end) dans un même doc

Usage :
    python verify_outputs.py ner/reconstructed_outputs/*_reconstructed.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def check_file(path: Path):
    with path.open("r", encoding="utf-8") as f:
        docs = json.load(f)

    errors     = []
    duplicates = []

    for doc in docs:
        text      = doc["text"]
        doc_id    = doc.get("doc_id", "UNKNOWN")
        seen_spans = set()  # pour détecter les doublons

        for ent in doc.get("entities", []):
            start = ent["start"][0]
            end   = ent["end"][0]
            span  = (start, end)

            # ---- 1) correspondance texte / offsets ------------------------
            if text[start:end] != ent["text"]:
                errors.append(
                    (doc_id, start, end, ent["text"], text[start:end])
                )

            # ---- 2) doublons (même span) ----------------------------------
            if span in seen_spans:
                duplicates.append((doc_id, start, end, ent["text"]))
            else:
                seen_spans.add(span)

    return errors, duplicates


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_outputs.py <glob_or_files>")
        sys.exit(1)

    files = [Path(p) for arg in sys.argv[1:] for p in Path().glob(arg)]
    if not files:
        print("Aucun fichier correspondant.")
        sys.exit(1)

    total_err, total_dup = 0, 0

    for fp in files:
        errs, dups = check_file(fp)
        total_err += len(errs)
        total_dup += len(dups)

        if errs or dups:
            print(f"\n❌ Problèmes dans {fp}:")
            for d in errs[:5]:
                print(f"   • Offset mismatch doc={d[0]} [{d[1]}:{d[2]}]")
                print(f"     ent_text='{d[3]}' | text_slice='{d[4]}'")
            for d in dups[:5]:
                print(f"   • Doublon doc={d[0]} span=({d[1]},{d[2]}) '{d[3]}'")
            if len(errs) > 5 or len(dups) > 5:
                print(f"   … {len(errs)+len(dups)-10} autres problème(s) omis")

    # ----- Récapitulatif global ---------------------------------------------
    if total_err or total_dup:
        print("\n=== RÉCAP ===")
        print(f"Offsets incorrects : {total_err}")
        print(f"Doublons           : {total_dup}")
        sys.exit(1)
    else:
        print("✅ Tous les fichiers sont conformes.")


if __name__ == "__main__":
    main()
