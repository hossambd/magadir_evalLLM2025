#!/usr/bin/env python3
"""
Balaye les fichiers *_file_info.json (créés par prepare_batches.py),
retrouve pour chacun le batch auquel il a été associé, puis
écrit un mapping {input_file_id: batch_id} dans input_to_batch.json.
"""

import json
from pathlib import Path
from openai import OpenAI

# ── PARAMÈTRES ───────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("event/openai_outputs")   # même que dans les autres scripts
MAPPING_FILE = OUTPUT_DIR / "input_to_batch.json"     # fichier de sortie
PAGE_SIZE    = 100                                    # pagination API

# ── INIT ─────────────────────────────────────────────────────────────────────
client = OpenAI()

# ── LECTURE DES input_file_id ───────────────────────────────────────────────
file_meta_paths = sorted(OUTPUT_DIR.glob("*_file_info.json"))
if not file_meta_paths:
    raise SystemExit("❌ Aucun *_file_info.json trouvé – lancez d’abord prepare_batches.py")

input_ids = []
for p in file_meta_paths:
    meta = json.loads(p.read_text(encoding="utf-8"))
    input_ids.append(meta["file_id"])

print(f"🔍 {len(input_ids)} input_file_id à mapper…")

# ── RÉCUPÉRATION DES BATCHES EXISTANTS ──────────────────────────────────────
# On charge tous les batchs de l’orga (paginated) et on les range par input_file_id
batches_by_input = {}
cursor = None
while True:
    resp = client.batches.list(limit=PAGE_SIZE, after=cursor)
    for b in resp.data:
        batches_by_input[b.input_file_id] = b.id
    if resp.has_more:
        cursor = resp.data[-1].id
    else:
        break
print(f"📋 {len(batches_by_input)} batch(s) trouvés sur l’API.")

# ── CONSTITUTION DU MAPPING ─────────────────────────────────────────────────
mapping = {}
missing = []
for fid in input_ids:
    bid = batches_by_input.get(fid)
    if bid:
        mapping[fid] = bid
    else:
        missing.append(fid)

# ── SORTIE ───────────────────────────────────────────────────────────────────
with open(MAPPING_FILE, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)
print(f"💾 Mapping écrit → {MAPPING_FILE}")

if missing:
    print(f"⚠️  Aucun batch trouvé pour {len(missing)} fichier(s) :")
    for fid in missing:
        print("   •", fid)
    print("   (Le batch n’est peut-être pas encore lancé ou est supprimé.)")
