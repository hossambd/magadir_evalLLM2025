#!/usr/bin/env python3
"""
Télécharge les résultats de batchs OpenAI à partir du mapping
input_file_id → batch_id enregistré dans input_to_batch.json.

Le script :
1. Charge chaque meta-fichier *_file_info.json pour retrouver la stratégie
   et le numéro de part (ex. retail_part3).
2. Récupère le batch (via batch_id), attend qu'il soit terminé si besoin.
3. Télécharge le output_file_id et écrit/concatène les réponses dans
   ner/batch_results/<strategy>_outputs.jsonl
"""

import json
import time
from pathlib import Path
from openai import OpenAI

# ── PARAMÈTRES ───────────────────────────────────────────────────────────────
BASE_DIR           = Path("event")                        # racine de vos données
BATCH_DIR          = BASE_DIR / "openai_outputs"  # même que précédemment
META_GLOB          = "*_file_info.json"                 # fichiers créés par prepare_batches.py
MAPPING_FILE       = BATCH_DIR / "input_to_batch.json"  # créé par map_input_to_batch.py
RESULTS_DIR        = BASE_DIR / "batch_results"         # nouveau dossier résultats
POLL_DELAY_SECONDS = 60                                 # si on doit attendre

# ── INIT ─────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
client = OpenAI()
TERMINAL = {"completed", "failed", "cancelled", "expired"}

# ── CHARGER LE MAPPING ───────────────────────────────────────────────────────
if not MAPPING_FILE.exists():
    raise SystemExit(f"❌ {MAPPING_FILE} introuvable. Lancez map_input_to_batch.py d'abord.")

mapping = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
print(f"🔗 {len(mapping)} couples input_file_id → batch_id chargés.")

# ── INDEXER LES META-FICHIERS POUR OBTENIR LES STRATÉGIES ───────────────────
meta_by_file_id = {}
for p in BATCH_DIR.glob(META_GLOB):
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta_by_file_id[meta["file_id"]] = meta       # contient strategy + part

# ── FONCTIONS UTILITAIRES ───────────────────────────────────────────────────
def wait_for_batch(batch_id: str):
    while True:
        b = client.batches.retrieve(batch_id)
        if b.status in TERMINAL:
            return b
        print(f"⏳ Batch {batch_id} status {b.status}… nouvelle vérif dans {POLL_DELAY_SECONDS}s")
        time.sleep(POLL_DELAY_SECONDS)

def parse_record(record: dict):
    """
    Extrait (prompt_id, output|error) du format Batch v1
    """
    pid = record.get("custom_id")
    body = record.get("response", {}).get("body")
    if not (pid and body):
        return pid, {"error": "missing_body"}

    try:
        txt = body["output"][0]["content"][0]["text"]
        parsed = json.loads(txt)
        return pid, {"output": parsed}
    except Exception as e:
        return pid, {"error": f"parse_error: {e}"}

# ── TRAITEMENT DE CHAQUE BATCH ───────────────────────────────────────────────
for input_file_id, batch_id in mapping.items():

    # ---- infos de contexte ----
    meta = meta_by_file_id.get(input_file_id)
    if not meta:
        print(f"⚠️  Pas de meta pour file_id {input_file_id} – ignoré.")
        continue
    strategy  = meta["strategy"]
    part      = meta["part"]
    tag       = f"{strategy}_part{part}"
    print(f"\n🔍 Téléchargement batch {tag} (batch_id={batch_id})")

    # ---- récupérer / attendre le batch ----
    batch = client.batches.retrieve(batch_id)
    if batch.status not in TERMINAL:
        print(f"⏳ Batch encore {batch.status} – on attend...")
        batch = wait_for_batch(batch_id)

    if batch.status != "completed":
        print(f"❌ Batch {tag} terminé mais non complété : {batch.status}")
        continue

    output_fid = batch.output_file_id
    if not output_fid:
        print(f"⚠️  Pas de output_file_id pour {tag}")
        continue

    # ---- télécharger le fichier de sortie ----
    file_stream = client.files.content(output_fid)
    results = []
    for line in file_stream.iter_lines():
        line = line.decode("utf-8") if isinstance(line, bytes) else line
        pid, content = parse_record(json.loads(line))
        if pid:
            entry = {"id": pid}
            entry.update(content)   # merge output|error
            results.append(entry)

    # ---- écrire / concaténer par stratégie ----
    out_path = RESULTS_DIR / f"{strategy}_outputs.jsonl"
    mode = "a" if out_path.exists() else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ {len(results)} réponses ajoutées → {out_path}")

print("\n🏁 Téléchargement terminé pour tous les batchs.")
