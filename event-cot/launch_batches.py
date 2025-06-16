#!/usr/bin/env python3
"""
Lance un ou plusieurs batchs OpenAI à partir de leur nom
(ex. python launch_batches.py retail_part2 legal_part3).
Les meta-fichiers _file_info.json doivent exister (générés par prepare_batches.py).
"""

import json
import sys
import time
from pathlib import Path
import tiktoken
from openai import OpenAI

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODEL = "gpt-4.1"
POLL_DELAY_SECONDS = 240
OUTPUT_DIR = Path("event-cot/openai_outputs")  # même dossier que précédemment

# ── INIT ─────────────────────────────────────────────────────────────────────
try:
    ENCODING = tiktoken.encoding_for_model(MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("o200k_base")

client = OpenAI()
TERMINAL = {"completed", "failed", "cancelled", "expired"}

# ── FONCTIONS ────────────────────────────────────────────────────────────────
def wait(batch_id: str):
    while True:
        b = client.batches.retrieve(batch_id)
        status = b.status
        if status in TERMINAL:
            print(f"✅ Batch {batch_id} terminé — {status}")
            return status
        print(f"⏳ Batch {batch_id} toujours {status}… (prochain check dans {POLL_DELAY_SECONDS}s)")
        time.sleep(POLL_DELAY_SECONDS)

def launch_one(batch_name: str):
    meta_path = OUTPUT_DIR / f"{batch_name}_file_info.json"
    if not meta_path.exists():
        print(f"⚠️  Meta-fichier introuvable : {meta_path}")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    file_id = meta["file_id"]

    # Création du batch (réservation des tokens)
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"batch_name": batch_name}
    )
    print(f"🚀 Batch lancé — id {batch.id}")
    # wait(batch.id)

# ── SCRIPT ───────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python launch_batches.py <batch_name1> [batch_name2 ...]")
        sys.exit(1)

    for name in sys.argv[1:]:
        launch_one(name)

if __name__ == "__main__":
    main()
