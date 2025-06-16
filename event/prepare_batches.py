#!/usr/bin/env python3
"""
Découpe les prompts en lots, génère les .jsonl et
uploade chaque fichier sur OpenAI (purpose="batch").
Ne crée PAS les batchs : chaque lot reste prêt à être lancé.
"""

import json
from pathlib import Path
from tqdm import tqdm
import tiktoken
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL = "gpt-4.1"
TOKEN_LIMIT_PER_BATCH = 1_200_000   # marge
PROMPT_ROOT_DIR = Path("event/generated_prompts/events")
OUTPUT_DIR = Path("event/openai_outputs")
BATCH_INPUT_DIR = OUTPUT_DIR / "batch_inputs"

# ── INIT ─────────────────────────────────────────────────────────────────────
try:
    ENCODING = tiktoken.encoding_for_model(MODEL)
except KeyError:
    ENCODING = tiktoken.get_encoding("o200k_base")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)
client = OpenAI()

# ── SCHÉMA JSON ──────────────────────────────────────────────────────────────
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "attribute": {"type": "string"},
                        "value": {"type": "string"}
                    },
                    "required": ["attribute", "value"],
                    "additionalProperties": False
                }
            }
        }
    },
    "required": ["events"],
    "additionalProperties": False
}

# ── UTILS ────────────────────────────────────────────────────────────────────
def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

def build_batch_request(prompt_text: str, prompt_id: str) -> dict:
    return {
        "custom_id": prompt_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": [
                {"role": "system",
                "content": "Tu es un assistant d'extraction d'événements. Tu dois retourner un JSON de la forme : {\"events\": [[{\"attribute\": ..., \"value\": ...}, ...], ...]}"
                    },
                {"role": "user", "content": prompt_text}
            ],
            "text": {"format": {
                    "type": "json_schema",
                    "name": "event_response",
                    "schema": EVENT_SCHEMA,
                    "strict": True
                }}
        }
    }

# ── TRAITEMENT ───────────────────────────────────────────────────────────────
def prepare_one_strategy(strategy_dir: Path):
    strategy = strategy_dir.name
    prompts  = sorted(strategy_dir.glob("prompt_*.txt"))
    print(f"\n📂 {strategy}: {len(prompts)} prompts trouvés")

    batches, cur_batch, token_sum = [], [], 0
    for pf in tqdm(prompts, desc=f"Découpage {strategy}"):
        txt   = pf.read_text(encoding="utf-8")
        req   = build_batch_request(txt, pf.stem)
        tokens = count_tokens(txt)
        if token_sum + tokens > TOKEN_LIMIT_PER_BATCH:
            batches.append(cur_batch)
            cur_batch, token_sum = [], 0
        cur_batch.append(req)
        token_sum += tokens
    if cur_batch:
        batches.append(cur_batch)

    print(f"🔢 {len(batches)} lot(s) généré(s)")

    for idx, lines in enumerate(batches, start=1):
        batch_name = f"{strategy}_part{idx}"
        jsonl_path = BATCH_INPUT_DIR / f"{batch_name}.jsonl"

        # 1. Écriture
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for line in lines:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")
        print(f"📄 {jsonl_path} écrit")

        # 2. Upload (mais pas de batch !)
        file_obj = client.files.create(file=open(jsonl_path, "rb"),
                                       purpose="batch")
        print(f"📤 Upload OK — file_id {file_obj.id}")

        # 3. Sauvegarde meta (servira à launch_batches.py)
        meta_path = OUTPUT_DIR / f"{batch_name}_file_info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "batch_name": batch_name,
                "file_id": file_obj.id,
                "strategy": strategy,
                "part": idx
            }, f, indent=2, ensure_ascii=False)
        print(f"💾 Meta sauvegardé → {meta_path}")

def main():
    dirs = [d for d in PROMPT_ROOT_DIR.iterdir() if d.is_dir()]
    if not dirs:
        print("❌ Aucun dossier dans generated_prompts/")
        return
    for d in dirs:
        prepare_one_strategy(d)

if __name__ == "__main__":
    main()
