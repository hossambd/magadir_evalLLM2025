import json
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# === CONFIGURATION ==========================================================
MODEL = "gpt-4.1"

# Choix robuste de l’encodage ------------------------------------------------
try:
    ENCODING = tiktoken.encoding_for_model(MODEL)
except KeyError:                             # gpt-4.1 pas encore mappé ?
    ENCODING = tiktoken.get_encoding("o200k_base")
# ----------------------------------------------------------------------------

TOKEN_LIMIT_PER_BATCH = 1_200_000           # marge de sécurité
POLL_DELAY_SECONDS   = 240                   # délai entre deux checks

PROMPT_ROOT_DIR = Path("ner/generated_prompts")
OUTPUT_DIR      = Path("ner/openai_outputs_batches_2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_INPUT_DIR = OUTPUT_DIR / "batch_inputs"
BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI()

# === JSON Schema attendu ====================================================
NER_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string"}
                },
                "required": ["text", "label"],
                "additionalProperties": False
            }
        }
    },
    "required": ["entities"],
    "additionalProperties": False
}

# === OUTILS =================================================================
TERMINAL_STATUS = {"completed", "failed", "cancelled", "expired"}

def wait_until_done(batch_id: str, sleep_s: int = POLL_DELAY_SECONDS) -> str:
    """Interroge l’API jusqu’à ce que le batch passe dans un état terminal.
       Retourne le statut final."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        if status in TERMINAL_STATUS:
            print(f"✅ Batch {batch_id} terminé – status: {status}")
            return status
        print(f"⏳ Batch {batch_id} toujours {status}… nouvelle vérification dans {sleep_s}s")
        time.sleep(sleep_s)

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
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant NER. Réponds uniquement avec un JSON du type : "
                        "{\"entities\": [{\"text\": ..., \"label\": ...}]}"
                    )
                },
                {"role": "user", "content": prompt_text}
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ner_response",
                    "schema": NER_SCHEMA,
                    "strict": True
                }
            }
        }
    }

# === TRAITEMENT PAR DOSSIER DE STRATÉGIE ====================================
def process_strategy_folder_batch(strategy_dir: Path, strategy_name: str):
    prompts = sorted(strategy_dir.glob("prompt_*.txt"))
    print(f"\n📂 {strategy_name} : {len(prompts)} prompts trouvés")

    batches = []
    current_batch, token_count = [], 0

    # Découpage en batches <= TOKEN_LIMIT_PER_BATCH --------------------------
    for prompt_file in tqdm(prompts, desc=f"🧮 Découpage batchs {strategy_name}"):
        prompt_text = prompt_file.read_text(encoding="utf-8")
        prompt_id   = prompt_file.stem
        tokens      = count_tokens(prompt_text)
        request     = build_batch_request(prompt_text, prompt_id)

        if token_count + tokens > TOKEN_LIMIT_PER_BATCH:
            batches.append(current_batch)
            current_batch, token_count = [], 0

        current_batch.append(request)
        token_count += tokens

    if current_batch:
        batches.append(current_batch)

    print(f"🔢 {len(batches)} batch(s) généré(s) pour {strategy_name}")

    # Lancement SÉQUENTIEL des batches --------------------------------------
    for i, batch_lines in enumerate(batches, start=1):
        batch_name       = f"{strategy_name}_part{i}"
        batch_input_path = BATCH_INPUT_DIR / f"{batch_name}.jsonl"

        # 1. Écriture du .jsonl
        with open(batch_input_path, "w", encoding="utf-8") as f:
            for line in batch_lines:
                json.dump(line, f, ensure_ascii=False)
                f.write("\n")
        print(f"📄 Batch {i} écrit : {batch_input_path}")

        # 2. Upload du fichier
        uploaded_file = client.files.create(
            file=open(batch_input_path, "rb"),
            purpose="batch"
        )
        print(f"📤 Fichier uploadé – ID : {uploaded_file.id}")

        # 3. Création du batch
        batch = client.batches.create(
            input_file_id   = uploaded_file.id,
            endpoint        = "/v1/responses",
            completion_window = "24h",
            metadata        = {"strategy": strategy_name, "part": str(i)}
        )
        print(f"🚀 Batch {i} lancé – ID : {batch.id}")

        # 4. Sauvegarde meta locale
        batch_info_path = OUTPUT_DIR / f"{batch_name}_batch_info.json"
        with open(batch_info_path, "w", encoding="utf-8") as f:
            json.dump({
                "strategy": strategy_name,
                "batch_index": i,
                "batch_id": batch.id,
                "input_file_id": uploaded_file.id
            }, f, ensure_ascii=False, indent=2)
        print(f"💾 Infos batch sauvegardées → {batch_info_path}")

        # 5. *** ATTENTE JUSQU’À FIN DU BATCH ***
        wait_until_done(batch.id)        # ← blocage ici tant que le batch n’est pas terminé

# === POINT D’ENTRÉE =========================================================
def main():
    strategy_dirs = [d for d in PROMPT_ROOT_DIR.iterdir() if d.is_dir()]
    if not strategy_dirs:
        print("❌ Aucun dossier trouvé dans generated_prompts/.")
        return

    for strategy_dir in strategy_dirs:
        process_strategy_folder_batch(strategy_dir, strategy_dir.name)

if __name__ == "__main__":
    main()
