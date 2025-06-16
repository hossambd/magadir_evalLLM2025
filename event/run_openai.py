import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# === CONFIGURATION ===
MODEL = "gpt-4.1"
BASE_PROMPT_DIR = Path("event/generated_prompts/events")
OUTPUT_BASE_PATH = Path("event/openai_outputs")
OUTPUT_BASE_PATH.mkdir(parents=True, exist_ok=True)

client = OpenAI()

# === SCHEMA attendu pour les événements ===
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

def extract_structured_events(prompt_text: str, prompt_id: str) -> dict:
    try:
        response = client.responses.create(
            model=MODEL,
            input=[
                {
                    "role": "system",
                    "content": "Tu es un assistant d'extraction d'événements. Tu dois retourner un JSON de la forme : {\"events\": [[{\"attribute\": ..., \"value\": ...}, ...], ...]}"
                },
                {"role": "user", "content": prompt_text}
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "event_response",
                    "schema": EVENT_SCHEMA,
                    "strict": True
                }
            }
        )
        structured_output = json.loads(response.output_text)
        return {
            "id": prompt_id,
            "output": structured_output
        }
    except Exception as e:
        return {
            "id": prompt_id,
            "error": str(e)
        }

def run_on_folder(k_val: int):
    prompt_dir = BASE_PROMPT_DIR / f"events_k{k_val}"
    output_path = OUTPUT_BASE_PATH / f"events_outputs_k{k_val}.jsonl"

    if not prompt_dir.exists():
        print(f"⚠️ Dossier introuvable pour k={k_val} : {prompt_dir}")
        return

    prompts = sorted(prompt_dir.glob("prompt_*.txt"))
    print(f"\n📂 {len(prompts)} prompts trouvés dans '{prompt_dir}/'")

    results = []
    for prompt_file in tqdm(prompts, desc=f"🔍 Extraction événements (k={k_val})"):
        prompt_text = prompt_file.read_text(encoding="utf-8")
        prompt_id = prompt_file.stem
        result = extract_structured_events(prompt_text, prompt_id)
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ Résultats enregistrés pour k={k_val} dans : {output_path.name}")

def main():
    K_VALUES = [4, 6, 8]  # adapte ici si besoin
    for k in K_VALUES:
        run_on_folder(k)

if __name__ == "__main__":
    main()
