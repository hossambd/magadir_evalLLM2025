import json
from pathlib import Path
import random

# === Paramètres ===
MAIN_PROMPT_PATH = "event/prompt_elements/main.txt"
TRAIN_JSON_PATH = "event/datasets_textual_events/train.json"
TEST_JSON_PATH = "./datasets/test.json"
OUTPUT_BASE_DIR = Path("event/generated_prompts/events")
K_VALUES = [4, 8]  # différentes valeurs de few-shot

# === Fonctions ===

def load_main_prompt():
    with open(MAIN_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_fewshot_example(example):
    formatted = f'Texte: "{example["text"].strip()}"\nÉvénements:\n'
    for i, event in enumerate(example.get("events", []), 1):
        lines = []
        for attr in event:
            occurrences = attr.get("occurrences", [])
            first_occ = occurrences[0].strip() if occurrences else "[EMPTY]"
            lines.append(f"  - {attr['attribute']}: {first_occ}")
        formatted += f"  Event {i}:\n" + "\n".join(lines) + "\n"
    return formatted.strip()

def format_target_prompt(example):
    return f'Texte: "{example["text"].strip()}"\nÉvénements:'

def generate_prompt(main_prompt, fewshot_examples, target_text):
    dashed = "-" * 80
    fewshot_block = f"\n{dashed}\n".join(fewshot_examples) if fewshot_examples else ""
    return f"{main_prompt}\n\n{fewshot_block}\n\n{dashed}\n{target_text}"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_prompts(all_prompts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(all_prompts):
        output_file = output_dir / f"prompt_{i:03}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(prompt)
    print(f"✅ {len(all_prompts)} prompts saved in '{output_dir}/'.")

# === Exécution principale ===
if __name__ == "__main__":
    main_prompt = load_main_prompt()
    train_data = load_json(TRAIN_JSON_PATH)
    test_data = load_json(TEST_JSON_PATH)
    candidates = [ex for ex in train_data if ex.get("events")]

    for k in K_VALUES:
        print(f"\n🔧 Génération des prompts pour k={k}")
        if k > len(candidates):
            print(f"⚠️ Pas assez d'exemples pour k={k}. Skipping.")
            continue

        fewshot_raw = random.sample(candidates, k=k)
        fewshot_formatted = [format_fewshot_example(ex) for ex in fewshot_raw]

        prompts = []
        for example in test_data:
            target = format_target_prompt(example)
            prompt = generate_prompt(main_prompt, fewshot_formatted, target)
            prompts.append(prompt)

        save_prompts(prompts, OUTPUT_BASE_DIR / f"events_k{k}")
