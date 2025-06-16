import json
from pathlib import Path

# === Paramètres ===
MAIN_PROMPT_PATH = "ner/prompt_elements/main_prompt.txt"
TRAIN_DIR = Path("ner/demo_datasets")
TARGET_JSON_PATH = "test_segments.json"
OUTPUT_DIR = Path("ner/generated_prompts")
K_RANGE = [4, 6, 8]

# === Fonctions ===
def load_main_prompt():
    with open(MAIN_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_fewshot_examples(path, k):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        examples = content.split("-" * 80)
        return examples[:k]

def format_target_prompt(example):
    text = example["text"].strip()
    return f'Texte: "{text}"\nEntités:'

def generate_prompt(main_prompt, fewshot_block, target_text):
    return f"{main_prompt}\n\n{fewshot_block}\n\n{'-'*80}\n{target_text}"

def load_target_data():
    with open(TARGET_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_prompts(all_prompts, strategy_k):
    out_dir = OUTPUT_DIR / strategy_k
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(all_prompts):
        output_file = out_dir / f"prompt_{i:03}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(prompt)
    print(f"✅ {len(all_prompts)} prompts saved in '{out_dir}/'.")

# === Exécution principale ===
if __name__ == "__main__":
    main_prompt = load_main_prompt()
    target_data = load_target_data()

    for train_file in TRAIN_DIR.glob("*_train.txt"):
        strategy = train_file.stem.replace("_train", "")

        for k in K_RANGE:
            fewshot_examples = load_fewshot_examples(train_file, k)
            fewshot_block = "\n" + ("-" * 80 + "\n").join(fewshot_examples)

            all_prompts = []
            for example in target_data:
                target = format_target_prompt(example)
                prompt = generate_prompt(main_prompt, fewshot_block, target)
                all_prompts.append(prompt)

            strategy_k = f"{strategy}_k{k}"
            save_prompts(all_prompts, strategy_k)
