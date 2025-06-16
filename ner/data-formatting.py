import json
from pathlib import Path

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def prepare_prompt(example):
    text = example["text"]
    entities = example.get("entities", [])

    formatted_entities = []
    for ent in entities:
        ent_text = ent["text"]
        ent_label = ent["label"]
        formatted_entities.append(f"- {ent_text} ({ent_label})")

    entity_block = "\n".join(formatted_entities)
    prompt = f"Texte: \"{text.strip()}\"\nEntités:\n{entity_block}\n"
    return prompt

def process_file(filepath):
    data = load_json(filepath)
    return [prepare_prompt(example) for example in data]

if __name__ == "__main__":
    base_dir = Path("ner/demo_datasets")
    json_files = sorted(base_dir.glob("*.json"))

    for json_file in json_files:
        prompts = process_file(json_file)
        output_path = json_file.with_suffix(".txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for prompt in prompts:
                f.write(prompt + "\n" + "-" * 80 + "\n")

        print(f"✅ Prompts saved to {output_path}")
