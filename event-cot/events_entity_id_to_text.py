import json
from pathlib import Path

def replace_entity_ids_and_sort(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = []

    for doc in data:
        id_to_text = {ent["id"]: ent["text"] for ent in doc.get("entities", [])}

        # Copier le doc pour ne pas modifier l'original
        new_doc = dict(doc)
        new_doc["events"] = []

        for event in doc.get("events", []):
            new_event = []
            for attr in event:
                new_occurrences = [
                    id_to_text.get(occ, f"[UNKNOWN_ID:{occ}]")
                    for occ in attr.get("occurrences", [])
                ]
                new_event.append({
                    "attribute": attr["attribute"],
                    "occurrences": new_occurrences
                })
            new_doc["events"].append(new_event)

        updated_data.append(new_doc)

    # Trier par nombre décroissant d'événements
    updated_data.sort(key=lambda x: len(x.get("events", [])), reverse=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier généré (trié par #événements) : {output_path}")

# === Nouveau dossier de sortie ===
OUTPUT_DIR = Path("event/datasets_textual_events")
OUTPUT_DIR.mkdir(exist_ok=True)

# === Traitement des fichiers ===
replace_entity_ids_and_sort("./datasets/train.json", OUTPUT_DIR / "train.json")
# replace_entity_ids_and_sort("./datasets/train.json", OUTPUT_DIR / "test_preds.json")
