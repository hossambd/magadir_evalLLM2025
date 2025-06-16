import json
import re
import difflib
from pathlib import Path
from collections import defaultdict

# === Chemins ===
ENTITIES_PATH = Path("../reconstructed_outputs/density_reconstructed.json")
EVENTS_PATH = Path("openai_outputs/events_outputs.jsonl")
OUTPUT_PATH = Path("final_outputs/entities_and_event_ids_enriched_matching.json")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Charger les entités ===
with open(ENTITIES_PATH, "r", encoding="utf-8") as f:
    entities_data = json.load(f)
doc_map = {doc["doc_id"]: doc for doc in entities_data}

# === Charger les prédictions d’événements ===
with open(EVENTS_PATH, "r", encoding="utf-8") as f:
    event_predictions = [json.loads(line) for line in f if line.strip()]

# === Utilitaires de matching ===

def match_exact(val_text, entity_map):
    return entity_map.get(val_text, [])

def match_regex(val_text, entity_map):
    results = []
    for key in entity_map:
        if val_text in key or key in val_text or re.search(rf'\b{re.escape(val_text)}\b', key):
            results.extend(entity_map[key])
    return results

def match_fuzzy(val_text, entity_map, cutoff=0.85):
    candidates = list(entity_map.keys())
    matches = difflib.get_close_matches(val_text, candidates, n=3, cutoff=cutoff)
    matched_ids = []
    for m in matches:
        matched_ids.extend(entity_map[m])
    return matched_ids

# === Traitement principal ===
final_docs = []

for pred in event_predictions:
    pred_id = pred.get("id")
    if not pred_id or "output" not in pred:
        continue

    doc_index = int(pred_id.replace("prompt_", ""))
    doc_id = f"doc_{doc_index}"
    doc = doc_map.get(doc_id)
    if not doc:
        continue

    entity_map = defaultdict(list)
    for ent in doc["entities"]:
        norm = ent["text"].strip().lower()
        entity_map[norm].append(ent["id"])

    structured_events = []
    for event in pred["output"].get("events", []):
        event_block = []
        for attr in event:
            attr_name = "evt:" + attr.get("attribute", "").strip()
            val_text = attr.get("value", "").strip().lower()

            ids = match_exact(val_text, entity_map)
            if not ids:
                ids = match_regex(val_text, entity_map)
            if not ids:
                ids = match_fuzzy(val_text, entity_map)

            if ids:
                event_block.append({
                    "attribute": attr_name,
                    "occurrences": sorted(set(ids))
                })

        if event_block:
            structured_events.append(event_block)

    final_docs.append({
        "doc_id": doc_id,
        "text": doc["text"],
        "entities": doc["entities"],
        "events": structured_events
    })

# === Sauvegarde du fichier enrichi ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_docs, f, ensure_ascii=False, indent=2)

print(f"\n✅ Fichier enrichi généré : {OUTPUT_PATH.name}")
