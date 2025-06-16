import json

def split_text_with_entities(document, doc_index):
    original_text = document['text']
    entities = document['entities']

    # Découpe par saut de ligne
    segments = original_text.split('\n')
    results = []
    current_offset = 0  # position globale dans le texte original

    for segment in segments:
        segment_len = len(segment)
        segment_start = current_offset
        segment_end = current_offset + segment_len

        # Collecter les entités qui tombent dans ce segment
        local_entities = []
        for ent in entities:
            for s, e in zip(ent['start'], ent['end']):
                if segment_start <= s < segment_end:
                    # Recalcule les offsets localement pour ce segment
                    new_start = s - segment_start
                    new_end = e - segment_start
                    local_entities.append({
                        "text": original_text[s:e],
                        "start": [new_start],
                        "end": [new_end],
                        "label": ent["label"],
                        "id": ent["id"]
                    })

        # Ajouter le segment seulement s’il n’est pas vide
        if segment.strip():
            results.append({
                "doc_id": f"doc_{doc_index}",
                "text": segment,
                "entities": local_entities
            })

        # Avancer le curseur dans le texte (ajouter 1 pour le \n consommé)
        current_offset = segment_end + 1

    return results

# === Exemple d'utilisation ===
input_path = "datasets/test.json"
output_path = "test_segments.json"

with open(input_path, "r") as f:
    data = json.load(f)

split_data = []
for i, doc in enumerate(data):
    split_data.extend(split_text_with_entities(doc, i))

with open(output_path, "w") as f:
    json.dump(split_data, f, ensure_ascii=False, indent=2)

print(f"✅ Fichier '{output_path}' généré avec {len(split_data)} segments.")
