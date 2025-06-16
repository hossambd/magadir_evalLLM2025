import json
from pathlib import Path
from collections import defaultdict

# === Chemins d'entrée ===
INPUT_DIR = Path("ner/aligned_outputs_3")
OUTPUT_DIR = Path("ner/reconstructed_outputs_2")
ORIGINAL_DOC_PATH = Path("datasets/test.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger les documents originaux pour retrouver les textes complets
with open(ORIGINAL_DOC_PATH, "r", encoding="utf-8") as f:
    original_docs = {f"doc_{i}": doc for i, doc in enumerate(json.load(f))}

def recompose_predictions_with_alignment(segmented_data):
    docs = defaultdict(list)
    for segment in segmented_data:
        docs[segment["doc_id"]].append(segment)

    reconstructed = []

    for doc_id, segments in docs.items():
        original_text = original_docs[doc_id]["text"]
        segments_sorted = sorted(segments, key=lambda s: segmented_data.index(s))

        segments_sorted = segments

        # print(f"**************************")
        # print(f"Original Segment {segments}")
        # print(f"Ordered Segments: {segments_sorted}")

        current_offset = 0
        entities = []

        for segment in segments_sorted:
            seg_text = segment["text"]
            # print("***********")
            # print(f"Orignial text: {original_text}")
            # print(f"Segmented text: {seg_text}")
            # print(f"current_offset: {current_offset}")
            found_at = original_text.find(seg_text, current_offset)
            # print(f"Found at: {found_at}")
            if found_at == -1:
                raise ValueError(f"⚠️ Segment introuvable dans le texte original :\n{seg_text[:50]}...")

            # Recalibrer les entités avec les bons offsets globaux
            for ent in segment.get("entities", []):
                new_ent = {
                    "text": ent["text"],
                    "start": [s + found_at for s in ent["start"]],
                    "end": [e + found_at for e in ent["end"]],
                    "label": ent["label"],
                    "id": ent["id"]
                }
                entities.append(new_ent)

            current_offset = found_at + len(seg_text)

        reconstructed.append({
            "doc_id": doc_id,
            "text": original_text,
            "entities": entities
        })

    return reconstructed

# === Traitement ===
for json_file in sorted(INPUT_DIR.glob("*_with_offsets.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        segmented_data = json.load(f)

    output = recompose_predictions_with_alignment(segmented_data)
    output_path = OUTPUT_DIR / json_file.name.replace("_with_offsets.json", "_reconstructed.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Fichier reconstruit : {output_path}")
