#!/usr/bin/env python3
import json, re, uuid
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ==========================================================
SEGMENT_PATH        = Path("test_segments.json")
OPENAI_OUTPUTS_DIR  = Path("ner/batch_results_ordered")
ALIGNMENTS_DIR      = Path("ner/aligned_outputs_3")
ALIGNMENTS_DIR.mkdir(exist_ok=True)

# === Étape 1 : offsets locaux "gold" (identique) ============================
with SEGMENT_PATH.open("r", encoding="utf-8") as f:
    test_segments = json.load(f)

for seg in test_segments:
    for ent in seg["entities"]:
        m = re.search(re.escape(ent["text"]), seg["text"])
        if m:
            ent["start"], ent["end"] = [m.start()], [m.end()]

Path("test_segments_local_offsets.json").write_text(
    json.dumps(test_segments, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print("✅ Offsets locaux recalculés pour test_segments.")

# === Étape 2 : aligner les sorties ==========================================
jsonl_files = sorted(OPENAI_OUTPUTS_DIR.glob("*_outputs.jsonl"))
if not jsonl_files:
    raise SystemExit("❌ Aucun *_outputs.jsonl dans ner/batch_results_ordered/")

def next_after(text, segment, start_from):
    """Renvoie la 1ʳᵉ occurrence de ‹text› à partir de start_from, sinon None."""
    m = re.search(re.escape(text), segment[start_from:])
    if not m:
        return None
    return m.start() + start_from, m.end() + start_from

for file_path in jsonl_files:
    strategy = file_path.stem.replace("_outputs", "")
    out_path = ALIGNMENTS_DIR / f"{strategy}_with_offsets.json"

    preds = [json.loads(l) for l in file_path.read_text(encoding="utf-8").splitlines()]
    aligned = []

    for i, pred in enumerate(tqdm(preds, desc=f"🔁 Align {strategy}")):
        seg_gold   = test_segments[i]           # correspondance index → segment
        txt_seg    = seg_gold["text"]

        cur_pos = 0                             # où commence la recherche
        entities = []

        for ent in pred.get("output", {}).get("entities", []):
            span = next_after(ent["text"], txt_seg, cur_pos)
            if not span:
                # rien après cur_pos ⇒ on réessaye depuis le début pour sauver
                span = next_after(ent["text"], txt_seg, 0)
                if not span:                    # vraiment introuvable : on skip
                    continue
            start, end = span
            cur_pos = end                       # on décale le pointeur

            entities.append({
                "text":  ent["text"],
                "start": [start],
                "end":   [end],
                "label": ent["label"],
                "id":    str(uuid.uuid4())
            })

        aligned.append({
            "doc_id":  seg_gold["doc_id"],
            "text":    txt_seg,
            "entities": entities
        })

    out_path.write_text(json.dumps(aligned, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"✅ Alignement enregistré → {out_path}")
