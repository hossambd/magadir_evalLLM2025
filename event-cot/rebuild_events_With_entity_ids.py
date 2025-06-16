import json
from pathlib import Path
from collections import defaultdict

# === Dossiers ===
RECONSTRUCTED_DIR = Path("ner/reconstructed_outputs")  # contient les fichiers d'entités
EVENTS_DIR = Path("event-cot/batch_results")                # contient les outputs d'inférence
OUTPUT_DIR = Path("event-cot/final_outputs")                 # où écrire les fichiers finaux
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Liste des valeurs de k à considérer ===
K_RANGE = [4,  8]  # adapter si besoin

# === Parcours de chaque fichier d'entités ===
for ent_file in sorted(RECONSTRUCTED_DIR.glob("*.json")):
    print(f"\n📂 Traitement fichier entités : {ent_file.name}")
    with open(ent_file, "r", encoding="utf-8") as f:
        entities_data = json.load(f)
    doc_map = {doc["doc_id"]: doc for doc in entities_data}

    print(f"✅ {len(doc_map)} documents trouvés.")

    # === Parcours de chaque valeur de k ===
    for k in K_RANGE:
        print(f"\n➡️ Traitement pour k = {k}")

        events_path = EVENTS_DIR / f"events_k{k}_outputs.jsonl"
        if not events_path.exists():
            print(f"❌ Fichier événements manquant : {events_path.name}")
            continue

        print(f"🔹 Lecture des événements depuis : {events_path}")
        event_predictions = []
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    event_predictions.append(json.loads(line))

        final_docs = []
        for pred in event_predictions:
            pred_id = pred.get("id")
            if not pred_id or "output" not in pred:
                print(f"⚠️ Prédiction ignorée (id ou output manquant).")
                continue

            # Convertir prompt_000 → doc_0
            if pred_id.startswith("prompt_"):
                try:
                    doc_index = int(pred_id.replace("prompt_", ""))
                    doc_id = f"doc_{doc_index}"
                except ValueError:
                    continue
            else:
                continue

            doc = doc_map.get(doc_id)
            if not doc:
                print(f"⚠️ Aucun document trouvé pour {doc_id}")
                continue

            # === Création du dictionnaire texte → ID
            entity_map = defaultdict(list)
            for ent in doc["entities"]:
                norm_text = ent["text"].strip().lower()
                entity_map[norm_text].append(ent["id"])

            # === Appariement des événements
            structured_events = []
            for event in pred["output"].get("events", []):
                event_block = []
                for attr in event:
                    attr_name = "evt:" + attr.get("attribute", "").strip()
                    val_text = attr.get("value", "").strip().lower()
                    matched_ids = entity_map.get(val_text, [])
                    if matched_ids:
                        event_block.append({
                            "attribute": attr_name,
                            "occurrences": matched_ids
                        })
                    else:
                        print(f"🔍 Aucun match pour '{val_text}' dans {doc_id}")
                if event_block:
                    structured_events.append(event_block)

            doc_with_events = {
                "doc_id": doc_id,
                "text": doc["text"],
                "entities": doc["entities"],
                "events": structured_events
            }
            final_docs.append(doc_with_events)
            print(f"✅ {doc_id} : {len(structured_events)} événements ajoutés.")

        # === Sauvegarde du résultat
        out_name = f"{ent_file.stem}_with_event_ids_k{k}.json"
        output_path = OUTPUT_DIR / out_name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_docs, f, ensure_ascii=False, indent=2)
        print(f"📁 Fichier sauvegardé : {output_path}")
