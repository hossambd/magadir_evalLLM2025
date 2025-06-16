import json
from collections import Counter
from pathlib import Path

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_entity_labels(data):
    """Retourne un Counter global pour tous les labels"""
    label_counter = Counter()
    for item in data:
        labels = [ent['label'] for ent in item['entities']]
        label_counter.update(labels)
    return label_counter

def score_instance(instance, global_counts):
    """Calcule différents critères pour l'ordonnancement"""
    labels = [ent['label'] for ent in instance['entities']]
    unique_labels = set(labels)
    text_len = len(instance['text'])

    # Stratégie 1 : diversité de labels
    label_diversity = len(unique_labels)

    # Stratégie 2 : rareté inverse (on somme les 1/fréquence pour chaque label)
    rarity_score = sum(1 / global_counts[label] for label in unique_labels)

    # Stratégie 3 : densité d'entités
    density = len(labels) / text_len if text_len > 0 else 0

    return {
        "label_diversity": label_diversity,
        "rarity_score": rarity_score,
        "density": density
    }

def reorder_dataset(data, strategy, global_counts):
    scored = []

    for item in data:
        scores = score_instance(item, global_counts)
        item['_scores'] = scores
        scored.append(item)

    # Tri selon la stratégie choisie
    if strategy == "diversity":
        scored.sort(key=lambda x: x['_scores']['label_diversity'], reverse=True)
    elif strategy == "rarity":
        scored.sort(key=lambda x: x['_scores']['rarity_score'], reverse=True)
    elif strategy == "density":
        scored.sort(key=lambda x: x['_scores']['density'], reverse=True)
    elif strategy == "combined":
        scored.sort(key=lambda x: (
            x['_scores']['label_diversity'] * 2 +
            x['_scores']['rarity_score'] * 3 +
            x['_scores']['density']
        ), reverse=True)

    # Nettoyage
    for item in scored:
        item.pop('_scores', None)

    return scored

def save_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# === Exécution principale ===
if __name__ == "__main__":
    train_path = "datasets/train.json"
    output_dir = Path("ner/demo_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(train_path)
    global_counts = count_entity_labels(data)

    strategies = ["diversity", "density"]

    for strategy in strategies:
        ordered_data = reorder_dataset(data, strategy, global_counts)
        output_path = output_dir / f"{strategy}_train.json"
        save_data(ordered_data, output_path)
        print(f"✅ Fichier '{output_path.name}' créé avec {len(ordered_data)} exemples.")
