
# MAGADIR EvalLLM2025

**MAGADIR** est un système complet pour la reconnaissance d'entités nommées (NER) et la détection d’événements dans des documents de santé publique ou de défense, en s’appuyant sur des modèles de langage (LLMs). Il a été développé dans le cadre du challenge EvalLLM 2025.

## 🧩 Project Structure

```
magadir_evalLLM2025-main/
├── ner/                            # NER System pipeline
│   ├── text_to_segments.py
│   ├── demo_preparation.py
│   ├── data_formatting.py
│   ├── generate_prompt_batches.py
│   ├── run_openai.py
│   ├── transform_openai_predictions.py
│   └── reconstruct_prediction.py
│
├── event/                          # Event detection pipeline
│   ├── events_entity_id_to_text.py
│   ├── generate_events_prompts.py
│   ├── run_openai.py
│   └── rebuild_events_with_entity_ids.py
│
├── matching/                       # Exact matching and evaluation
│   └── evaluation_tools_events.py
│
└── README.md                       # Project documentation
```

## 🚀 Setup

1. Clone or unzip the repository:

```bash
unzip magadir_evalLLM2025-main.zip
cd magadir_evalLLM2025-main
```

2. (Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> Si `requirements.txt` est absent, installe les bibliothèques suivantes : `openai`, `tqdm`, `jsonlines`, `pandas`, etc.

## ⚙️ Usage

### 🔹 NER Pipeline

```bash
cd ner/

python text_to_segments.py
python demo_preparation.py
python data_formatting.py
python generate_prompt_batches.py
python run_openai.py
python transform_openai_predictions.py
python reconstruct_prediction.py
```

### 🔸 Event Detection Pipeline

```bash
cd event/

python events_entity_id_to_text.py
python generate_events_prompts.py
python run_openai.py
python rebuild_events_with_entity_ids.py
```

### ✅ Exact Matching & Evaluation

```bash
cd matching/

python evaluation_tools_events.py --gold data/gold.jsonl --predicted data/predictions.jsonl
```
