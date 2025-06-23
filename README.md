
# MAGADIR EvalLLM2025

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

### ✅ Event Detection with CoT

```bash
cd event-cot/

```
