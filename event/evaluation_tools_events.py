"""Evaluation module for NER and event extraction tasks."""

import copy
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd
from nervaluate import Evaluator

# Define types
Entity = dict[str, str | int]
Event = list[dict[str, str | set]]
AnnotationData = list[dict[str, Any]]


@dataclass
class EvaluationConfig:
    """Configuration parameters for evaluation."""

    discontinuous_spans: bool = True
    eval_schema: str = "strict"
    completeness_threshold: float = 0.5
    strict_loading: bool = False


class DataHandler:
    """Centralized data handling."""

    @staticmethod
    def _load_raw_data(source: str | list[dict]) -> list[dict]:
        """Load raw data from file or raw data."""
        if isinstance(source, str):
            with open(source, encoding="utf-8") as f:
                return json.load(f)
        return copy.deepcopy(source)

    @staticmethod
    def load_entities(
        source: str | AnnotationData,
        config: EvaluationConfig,
    ) -> tuple[list[list[Entity]], list[str]]:
        """Import data and process them into a standard format."""
        data = DataHandler._load_raw_data(source)
        return DataHandler._process_entities(data, config)

    @staticmethod
    def _process_entities(
        data: AnnotationData,
        config: EvaluationConfig,
    ) -> tuple[list[list[Entity]], list[str]]:
        """Transform raw data into standard format for entity evaluation."""
        id_mapping = {}
        entities = []
        unique_tags = set()

        for doc in data:
            doc_entities = []
            for annotation in doc.get("entities", []):
                # Generate a unique ID for each entity
                key = f"{annotation['start']}-{annotation['end']}-{annotation['text']}"
                new_id = hashlib.sha256(key.encode()).hexdigest()[:8]
                id_mapping[annotation.get("id")] = new_id

                label = annotation["label"]
                unique_tags.add(label)

                if config.discontinuous_spans and len(annotation["start"]) > 1:
                    doc_entities.extend(
                        [
                            {"label": label, "start": start, "end": end}
                            for start, end in zip(
                                annotation["start"],
                                annotation["end"],
                                strict=False,
                            )
                        ],
                    )
                else:
                    doc_entities.append(
                        {
                            "label": label,
                            "start": annotation["start"][0],
                            "end": annotation["end"][-1],
                        },
                    )
            entities.append(doc_entities)

        return entities, list(unique_tags), id_mapping

    @staticmethod
    def load_events(source: str | AnnotationData, id_mapping: dict[str, str]) -> list[list[Event]]:
        """Import data and process them into a standard format."""
        data = DataHandler._load_raw_data(source)
        return DataHandler._process_events(data, id_mapping)

    @staticmethod
    def _process_events(data: AnnotationData, id_mapping: dict[str, str]) -> list[list[Event]]:
        """Transform raw data into standard format for event and document evaluation."""
        processed = []

        for doc in data:
            doc_events = []
            for event_group in doc.get("events", []):
                group = []
                for elem in event_group:
                    new_elem = {
                        "attribute": elem["attribute"],
                        "occurrences": {
                            id_mapping.get(old_id, "EVAL_INVALID_ID")
                            for old_id in elem["occurrences"]
                        },
                    }
                    group.append(new_elem)
                doc_events.append(group)
            processed.append(doc_events)

        return processed


class NEREvaluator:
    """Handles Named Entity Recognition evaluation."""

    @staticmethod
    def calculate_metrics(y_true: list, y_pred: list, tags: list) -> tuple[dict, dict]:
        """Calculate NER metrics using nervaluate.

        Also computes micro-based global metrics, by pooling all individual predictions.
        """
        evaluator = Evaluator(y_true, y_pred, tags)
        overall, by_tag, _, _ = evaluator.evaluate()
        return overall, by_tag

    @staticmethod
    def calculate_support(y_true: list[list]) -> int:
        """Calculate the support for each entity in the true labels."""
        support = {}
        for doc in y_true:
            for entity in doc:
                if support.get(entity["label"], None) is None:
                    support[entity["label"]] = 1
                else:
                    support[entity["label"]] += 1
        return support

    @staticmethod
    def get_macro_metrics(
        results_per_tags: dict[dict],
        metric: str = "f1",
        eval_schema: str = "strict",
    ) -> float:
        """Compute macro metrics from nervaluate.

        Each tag contributes equally regardless of its frequency.
        eval_schema follows nervaluate evaluation schema and can be:
            - strict : exact boundary surface string match and entity type
            - exact : exact boundary match over the surface string, regardless of the type
            - partial : partial boundary match over the surface string, regardless of the type
            - type : some overlap is required
        """
        macro = 0
        for val in results_per_tags.values():
            macro += val[eval_schema][metric]
        return round(macro * 100 / len(results_per_tags), 2)

    @staticmethod
    def get_weighted_metrics(
        results_per_tags: dict[dict],
        support: dict,
        metric: str = "f1",
        eval_schema: str = "strict",
    ) -> float:
        """Compute weighted metrics over the results.

        Each tag contributes proportionally to its frequency.
        eval_schema follows nervaluate evaluation schema and can be:
            - strict : exact boundary surface string match and entity type
            - exact : exact boundary match over the surface string, regardless of the type
            - partial : partial boundary match over the surface string, regardless of the type
            - type : some overlap is required
        """
        total_support = sum(support.values())
        weighted = sum(
            results_per_tags[tag][eval_schema][metric] * support_weight
            for tag, support_weight in support.items()
        )
        return round(weighted * 100 / total_support, 2)


class EventEvaluator:
    """Handles event extraction evaluation."""

    @classmethod
    def find_best_match(
        cls,
        true_event: list[dict],
        pred_doc: list[list],
        max_false_occurrences: int = 0,
    ) -> dict:
        """Find the best match between a true event and a list of predicted events."""
        best_scores = {"tp": 0, "fp": 0, "fn": 0}
        for i, pred_event in enumerate(pred_doc):
            tp, fp, fn = 0, 0, 0

            # Count true positives and false positives
            for t_elem in true_event:
                matched = any(
                    p_elem["attribute"] == t_elem["attribute"]
                    and len(p_elem["occurrences"] & t_elem["occurrences"]) > 0
                    and len(p_elem["occurrences"] - t_elem["occurrences"]) <= max_false_occurrences
                    for p_elem in pred_event
                )
                if matched:
                    tp += 1
                if not matched:
                    fn += 1

            # Count false positives
            for p_elem in pred_event:
                if not any(
                    t_elem["attribute"] == p_elem["attribute"]
                    and len(p_elem["occurrences"] & t_elem["occurrences"]) > 0
                    and len(p_elem["occurrences"] - t_elem["occurrences"]) <= max_false_occurrences
                    for t_elem in true_event
                ):
                    fp += 1

            # Keep only the best match which maximize f1-score
            current_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            if current_f1 > best_scores.get("f1", 0):
                best_scores = {"tp": tp, "fp": fp, "fn": fn, "f1": current_f1}
                best_index = i

        if best_scores["tp"] == 0:
            return best_scores, None
        return best_scores, best_index

    @staticmethod
    def compute_micro_metrics(tp: int, fp: int, fn: int) -> dict:
        """Compute precision, recall and F1 metrics."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    @staticmethod
    def event_level_metrics(true_events: list, pred_events: list) -> dict:
        """Compute event-level metrics."""
        macro_precision = []
        macro_recall = []
        macro_f1 = []
        overall_micro_metrics = {"tp": 0, "fp": 0, "fn": 0}

        for true_doc, pred_doc in zip(true_events, pred_events, strict=False):
            if pred_doc == [] and true_doc != []:
                for true_event in true_events:
                    fn = len(true_event)
                    evt_metrics = EventEvaluator.compute_micro_metrics(0, 0, fn)

                    macro_precision.append(evt_metrics["precision"])
                    macro_recall.append(evt_metrics["recall"])
                    macro_f1.append(evt_metrics["f1"])

                    overall_micro_metrics["fn"] += fn
            else:
                used_pred_indices = set()
                for true_event in true_doc:
                    best_scores, best_index = EventEvaluator.find_best_match(true_event, pred_doc)
                    if best_index is not None:
                        used_pred_indices.add(best_index)
                        evt_metrics = EventEvaluator.compute_micro_metrics(
                            best_scores["tp"],
                            best_scores["fp"],
                            best_scores["fn"],
                        )

                        macro_precision.append(evt_metrics["precision"])
                        macro_recall.append(evt_metrics["recall"])
                        macro_f1.append(evt_metrics["f1"])

                        overall_micro_metrics = {
                            key: overall_micro_metrics[key] + best_scores[key]
                            for key in overall_micro_metrics
                        }
                    else:
                        fn = len(true_event)
                        evt_metrics = EventEvaluator.compute_micro_metrics(0, 0, fn)

                        macro_precision.append(evt_metrics["precision"])
                        macro_recall.append(evt_metrics["recall"])
                        macro_f1.append(evt_metrics["f1"])

                        overall_micro_metrics["fn"] += fn

                # Si un évènement prédit est faux positif alors tous ses éléments sont faux positifs
                for i, pred_event in enumerate(pred_doc):
                    if i not in used_pred_indices:
                        fp = len(pred_event)
                        evt_metrics = EventEvaluator.compute_micro_metrics(0, fp, 0)

                        macro_precision.append(evt_metrics["precision"])
                        macro_recall.append(evt_metrics["recall"])
                        macro_f1.append(evt_metrics["f1"])

                        overall_micro_metrics["fp"] += fp

        macro_precision_avg = sum(macro_precision) / len(macro_precision) if macro_precision else 0
        macro_recall_avg = sum(macro_recall) / len(macro_recall) if macro_recall else 0
        macro_f1_avg = sum(macro_f1) / len(macro_f1) if macro_f1 else 0

        overall_micro_metrics = EventEvaluator.compute_micro_metrics(
            overall_micro_metrics["tp"],
            overall_micro_metrics["fp"],
            overall_micro_metrics["fn"],
        )

        return {
            "macro": {
                "precision": macro_precision_avg,
                "recall": macro_recall_avg,
                "f1": macro_f1_avg,
                "support": len(macro_precision),
            },
            "micro": overall_micro_metrics,
        }

    @classmethod
    def compute_completeness(cls, true_events: list, pred_events: list, threshold: float) -> dict:
        """Computes event completeness metrics."""
        complete_strict = complete_relaxed = 0

        for true_doc, pred_doc in zip(true_events, pred_events, strict=True):
            for true_event in true_doc:
                best_match = max(
                    (
                        sum(
                            any(
                                p_elem["attributee"] == t_elem["attributee"]
                                and p_elem["occurrences"] & t_elem["occurrences"]
                                for p_elem in pred_event
                            )
                            for t_elem in true_event
                        )
                        / len(true_event)
                    )
                    for pred_event in pred_doc
                )

                complete_strict += best_match >= 1.0
                complete_relaxed += best_match >= threshold

        return {
            "strict_completeness": complete_strict / len(true_events),
            "relaxed_completeness": complete_relaxed / len(true_events),
        }

    @staticmethod
    def doc_level_metrics(true_events: list, pred_events: list) -> float:
        """Document-level evaluation with set-based operations."""
        macro_precision = []
        macro_recall = []
        macro_f1 = []
        overall_micro_metrics = {"tp": 0, "fp": 0, "fn": 0}

        for true_doc, pred_doc in zip(true_events, pred_events, strict=False):
            true_ce = {
                frozenset(e["occurrences"])
                for true_evt in true_doc
                for e in true_evt
                if e["attribute"] == "evt:central_element"
            }
            pred_ce = {
                frozenset(e["occurrences"])
                for pred_evt in pred_doc
                for e in pred_evt
                if e["attribute"] == "evt:central_element"
            }

            # No event in golden annotation and no event predicted
            if len(true_ce) == len(pred_ce) == 0:
                continue
            # No event in golden annotation but event predicted
            if len(true_ce) == 0 and len(pred_ce) > 0:
                fp = len(pred_ce)
                tp, fn = 0, 0
            elif len(true_ce) > 0:
                tp = sum(1 for p in pred_ce if any(p <= t for t in true_ce))
                fp = len(pred_ce) - tp
                fn = sum(1 for t in true_ce if not any(p <= t for p in pred_ce))

                # Add count-based penalty: difference between predicted/true events
                count_diff = len(pred_ce) - len(true_ce)
                fp += max(count_diff, 0)  # Penalize over-detection
                fn += max(-count_diff, 0)  # Penalize under-detection

            doc_metrics = EventEvaluator.compute_micro_metrics(tp, fp, fn)

            macro_precision.append(doc_metrics["precision"])
            macro_recall.append(doc_metrics["recall"])
            macro_f1.append(doc_metrics["f1"])

            overall_micro_metrics["tp"] += tp
            overall_micro_metrics["fp"] += fp
            overall_micro_metrics["fn"] += fn

        macro_precision_avg = sum(macro_precision) / len(macro_precision) if macro_precision else 0
        macro_recall_avg = sum(macro_recall) / len(macro_recall) if macro_recall else 0
        macro_f1_avg = sum(macro_f1) / len(macro_f1) if macro_f1 else 0

        overall_micro_metrics = EventEvaluator.compute_micro_metrics(
            overall_micro_metrics["tp"],
            overall_micro_metrics["fp"],
            overall_micro_metrics["fn"],
        )

        return {
            "macro": {
                "precision": macro_precision_avg,
                "recall": macro_recall_avg,
                "f1": macro_f1_avg,
                "support": len(macro_precision),
            },
            "micro": overall_micro_metrics,
        }


class EvaluationPipeline:
    """Pipeline for evaluating the performance of a NER and event extraction model."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the EvaluationPipeline."""
        self.config = config
        self.results = defaultdict(dict)
        self.metrics_registry = []

    def run_ner_eval(self, true_entities: Entity, pred_entities: Entity, tags: list) -> dict:
        """Run the NER evaluation."""
        metrics = defaultdict(float)
        results, results_per_tag = NEREvaluator.calculate_metrics(
            true_entities,
            pred_entities,
            tags,
        )
        support = NEREvaluator.calculate_support(true_entities)

        metrics["entity_micro_precision"] = round(
            results[self.config.eval_schema]["precision"] * 100,
            2,
        )
        metrics["entity_micro_recall"] = round(results[self.config.eval_schema]["recall"] * 100, 2)
        metrics["entity_micro_f1"] = round(results[self.config.eval_schema]["f1"] * 100, 2)
        for metric_type in ["macro", "weighted"]:
            for measure in ["precision", "recall", "f1"]:
                key = f"entity_{metric_type}_{measure}"
                if metric_type == "macro":
                    value = NEREvaluator.get_macro_metrics(
                        results_per_tag,
                        measure,
                        self.config.eval_schema,
                    )
                else:
                    value = NEREvaluator.get_weighted_metrics(
                        results_per_tag,
                        support,
                        measure,
                        self.config.eval_schema,
                    )
                metrics[key] = round(value, 2)
        return metrics

    def run_event_eval(self, true_events: Event, pred_events: Event) -> dict:
        """Run the NER evaluation."""
        metrics = defaultdict(float)
        event_metrics = EventEvaluator.event_level_metrics(true_events, pred_events)
        for scope, sc_metrics in event_metrics.items():
            for k, v in sc_metrics.items():
                metrics[f"event_{scope}_{k}"] = round(v * 100, 2) if isinstance(v, float) else v

        # Document Evaluation
        doc_metrics = EventEvaluator.doc_level_metrics(true_events, pred_events)
        for scope, sc_metrics in doc_metrics.items():
            for k, v in sc_metrics.items():
                metrics[f"doc_{scope}_{k}"] = round(v * 100, 2) if isinstance(v, float) else v

        return metrics

    def run_pipeline(
        self,
        input_folder: str,
        output_csv: str,
        truth_path: str,
    ) -> None:
        """Export results to a CSV file."""
        all_results = []

        # Get golden annotations
        true_entities, true_tags, id_mapping = DataHandler.load_entities(truth_path, self.config)
        true_events = DataHandler.load_events(truth_path, id_mapping)

        # Iterate on each prediction file
        for filename in os.listdir(input_folder):
            pred_path = os.path.join(input_folder, filename)  # noqa: PTH118
            metrics = {
                "pred_file": filename,
                "eval_schema": self.config.eval_schema,
            }
            try:
                pred_entities, pred_tags, id_mapping = DataHandler.load_entities(
                    pred_path, self.config
                )
                pred_events = DataHandler.load_events(pred_path, id_mapping)

                missing_tags = set(true_tags) - set(pred_tags)
                metrics["missing_tags"] = ",".join(missing_tags) if missing_tags else None
                metrics["num_missing_tags"] = len(missing_tags)

                metrics.update(self.run_ner_eval(true_entities, pred_entities, true_tags))
                metrics.update(self.run_event_eval(true_events, pred_events))

                all_results.append(metrics)

            except (OSError, ValueError) as e:
                print(f"Error processing {filename}: {e!s}")  # noqa: T201
                continue

        # Save to CSV
        if all_results:
            results = pd.DataFrame(all_results)
            results.to_csv(output_csv, index=False)
            print(f"Saved results for {len(all_results)} files to {output_csv}")  # noqa: T201
        else:
            print("No files processed")  # noqa: T201


if __name__ == "__main__":
    TRUTH = "./datasets/test_preds.json"
    INPUT_FOLDER = "event/final_outputs"
    OUTPUT = "event/evaluation/results_evalLLM_event_4-1.csv"
    config = EvaluationConfig()
    pipeline = EvaluationPipeline(config)
    pipeline.run_pipeline(INPUT_FOLDER, OUTPUT, TRUTH)
