from typing import Dict, List

from lightning.pytorch.callbacks import Callback

from relik.reader.data.relik_reader_sample import RelikReaderSample
from relik.reader.utils.relik_reader_predictor import RelikReaderPredictor
from relik.reader.utils.metrics import f1_measure, safe_divide
from relik.reader.utils.special_symbols import NME_SYMBOL
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager
from relik.reader.data.relik_reader_sample import load_relik_reader_samples
# from sklearn.metrics import adjusted_rand_score
# import itertools
from corefeval import get_metrics
from corefeval import Scorer, Document
# def compute_ari(predicted_clusters, gold_clusters):
#     def flatten_clusters(clusters):
#         # Converts a list of clusters to a list of element pairs.
#         # Each element is assigned a unique cluster label.
#         element_cluster_map = {}
#         for idx, cluster in enumerate(clusters):
#             for pair in cluster:
#                 for item in pair:
#                     element_cluster_map[item] = idx
#         return element_cluster_map

#     def clusters_to_labels(clusters, all_items):
#         # Convert clusters to a flat list of labels
#         cluster_map = flatten_clusters(clusters)
#         return [cluster_map.get(item, -1) for item in all_items]

#     def unique_items(pred_clusters, gold_clusters):
#         # Find all unique items from both cluster lists
#         pred_items = set(itertools.chain(*itertools.chain(*pred_clusters)))
#         gold_items = set(itertools.chain(*itertools.chain(*gold_clusters)))
#         return sorted(list(pred_items.union(gold_items)))

#     # Find all unique items
#     all_items = unique_items(predicted_clusters, gold_clusters)

#     # Convert clusters to labels
#     pred_labels = clusters_to_labels(predicted_clusters, all_items)
#     gold_labels = clusters_to_labels(gold_clusters, all_items)

#     # Compute Adjusted Rand Index
#     return adjusted_rand_score(gold_labels, pred_labels)

class StrongMatching:
    def __call__(self, predicted_samples: List[RelikReaderSample]) -> Dict:
        # accumulators
        scorer = Scorer()
        correct_predictions = 0
        correct_predictions_at_k = 0
        total_predictions = 0
        total_gold = 0
        total_conll_f1 = 0
        correct_span_predictions = 0
        miss_due_to_candidates = 0

        # prediction index stats
        avg_correct_predicted_index = []
        avg_wrong_predicted_index = []
        less_index_predictions = []

        # collect data from samples
        for sample in predicted_samples:
            predicted_annotations = sample.predicted_window_labels_chars
            predicted_annotations_probabilities = sample.probs_window_labels_chars
            predicted_clusters = list(sample.predicted_clusters.values())
            gold_annotations = {
                (ss, se, entity)
                for ss, se, entity in sample.window_labels
                if entity != NME_SYMBOL
            }
            # create gold clusters
            set_entities = list(set([entity for _, _, entity in gold_annotations]))
            gold_clusters = [[] for _ in range(len(set_entities))]
            for ss, se, entity in gold_annotations:
                gold_clusters[set_entities.index(entity)].append((ss, se))
            # remove NME from predicted clusters and the titles, since they are not part of the evaluation
            predicted_clusters_filtered = []
            for cluster in predicted_clusters:
                filtered_cluster = []
                for span in cluster:
                    if span[2] != "--NME--":
                        filtered_cluster.append((span[0], span[1]))
                if len(filtered_cluster) > 0:
                    predicted_clusters_filtered.append(filtered_cluster)
            # compute ARI
            doc = Document(
                predicted=predicted_clusters_filtered,
                truth=gold_clusters,
            )
            scorer.update(doc)
            # conll_f1, metrics = get_metrics(predicted_clusters_filtered, gold_clusters, verbose=False)
            # total_conll_f1 += conll_f1
            total_predictions += len(predicted_annotations)
            total_gold += len(gold_annotations)

            # correct named entity detection
            predicted_spans = {(s, e) for s, e, _ in predicted_annotations}
            gold_spans = {(s, e) for s, e, _ in gold_annotations}
            correct_span_predictions += len(predicted_spans.intersection(gold_spans))

            # correct entity linking
            correct_predictions += len(
                predicted_annotations.intersection(gold_annotations)
            )

            for ss, se, ge in gold_annotations.difference(predicted_annotations):
                if ge not in sample.span_candidates:
                    miss_due_to_candidates += 1
                if ge in predicted_annotations_probabilities.get((ss, se), set()):
                    correct_predictions_at_k += 1

            # indices metrics
            predicted_spans_index = {
                (ss, se): ent for ss, se, ent in predicted_annotations
            }
            gold_spans_index = {(ss, se): ent for ss, se, ent in gold_annotations}

            for pred_span, pred_ent in predicted_spans_index.items():
                gold_ent = gold_spans_index.get(pred_span)

                if pred_span not in gold_spans_index:
                    continue

                # missing candidate
                if gold_ent not in sample.span_candidates:
                    continue

                gold_idx = sample.span_candidates.index(gold_ent)
                if gold_idx is None:
                    continue
                pred_idx = sample.span_candidates.index(pred_ent)

                if gold_ent != pred_ent:
                    avg_wrong_predicted_index.append(pred_idx)

                    if gold_idx is not None:
                        if pred_idx > gold_idx:
                            less_index_predictions.append(0)
                        else:
                            less_index_predictions.append(1)

                else:
                    avg_correct_predicted_index.append(pred_idx)

        # compute NED metrics
        span_precision = safe_divide(correct_span_predictions, total_predictions)
        span_recall = safe_divide(correct_span_predictions, total_gold)
        span_f1 = f1_measure(span_precision, span_recall)

        # compute EL metrics
        precision = safe_divide(correct_predictions, total_predictions)
        recall = safe_divide(correct_predictions, total_gold)
        recall_at_k = safe_divide(
            (correct_predictions + correct_predictions_at_k), total_gold
        )

        f1 = f1_measure(precision, recall)

        total_conll_f1, metrics = scorer.detailed_score(
            modelname="Undefined model", dataset="Undefined data", verbose=False
        )

        wrong_for_candidates = safe_divide(miss_due_to_candidates, total_gold)

        out_dict = {
            "span_precision": span_precision,
            "span_recall": span_recall,
            "span_f1": span_f1,
            "core_precision": precision,
            "core_recall": recall,
            "core_recall-at-k": recall_at_k,
            "core_f1": round(f1, 4),
            "conll_f1": round(total_conll_f1, 4),
            "wrong-for-candidates": wrong_for_candidates,
            "index_errors_avg-index": safe_divide(
                sum(avg_wrong_predicted_index), len(avg_wrong_predicted_index)
            ),
            "index_correct_avg-index": safe_divide(
                sum(avg_correct_predicted_index), len(avg_correct_predicted_index)
            ),
            "index_avg-index": safe_divide(
                sum(avg_correct_predicted_index + avg_wrong_predicted_index),
                len(avg_correct_predicted_index + avg_wrong_predicted_index),
            ),
            "index_percentage-favoured-smaller-idx": safe_divide(
                sum(less_index_predictions), len(less_index_predictions)
            ),
        }

        return {k: round(v, 5) for k, v in out_dict.items()}


class ELStrongMatchingCallback(Callback):
    def __init__(
            self, dataset_path: str, dataset_conf, log_metric: str = "val_", skip_first: int = 0
        ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_conf = dataset_conf
        self.strong_matching_metric = StrongMatching()
        self.log_metric = log_metric
        self.tokenizer = SpacyTokenizer(language="en", use_gpu=False)

        self.sentence_splitter = WindowSentenceSplitter(
            window_size=32, window_stride=16
        )
        self.window_manager = WindowManager(
                        self.tokenizer, self.sentence_splitter
                    )
        self.skip_first = skip_first

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        if trainer.current_epoch < self.skip_first:
            pl_module.log(f"{self.log_metric}doc_core_f1", 0.0)
            return

        samples = list(load_relik_reader_samples(self.dataset_path))

        predicted_samples = pl_module.relik_reader_core_model.read(
            samples=samples,
            progress_bar=False,
        )
        predicted_samples = list(predicted_samples)
        for k, v in self.strong_matching_metric(predicted_samples).items():
            pl_module.log(f"{self.log_metric}{k}", v)
        merged_windows = self.window_manager.merge_windows(predicted_samples)
        empty_candidates = []
        for idx, sample in enumerate(merged_windows):
            sample.predicted_window_labels_chars = set(sample.predicted_spans)
            # sample._d["span_candidates"] = list(set(element[0] for span_list in sample.probs_window_labels_chars.values() for element in span_list[:3]) - set(["--NME--"]))
            sample._d["span_candidates"] = list(set([span[2] for span in sample.predicted_spans]) - set(["--NME--"]))
            if len(sample._d["span_candidates"]) == 0:
                sample.predicted_clusters = {}
                empty_candidates.append(idx)
            tokens  = self.tokenizer(sample.text)
            token2char_start = {str(i): w.idx for i, w in enumerate(tokens)}
            token2char_end = {
                            str(i): w.idx + len(w.text) for i, w in enumerate(tokens)
                        }
            char2token_start = {str(w.idx): w.i for i, w in enumerate(tokens)}
            char2token_end = {
                            str(w.idx + len(w.text)): w.i for i, w in enumerate(tokens)
                        }
            sample._d["tokens"] = [w.text for w in tokens]
            sample._d["token2char_start"] = token2char_start
            sample._d["token2char_end"] = token2char_end
            sample._d["char2token_start"] = char2token_start
            sample._d["char2token_end"] = char2token_end
            sample._d["window_id"] = 0
            sample._d["offset"] = 0
            if "_mixin_prediction_position" in sample._d:
                del sample._d["_mixin_prediction_position"]
            sample._mixin_prediction_position = None
        empty_candidates_sample = []
        for idx in empty_candidates:
            empty_candidates_sample.append(merged_windows[idx])
            del merged_windows[idx]
        prev_max_length = pl_module.relik_reader_core_model.dataset.max_length
        pl_module.relik_reader_core_model.dataset.max_length = 8192
        predicted_samples = pl_module.relik_reader_core_model.read(
            samples=merged_windows, token_batch_size=8192, max_length=8192, progress_bar=False
        )
        predicted_samples = list(predicted_samples) + empty_candidates_sample
        
        for k, v in self.strong_matching_metric(predicted_samples).items():
            pl_module.log(f"{self.log_metric}doc_{k}", v)

        pl_module.relik_reader_core_model.dataset.max_length = prev_max_length