import json
import math
from pathlib import Path

from django.test import SimpleTestCase

from .agent_eval import score_answer
from .eval_metrics import (
    BLEU1,
    BLEU2,
    BLEU4,
    bleu,
    mean_average_precision,
    mrr,
    ndcg,
    precision,
    recall,
    rouge,
    split_into_words,
    split_sentences,
)
from .interview_eval_metrics import (
    gold_hits,
    gold_size,
    ranking_metrics,
    summarize,
)


class GoldHitsTests(SimpleTestCase):
    def test_matches_by_source_title_and_evidence(self):
        gold = {"expected_sources": ["p1"], "expected_titles": ["手册"], "evidence": ["每 30 秒上报"]}
        self.assertEqual(gold_hits({"knowledge_source": "p1"}, gold), {"source:p1"})
        self.assertEqual(gold_hits({"knowledge_title": "手册"}, gold), {"title:手册"})
        self.assertEqual(gold_hits({"content": "ES-20 每 30\n秒上报温度"}, gold), {"evidence:每 30 秒上报"})
        self.assertEqual(gold_hits({"knowledge_source": "p2", "content": "无关"}, gold), set())
        self.assertEqual(gold_size(gold), 3)


class RankingMetricsTests(SimpleTestCase):
    def test_single_gold_at_rank_three(self):
        metrics = ranking_metrics([set(), set(), {"a"}, set()], total_gold=1)

        self.assertAlmostEqual(metrics["reciprocal_rank"], 1 / 3)
        self.assertEqual((metrics["hit_at_1"], metrics["hit_at_3"]), (0.0, 1.0))
        self.assertAlmostEqual(metrics["ndcg_at_5"], 0.5)

    def test_recall_counts_distinct_gold_items_once(self):
        metrics = ranking_metrics([{"a"}, {"a"}, {"b"}], total_gold=4)

        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_3"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 0.5)

    def test_perfect_ranking_has_ndcg_one_and_miss_has_zero(self):
        self.assertAlmostEqual(ranking_metrics([{"a"}, {"b"}], total_gold=2)["ndcg_at_10"], 1.0)
        miss = ranking_metrics([set(), set()], total_gold=2)
        self.assertEqual((miss["reciprocal_rank"], miss["hit_at_10"], miss["ndcg_at_10"]), (0.0, 0.0, 0.0))

    def test_one_result_matching_several_gold_items_keeps_ndcg_at_most_one(self):
        metrics = ranking_metrics([{"a", "b"}, set()], total_gold=2)

        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertLessEqual(metrics["ndcg_at_1"], 1.0)
        self.assertLessEqual(metrics["ndcg_at_10"], 1.0)

    def test_summarize_averages_over_questions(self):
        rows = [
            {**ranking_metrics([{"a"}], 1), "latency_ms": 10},
            {**ranking_metrics([set(), {"a"}], 1), "latency_ms": 30},
        ]
        summary = summarize(rows)

        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["mrr"], 0.75)
        self.assertEqual(summary["hit_at_1"], 0.5)
        self.assertEqual(summary["avg_latency_ms"], 20.0)
        self.assertNotIn("avg_result_count", summary)

    def test_summarize_averages_result_count_when_present(self):
        rows = [
            {**ranking_metrics([{"a"}], 1), "latency_ms": 10, "result_count": 4},
            {**ranking_metrics([set()], 1), "latency_ms": 20},
        ]

        self.assertEqual(summarize(rows)["avg_result_count"], 2.0)


class WeKnoraBLEUTests(SimpleTestCase):
    def test_bleu_metric_compute_zero_precision(self):
        cases = [
            ("zero bigram precision", "a b c d", "a d c b", False, BLEU2, 0.0),
            ("zero trigram precision", "a b c d", "a b d c", False, (0, 0, 1, 0), 0.0),
            ("zero fourgram precision", "a b c d", "a b c e", False, BLEU4, 0.0),
            ("short candidate", "a", "a", False, BLEU4, 0.0),
            ("zero weight ignores missing orders", "a", "a", False, BLEU1, 1.0),
            ("positive weighted precisions", "a b c", "a b d", False, BLEU2, 0.5773502691896257),
            ("perfect match", "a b c d", "a b c d", False, BLEU4, 1.0),
            ("no match", "a b c d", "e f g h", False, BLEU4, 0.0),
            ("empty candidate", "", "a", False, BLEU1, 0.0),
            ("empty reference", "a", "", False, BLEU1, 0.0),
            ("add-one smoothing unchanged", "a b c d", "a d c b", True, BLEU2, 0.5),
            ("smoothed short candidate unchanged", "a", "a", True, BLEU4, 1.0),
        ]
        for name, candidate, reference, smoothing, weights, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    bleu(candidate, reference, weights, smoothing),
                    expected,
                    delta=1e-9,
                )

    def test_text_metrics_include_every_sentence(self):
        self.assertAlmostEqual(
            bleu("alpha.gamma", "alpha.beta", BLEU1),
            0.5,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            rouge("alpha.gamma", "alpha.beta", "rouge-1", "p"),
            0.5,
            delta=1e-9,
        )


class WeKnoraTextSplittingTests(SimpleTestCase):
    def test_split_sentences(self):
        cases = [
            ("", []),
            (" \n\t ", []),
            ("One. Two. Three. Four", ["One", "Two", "Three", "Four"]),
            ("第一句。第二句。第三句。", ["第一句", "第二句", "第三句"]),
            (" One。 Two. Third 。", ["One", "Two", "Third"]),
            (".。One..。Two。", ["One", "Two"]),
            (" single sentence ", ["single sentence"]),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(split_sentences(text), expected)

    def test_split_into_words(self):
        cases = [
            ("nil", None, []),
            ("empty", ["", " \n\t"], []),
            (
                "words",
                ["Hello world_123! 42,foo."],
                ["Hello", "world_123!", "42,foo."],
            ),
            ("whitespace", ["go\tone", "two\nthree"], ["go", "one", "two", "three"]),
            ("punctuation", ["A:B;C，D"], ["A", ":", "B", ";", "C", "，", "D"]),
            ("mixed scripts", ["中A文!"], ["中", "A", "文", "!"]),
            ("symbols", ["abc 😀 def + xyz"], ["abc", "def", "xyz"]),
            ("sentence order", ["first", "second"], ["first", "second"]),
        ]
        for name, sentences, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(split_into_words(sentences), expected)


class WeKnoraRetrievalMetricTests(SimpleTestCase):
    def test_ndcg_metric_compute(self):
        cases = [
            ("one of three relevant documents", 3, [[1, 2, 3]], [1], 0.46927872602275644),
            ("two of three relevant documents", 3, [[1, 2, 3]], [1, 2], 0.7653606369886217),
            ("ideal ranking respects cutoff", 1, [[1, 2, 3]], [1], 1.0),
            ("complete retrieval below cutoff", 3, [[1, 2]], [1, 2], 1.0),
            ("relevant result beyond cutoff", 2, [[1]], [9, 8, 1], 0.0),
            ("rank discount", 3, [[1]], [9, 1], 0.6309297535714574),
            ("distinct relevant documents", 3, [[1, 1], [1]], [1, 9, 8], 1.0),
            ("no ground truth", 3, None, [1], 0.0),
            ("empty ground truth set", 3, [[]], [1], 0.0),
            ("empty retrieval", 3, [[1, 2]], None, 0.0),
            ("zero cutoff", 0, [[1]], [1], 0.0),
        ]
        for name, cutoff, ground_truth, ids, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    ndcg(ground_truth, ids, cutoff),
                    expected,
                    delta=1e-9,
                )

    def test_mrr_metric_compute(self):
        cases = [
            ("perfect match - first position", [[1, 2]], [1, 2, 3], 1.0),
            ("match at second position", [[1, 2]], [3, 1, 2], 0.5),
            ("no match", [[1, 2]], [3, 4], 0.0),
            ("multiple queries", [[1, 2], [3, 4]], [1, 3, 2, 4], 0.75),
            ("empty ground truth", [], [1, 2], 0.0),
        ]
        for name, ground_truth, ids, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    mrr(ground_truth, ids),
                    expected,
                    delta=1e-9,
                )

    def test_map_metric_compute(self):
        cases = [
            ("total match", [[2, 4, 6]], [2, 4, 6], 1.0),
            ("no match", [[1, 2]], [3, 4], 0.0),
            ("partial match", [[1, 2, 3]], [2, 5, 1, 3], 0.8055555555555555),
            ("empty ground truth", [], [1, 2], 0.0),
            ("only one of four relevant documents retrieved", [[1, 2, 3, 4]], [1], 0.25),
            ("incomplete recall with a nonrelevant result first", [[1, 2, 3]], [99, 1, 2], 7.0 / 18),
            ("duplicate retrieved documents count once", [[1, 2]], [1, 1, 2], 5.0 / 6),
            ("duplicate hits cannot inflate the normalized score", [[1]], [1, 1], 1.0),
            ("duplicate ground truth documents count once", [[1, 1, 2]], [1], 0.5),
            ("empty ground truth group", [[]], [1], 0.0),
            ("empty retrieved results", [[1, 2]], None, 0.0),
            ("multiple queries with incomplete recall", [[1, 2], [3, 4]], [1, 3], 0.375),
            ("multiple queries", [[1, 2], [3, 4]], [1, 3, 2, 4], 0.6666666666666666),
        ]
        for name, ground_truth, ids, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    mean_average_precision(ground_truth, ids),
                    expected,
                    delta=1e-9,
                )

    def test_precision_metric_compute(self):
        cases = [
            ("perfect match", [[1, 3, 5]], [1, 3, 5], 1.0),
            ("half match", [[1, 2, 3]], [1, 4, 2], 0.6666666666666666),
            ("no match", [[1, 2, 3]], [4, 5, 6], 0.0),
            ("empty retrieval", [[1, 2, 3]], [], 0.0),
            ("multiple ground truths", [[1, 2], [3, 4]], [1, 3, 5], 0.3333333333333333),
        ]
        for name, ground_truth, ids, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    precision(ground_truth, ids),
                    expected,
                    delta=1e-9,
                )

    def test_recall_metric_compute(self):
        cases = [
            ("perfect recall - all ground truth retrieved", [[1, 2, 3]], [1, 2, 3, 4], 1.0),
            ("partial recall - some ground truth retrieved", [[1, 2, 3], [4, 5]], [1, 4, 6], 0.41666666666666663),
            ("no recall - no ground truth retrieved", [[1, 2, 3]], [4, 5, 6], 0.0),
            ("empty retrieval list", [[1, 2, 3]], [], 0.0),
            ("multiple ground truth sets", [[1, 2], [3, 4], [5, 6]], [1, 3, 7], 0.3333333333333333),
        ]
        for name, ground_truth, ids, expected in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    recall(ground_truth, ids),
                    expected,
                    delta=1e-9,
                )

    def test_retrieval_metrics_with_duplicates(self):
        ideal_two = 1 + 1 / math.log2(3)
        cases = [
            ("duplicate cannot exceed full credit", [[1]], [1, 1], 3, 0.5, 1.0, 1.0),
            ("duplicate does not cover missing evidence", [[1, 2]], [1, 1], 3, 0.5, 0.5, 1 / ideal_two),
            ("duplicate preserves later document rank", [[1, 2]], [1, 1, 2], 3, 2.0 / 3, 1.0, 1.5 / ideal_two),
            ("nonrelevant duplicates preserve rank", [[1]], [9, 9, 1], 3, 1.0 / 3, 1.0, 0.5),
            ("duplicates do not move evidence into cutoff", [[1, 2]], [1, 1, 2], 2, 2.0 / 3, 1.0, 1 / ideal_two),
            ("ground truth groups stay independent", [[1, 1], [1, 2]], [1, 1, 2], 3, 0.5, 1.0, 1.5 / ideal_two),
        ]
        for name, ground_truth, ids, cutoff, want_precision, want_recall, want_ndcg in cases:
            with self.subTest(name=name):
                self.assertAlmostEqual(
                    precision(ground_truth, ids),
                    want_precision,
                    delta=1e-9,
                )
                self.assertAlmostEqual(
                    recall(ground_truth, ids),
                    want_recall,
                    delta=1e-9,
                )
                self.assertAlmostEqual(
                    ndcg(ground_truth, ids, cutoff),
                    want_ndcg,
                    delta=1e-9,
                )

    def test_retrieval_metric_inputs_are_not_mutated(self):
        ground_truth = [[1, 1], [1, 2]]
        ids = [1, 1, 2, 9]
        original_ground_truth = [group[:] for group in ground_truth]
        original_ids = ids[:]

        first_results = (
            precision(ground_truth, ids),
            recall(ground_truth, ids),
            ndcg(ground_truth, ids, 3),
        )
        second_results = (
            precision(ground_truth, ids),
            recall(ground_truth, ids),
            ndcg(ground_truth, ids, 3),
        )

        self.assertEqual(first_results, second_results)
        self.assertEqual(ground_truth, original_ground_truth)
        self.assertEqual(ids, original_ids)


class AgentAnswerScoringTests(SimpleTestCase):
    def test_task_references_score_correct_and_refusal_matches_answerability(self):
        path = Path(__file__).resolve().parents[1] / "interview_materials" / "agent_eval_tasks.json"
        tasks = json.loads(path.read_text(encoding="utf-8"))
        refusal = "知识库中没有相关信息。"

        for task in tasks:
            with self.subTest(task=task["id"], answer="reference"):
                self.assertTrue(score_answer(task, task["reference"]))
            with self.subTest(task=task["id"], answer="refusal"):
                self.assertEqual(
                    score_answer(task, refusal),
                    bool(task.get("unanswerable")),
                )
