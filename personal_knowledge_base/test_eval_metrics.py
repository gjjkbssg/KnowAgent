from django.test import SimpleTestCase

from .eval_metrics import gold_hits, gold_size, ranking_metrics, summarize


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
