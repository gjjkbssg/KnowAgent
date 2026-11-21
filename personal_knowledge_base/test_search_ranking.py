from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from personal_knowledge_base import search
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.search import _fts_search, apply_mmr, deduplicate_results, hybrid_search, index_chunk, rrf_fuse, update_chunk_index_scope


class SearchRankingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="ranker")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        self.other_kb = KnowledgeBase.objects.create(tenant=self.tenant, name="other")

    def add_chunk(self, content, kb=None, title="doc", index=0):
        kb = kb or self.kb
        knowledge = Knowledge.objects.create(tenant=self.tenant, knowledge_base=kb, title=title, parse_status="completed")
        chunk = Chunk.objects.create(
            tenant=self.tenant,
            knowledge_base=kb,
            knowledge=knowledge,
            content=content,
            chunk_index=index,
        )
        index_chunk(chunk)
        return chunk

    def test_fts_returns_best_bm25_matches_first_even_when_inserted_last(self):
        for i in range(30):
            self.add_chunk(f"filler {i} mentions kiwi once among many other unrelated words " + "padding " * 40, title=f"weak-{i}")
        best = self.add_chunk("kiwi kiwi kiwi kiwi", title="best")

        scores = _fts_search(self.tenant.id, {self.kb.id}, "kiwi", 2)

        self.assertIn(best.id, scores)
        self.assertEqual(max(scores, key=scores.get), best.id)
        self.assertTrue(all(score > 0 for score in scores.values()))

    def test_moved_chunk_is_found_only_in_its_new_knowledge_base_without_re_embedding(self):
        chunk = self.add_chunk("kiwi orchard notes")
        chunk.knowledge_base = self.other_kb
        chunk.save(update_fields=["knowledge_base", "updated_at"])

        with patch("personal_knowledge_base.model_providers.embedding") as embed:
            update_chunk_index_scope(chunk)

        embed.assert_not_called()
        self.assertEqual(_fts_search(self.tenant.id, {self.kb.id}, "kiwi", 5), {})
        self.assertEqual(list(_fts_search(self.tenant.id, {self.other_kb.id}, "kiwi", 5)), [chunk.id])
        self.assertEqual([item["chunk_id"] for item in hybrid_search(self.tenant.id, [self.other_kb.id], "kiwi", 3)], [chunk.id])

    def test_fts_applies_knowledge_base_filter_before_limit(self):
        for i in range(20):
            self.add_chunk("mango mango mango", kb=self.other_kb, title=f"other-{i}")
        wanted = self.add_chunk("a single mango " + "padding " * 40, title="wanted")

        scores = _fts_search(self.tenant.id, {self.kb.id}, "mango", 1)

        self.assertEqual(set(scores), {wanted.id})

    def test_rrf_uses_ranks_not_raw_scores(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])

        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused["a"], fused["b"])
        self.assertGreater(fused["a"], fused["c"])
        self.assertGreater(fused["c"], fused["d"] - 1e-9)

    def test_mmr_drops_near_duplicate_when_k_is_smaller_than_candidates(self):
        results = [
            {"chunk_id": "a", "content": "alpha beta gamma delta", "score": 1.0},
            {"chunk_id": "b", "content": "alpha beta gamma delta epsilon", "score": 0.95},
            {"chunk_id": "c", "content": "zeta eta theta iota", "score": 0.9},
            {"chunk_id": "d", "content": "kappa lambda mu nu", "score": 0.1},
        ]

        selected = apply_mmr(results, k=2, lambda_param=0.7)

        self.assertEqual([item["chunk_id"] for item in selected], ["a", "c"])

    def test_hybrid_search_reranks_graph_candidates_instead_of_pinning_them(self):
        relevant = self.add_chunk("papaya harvest season is in early summer", title="relevant")
        graph_hit = self.add_chunk("completely unrelated entity mention", title="graph")
        graph_rows = [{"chunk_id": graph_hit.id}]
        seen_candidates = []

        def fake_rerank(query, results, top_k=None, tenant=None):
            seen_candidates.extend(item["chunk_id"] for item in results)
            ordered = sorted(results, key=lambda item: item["chunk_id"] != relevant.id)
            return [{**item, "score": 0.9 if item["chunk_id"] == relevant.id else 0.1} for item in ordered][:top_k]

        with (
            patch("personal_knowledge_base.graph_rag.graph_search_results", return_value=graph_rows),
            patch("personal_knowledge_base.model_providers.rerank", side_effect=fake_rerank),
            patch.object(search, "_vector_search", return_value={}),
        ):
            results = hybrid_search(self.tenant.id, [self.kb.id], "papaya harvest", 5)

        self.assertIn(graph_hit.id, seen_candidates)
        self.assertEqual([item["chunk_id"] for item in results], [relevant.id, graph_hit.id])
        self.assertEqual(results[1]["match_type"], "graph")
        self.assertLess(results[1]["score"], results[0]["score"])

    def test_hybrid_search_returns_at_most_top_k(self):
        topics = ["soil acidity", "pruning schedule", "pest control", "export prices", "juice recipes", "seed storage", "frost damage", "pollination"]
        for i, topic in enumerate(topics):
            self.add_chunk(f"guava {topic} notes: {topic} {topic} chapter{i} section{i} appendix{i}", title=f"g-{i}")

        with patch.object(search, "_vector_search", return_value={}):
            results = hybrid_search(self.tenant.id, [self.kb.id], "guava", 3)

        self.assertEqual(len(results), 3)


class DeduplicateResultsTests(SimpleTestCase):
    FULL = (
        "智能会议室由中枢网关 XH-100 协调以下设备：环境传感器 ES-20，每 30 秒上报温度；"
        "空气净化器 AP-50，支持静音、自动和强效三种模式；智能灯组 LG-8，支持亮度调节；"
        "会议屏 DS-65，与 XH-100 处于同一局域网时才能使用无线投屏。"
    )
    LABELS = "ES-20\nXH-100\nAP-50\nLG-8\nDS-65"

    def test_short_chunk_does_not_replace_the_chunk_that_contains_it(self):
        results = deduplicate_results([
            {"chunk_id": "labels", "content": self.LABELS, "score": 0.9},
            {"chunk_id": "full", "content": self.FULL, "score": 0.5},
        ])

        self.assertEqual([item["chunk_id"] for item in results], ["full"])

    def test_containing_chunk_wins_in_either_order(self):
        results = deduplicate_results([
            {"chunk_id": "full", "content": self.FULL, "score": 0.9},
            {"chunk_id": "labels", "content": self.LABELS, "score": 0.5},
        ])

        self.assertEqual([item["chunk_id"] for item in results], ["full"])

    def test_near_duplicates_of_similar_length_keep_the_higher_score(self):
        results = deduplicate_results([
            {"chunk_id": "low", "content": self.FULL, "score": 0.2},
            {"chunk_id": "high", "content": self.FULL + " 详见拓扑图。", "score": 0.8},
            {"chunk_id": "other", "content": "E17 表示环境传感器数据过期，应暂停依赖环境数据的自动化规则。", "score": 0.1},
        ])

        self.assertEqual([item["chunk_id"] for item in results], ["high", "other"])
