import re
from unittest.mock import Mock, patch

from django.db import connection
from django.test import SimpleTestCase, TestCase, override_settings

from personal_knowledge_base import search
from personal_knowledge_base.model_providers import rerank
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant
from personal_knowledge_base.search import (
    _fts_search,
    apply_mmr,
    apply_rerank_threshold,
    apply_version_aware_mmr,
    apply_version_policy,
    build_fts_query,
    cjk_bigram_text,
    deduplicate_results,
    hybrid_search,
    index_chunk,
    query_wants_history,
    retokenize_fts_rows,
    rrf_channel_weights,
    rrf_fuse,
    update_chunk_index_scope,
)


class SearchRankingTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="ranker")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb")
        self.other_kb = KnowledgeBase.objects.create(tenant=self.tenant, name="other")

    def add_chunk(self, content, kb=None, title="doc", index=0, metadata=None):
        kb = kb or self.kb
        knowledge = Knowledge.objects.create(
            tenant=self.tenant,
            knowledge_base=kb,
            title=title,
            parse_status="completed",
            metadata=metadata or {},
        )
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

    def test_chinese_bigram_fts_finds_paraphrased_query(self):
        chunk = self.add_chunk("冷水机组高压报警时应先检查冷却水流量", title="冷水机组")

        scores = _fts_search(self.tenant.id, {self.kb.id}, "高压报警应该怎么处理", 10)

        self.assertIn(chunk.id, scores)

    def test_english_fts_search_still_finds_identifier_content(self):
        chunk = self.add_chunk("The E-102 cooling pump alarm indicates low coolant pressure")

        scores = _fts_search(self.tenant.id, {self.kb.id}, "cooling pump", 10)

        self.assertIn(chunk.id, scores)

    def test_identifier_fallback_matches_hyphenated_identifier(self):
        chunk = self.add_chunk("E-102 motor code")

        scores = _fts_search(self.tenant.id, {self.kb.id}, "E102 是什么故障", 10)

        self.assertIn(chunk.id, scores)

    def test_retokenize_fts_rows_rewrites_raw_text_idempotently(self):
        search.ensure_search_tables()
        chunk_id = "retokenize-test"
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chunks_fts WHERE chunk_id = %s", [chunk_id])
            cursor.execute(
                """
                INSERT INTO chunks_fts(chunk_id, tenant_id, knowledge_base_id, knowledge_id, title, content)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [chunk_id, self.tenant.id, self.kb.id, "knowledge", "冷水机组", "高压报警"],
            )
            with patch.object(cursor, "executemany", wraps=cursor.executemany) as batch_updates:
                retokenize_fts_rows(cursor)
                self.assertGreater(batch_updates.call_count, 0)
            cursor.execute("SELECT title, content FROM chunks_fts WHERE chunk_id = %s", [chunk_id])
            first = cursor.fetchone()
            self.assertEqual(first, ("冷水 水机 机组", "高压 压报 报警"))

            with patch.object(cursor, "executemany", wraps=cursor.executemany) as batch_updates:
                retokenize_fts_rows(cursor)
                batch_updates.assert_not_called()
            cursor.execute("SELECT title, content FROM chunks_fts WHERE chunk_id = %s", [chunk_id])
            self.assertEqual(cursor.fetchone(), first)

    def test_rrf_uses_ranks_not_raw_scores(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])

        self.assertAlmostEqual(fused["a"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused["a"], fused["b"])
        self.assertGreater(fused["a"], fused["c"])
        self.assertGreater(fused["c"], fused["d"] - 1e-9)

    def test_rrf_defaults_to_equal_weights_and_supports_weighted_channels(self):
        rankings = [["vector", "keyword"], ["keyword", "vector"]]

        self.assertEqual(rrf_fuse(rankings), rrf_fuse(rankings, weights=[1.0, 1.0]))
        weighted = rrf_fuse(rankings, weights=[0.7, 0.3])
        self.assertGreater(weighted["vector"], weighted["keyword"])

    def test_rrf_rejects_a_weight_count_mismatch(self):
        with self.assertRaises(ValueError):
            rrf_fuse([["a"], ["b"]], weights=[0.7])

    def test_rrf_channel_weights_read_overridden_settings(self):
        with override_settings(
            RETRIEVAL_RRF_VECTOR_WEIGHT=0.7,
            RETRIEVAL_RRF_KEYWORD_WEIGHT=0.3,
            RETRIEVAL_RRF_GRAPH_WEIGHT=0.3,
        ):
            self.assertEqual(rrf_channel_weights(), {"vector": 0.7, "keyword": 0.3, "graph": 0.3})

    def test_hybrid_search_applies_configured_rrf_channel_weights(self):
        fts_chunk = self.add_chunk("fts result", title="fts")
        vector_chunk = self.add_chunk("vector result", title="vector")

        with (
            override_settings(RETRIEVAL_RRF_VECTOR_WEIGHT=0.7, RETRIEVAL_RRF_KEYWORD_WEIGHT=0.3),
            patch.object(search, "_fts_search", return_value={fts_chunk.id: 1.0}),
            patch.object(search, "_vector_search", return_value={vector_chunk.id: 1.0}),
            patch("personal_knowledge_base.graph_rag.graph_search_results", return_value=[]),
            patch("personal_knowledge_base.graph_rag.expand_relation_context", return_value=[]),
            patch("personal_knowledge_base.model_providers.rerank", side_effect=lambda query, results, top_k=None, tenant=None: results),
        ):
            results = hybrid_search(self.tenant.id, [self.kb.id], "test", 2)

        self.assertEqual([item["chunk_id"] for item in results], [vector_chunk.id, fts_chunk.id])

    def test_hybrid_search_drops_results_below_rerank_fallback_score(self):
        chunk = self.add_chunk("retrieval result", title="candidate")

        with (
            override_settings(RETRIEVAL_RERANK_THRESHOLD=0.2, RETRIEVAL_RERANK_FALLBACK_MIN_SCORE=0.15),
            patch.object(search, "_fts_search", return_value={chunk.id: 1.0}),
            patch.object(search, "_vector_search", return_value={}),
            patch.object(search, "expand_query", return_value=[]),
            patch("personal_knowledge_base.graph_rag.graph_search_results", return_value=[]),
            patch("personal_knowledge_base.graph_rag.expand_relation_context", return_value=[]),
            patch(
                "personal_knowledge_base.model_providers.rerank",
                side_effect=lambda query, results, top_k=None, tenant=None: [
                    {**item, "rerank_score": 0.1} for item in results
                ],
            ),
        ):
            results = hybrid_search(self.tenant.id, [self.kb.id], "retrieval", 5)

        self.assertEqual(results, [])

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

    def test_hybrid_search_orders_current_and_historical_versions_by_query_intent(self):
        superseded = self.add_chunk(
            "chiller legacy setup uses pressure limit alpha and manual valve control",
            title="legacy setup",
            metadata={"lifecycle": {"status": "superseded", "superseded_by": "current-id"}},
        )
        current = self.add_chunk(
            "chiller current configuration uses temperature target beta and automatic valve control",
            title="current setup",
        )
        scores = {superseded.id: 1.0, current.id: 0.9}

        def search_for(query):
            with (
                patch.object(search, "_fts_search", return_value=scores),
                patch.object(search, "_vector_search", return_value={}),
                patch("personal_knowledge_base.graph_rag.graph_search_results", return_value=[]),
                patch("personal_knowledge_base.graph_rag.expand_relation_context", return_value=[]),
                patch(
                    "personal_knowledge_base.model_providers.rerank",
                    side_effect=lambda _query, rows, top_k=None, tenant=None: [
                        {**row, "score": 1.0} for row in rows
                    ],
                ),
            ):
                return hybrid_search(self.tenant.id, [self.kb.id], query, 2)

        neutral = search_for("chiller setup")
        historical = search_for("旧版 chiller setup")

        self.assertEqual([item["chunk_id"] for item in neutral], [current.id, superseded.id])
        self.assertEqual([item["chunk_id"] for item in historical], [superseded.id, current.id])
        self.assertEqual(neutral[1]["lifecycle_status"], "superseded")

    def test_version_aware_mmr_keeps_preferred_version_at_top_k(self):
        superseded = {"id": "old", "score": 0.9, "content": "old rule", "lifecycle_status": "superseded"}
        current = {"id": "current", "score": 0.6, "content": "current rule", "lifecycle_status": "current"}

        current_first = apply_version_aware_mmr([superseded, current], "current rule", 1)
        history_first = apply_version_aware_mmr([superseded, current], "旧版规则", 1)

        self.assertEqual(current_first, [current])
        self.assertEqual(history_first, [superseded])

    def test_relation_expansion_rows_receive_knowledge_lifecycle_status(self):
        current = self.add_chunk("current chiller valve setting with automatic controls", title="current")
        superseded = self.add_chunk(
            "old pressure regulation notes for the manual pump",
            title="superseded",
            metadata={"lifecycle": {"status": "superseded", "superseded_by": "current-id"}},
        )
        extra = {
            "chunk_id": superseded.id,
            "id": superseded.id,
            "content": superseded.content,
            "knowledge_id": superseded.knowledge_id,
            "knowledge_base_id": self.kb.id,
            "knowledge_title": superseded.knowledge.title,
            "knowledge_base_name": self.kb.name,
            "match_type": "graph",
            "metadata": {},
        }

        with (
            patch.object(search, "_fts_search", return_value={current.id: 1.0}),
            patch.object(search, "_vector_search", return_value={}),
            patch("personal_knowledge_base.graph_rag.graph_search_results", return_value=[]),
            patch("personal_knowledge_base.graph_rag.expand_relation_context", return_value=[extra]),
            patch(
                "personal_knowledge_base.model_providers.rerank",
                side_effect=lambda _query, rows, top_k=None, tenant=None: [
                    {**row, "score": 1.0} for row in rows
                ],
            ),
        ):
            results = hybrid_search(self.tenant.id, [self.kb.id], "neutral query", 2)

        self.assertEqual([item["chunk_id"] for item in results], [current.id, superseded.id])
        self.assertEqual(results[1]["lifecycle_status"], "superseded")


class CJKTextTests(SimpleTestCase):
    def test_cjk_runs_are_replaced_with_overlapping_bigrams(self):
        self.assertEqual(cjk_bigram_text("中文检索系统"), "中文 文检 检索 索系 系统")
        self.assertEqual(cjk_bigram_text("E-102报警"), "E-102 报警")
        self.assertEqual(cjk_bigram_text("中"), "中")

    def test_bigram_text_is_idempotent_and_preserves_english(self):
        for text in ("中文检索系统", "E-102报警", "A中文B"):
            transformed = cjk_bigram_text(text)
            self.assertEqual(cjk_bigram_text(transformed), transformed)
        self.assertEqual(cjk_bigram_text("English query E-102"), "English query E-102")

    def test_build_fts_query_removes_stopwords_and_prefixes_single_cjk(self):
        query = build_fts_query("什么是冷水机组")

        self.assertNotIn('"什么"', query)
        self.assertIn('"冷水"', query)
        self.assertEqual(build_fts_query("中"), '"中"*')

    def test_build_fts_query_escapes_quotes_and_skips_empty_queries(self):
        with patch.object(search, "TOKEN_RE", re.compile(r'a"b')):
            self.assertEqual(build_fts_query('a"b'), '"a""b"')
        self.assertEqual(build_fts_query(""), "")
        self.assertEqual(build_fts_query("什么 是 怎么"), "")


class RerankThresholdTests(SimpleTestCase):
    def test_keeps_items_meeting_threshold_in_order(self):
        results = [
            {"id": "first", "rerank_score": 0.8},
            {"id": "below", "rerank_score": 0.19},
            {"id": "second", "rerank_score": 0.4},
        ]

        self.assertEqual(
            [item["id"] for item in apply_rerank_threshold(results, 0.2, 0.15)],
            ["first", "second"],
        )

    def test_degrades_threshold_once_for_high_thresholds(self):
        results = [{"id": "degraded", "rerank_score": 0.4}, {"id": "low", "rerank_score": 0.2}]

        self.assertEqual(
            [item["id"] for item in apply_rerank_threshold(results, 0.5, 0.15)],
            ["degraded"],
        )

    def test_keeps_the_best_item_when_it_meets_fallback_score(self):
        results = [{"id": "best", "rerank_score": 0.16}, {"id": "lower", "rerank_score": 0.14}]

        self.assertEqual(apply_rerank_threshold(results, 0.5, 0.15), [results[0]])

    def test_returns_empty_when_all_scores_are_below_fallback(self):
        results = [{"id": "first", "rerank_score": 0.14}, {"id": "second", "rerank_score": 0.1}]

        self.assertEqual(apply_rerank_threshold(results, 0.5, 0.15), [])

    def test_unscored_results_are_returned_unchanged(self):
        results = [{"id": "first"}, {"id": "second"}]

        self.assertIs(apply_rerank_threshold(results, 0.2, 0.15), results)

    def test_missing_scores_are_below_threshold_and_input_is_not_mutated(self):
        results = [{"id": "unscored"}, {"id": "scored", "rerank_score": 0.8}]
        before = [dict(item) for item in results]

        filtered = apply_rerank_threshold(results, 0.2, 0.15)

        self.assertEqual([item["id"] for item in filtered], ["scored"])
        self.assertEqual(results, before)


class RerankProviderTests(SimpleTestCase):
    def test_rerank_preserves_input_metadata_and_records_rerank_score(self):
        metadata = {"source": "test"}
        results = [{"content": "relevant passage", "metadata": metadata, "score": 0.2}]
        response = Mock()
        response.json.return_value = {"results": [{"index": 0, "relevance_score": 0.87}]}
        cfg = {"enabled": True, "configured": True, "base_url": "https://rerank.example/v1", "model": "test-rerank"}

        with (
            patch("personal_knowledge_base.model_providers._role_config", return_value=cfg),
            patch("personal_knowledge_base.model_providers.requests.post", return_value=response),
            patch(
                "personal_knowledge_base.model_providers.usage_from_response",
                return_value={"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            ),
            patch("personal_knowledge_base.model_providers.record_model_usage"),
        ):
            ranked = rerank("query", results)

        self.assertEqual(results[0]["metadata"], {"source": "test"})
        self.assertIs(results[0]["metadata"], metadata)
        self.assertEqual(ranked[0]["metadata"], {"source": "test", "rerank_model": "test-rerank"})
        self.assertIsNot(ranked[0]["metadata"], metadata)
        self.assertEqual(ranked[0]["rerank_score"], 0.87)


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

    def test_containing_chunk_replaces_every_fragment_it_contains(self):
        results = deduplicate_results([
            {"chunk_id": "labels-a", "content": "ES-20\nXH-100\nAP-50", "score": 0.9},
            {"chunk_id": "labels-b", "content": "LG-8\nDS-65", "score": 0.8},
            {"chunk_id": "full", "content": self.FULL, "score": 0.5},
        ])

        self.assertEqual([item["chunk_id"] for item in results], ["full"])

    def test_near_duplicates_of_similar_length_keep_the_higher_score(self):
        results = deduplicate_results([
            {"chunk_id": "low", "content": self.FULL, "score": 0.2},
            {"chunk_id": "high", "content": self.FULL + " 详见拓扑图。", "score": 0.8},
            {"chunk_id": "other", "content": "E17 表示环境传感器数据过期，应暂停依赖环境数据的自动化规则。", "score": 0.1},
        ])

        self.assertEqual([item["chunk_id"] for item in results], ["high", "other"])


class VersionPolicyTests(SimpleTestCase):
    def test_query_wants_history_matches_historical_terms_only(self):
        for query in ("查旧版", "历史规则", "deprecated procedure"):
            with self.subTest(query=query):
                self.assertTrue(query_wants_history(query))
        for query in ("现在应该怎么做", "按现行规则处理", "上线之前要检查什么", "以前没装传感器的会议室怎么办"):
            with self.subTest(query=query):
                self.assertFalse(query_wants_history(query))

    def test_apply_version_policy_stably_partitions_without_dropping_results(self):
        results = [
            {"id": "current-1"},
            {"id": "old-1", "lifecycle_status": "superseded"},
            {"id": "current-2", "lifecycle_status": "current"},
            {"id": "old-2", "lifecycle_status": "superseded"},
        ]

        current_first = apply_version_policy(results, "current rule")
        history_first = apply_version_policy(results, "旧版 rule")

        self.assertEqual([item["id"] for item in current_first], ["current-1", "current-2", "old-1", "old-2"])
        self.assertEqual([item["id"] for item in history_first], ["old-1", "old-2", "current-1", "current-2"])
        self.assertEqual(len(current_first), len(results))
        self.assertEqual([item["lifecycle_status"] for item in current_first if item["id"].startswith("old")], ["superseded", "superseded"])
