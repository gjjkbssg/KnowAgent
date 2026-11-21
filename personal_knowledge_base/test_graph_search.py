from unittest.mock import PropertyMock, patch

from django.test import TestCase

from personal_knowledge_base import graph_rag
from personal_knowledge_base.agent_tools import QueryKnowledgeGraphTool
from personal_knowledge_base.graph_rag import Neo4jGraphRepository, graph_search
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant


class FakeRelationship:
    def __init__(self, rel_type):
        self.type = rel_type


class FakeTx:
    def __init__(self, records):
        self.records = records

    def run(self, query, **params):
        return self.records


class GraphSearchTests(TestCase):
    def setUp(self):
        self.tenant = Tenant.objects.create(name="graph")
        self.kb = KnowledgeBase.objects.create(tenant=self.tenant, name="kb", indexing_strategy={"graph_enabled": True})
        knowledge = Knowledge.objects.create(tenant=self.tenant, knowledge_base=self.kb, title="故障复盘", parse_status="completed")
        self.chunk = Chunk.objects.create(
            tenant=self.tenant, knowledge_base=self.kb, knowledge=knowledge, content="E17 由 CO2 传感器掉线引起", chunk_index=0
        )

    def search(self, graph):
        with patch.object(Neo4jGraphRepository, "available", new_callable=PropertyMock, return_value=True), patch.object(
            graph_rag, "extract_query_entities", return_value=["E17"]
        ), patch.object(graph_rag.graph_repository, "search_node", return_value=graph):
            return graph_search(self.tenant.id, [self.kb.id], "E17 会影响什么？")

    def test_relation_direction_follows_the_stored_edge_not_the_match_order(self):
        matched = {"name": "E17", "chunks": ["c1"]}
        neighbour = {"name": "CO2 传感器", "chunks": ["c2"]}
        records = [
            {"n": matched, "m": neighbour, "r": FakeRelationship("causes"), "forward": False},
            {"n": matched, "m": {"name": "新风规则", "chunks": []}, "r": FakeRelationship("affects"), "forward": True},
        ]

        graph = Neo4jGraphRepository._search_node(FakeTx(records), "`kb`", ["E17"])

        self.assertEqual(
            graph["relation"],
            [
                {"node1": "CO2 传感器", "node2": "E17", "type": "causes"},
                {"node1": "E17", "node2": "新风规则", "type": "affects"},
            ],
        )

    def test_graph_search_returns_chunks_and_deduplicated_relations(self):
        relation = {"node1": "CO2 传感器", "node2": "E17", "type": "causes"}
        graph = {"node": [{"name": "E17", "chunks": [self.chunk.id]}], "relation": [relation, dict(relation)]}

        results, relations = self.search(graph)

        self.assertEqual([item["chunk_id"] for item in results], [self.chunk.id])
        self.assertEqual(relations, [relation])

    def test_graph_tool_shows_the_relation_chain_to_the_model(self):
        relation = {"node1": "CO2 传感器", "node2": "E17", "type": "causes"}
        hit = {"chunk_id": self.chunk.id, "knowledge_title": "故障复盘", "content": self.chunk.content, "match_type": "graph"}
        context = {"tenant_id": self.tenant.id, "kb_ids": [self.kb.id]}
        with patch.object(graph_rag, "graph_search", return_value=([hit], [relation])), patch.object(
            graph_rag, "expand_relation_context", return_value=[]
        ):
            result = QueryKnowledgeGraphTool().execute({"query": "E17 的原因"}, context)

        self.assertIn("- CO2 传感器 -[causes]-> E17", result.output)
        self.assertIn("[1] 故障复盘 (graph)", result.output)
        self.assertEqual(result.data, [hit])
