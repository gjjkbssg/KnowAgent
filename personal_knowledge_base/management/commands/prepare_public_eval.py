import hashlib
import json
import random
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from services.knowledge import delete_knowledge_content
from personal_knowledge_base.model_providers import embedding
from personal_knowledge_base.models import Chunk, Knowledge, KnowledgeBase, Tenant, User
from personal_knowledge_base.search import index_chunk, pack_embedding

PUBLIC_KB_NAME = "[评测] DuRetrieval 子集"
DATASET = "C-MTEB/DuRetrieval"
EMBEDDING_BATCH = 10


def sample_dataset(seed: int, query_count: int, negative_count: int, max_chars: int) -> dict:
    """
    按固定种子从 DuRetrieval 里抽样：若干问题、它们的全部正例段落、再加随机负例段落。
    只保留不超过 max_chars 的段落，这样一个段落正好是一个 Chunk。
    """
    from datasets import load_dataset

    corpus = load_dataset(DATASET, split="corpus")
    queries = load_dataset(DATASET, split="queries")
    qrels = load_dataset(f"{DATASET}-qrels", split="dev")

    lengths = dict(zip(corpus["id"], (len(text) for text in corpus["text"])))
    positives: dict[str, list[str]] = {}
    for qid, pid, score in zip(qrels["qid"], qrels["pid"], qrels["score"]):
        if score > 0 and lengths.get(pid, max_chars + 1) <= max_chars:
            positives.setdefault(qid, []).append(pid)

    rng = random.Random(seed)
    qids = sorted(rng.sample(sorted(positives), query_count))
    positive_pids = {pid for qid in qids for pid in positives[qid]}
    all_positive_pids = {pid for pids in positives.values() for pid in pids}
    candidates = sorted(pid for pid, size in lengths.items() if size <= max_chars and pid not in all_positive_pids)
    negative_pids = set(rng.sample(candidates, negative_count))

    wanted = positive_pids | negative_pids
    passages = {pid: text for pid, text in zip(corpus["id"], corpus["text"]) if pid in wanted}
    query_text = dict(zip(queries["id"], queries["text"]))
    return {
        "passages": passages,
        "questions": [
            {
                "id": qid,
                "category": "DuRetrieval",
                "question": query_text[qid],
                "expected_sources": sorted(positives[qid]),
            }
            for qid in qids
        ],
    }


class Command(BaseCommand):
    help = "Build a retrieval-only evaluation knowledge base from a seeded sample of the public DuRetrieval benchmark."

    def add_arguments(self, parser):
        parser.add_argument("--seed", type=int, default=20261122)
        parser.add_argument("--queries", type=int, default=100)
        parser.add_argument("--negatives", type=int, default=1000)
        parser.add_argument("--max-chars", type=int, default=600)
        parser.add_argument("--questions-out", default="media/public_eval/questions.json")
        parser.add_argument("--rebuild", action="store_true")

    def handle(self, *args, **options):
        tenant = Tenant.objects.order_by("created_at").first()
        if not tenant:
            raise CommandError("No tenant exists. Run auth auto-setup first.")
        user = User.objects.filter(tenant=tenant).order_by("created_at").first()

        existing = KnowledgeBase.objects.filter(tenant=tenant, name=PUBLIC_KB_NAME).first()
        if existing and not options["rebuild"]:
            raise CommandError("Public eval knowledge base already exists; pass --rebuild to replace it.")
        if existing:
            for item in Knowledge.objects.filter(knowledge_base=existing):
                delete_knowledge_content(item)
            existing.delete()

        sample = sample_dataset(options["seed"], options["queries"], options["negatives"], options["max_chars"])
        passages = sample["passages"]
        self.stdout.write(f"sampled {len(sample['questions'])} questions, {len(passages)} passages")

        kb = KnowledgeBase.objects.create(
            tenant=tenant,
            name=PUBLIC_KB_NAME,
            description="DuRetrieval 公开检索基准的固定种子抽样，只用于检索评测。一个段落一个 Chunk，不建图谱和 Wiki。",
            type="document",
            chunking_config={"chunk_size": options["max_chars"], "chunk_overlap": 0},
            indexing_strategy={
                "vector_enabled": True,
                "keyword_enabled": True,
                "wiki_enabled": False,
                "graph_enabled": False,
            },
            creator_id=user.id if user else "",
        )

        pids = sorted(passages)
        for start in range(0, len(pids), EMBEDDING_BATCH):
            batch = pids[start : start + EMBEDDING_BATCH]
            # Embedding 是网络请求，放在写事务外面
            vectors = embedding(tenant, [passages[pid] for pid in batch])
            with transaction.atomic():
                for pid, vector in zip(batch, vectors):
                    text = passages[pid]
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    item = Knowledge.objects.create(
                        tenant=tenant,
                        knowledge_base=kb,
                        type="file",
                        title=f"DuRetrieval {pid[:12]}",
                        source=pid,
                        parse_status="completed",
                        file_name=f"{pid}.txt",
                        file_type="txt",
                        file_size=len(text.encode("utf-8")),
                        file_hash=digest,
                    )
                    chunk = Chunk.objects.create(
                        tenant=tenant,
                        knowledge_base=kb,
                        knowledge=item,
                        content=text,
                        chunk_index=0,
                        start_at=0,
                        end_at=len(text),
                        metadata={"title": item.title, "source": pid},
                        content_hash=digest,
                    )
                    index_chunk(chunk, pack_embedding(vector))
            if (start // EMBEDDING_BATCH) % 20 == 0:
                self.stdout.write(f"indexed {min(start + EMBEDDING_BATCH, len(pids))}/{len(pids)}")

        questions_path = Path(options["questions_out"]).resolve()
        questions_path.parent.mkdir(parents=True, exist_ok=True)
        questions_path.write_text(json.dumps(sample["questions"], ensure_ascii=False, indent=2), encoding="utf-8")
        self.stdout.write(
            self.style.SUCCESS(
                f"public eval ready: kb_id={kb.id} chunks={Chunk.objects.filter(knowledge_base=kb).count()} "
                f"questions={questions_path}"
            )
        )
