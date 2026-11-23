# KnowAgent 的 RAG 和 Agent：一次问答从头到尾怎么走

这份文档带你走完两条链路：一次普通 RAG 问答，一次 Agent 问答。每一步给出三样东西：

- **我们的代码**：文件和行号（以当前 `main` 为准）。
- **WeKnora 的对应实现**：腾讯开源项目 [Tencent/WeKnora](https://github.com/Tencent/WeKnora)，对照的是 `7a3ca49`。
- **为什么这样设计**：面试官真正想听的部分。

先记住一句话：**RAG 是“先检索、再回答”的固定流水线；Agent 是“模型自己决定要不要查、查什么、查几次”的循环。** 两者用的是同一套检索函数 `hybrid_search`。区别只在于谁来调用它、调用几次。

---

## 0. 全景图

```text
离线入库：文件 → 解析 → 切块 → 向量化 + 全文索引 + 图谱抽取 → SQLite / sqlite-vec / Neo4j

在线问答：
  RAG   ：问题 → 意图识别/改写 → hybrid_search(一次) → 拼上下文 → 模型生成
  Agent ：问题 → [模型思考 → 调工具(可能是 hybrid_search) → 看结果]×N → 模型给最终答案
```

| 能力 | 我们 | WeKnora |
|---|---|---|
| 语言 | Python（Django） | Go |
| RAG 流水线 | `rag_pipeline.run_rag_pipeline` + `search.hybrid_search` | `chat_pipeline` 插件链（`types/chat_manage.go:446` 的 `Pipeline`） |
| Agent 循环 | `agent_engine.AgentEngine._execute` | `agent/engine.go` 的 `executeLoop` / `runReActIteration` |
| 框架 | 都没用 LangChain、LlamaIndex，全部手写 | 同左 |

---

## 1. 离线入库：一份文档怎么变成可检索的数据

| 步骤 | 我们 | WeKnora | 要点 |
|---|---|---|---|
| 解析 | `document_processing.extract_text_from_bytes` | `docreader` 服务 | 把 PDF/Word/Markdown 变成纯文本块 |
| 切块 | `split_text`（`document_processing.py:57`）：512 字符，重叠 50，优先在换行或句号处断开，最后一小段并入前一块 | `chunker/splitter.go:105`：512 字符，重叠 80 | 块太大，检索不准；块太小，丢上下文。重叠是为了防止一句话被切断 |
| 向量 | Embedding 接口返回 1024 维，做 L2 归一化后存进 sqlite-vec | 存进向量库（Milvus/pgvector/SQLite 等） | 归一化后，内积就等于余弦相似度 |
| 全文索引 | SQLite FTS5，中文先切成重叠二元组（`cjk_bigram_text`） | SQLite 后端同样用 `tokenizeCJKBigram` | FTS5 的默认分词器会把一整串中文当成一个词，不切二元组就几乎搜不到中文 |
| 图谱 | 用模型抽取实体和关系，写入 Neo4j | 同样有实体抽取和图谱检索 | 给多跳问题用，但评测显示收益不明显（见第 5 节） |

**面试怎么说**：“入库时一份文档会建三种索引：向量索引负责语义相近，全文索引负责精确的型号和错误码，图谱负责实体关系。中文全文检索必须先切二元组，这是我们和 WeKnora 都踩过的坑。”

---

## 2. 一次 RAG 问答

WeKnora 把 RAG 写成一串阶段（`types/chat_manage.go:464` 的 `rag_stream`）。我们按同样的顺序对照：

| WeKnora 阶段 | 做什么 | 我们的代码 |
|---|---|---|
| `LOAD_HISTORY` | 读最近几轮对话 | `rag_pipeline._recent_dialogue` |
| `QUERY_UNDERSTAND` | 判断意图（闲聊还是查知识库），把“它的排查顺序呢？”这类追问改写成完整问题 | `_quick_intent_detect`（规则快判）+ `understand_query`（模型改写），`rag_pipeline.py:84-140` |
| `CHUNK_SEARCH_PARALLEL` | 关键词检索和向量检索并行 | `hybrid_search`，`search.py:624` |
| `CHUNK_RERANK` | 用重排模型给候选重新打分 | `search.py:721` 调 `qwen3-rerank` |
| `CHUNK_MERGE` / `FILTER_TOP_K` | 去重、合并相邻片段、截断到 Top-K | `deduplicate_results`、`diversify_by_knowledge`、`apply_mmr`、`expand_short_chunks` |
| `INTO_CHAT_MESSAGE` | 把片段拼成提示词 | `_build_user_prompt` |
| `CHAT_COMPLETION_STREAM` | 调模型流式生成 | `services/chat_generation.py` 的非 Agent 分支 |

### 2.1 `hybrid_search` 逐行拆解（最重要，必须讲得出来）

```python
def hybrid_search(tenant_id, kb_ids, query, top_k=10):
    fts   = _fts_search(...)      # ① BM25 全文检索，每路取 top_k*4
    vec   = _vector_search(...)   # ② sqlite-vec 向量检索，每路取 top_k*4
    rankings = [rank(fts), rank(vec)]           # 每路只保留“名次”，不看原始分数
    if 召回太少: 追加查询扩展的全文结果          # ③ 查询扩展
    rankings.append(graph_search(...))           # ④ 图谱命中的片段作为第三路
    scores = rrf_fuse(rankings, weights)         # ⑤ 加权 RRF 融合
    候选 = 前 top_k*4 条 + 图谱关系扩展的片段
    候选 = rerank(query, 候选, top_k*2)          # ⑥ 重排模型
    候选 = 丢掉低于阈值 0.4 的                    # ⑦ Rerank 阈值
    候选 = 已废止文档排到现行版本后面             # ⑧ 版本感知（我们独有）
    候选 = 每篇文档最多 2 条                      # ⑨ 文档级多样性
    候选 = MMR(λ=0.7) 选出 top_k                  # ⑩ 去冗余
    return 过短片段补全上下文(候选)               # ⑪ 短块扩展
```

逐项解释：

- **为什么用 RRF 而不是把分数相加？** BM25 分数可以是 15，余弦相似度在 0 到 1 之间，量纲不同，直接相加没有意义。RRF 只看名次：`score = Σ wᵢ / (k + rankᵢ)`，k=60。WeKnora 同样用 k=60（`types/retrieval_config.go:93`），默认权重向量 0.7、关键词 0.3（同文件 :102）。我们照抄了这组默认值。
- **为什么还要 Rerank？** 召回阶段的向量模型是“双塔”结构：问题和片段分开编码，速度快但精度一般。重排模型把问题和片段放在一起编码（交叉编码），更准但更慢，所以只对少量候选使用。
- **Rerank 阈值**：WeKnora 默认 0.2，全部候选都被拒绝时，保留分数高于 0.15 的最好一条（`reranking/rerank.go:24-28`）。我们实测 qwen3-rerank 的打分整体偏高，0.2 一条都过滤不掉，所以根据分数分布改成 0.4。
- **MMR**：λ=0.7 表示 70% 看相关性、30% 看与已选片段的差异，避免前几条都是同一段话。WeKnora 的 `reranking/mmr.go:12` 也是 `DefaultMMRLambda = 0.7`。
- **版本感知**：这是我们在评测中发现问题后加的，WeKnora 没有。已废止的 v2.3 手册不删除，只排在 v2.4 后面；问题里出现“旧版”“上一版”时，再把旧版排到前面。

**面试怎么说**：“一次检索走三路召回、RRF 融合、重排、阈值过滤和 MMR 去冗余。每一步的默认参数都对照过 WeKnora，偏离的地方，比如重排阈值 0.4，是用评测数据校准出来的。”

---

## 3. 一次 Agent 问答

### 3.1 循环本身（ReAct / Function Calling）

我们的 `_execute`（`agent_engine.py:553-819`）简化后是这样：

```python
messages = [system_prompt, *history, user_question]
for iteration in range(1, max_iterations + 1):           # 默认 20 轮，WeKnora 也是 20
    messages = manage_context_window(messages)           # 超过 50% 上限时用模型做摘要，超过 80% 时按滑动窗口截断
    reply = llm(messages, tools=工具的 JSON Schema)       # Think：模型决定说话还是调工具
    if not reply.tool_calls:                             # 不调工具 = 给出最终答案
        if 内容为空: 追加“请提供完整回答”，重试；continue
        return 最终答案 + 结构化引用
    if 和上一轮的工具调用完全相同: 判定卡死，停下用已有结果作答
    if 有需要审批的 MCP 工具: 保存检查点，暂停，等用户批准
    results = 并行执行工具(reply.tool_calls)              # Act：线程池并行，每个工具有超时
    messages += 工具结果（每条截断到 16 KB）               # Observe：结果作为 tool 消息追加回对话
    if 连续 3 轮工具全部失败: 停止，用已有信息作答
return 达到最大轮数，基于已有信息作答
```

WeKnora 的 `runReActIteration`（`agent/engine.go:860`）注释里写的就是 “think → analyze → act → observe”，结构相同。

### 3.2 停止条件对照（面试高频）

| 防护 | 我们 | WeKnora（`agent/const.go`） |
|---|---|---|
| 最大轮数 | 20 | `DefaultAgentMaxIterations = 20` |
| 卡死检测 | 连续 2 轮发出完全相同的工具调用 | 连续 2 轮回复内容相同（`maxRepeatedResponseRounds = 2`） |
| 工具连续失败 | 连续 3 轮工具全部失败就停 | 同一调用连续失败 3 次（`maxConsecutiveFailedToolCalls = 3`） |
| 模型调用重试 | 按状态码和 `Retry-After` 退避重试 | `maxLLMRetries = 2`，`Retry-After` 最多等 30 秒 |
| 工具超时 | 有 | `defaultToolExecTimeout = 60s` |
| 空回复 | 追加提示后重试 | `maxEmptyResponseRetries = 2` |
| 上下文过长 | 超过 50% 摘要压缩，超过 80% 滑动窗口截断 | `compaction` 包：超过阈值时生成摘要，并存成检查点 |
| 危险操作 | MCP 工具调用前需人工审批，暂停期间状态存数据库，批准后从断点继续 | `approval` + `hitl` 包，思路相同 |

**为什么需要这些防护？** 模型不保证收敛。它可能反复调用同一个工具，可能一直拿到空结果还继续重试，也可能对着失败的接口无限循环。每一条防护都对应一种真实的失控方式。

### 3.3 工具

- 默认工具列表（`agent_tools.py:1080` 的 `DEFAULT_ALLOWED_TOOLS`）：Wiki 4 个、知识库 4 个、图谱 1 个，推理用的 `thinking`、`todo_write`，派子 Agent 用的 `actor`，以及外部工具 `database_query`（只读）、`web_search`、`web_fetch`。
- 核心工具 `knowledge_search`（`agent_tools.py:206`）内部调用的就是第 2 节的 `hybrid_search`。它多做了一件事：同一次执行里已经返回过的片段，再次命中时只显示标题，标注“已返回过”，以节省上下文。WeKnora 的 `seenChunks` 做法相同。
- 已废止的文档在工具输出里会标上“（已废止）”，工具描述里也告诉模型以现行版本为准。

### 3.4 多 Agent（子 Agent）

产品的 Agent 模式下，主 Agent 只拿到 `actor` 和 `thinking` 两个工具，通过 `actor` 把子任务派给 4 种子 Agent（`agent_actor.py:17`）：

| 子 Agent | 可用工具 | 最多轮数 |
|---|---|---|
| `doc_retriever` 文档检索 | knowledge_search、grep_chunks、list_knowledge_docs、get_document_info | 6 |
| `wiki_researcher` Wiki 研究 | wiki 系列 4 个 | 6 |
| `graph_reasoner` 图谱推理 | query_knowledge_graph、wiki、knowledge_search | 6 |
| `answer_writer` 答案综合 | thinking | 4 |

子 Agent 不能再派子 Agent，这样调用深度最多两层。所有子 Agent 检索到的片段会汇总到同一个 `ReferenceCollector`，最终答案的引用来自这里。

---

## 4. RAG 和 Agent 各自适合什么问题

最直观的例子是人物多跳题（`interview_materials/06_Agent_vs_RAG人物实证.md`）：

> 谁把北辰项目端侧推理 P95 从 180ms 降到 70ms？这个人后来负责哪个项目，他的导师是谁？

- **RAG**：只用原问题检索一次。Top-5 里有性能复盘（能知道是林晓），但没有林晓的访谈，所以后两问答不出来。
- **Agent**：第一次检索得到“林晓”，再拿“林晓”检索一次，找到访谈，三问全部答对。

**原理**：第二跳的查询词（“林晓”）是第一跳检索之后才知道的，固定流水线无法提前构造这个查询。Agent 的价值就在这里。代价是更多的模型调用、更高的延迟和更多 token。

第 5 节的评测用 31 道题量化这个取舍。

---

## 5. 评测：直接用 WeKnora 的指标

WeKnora 的评测（`internal/application/service/metric/`）只有检索和生成两类指标，我们按原逻辑移植成了 Python（`personal_knowledge_base/eval_metrics.py`），并把它的 Go 单元测试用例全部搬过来，证明数值一致。

| 指标 | 含义（以一道题为例） |
|---|---|
| Precision | 返回的结果里，有多少比例是正确证据 |
| Recall | 正确证据里，有多少比例被找到 |
| MRR | 第一条正确证据排第几，取倒数（排第 1 得 1 分，排第 2 得 0.5 分） |
| MAP | 每找到一条正确证据，就计算一次当时的 Precision，再取平均；同时兼顾“找全”和“排前” |
| NDCG@3 / @10 | 前 3 / 前 10 名的排序质量，越靠前权重越大，满分为 1 |
| BLEU-1/2/4 | 生成答案和标准答案的 1/2/4 元词组重合程度（jieba 分词） |
| ROUGE-1/2/L | 同上，但偏向召回；L 是最长公共子序列 |

WeKnora 没有 Agent 指标，我们另外补了 4 个：**答案正确率**（关键信息是否都答出来）、**平均工具调用次数**、**耗时**、**token 成本**。

注意：BLEU/ROUGE 只看字面重合，模型回答一长，分数就偏低。它们适合横向比较不同方案，不能当成准确率。

### 5.1 Agent 评测结果（31 题 × 3 种方案，真实模型，每种方案跑 1 次）

语料：智能楼宇 16 份文档 + 人物多跳 10 份，再混入 DuRetrieval 的 5,489 段作为噪声。题目在 `interview_materials/agent_eval_tasks.json`：单跳 6、多跳 11、版本冲突 3、汇总对比 4、根因分析 3、无法回答 4。

**答案正确率按人工核对计**。关键词自动打分有 5 处误判，都是“答案里出现了关键词，但结论是错的”，明细在 `results/agent_eval/human_audit.json`。

| 方案 | 正确率（人工核对） | 关键词打分 | Recall | MRR | MAP | NDCG@3 | 平均 token | 平均 / 中位耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 普通 RAG | **83.9%**（26/31） | 93.5% | 0.932 | 0.858 | 0.798 | 0.831 | 6.9k | 23.9 / 21.7 s |
| 单 Agent | **93.5%**（29/31） | 93.5% | 0.944 | 0.920 | 0.862 | 0.866 | 21.3k | 30.7 / 22.0 s |
| 多 Agent（产品默认的 Agent 模式） | **96.8%**（30/31） | 96.8% | 0.907 | 0.781 | 0.678 | 0.684 | 75.3k | 73.3 / 63.9 s |

检索指标按文档计：Agent 的“检索结果”是它各次工具调用返回的片段，按首次出现的顺序排列，所以和 RAG 的单次 Top-5 不完全可比。

按题型（人工核对）：

| 题型 | 普通 RAG | 单 Agent | 多 Agent |
|---|---:|---:|---:|
| 单跳事实（6） | 6 | 5 | 6 |
| 多跳推理（11） | **7** | 10 | 11 |
| 版本冲突（3） | 2 | 3 | 3 |
| 汇总对比（4） | 4 | 4 | 4 |
| 根因分析（3） | 3 | 3 | 3 |
| 无法回答（4） | 4 | 4 | 3 |

能得出的结论：

1. **差距集中在多跳题**。RAG 在多跳题上答对 7/11。丢掉的 4 道里，3 道是缺一份必需的文档：人物题缺林晓的访谈，B210 缺 ES-21 规格，C102 缺设备清单。第 4 道（A305 关灯）两份文档都检索到了，但模型读错了 v2.4 的规则。单跳和汇总题上，三种方案没有区别。
2. **Agent 的代价是成本**：单 Agent 的 token 约为 RAG 的 3 倍，多 Agent 约 11 倍，耗时约 3 倍。多 Agent 只比单 Agent 多答对 1 题。
3. **BLEU/ROUGE 反而是 RAG 最高**（BLEU-4：RAG 0.083、单 Agent 0.024、多 Agent 0.012；ROUGE-L：0.384、0.207、0.185），因为 RAG 的回答短、贴近原文。这说明 BLEU/ROUGE 衡量的是“像不像参考答案的措辞”，不是对不对。
4. **限制**：每种方案只跑了 1 次，温度 0.7；1 道题就差 3.2 个百分点；题目是自己写的。正确率差异只能看趋势，成本倍数比较稳定。

### 5.2 评测发现并修复的 Agent 缺陷

单 Agent 答错的 2 道题都不是模型能力问题，而是循环本身的缺陷：

- **达到最大轮数时不作答**：原来直接返回“已达到最大推理轮数，基于当前信息给出回答。”，却没有真的给出回答。现在照 WeKnora `finalize.go` 的 `handleMaxIterations`：撤掉工具，让模型基于已有结果强制作答。
- **工具标签被当成最终答案**：模型输出了 `</th> </invoke>`。现在去掉标签后没有正文的回复，按空回复处理，追加提示后重试。

修复后单独重跑了单 Agent 一组，人工核对的正确率仍是 29/31。原来那 2 道答对了，但另有 2 道触发了最大轮数：一道强制作答用了已废止的 v2.3 数值；另一道强制作答返回空，降级逻辑把原始工具输出直接拼成了“答案”。说明修复解决了“不作答”的问题，但“答得对不对”还取决于前面几轮查到了什么。降级时拼接原始工具输出的做法也应该改掉，WeKnora 在这种情况下返回明确的失败提示。

### 5.3 检索评测换成 WeKnora 口径（领域困难集 46 题）

| 方案 | Precision | Recall | NDCG@3 | NDCG@10 | MRR | MAP |
|---|---:|---:|---:|---:|---:|---:|
| 仅全文检索 | 0.115 | 0.924 | 0.825 | 0.839 | 0.844 | 0.792 |
| 仅向量 | 0.117 | 0.957 | 0.792 | 0.837 | 0.829 | 0.778 |
| RRF 融合 | 0.120 | 0.967 | 0.845 | 0.871 | 0.868 | 0.823 |
| + Rerank | 0.117 | 0.960 | 0.854 | 0.882 | 0.887 | 0.839 |
| + Rerank 阈值 0.4 | **0.308** | 0.960 | 0.854 | 0.882 | 0.887 | 0.839 |
| 完整流水线（含图谱） | 0.245 | 0.960 | 0.854 | 0.881 | 0.886 | 0.838 |

Rerank 阈值不改变排序，只丢掉低分片段：平均返回条数从 10 降到 5.3，Precision 从 0.117 升到 0.308，Recall 不变。图谱那一路没有带来排序收益。

---

## 6. 面试追问和参考回答

1. **为什么不用 LangChain？**
   Agent 循环本身只有一两百行，核心是“调模型 → 执行工具 → 追加结果”。手写的好处是每一条停止条件、每一次重试都看得见、可测试。WeKnora 这样的生产项目也是手写的。

2. **RRF 的 k=60 是什么意思？**
   k 用来平滑名次差距。k 越大，第 1 名和第 10 名的分数差越小，融合时越“民主”。60 是 RRF 原论文的经验值，WeKnora 和我们都用它。

3. **向量权重为什么是 0.7？**
   照抄 WeKnora 的默认值。我们在两套评测上都测过：领域集上加权更好，DuRetrieval 上等权略好。最终保留 0.7/0.3，并在 README 里写明了理由。

4. **Rerank 阈值 0.4 会不会过拟合？**
   会有风险。这个阈值是在评测集上看分数分布选出来的，换了重排模型就要重新校准。

5. **Agent 怎么防止死循环？**
   五道防线：最大轮数 20、相同工具调用连续两轮判定卡死、工具连续 3 轮失败就停、空回复重试上限，以及超时。见 3.2 的表。

6. **Agent 什么时候比 RAG 差？**
   单跳事实题。评测里单跳题 RAG 6/6、单 Agent 5/6，Agent 却多花了约 3 倍 token。Agent 的优势只在多跳题上（RAG 7/11，单 Agent 10/11）。

7. **子 Agent 有什么用？**
   把检索和综合拆开，每个子 Agent 的工具更少、提示词更专注。代价是调用链更长，延迟和 token 成本更高。

8. **MCP 工具调用怎么保证安全？**
   默认每次调用都要人工确认，服务端自称“只读”也不信。暂停时把 Agent 的完整状态存进数据库，批准后从断点继续；如果工具可能已经执行过一半，恢复时不再重试，避免重复执行。

9. **上下文窗口满了怎么办？**
   超过 50% 用模型生成摘要，超过 80% 按滑动窗口截断。旧的知识库检索结果会先被脱敏缩短，因为它们通常最长，也最没必要保留原文。

10. **你们的评测可信吗？**
    检索评测有一套公开数据（DuRetrieval 子集）和一套自己写的领域题。Agent 题是自己写的，每类只有 3 到 11 道，只能看趋势。指标算法和 WeKnora 一致，有单元测试保证。

---

## 7. 已知短板（要主动说，不要被问出来）

- Agent 评测题是实现者自己写的，题量小，每种方案只跑了 1 次，只能看趋势。
- Agent 达到最大轮数时，强制作答的质量取决于前面查到了什么；强制作答返回空时，降级逻辑会拼接原始工具输出，应改成明确的失败提示。
- 图谱在多跳题上没有测出收益：领域集上关掉 RAG 检索里的图谱通道后，各项指标完全不变，平均检索耗时从 4.93 秒降到 4.13 秒（多跳题从 4.58 秒降到 3.25 秒），所以 `RETRIEVAL_GRAPH_IN_HYBRID` 默认关闭，多跳问题交给 Agent 的图谱工具。
- 版本状态需要人工标记，系统不会自动推断哪份文档已被取代。
- BLEU/ROUGE 只看字面重合，答案正确率用的是关键词匹配，都不等于语义层面的正确。
