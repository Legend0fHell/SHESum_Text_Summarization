# Academic Paper Plan Mode Artifact

Mode: `academic-paper` / `plan`

Oversight: very high

Originality spectrum: originality-oriented, but citation-locked. Do not introduce unverified references.

Current status: experiments are not yet run. Methods, related work, abstract, and experimental design can be drafted first; results, discussion of observed findings, and claims of superiority must remain placeholders until real runs are complete.

## User Checkpoint Decisions

- Evaluation size means **100 dataset samples per dataset**, not 100 semantic chunks.
- Use only existing entries in `final_report/latex/custom.bib`; missing citations must remain TODO comments or uncited implementation details.
- BGE-M3 is the embedding model. Qwen3.6-35B and Gemma4-12B are replaceable LLM choices in the experiment section, not the methodological backbone.
- Proceed with drafting Abstract, Introduction, Related Work, Method, and Experimental Setup in `final_report/latex/acl_latex.tex`.
- Main reported framework setting is Graph E2b. E1, E2a, and E2b comparisons are salience ablations, not three equal main systems.

## Paper Configuration Record

- Working title: Graph-Guided Extract-Support Summarization for Vietnamese Long and Multi-Document Texts
- Paper type: empirical NLP systems paper
- Venue/template target: ACL-style LaTeX template in `final_report/latex/acl_latex.tex`
- Bibliography file: `final_report/latex/custom.bib`
- Main language: English
- Datasets: ViMs, VN-MDS, Multi-News
- Planned evaluation size: 100 dataset samples per dataset
- Primary metrics: ROUGE-1, ROUGE-2, ROUGE-L
- Efficiency metrics: input tokens, output tokens, LLM calls, runtime seconds
- Diagnostic metrics: chunk count, community count, selected reference index, reference count, generated summary examples
- LLM choices: Qwen3.6-35B and/or Gemma4-12B through OpenAI-compatible endpoints
- Embeddings: BGE-M3
- Vietnamese entity/factual phrase extraction: Underthesea plus regex facts
- English entity/factual phrase extraction: spaCy `en_core_web_sm` plus regex facts
- Non-reportable debug settings: dry-run LLM and hash embeddings

## Terminology Lock

Use `agent_docs/terminology.md` as binding terminology for the final report.

- Use **chunk community** for Leiden output. Avoid `topic`, `topic group`, and `topical community` unless explicitly explaining why those terms are avoided.
- Use **segment** for the evidence unit constructed from up to three consecutive splitter outputs. Avoid `tri-sentence unit` in final prose.
- Use **splitter output** for raw sentence-like spans emitted by the rule-based boundary splitter.
- Use **semantic chunk** for larger chunk nodes used in the graph.

Recommended definitions:

> We use the term **chunk community** to denote a set of semantic chunks grouped by Leiden community detection on the weighted chunk graph. A chunk community is a graph-theoretic grouping induced by content, entity/factual overlap, and positional affinity; it is not assumed to correspond to a single linguistic topic.

> We define a **segment** as a sentence-like evidence unit produced from the rule-based boundary splitter. Each segment is constructed from up to three consecutive splitter outputs using an overlapping window. Because Vietnamese news text may contain headlines, clauses, fragments, or over-segmented spans, a segment is not guaranteed to be a grammatical sentence. The three-output window is a best-effort mechanism for preserving local context, especially when short spans depend on neighboring text.

## Core Claim Under Construction

The paper should not claim that graph clustering discovers linguistic topics. The defensible claim is narrower and stronger:

> A training-free summarization pipeline can reduce reliance on repeated long-context LLM calls by using source-grounded segment evidence, semantic chunks, sparse chunk-community detection, and source-only support propagation before abstractive generation.

The empirical claim remains conditional until experiments are run:

> We test whether chunk-community grouping and source-only Extract-Support prompting improve ROUGE and efficiency over a sequential no-graph baseline and a pure full-text LLM baseline.

## Chapter Plan

### Abstract

Purpose: State problem, method, datasets, metrics, and pending/observed results without overclaiming.

Required content:

- Vietnamese long-document and multi-document summarization is the motivating setting.
- Training-free means no task-specific summarization training; pretrained embeddings, NLP tools, graph algorithms, and LLM prompting are allowed.
- Method: rule-based splitter outputs -> overlapping segments -> semantic chunks -> chunk graph -> Leiden chunk communities -> Extract-Support generation -> source-only support propagation.
- Datasets: ViMs, VN-MDS, Multi-News.
- Metrics: ROUGE-1, ROUGE-2, ROUGE-L, input/output tokens, LLM calls, runtime.
- Baselines: no-graph Extract-Support and pure LLM full-text baseline.
- Results sentence remains a placeholder until experiments finish.

Oversight note: Do not write `improves` or `outperforms` in the abstract until real results exist. Use `we evaluate` or `we test`.

### 1. Introduction

Purpose: Establish why Vietnamese long/multi-document summarization needs a low-training, low-LLM-call framework.

Planned paragraph sequence:

1. Long and multi-document summarization problem: distributed evidence, long contexts, risk of unsupported generation.
2. Vietnamese setting: fewer task-specific resources, practical need for news/report/social monitoring summarization.
3. Limitation of pure full-text prompting and naive hierarchical merging: either too context-heavy or prone to losing source details.
4. Proposed principle: keep source-originated support separate from generated summaries.
5. Proposed method: chunk graph and Leiden chunk communities, but defined as mathematical communities rather than linguistic topics.
6. Contributions.

Contribution wording:

- A training-free graph-guided Extract-Support framework for Vietnamese long-document and multi-document summarization.
- A segment-based source evidence representation that compensates for rule-based splitter over-segmentation.
- A sparse weighted chunk graph combining positional, entity/factual phrase, and embedding similarity.
- A source-only support propagation rule across community-level and final merge stages.
- A controlled evaluation design on ViMs, VN-MDS, and Multi-News with pure LLM and no-graph baselines.

### 2. Related Work

Purpose: Position the method without framing the paper as a direct competition with MoA.

Recommended subsections:

#### 2.1 Hierarchical and Context-Aware Long-Document Summarization

Use `ou-lapata-2025-hierarchical` for Context-Aware Hierarchical Merging. Explain Extract-Support as the direct motivation for source-derived support at later stages.

Citation status: available in `custom.bib`.

#### 2.2 Structure-Aware Segmentation and Semantic Chunking

Use `chen-2025-cothssum` for structured long-document summarization and hierarchical segmentation.

Citation status: available in `custom.bib`.

Citation gap: LangChain SemanticChunker is software/documentation, not currently in `custom.bib`. Decide whether to cite it formally or describe it as implementation detail.

#### 2.3 Graph-Based Clustering and Chunk Communities

Use `traag-2019-leiden` for Leiden community detection and `ashkenazi-2025-d2cs` for document graph clustering precedent.

Citation status: both available in `custom.bib`.

Argument boundary: D2CS is a precedent for graph construction plus clustering, not a direct summarization baseline.

#### 2.4 Extractive Support Selection

Use `zheng-lapata-2019-pacsum` for PACSUM.

Citation status: available in `custom.bib`.

Need to explain that the implementation adapts PACSUM with BGE-M3 cosine scores and a token budget instead of fixed extract count.

#### 2.5 Lightweight Entity and Factual Phrase Processing

Use `zhao-2025-e2graphrag` for efficient graph/entity retrieval motivation.

Use `tuan-2026-moa` only as inspiration or optional baseline, not as the main frame.

Citation status: both available in `custom.bib`.

Citation gaps: Underthesea, spaCy, and possibly Vietnamese NER tool references are not currently in `custom.bib`.

### 3. Method

Purpose: Make the implementation reproducible and conceptually precise.

Recommended subsections:

#### 3.1 Overview

Pipeline:

```text
documents
-> splitter outputs
-> overlapping segments
-> semantic chunks
-> sparse weighted chunk graph
-> Leiden chunk communities
-> community-level Extract-Support summaries
-> final source-supported merge
```

#### 3.2 Segment Construction

Define splitter outputs and segments. Emphasize best-effort local context preservation, not grammatical sentence claims.

Mathematical notation:

```text
g_i = concat(x_{i-1}, x_i, x_{i+1})
```

where `x_i` is a splitter output and boundary segments use available neighbors.

#### 3.3 Semantic Chunking

Describe BGE-M3 embeddings and LangChain SemanticChunker, with paragraph preservation as structural hint. State chunk target range as 1,000-1,500 whitespace tokens.

Citation status: BGE-M3 has `bge_m3` in `custom.bib`; SemanticChunker citation missing if needed.

#### 3.4 Entity and Factual Phrase Extraction

Describe Underthesea/spaCy NER plus regex facts. State chunk-level extraction. Describe alias consolidation using embeddings only for named entities; numeric/date/money facts are not embedding-merged.

Citation gaps: Underthesea, spaCy.

#### 3.5 Source Support Selection

Describe E2b as the default support selector and E1/E2a/E2b as salience ablations.

- E1: segment-to-chunk centroid cosine.
- E2a: PACSUM over all segment candidates.
- E2b: PACSUM over stride-2 segment candidates.

Use `segment` terminology throughout.

#### 3.6 Sparse Weighted Chunk Graph

Define edge candidates and weights:

```text
w_ij = alpha P_ij + beta E_ij + gamma C_ij
```

with:

- `P_ij`: document-relative positional similarity.
- `E_ij`: typed TF-IDF cosine over canonical phrase sets.
- `C_ij`: embedding cosine similarity.

Grid:

- alpha in {0.00, 0.05, 0.10, 0.15, 0.20}
- beta from 0 to 1-alpha, step 0.10, including endpoint
- gamma = 1-alpha-beta

#### 3.7 Leiden Chunk Communities

Define chunk communities using the terminology lock. Explain that chunks are sorted back into source order before prompting.

#### 3.8 Community-Level Extract-Support Generation

Describe prompt inputs: ordered community chunks plus compacted source support. Avoid claiming chain-of-thought unless prompts actually request private reasoning; current implementation does not rely on CoT output.

#### 3.9 Source-Only Higher-Level Support Propagation

State:

```text
E_parent subset union(E_child)
```

This is the key factuality-oriented mechanism.

### 4. Experimental Setup

Purpose: Pre-register the comparison before results exist.

Recommended subsections:

#### 4.1 Datasets

- ViMs: local `datasets/ViMs/original/Cluster_*/original/*.txt` as documents; `datasets/ViMs/summary/Cluster_*/*.gold.txt` as two annotator references.
- VN-MDS: local `datasets/VN-MDS/clusters/cluster_*/*.body.txt` as documents; raw `cluster_*.ref1.txt` and `cluster_*.ref2.txt` as references; tokenized `.tok.txt` files excluded.
- Multi-News: HuggingFace `Awesome075/multi_news_parquet`, `test` split.

Planned test size: 100 dataset samples per dataset.

Citation status: ViMs and Multi-News entries are available. VN-MDS metadata is not yet a BibTeX entry.

#### 4.2 Systems Compared

- Pure LLM: feed all source text directly to the LLM; no framework modifications.
- No-Graph E2b: semantic chunks and Extract-Support, but each chunk is summarized sequentially without Leiden grouping.
- Graph E2b: main proposed setting with chunk communities and stride-2 PACSUM segments.
- Salience ablation: Graph E1, Graph E2a, and Graph E2b under matched settings.
- Optional MoA baseline: only if fairly reproduced or clearly separated as a reference baseline.

#### 4.3 Model and Runtime Configuration

Report:

- LLM model and serving endpoint family.
- Temperature and decoding settings.
- Embedding backend and model.
- Entity backend per language.
- Hardware if runtime is reported.

#### 4.4 Evaluation

Metrics:

- ROUGE-1, ROUGE-2, ROUGE-L.
- Input tokens, output tokens, LLM calls, runtime seconds.

Reference policy:

- Score each generated summary against both human references when two are available.
- Select one reference by maximum mean of ROUGE-2 and ROUGE-L.
- Report ROUGE-1, ROUGE-2, and ROUGE-L from the same selected reference.

Use `lin-2004-rouge` for ROUGE.

### 5. Results

Purpose: Placeholder until real runs finish.

Tables to prepare:

- Main results by dataset and system.
- Efficiency comparison.
- Salience variant comparison.
- Graph weight ablation.
- English generalization on Multi-News.
- Qualitative examples using `generated_summary`.

Do not fill with dry-run outputs.

### 6. Discussion

Purpose: Interpret actual outcomes only after experiments.

Pre-planned analysis questions:

- Does graph community grouping reduce LLM calls or tokens relative to no-graph and pure LLM baselines?
- Does source-only support improve ROUGE under comparable LLM settings?
- Does the main E2b setting preserve quality while reducing redundancy or cost relative to E1 and E2a ablations?
- Which graph edge signal dominates: content, entity/factual phrase overlap, or position?
- Do Vietnamese conclusions transfer to Multi-News?

### 7. Limitations

Known limitations before experiments:

- Results pending.
- Pure LLM may exceed context windows for some samples.
- Token-bounded community splitting is not fully implemented.
- Entity layer is lightweight and lacks relation extraction.
- Vietnamese NER currently uses Underthesea; PhoNLP/VnCoreNLP are future alternatives.
- ROUGE measures lexical overlap and does not directly prove factual consistency.

### 8. Conclusion

Purpose: Summarize method and empirical implications after results. Before results, keep conclusion method-focused.

## INSIGHT Collection

INSIGHT 1: The main originality is not Leiden alone. It is the combination of chunk-community grouping with source-only Extract-Support propagation.

INSIGHT 2: The term `topic` creates an avoidable validity problem. `Chunk community` is more precise because the grouping is graph-theoretic, not necessarily linguistic.

INSIGHT 3: The segment definition protects the method from overclaiming sentence-level linguistic accuracy. The unit is best understood as an overlapping evidence window over rule-based splitter outputs.

INSIGHT 4: The strongest baseline is not only no-graph. The pure LLM full-text baseline is necessary to answer whether the framework adds value beyond direct long-context prompting.

INSIGHT 5: The two-reference evaluation policy should be transparent. Selecting a single best reference per prediction avoids mixing ROUGE-2 from one annotator and ROUGE-L from another.

INSIGHT 6: MoA should stay in related work and optional baseline framing. Direct competition framing would misrepresent the contribution and may overstate comparability.

INSIGHT 7: The paper can draft methods and related work now, but must delay empirical claims until 100-unit-per-dataset experiments are complete.

INSIGHT 8: The citation surface is currently the main integrity risk. Several implementation concepts need verified BibTeX entries before final drafting.

## Citation Inventory

Available in `final_report/latex/custom.bib`:

- `traag-2019-leiden`: Leiden community detection.
- `zheng-lapata-2019-pacsum`: PACSUM.
- `ou-lapata-2025-hierarchical`: Context-Aware Hierarchical Merging / Extract-Support motivation.
- `zhao-2025-e2graphrag`: efficient graph/entity RAG motivation.
- `ashkenazi-2025-d2cs`: graph clustering precedent.
- `chen-2025-cothssum`: structured long-document summarization / hierarchical segmentation.
- `tuan-2026-moa`: MoA inspiration / optional baseline.
- `lin-2004-rouge`: ROUGE.
- `tran2020vims`: ViMs.
- `fabbri2019multi`: Multi-News.
- `qwen36_35b_a3b`: Qwen model citation, if used.
- `gemma4_12b`: Gemma model citation, if used.
- `bge_m3`: BGE-M3 embeddings.

Citation gaps requiring user-provided BibTeX or permission to search/verify:

- VN-MDS / VietnameseMDS dataset citation.
- Underthesea software citation.
- spaCy software citation.
- LangChain SemanticChunker or LangChain Experimental citation, if mentioned as more than implementation detail.
- `rouge-score` Python package citation, if package-level provenance is cited separately from ROUGE.
- HuggingFace `Awesome075/multi_news_parquet` dataset mirror citation, if required beyond original Multi-News.
- igraph and leidenalg software citations, if implementation dependencies are cited.
- Python/pandas citation, only if required by venue or reproducibility policy.

Do not add any of these citations manually without verified metadata.

## LaTeX Drafting Plan

Target file: `final_report/latex/acl_latex.tex`

Bibliography file: `final_report/latex/custom.bib`

First safe drafting pass after checkpoint:

1. Replace the current unrelated title/abstract/introduction material with this paper's title and non-result abstract.
2. Draft Introduction, Related Work, and Method using only verified citation keys above.
3. Draft Experimental Setup with TODO markers for model endpoints, hardware, and actual counts.
4. Leave Results as tables with TODO cells.
5. Add a citation TODO block in comments for missing BibTeX items.

Do not write claims such as `outperforms`, `significantly improves`, or `state-of-the-art` until real experiment outputs are available.

## Socratic Checkpoint Questions

1. What exactly does `100 chunks of each dataset` mean for evaluation: 100 dataset samples, first samples until 100 semantic chunks are produced, or exactly 100 semantic chunks sampled across the dataset?
2. Should the first LaTeX draft be anonymous ACL-style or preprint-style with the current author block?
3. Which LLM should be treated as the main model in the paper: Qwen3.6-35B, Gemma4-12B, or both as equal backbones?
4. Should pure LLM be a main baseline table row for every dataset, or a separate context-window stress-test subsection?
5. Do you want the paper to emphasize Vietnamese-first contribution, or training-free low-call summarization more broadly with Vietnamese as the main evaluation setting?
6. May I search/verify missing citation metadata, or should I only use BibTeX entries you provide manually?
