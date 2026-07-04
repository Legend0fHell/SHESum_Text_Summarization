# Graph-Guided Extract-Support Summarization for Vietnamese Long and Multi-Document Texts

Working draft v0.1. Citation metadata and real experiment results are not yet verified. Do not submit this version.

## Abstract

Long-document and multi-document summarization remains difficult in Vietnamese because relevant evidence is often distributed across multiple source passages, while large language model summarization over long contexts can be costly and may introduce unsupported facts. This paper proposes a training-free graph-guided Extract-Support framework for Vietnamese long-document and multi-document summarization. The framework represents source texts as overlapping segment evidence units, forms paragraph-aware semantic chunks with BGE-M3 and LangChain SemanticChunker, extracts source-only salient support evidence, builds a sparse weighted chunk graph, and applies Leiden community detection to form chunk communities. Each chunk community is summarized by an LLM using both ordered source text and selected source-derived support evidence. Higher-level merging preserves the Extract-Support constraint by reselecting support only from inherited original source units, rather than from generated summaries. The primary evaluation targets VN-MDS and ViMs, with Multi-News used for English generalization. Primary metrics are ROUGE-1, ROUGE-2, and ROUGE-L, with input tokens, output tokens, LLM calls, and runtime as efficiency submetrics. The main framework setting uses E2b stride-2 PACSUM support selection. E1 chunk-centroid cosine, E2a PACSUM over all segments, and E2b are treated as salience ablations. Results are pending real LLM runs with Qwen3.6-35B and/or Gemma4-12B.

Keywords: Vietnamese summarization; long-document summarization; multi-document summarization; graph clustering; Leiden algorithm; Extract-Support; training-free NLP

## 1. Introduction

Summarization systems increasingly rely on large language models (LLMs), but long-document and multi-document summarization still exposes two practical limitations. First, documents can exceed the effective context budget or attention reliability of the model. Second, summaries generated through repeated abstraction may drift away from the source, especially when later steps condition mainly on previous summaries rather than original evidence. These issues are important for Vietnamese summarization, where datasets and task-specific training resources are more limited than in English, but practical applications such as news aggregation, public communication monitoring, social media synthesis, and report summarization still require scalable and factual summaries.

This paper studies a training-free framework for Vietnamese long-document and multi-document summarization. Training-free does not mean model-free. The framework uses pretrained multilingual embeddings, lightweight Vietnamese NLP, graph clustering, and LLM prompting, but it does not train a task-specific summarization model. The core idea is to separate low-cost source processing from expensive abstractive generation. Sentence splitting, tri-sentence unit construction, semantic chunking, entity and factual phrase extraction, salience ranking, graph construction, and Leiden clustering are performed before LLM summarization. LLM calls are reserved for topic-level abstraction and final merging.

The framework is inspired by Context-aware Hierarchical Merging, especially its Extract-Support setting, where source-derived support passages are provided alongside summaries during later merging steps [@context_aware_hm_2025]. The proposed framework adopts the same factuality-oriented principle but changes the structure around it. Instead of merging fixed groups of chunks, the method constructs a weighted chunk graph and applies Leiden community detection to identify topical communities. Topic groups are then summarized with source support selected from original tri-sentence units. At later hierarchy levels, support evidence remains source-only: a parent node can only select support from the union of support units inherited from its children.

The design also draws on hierarchical segmentation work, which argues that long-document summarization benefits from meaningful structure rather than arbitrary fixed-length chunks [@cothssum_2025]. Our implementation therefore starts from sentence and paragraph structure, stores overlapping tri-sentence units to reduce the harm of sentence splitter over-segmentation, and uses semantic chunking to form approximately 1,000 to 1,500-token chunks. The chunk graph combines three signals: document-relative position, entity or factual phrase overlap, and embedding-based semantic similarity. Leiden community detection then gives topic-level groups whose granularity can be controlled through graph weights and resolution.

The paper asks the following research questions:

RQ1. Can graph-guided topic formation improve Vietnamese long-document and multi-document summarization compared with sequential hierarchical summarization under comparable LLM settings?

RQ2. Which source-only salience extractor is most effective for Extract-Support prompting: centroid similarity, PACSUM over all tri-sentence units, or stride-2 PACSUM?

RQ3. How do positional, entity, and content edge weights affect ROUGE quality and efficiency metrics?

RQ4. Does the Vietnamese-focused framework generalize to English Multi-News when the entity extractor is switched from Underthesea to spaCy English NER?

The intended contributions are:

1. A training-free graph-guided Extract-Support architecture for Vietnamese long-document and multi-document summarization.
2. A source-only evidence propagation rule for hierarchical summarization, where support evidence at every level remains traceable to original tri-sentence units.
3. A sparse weighted chunk graph using positional, entity/factual phrase, and content similarity, clustered with Leiden.
4. A focused main evaluation using E2b, with E1/E2a/E2b salience ablations and graph edge-weight ablations.
5. An open implementation scaffold for reproducible experiments on VN-MDS, ViMs, and Multi-News.

## 2. Related Work

### 2.1 Long-Document Hierarchical Summarization

Hierarchical summarization reduces long inputs by summarizing chunks and recursively merging intermediate summaries. Context-aware hierarchical merging shows that later merging stages can benefit from additional source-derived context, and that Extract-Support variants can improve ROUGE-2 by grounding later summaries in source passages [@context_aware_hm_2025]. This finding motivates the central constraint in our framework: generated summaries should not become the only information path upward through the hierarchy.

However, hierarchical merging can also lose details if the lower-level summary omits minority viewpoints, numerical qualifications, or contrastive evidence. The proposed framework addresses this by storing source support units separately from generated summaries. Topic nodes contain generated text for abstraction, but also retain support unit IDs from the source. When merging topics, the support pool is reselected from original support units rather than copied from generated summaries.

### 2.2 Structure-Aware Segmentation

Long inputs are not uniformly structured. News articles, social media threads, reports, and policy documents often contain paragraphs, headings, lists, and topical shifts. CoTHSSum and related structured long-document summarization work motivate treating segmentation as part of the summarization problem rather than as preprocessing noise [@cothssum_2025]. In this paper, segmentation starts from conservative text normalization, paragraph detection, sentence splitting, and overlapping tri-sentence units. LangChain Experimental SemanticChunker is used to detect semantic boundaries, while paragraph boundaries act as structural hints.

The tri-sentence unit is a practical choice. Regex sentence splitters can over-segment short question-answer patterns or abbreviations. A single sentence such as “Yes” or “Có” is often meaningless without the preceding sentence. Overlapping three-sentence windows preserve local context and become the atomic support units for extraction.

### 2.3 Graph-Based Clustering for Text

Graph clustering has been used to organize text representations by semantic or distributional similarity. D2CS is a close precedent because it constructs a k-nearest-neighbor graph and uses Leiden community detection for clustering [@d2cs_2025]. The proposed framework differs in its node granularity and purpose. D2CS clusters documents or document-level representations, while this framework clusters semantic chunks inside a summarization pipeline. The graph is not the final output; it determines topic groups for Extract-Support summarization.

Leiden is appropriate because it produces well-connected communities and scales better than dense all-pairs grouping for larger graphs [@leiden_original_todo]. The implementation therefore constructs a sparse candidate edge set using semantic kNN plus adjacent chunks from the same source document. Edge weights are calculated only for candidates, reducing graph construction cost.

### 2.4 Extractive Salience and PACSUM

Extractive support selection determines which source units are provided to the LLM. A simple method is centroid similarity: rank each unit by cosine similarity to the chunk embedding. This is cheap and reproducible but may favor central repeated information. PACSUM ranks sentences by centrality using directional graph structure and is a strong unsupervised extractive baseline [@pacsum_2019].

This paper evaluates PACSUM with BGE-M3 embeddings in two variants. E2a uses all tri-sentence units. E2b uses stride-2 candidate units, corresponding to centers 2, 4, 6, and so on. The stride-2 variant reduces overlap between adjacent candidate windows and reduces pairwise PACSUM cost. We intentionally exclude more complicated protected-candidate logic in the first version to keep the ablation focused.

### 2.5 Lightweight Entity and Factual Phrase Processing

Entity information is used as a graph edge signal and as compact factual context. E2GraphRAG motivates the value of combining hierarchical summaries with lightweight entity indexes, without necessarily constructing a full relation-heavy knowledge graph [@e2graphrag_2025]. The proposed framework follows this middle path. It extracts Vietnamese named entities with Underthesea at the chunk level and augments them with rule-based factual phrase extraction for dates, percentages, and monetary expressions. Raw entity and factual phrase candidates are then consolidated into canonical aliases by embedding cosine similarity, following the same high-level consolidation idea as MoA's entity merging step. For English generalization on Multi-News, Underthesea is not used for NER; the framework switches to spaCy English NER.

MoA is treated as inspiration and an optional baseline, not as the main framing of this paper [@moa_2026]. The present framework has a different emphasis: fewer LLM calls, explicit source-only evidence propagation, and graph-guided topic formation rather than multi-agent LLM deliberation.

## 3. Method

### 3.1 Overview

Given a document collection, the framework builds a hierarchy of evidence-bearing nodes:

```text
documents
→ sentences
→ tri-sentence source units
→ semantic chunks
→ Leiden topic communities
→ topic summaries with source support
→ final merged summary
```

Every source unit has an ID and remains traceable through the pipeline. Generated summaries are used for abstraction, but source support units are stored separately and propagated upward.

### 3.2 Preprocessing and Tri-Sentence Units

The preprocessing stage applies Unicode normalization and whitespace normalization while preserving paragraph boundaries. Each document is split into paragraphs and sentences. For each sentence index i, the framework constructs a tri-sentence unit:

```text
u_i = (s_{i-1}, s_i, s_{i+1})
```

Boundary units use the available neighboring sentences. Each unit stores the document ID, sentence IDs, sentence texts, center index, raw text, and embedding. The implementation keeps all tri-sentence units even when an extraction variant evaluates only a subset.

### 3.3 Semantic Chunking

Chunks are formed with LangChain Experimental SemanticChunker using BGE-M3 embeddings. The default target chunk range is 1,000 to 1,500 whitespace tokens. A simple token chunker is retained only as an explicit ablation selected by configuration. Each chunk stores its text, source unit IDs, position within the document, embedding, and phrase set. Chunk embeddings are computed from the chunk text, while unit embeddings are computed from tri-sentence texts.

### 3.4 Entity and Factual Phrase Extraction

For Vietnamese, the initial entity extractor is Underthesea NER plus rule-based factual phrase extraction over chunk text. The rule layer detects structured phrases such as percentages, dates, and money amounts. Extracted phrases are normalized by whitespace and lowercasing. Across the sample, raw chunk-level named entities are embedded with the configured embedding model and merged into canonical aliases when cosine similarity exceeds the entity merge threshold, set to 0.85 by default. Numeric, date, and money phrases are not merged by embedding similarity because doing so could collapse distinct factual values. Canonical phrase sets are then attached back to chunks for graph construction.

For English datasets, the entity extractor uses spaCy `en_core_web_sm` plus the same regex factual phrase layer. The English model is a required dependency for English generalization experiments.

### 3.5 Salient Source Evidence Extraction

The framework evaluates three source-only extraction variants.

E1 ranks all tri-sentence units in a chunk by cosine similarity to the chunk embedding:

```text
score(u_i) = cos(e_{u_i}, e_c)
```

Units are selected in descending score order until the evidence token budget is reached.

E2a applies PACSUM to all tri-sentence units in the chunk. It constructs a unit similarity matrix, applies the reference PACSUM thresholding rule, calculates directional centrality scores, and selects top units under the same evidence budget. The default hyperparameters follow the reference CLI in `libs/PacSum/code/run.py`: beta = 0, lambda1 = 0, and lambda2 = 1. The adaptation in this paper uses BGE-M3 cosine similarities as edge scores.

E2b applies PACSUM only to stride-2 candidate units. In practice, this uses units with centers 2, 4, 6, and so on. The purpose is to reduce artificial centrality caused by heavily overlapping adjacent tri-sentence windows.

Selected units are compacted before prompting. Rather than printing every selected tri-sentence window independently, the system merges overlapping selections by source sentence ID and emits a sentence-ordered source span. This reduces prompt redundancy while retaining source provenance.

### 3.6 Sparse Chunk Graph Construction

Each chunk is a graph node. Candidate edges are built from semantic nearest neighbors and adjacent chunks in the same source document. For each candidate pair i, j, the edge weight is:

```text
w_ij = alpha P_ij + beta E_ij + gamma C_ij
```

where P is document-relative positional similarity, E is typed TF-IDF cosine over canonical entity and factual phrase aliases, and C is embedding cosine similarity. The weights satisfy:

```text
alpha + beta + gamma = 1
```

The default setting is alpha = 0.10, beta = 0.20, and gamma = 0.70. The planned grid uses alpha in {0.00, 0.05, 0.10, 0.15, 0.20}. For each alpha, beta ranges from 0 to 1-alpha in steps of 0.10, including the endpoint 1-alpha. Gamma is then set to 1-alpha-beta.

Positional similarity is calculated only within the same document:

```text
P_ij = exp(-|p_i - p_j| / tau), if d_i = d_j
P_ij = 0, otherwise
```

This prevents unrelated chunks from different documents from being treated as positionally close merely because their local indices are similar.

### 3.7 Leiden Topic Clustering

Leiden community detection is applied to the sparse weighted graph. Each community is treated as a topic group. Chunks are sorted back into source order before summarization so that clustering controls membership but does not scramble the presentation order. If graph clustering is disabled, each chunk becomes its own topic, giving a sequential no-graph baseline.

The current implementation includes Leiden through `igraph` and `leidenalg`, both of which are required for graph experiments. Token-bounded community splitting is part of the design but remains a planned extension for larger runs.

### 3.8 Topic-Level Extract-Support Summarization

Each topic is summarized with an explicit Extract-Support prompt. The prompt contains ordered source text and compacted source support. The LLM is instructed to preserve entities, factual phrases, numbers, dates, and contrastive claims while avoiding unsupported facts.

If the first clustering level yields multiple topic summaries, the framework performs a final merge. Crucially, merge support is not generated text. The support pool is built from the union of child source support units and reselected using the same salience variant. Thus, the parent support set satisfies:

```text
E_parent ⊆ union(E_child)
```

This source-only propagation rule is the main factuality safeguard of the hierarchy.

## 4. Experimental Design

### 4.1 Datasets

The primary Vietnamese datasets are VN-MDS and ViMs. Multi-News is used as an English generalization dataset. VN-MDS and ViMs evaluate the main Vietnamese setting. Multi-News tests whether the graph-guided Extract-Support structure is language-general when English entity extraction is substituted for Vietnamese NER.

Dataset statistics will be reported after final loader validation:

| Dataset | Language | Task type | Split used | Samples | Mean input tokens | Mean reference tokens |
| --- | --- | --- | --- | ---: | ---: | ---: |
| VN-MDS | Vietnamese | Multi-document summarization | TODO | TODO | TODO | TODO |
| ViMs | Vietnamese | Summarization | TODO | TODO | TODO | TODO |
| Multi-News | English | Multi-document summarization | TODO | TODO | TODO | TODO |

### 4.2 Models

Embeddings use BGE-M3. Vietnamese NER uses Underthesea 9.2.11 plus regex factual phrase extraction. English NER uses spaCy English when the model is installed. LLM summarization is planned with Qwen3.6-35B and/or Gemma4-12B served through an OpenAI-compatible endpoint.

Dry-run LLM and hash embeddings are used only for pipeline debugging. They are not paper results.

### 4.3 Compared Systems

The planned systems are:

| System | Graph | Salience | Notes |
| --- | --- | --- | --- |
| Pure LLM | No | None | Direct full-text prompt baseline |
| No-Graph E2b | No | E2b | Sequential chunk-community baseline without graph clustering |
| Graph E2b | Yes | E2b | Main proposed setting |
| Graph E1 | Yes | E1 | Salience ablation |
| Graph E2a | Yes | E2a | Salience ablation |
| Optional MoA baseline | External | External | Inspiration/baseline only if reproduced or fairly run |

The graph-weight grid is evaluated on E2b first. E1 and E2a are run as salience ablations when compute budget permits.

### 4.4 Metrics

Primary metrics are ROUGE-2 and ROUGE-L between the framework output and gold summaries. Efficiency submetrics are input tokens, output tokens, LLM calls, and runtime seconds. The implementation also records chunk count and topic count for diagnostic analysis.

ROUGE calculation uses the standard `rouge-score` package. The CSV records the backend to preserve evaluation provenance.

### 4.5 Reproducibility

All experiments should record dataset name, sample ID, salience method, graph weights, graph usage, chunking method, ROUGE scores, input tokens, output tokens, LLM calls, runtime, chunk count, and topic count. Result CSVs are aggregated with `scripts/summarize_results.py`.

Example command for a real LLM run:

```powershell
python scripts/run_graphsum.py --dataset vn_mds --data-root datasets --limit 5 --salience e1 --llm openai_compatible --model <model-name> --base-url http://localhost:8000/v1 --output runs/vn_mds_e1_real.csv
```

Example aggregation command:

```powershell
python scripts/summarize_results.py runs/vn_mds_e1_real.csv runs/vn_mds_e2a_real.csv runs/vn_mds_e2b_real.csv --output runs/vn_mds_summary.csv
```

## 5. Results

Real LLM results are pending. This section must not use dry-run or hash-embedding smoke outputs as evidence.

### 5.1 Main Vietnamese Results

| Dataset | System | ROUGE-2 | ROUGE-L | Input tokens | Output tokens | LLM calls | Runtime seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VN-MDS | Pure LLM | TODO | TODO | TODO | TODO | TODO | TODO |
| VN-MDS | No-Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |
| VN-MDS | Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |
| ViMs | Pure LLM | TODO | TODO | TODO | TODO | TODO | TODO |
| ViMs | No-Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |
| ViMs | Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |

### 5.2 Graph Weight Ablation

| Alpha | Beta | Gamma | Dataset | Salience | ROUGE-2 | ROUGE-L | LLM calls | Runtime seconds |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 5.3 English Generalization

| Dataset | System | ROUGE-2 | ROUGE-L | Input tokens | Output tokens | LLM calls | Runtime seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-News | Pure LLM | TODO | TODO | TODO | TODO | TODO | TODO |
| Multi-News | No-Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |
| Multi-News | Graph E2b | TODO | TODO | TODO | TODO | TODO | TODO |

## 6. Discussion

The framework is designed around a simple premise: the expensive generative model should not be responsible for every operation in a long-document summarization system. Embedding, graph clustering, entity extraction, salience selection, and evidence compaction can be performed before generation. This shifts the LLM role toward topic-level abstraction while preserving a source evidence path for factual grounding.

If Graph E2b outperforms No-Graph E2b, the result would support the usefulness of Leiden chunk-community formation. If Graph E2b outperforms E1 and E2a ablations under matched settings, the result would suggest that stride-2 PACSUM is a useful default support selector. If E2b achieves similar ROUGE with fewer input tokens or faster runtime than E2a, stride-2 PACSUM would be a useful efficiency variant for overlapping segment evidence units.

The entity edge weight ablation is especially important. A high beta may over-connect chunks that share common entities, while beta = 0 removes factual phrase information from clustering. The expected useful range is likely moderate, with content similarity remaining the dominant signal and positional similarity acting as a weak regularizer.

The framework may underperform when summaries require strong cross-document synthesis that is not captured by local chunk communities. It may also suffer when Underthesea misses important entities or when semantic chunking creates communities that exceed a practical LLM budget. These issues motivate future work on token-bounded Leiden refinement, PhoNLP as a Vietnamese NER backend, and richer entity normalization.

## 7. Limitations

This draft has several limitations.

First, real LLM results are not yet available. The current implementation has passed smoke tests, including one real BGE-M3 embedding run, but dry-run outputs cannot support empirical claims.

Second, the current entity similarity uses embedding-based alias consolidation followed by typed binary TF-IDF cosine over canonical entities and factual phrases. This is stronger than plain phrase overlap, but it still lacks relation extraction and a persistent cross-sample entity registry.

Third, token-bounded community splitting is not fully implemented. Leiden can create a community larger than the desired prompt budget, especially in collections dominated by one topic.

Fourth, ROUGE is computed with the required standard `rouge-score` package, and the CSV records the backend used for provenance.

Fifth, English generalization now uses `en_core_web_sm` when installed. Stronger English pipelines such as `en_core_web_trf` may improve entity quality but are outside the initial setup.

## 8. Conclusion

This paper proposes a training-free graph-guided Extract-Support framework for Vietnamese long-document and multi-document summarization. The method combines tri-sentence source evidence units, paragraph-aware semantic chunking, lightweight entity and factual phrase extraction, sparse weighted chunk graphs, Leiden topic clustering, and source-only evidence propagation. The framework is designed to reduce LLM calls and token usage while preserving source-grounded support for factual summarization. The next step is to complete real LLM experiments on VN-MDS and ViMs, evaluate English generalization on Multi-News, and run the planned salience and graph-weight ablations.

## Data Availability Statement

The experiments use local copies or loaders for VN-MDS, ViMs, and Multi-News. Dataset access details, versions, and preprocessing scripts must be finalized before submission. The implementation scaffold is located in the project repository under `graphsum/` and `scripts/`.

## Ethics Declaration

This study uses existing summarization datasets and does not involve new human-subject data collection. If social-media data are included in future experiments, the final paper should verify dataset licensing, anonymization, and platform terms.

## Author Contributions

CRediT roles are pending. Suggested roles: Conceptualization, Methodology, Software, Validation, Formal Analysis, Writing, Review and Editing.

## Conflict of Interest Statement

The authors declare no known conflicts of interest. This statement should be confirmed before submission.

## Funding Acknowledgment

Funding information is pending. If no external funding supported the work, state: “This research received no external funding.”

## AI Tool Usage Disclosure

AI assistance was used to help structure the research plan, draft manuscript text, and implement experiment scaffolding. All claims, citations, data, and results require human verification before submission.

## References to Verify

The following citation keys are placeholders tied to local papers or known method references. Full bibliographic metadata, DOI/URL, venue, and citation correctness must be verified in Stage 2.5 integrity review.

[@context_aware_hm_2025] 2025 ACL. Context-aware Hierarchical Merging for Long Document Summarization. Local file: `papers/2025_ACL_Context-aware Hierarchical Merging for Long Document Summarization.pdf`.

[@cothssum_2025] 2025 Journal. CoTHSSum: Structured long-document summarization via CoT reasoning and hierarchical segmentation. Local file: `papers/2025_Journal_CoTHSSum_Structured long-document summarization via CoT reasoning and hierarchical segmentation.pdf`.

[@d2cs_2025] 2025 EMNLP. D2CS. Local file: `papers/2025_EMNLP_D2CS.pdf`.

[@pacsum_2019] 2019 ACL. Sentence Centrality Revisited for Unsupervised Summarization. Local file: `papers/2019_ACL_PACSUM_Sentence Centrality Revisited for Unsupervised Summarization.pdf`.

[@e2graphrag_2025] 2025. E2GraphRAG. Local file: `papers/2025_E2GraphRAG.pdf`.

[@moa_2026] 2026 NCA. A Training-free Mixture-of-Agents Framework for Multi-document Summarization using LLMs and Knowledge Graphs. Local file: `papers/2026_NCA_MoA_A Training-free Mixture-of-Agents Framework for Multi-document Summarization using LLMs and Knowledge Graphs.pdf`.

[@leiden_original_todo] Original Leiden algorithm paper. Metadata to verify and add.
