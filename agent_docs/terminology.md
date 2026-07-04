# Terminology Notes for Final Report

Use this terminology when revising `agent_docs/draft.md` and writing the final report.

## Chunk Community

Use **chunk community** instead of **topic** or **topic group** when referring to Leiden output.

A **chunk community** is a set of semantic chunks grouped by Leiden community detection on the weighted chunk graph. A chunk community is a graph-theoretic grouping induced by content similarity, entity/factual overlap, and positional affinity. It is not assumed to correspond to a single linguistic topic.

Recommended wording:

> We use the term **chunk community** to denote a set of semantic chunks grouped by Leiden community detection on the weighted chunk graph. A chunk community is a graph-theoretic grouping induced by content, entity/factual overlap, and positional affinity; it is not assumed to correspond to a single linguistic topic.

Use these replacements:

- `topic` -> `chunk community`
- `topic summary` -> `community-level summary`
- `topic-level summarization` -> `community-level summarization`
- `topic merge` -> `community merge`
- `topic count` -> `community count`

## Segment

Use **segment** for the output unit produced by the rule-based boundary splitter. Avoid calling these units sentences in method descriptions, because the splitter can return sentence-like spans, headlines, clauses, fragments, or other over-segmented spans.

A **segment** is the local evidence unit constructed from the splitter output using an overlapping three-item window. The three consecutive splitter outputs are a best-effort context window, not a claim that the text contains three grammatical sentences. This helps preserve meaning when the splitter separates short spans that depend on neighboring text, such as `Does it work? Yes.`

Recommended wording:

> We define a **segment** as a sentence-like evidence unit produced from the rule-based boundary splitter. Each segment is constructed from up to three consecutive splitter outputs using an overlapping window. Because Vietnamese news text may contain headlines, clauses, fragments, or over-segmented spans, a segment is not guaranteed to be a grammatical sentence. The three-output window is a best-effort mechanism for preserving local context, especially when short spans depend on neighboring text.

Use these replacements:

- `sentence` -> `splitter output` when referring to raw boundary-split spans
- `tri-sentence unit` -> `segment`
- `tri-sentence evidence unit` -> `segment`
- `tri-sentence window` -> `overlapping segment window` only when emphasizing construction
- `sentence ID` -> `splitter-output ID` or `source span ID` where precise provenance is needed

Keep **SemanticChunker chunk** or **semantic chunk** distinct from **segment**. A segment is a local source-evidence unit; a chunk is a larger semantic unit used as a graph node.
