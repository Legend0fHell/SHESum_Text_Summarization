from __future__ import annotations


TOPIC_EXTRACT_SUPPORT_PROMPT = """[Instruction]
You are a multilingual, fact-grounded community summarization agent. Write in {language_name}. Your role is to summarize one chunk community using ordered source text and selected source support.

[Objective]
Given:
1. Ordered source text from one chunk community.
2. Source support selected from original source segments.

Produce a coherent community-level summary that:
- captures the central event, actors, actions, causes, consequences, and important context;
- preserves salient entities, numbers, dates, locations, and factual phrases;
- acknowledges disagreement, uncertainty, or contrast when the sources support it;
- remains strictly grounded in the provided input and does not add external knowledge.

[Dataset guidance]
{dataset_guidance}

[Length constraint]
{length_constraint}

[Process]
1. Identify Core Content
 - Determine the main event or situation, the central actors, and the most important actions or outcomes.
 - Use the ordered source text for coverage and narrative flow.
2. Verify Against Support
 - Prefer claims grounded in the selected source support when details are ambiguous.
 - Keep concrete facts only when supported by the source text or source support.
3. Compose
 - Write clear, neutral prose in inverted-pyramid style: most important information first, context after.
 - Merge repeated information once.
 - Do not mention "source text", "source support", "chunk", "community", "segment", "prompt", or the summarization process.

[Input]
Source text:
---
{source_text}
---

Source support:
---
{source_support}
---

[Output]
Return ONLY the final summary as plain text. Do not include labels, headings, bullet points, metadata, JSON, or analysis.
"""


MERGE_EXTRACT_SUPPORT_PROMPT = """[Instruction]
You are a multilingual, metadata-aware summary synthesizer. Write in {language_name}. Your task is to fuse multiple community-level summaries into one coherent final summary while using source-derived support only for factual verification.

[Objective]
Given:
1. Community-level summaries generated from chunk communities.
2. Source-derived support inherited from the original source segments.

Produce a final integrated summary that:
- preserves the main information from all important community summaries;
- emphasizes globally central facts and relationships;
- removes redundancy across communities;
- keeps numerical, temporal, and entity-specific details only when supported;
- explicitly preserves major disagreements, caveats, or uncertainty when supported.

[Dataset guidance]
{dataset_guidance}

[Length constraint]
{length_constraint}

[Process]
1. Rank and Integrate
 - Identify which community summaries contain the global lead, supporting context, consequences, and caveats.
 - Merge repeated claims once and preserve complementary details.
2. Verify and Resolve
 - Use source-derived support to check factual accuracy.
 - If summaries conflict, state the contrast only when the source-derived support justifies it; otherwise use cautious wording.
3. Compose Final Summary
 - Use inverted-pyramid structure: lead first, context and consequences after.
 - Do not reference "community summaries", "source support", "metadata", "chunks", "segments", or the summarization process.

[Input]
Community summaries:
---
{topic_summaries}
---

Source-derived support:
---
{source_support}
---

[Output]
Return ONLY one final summary as plain text. Do not include labels, headings, bullet points, metadata, JSON, or analysis.
"""


DIRECT_SUMMARY_PROMPT = """[Instruction]
You are a multilingual, fact-grounded multi-document summarization model. Write in {language_name}. Do not add information that is not present in the source documents.

[Objective]
Given a collection of related source documents, produce a concise, neutral summary that:
- captures the main point, key entities, events, relationships, numbers, dates, and causal links;
- merges repeated information across documents;
- preserves critical nuance, disagreement, uncertainty, or contrast when present;
- avoids opinion, hype, speculation, and hallucination.

[Dataset guidance]
{dataset_guidance}

[Length constraint]
{length_constraint}

[Process]
1. Multi-Document Scan
 - Identify recurring themes, central actors, core events, major claims, and salient factual details.
2. De-duplicate and Reconcile
 - Merge repeated details.
 - If conflicting claims exist, briefly note the disagreement without resolving it beyond what the documents state.
3. Compose
 - Write clear, neutral prose in inverted-pyramid style.
 - Do not mention "source documents", "prompt", or the summarization process.

[Input]
Source documents:
---
{source_text}
---

[Output]
Return ONLY the final summary as plain text. Do not include labels, headings, bullet points, metadata, JSON, or analysis.
"""


def language_name(language: str) -> str:
    return "Vietnamese" if language == "vi" else "English"


def render_topic_prompt(
    source_text: str,
    source_support: str,
    language: str,
    dataset: str = "",
    max_summary_words: int | None = None,
) -> str:
    return TOPIC_EXTRACT_SUPPORT_PROMPT.format(
        language_name=language_name(language),
        dataset_guidance=dataset_guidance(language, dataset),
        length_constraint=length_constraint(max_summary_words),
        source_text=source_text.strip(),
        source_support=source_support.strip(),
    )


def render_merge_prompt(
    topic_summaries: str,
    source_support: str,
    language: str,
    dataset: str = "",
    max_summary_words: int | None = None,
) -> str:
    return MERGE_EXTRACT_SUPPORT_PROMPT.format(
        language_name=language_name(language),
        dataset_guidance=dataset_guidance(language, dataset),
        length_constraint=length_constraint(max_summary_words),
        topic_summaries=topic_summaries.strip(),
        source_support=source_support.strip(),
    )


def render_direct_prompt(source_text: str, language: str, dataset: str = "", max_summary_words: int | None = None) -> str:
    return DIRECT_SUMMARY_PROMPT.format(
        language_name=language_name(language),
        dataset_guidance=dataset_guidance(language, dataset),
        length_constraint=length_constraint(max_summary_words),
        source_text=source_text.strip(),
    )


def dataset_guidance(language: str, dataset: str) -> str:
    if language != "vi":
        return "No additional dataset-level guidance."
    if dataset == "vn_mds":
        return (
            "VN-MDS is a Vietnamese multi-document news summarization dataset with extractive reference style. "
            "Preserve Vietnamese source wording for named entities, dates, numbers, locations, and core event phrases. "
            "Use graph/community/entity evidence only as a factual checklist; do not let graph-oriented phrasing dominate the final prose."
        )
    if dataset == "vims":
        return (
            "ViMs is a Vietnamese multi-document news summarization dataset with strong lead and source-wording bias. "
            "Preserve Vietnamese source wording for names, organizations, dates, numbers, campaign names, and core event phrases. "
            "Use graph/community/entity evidence only as a factual checklist; do not let graph-oriented phrasing dominate the final prose."
        )
    return (
        "Write entirely in Vietnamese. Preserve Vietnamese source wording for named entities, dates, numbers, locations, "
        "and core event phrases when they are salient and supported."
    )


def length_constraint(max_summary_words: int | None) -> str:
    if max_summary_words is None or max_summary_words <= 0:
        return "No explicit word limit."
    return f"The final summary must not exceed {max_summary_words} words. Prefer complete sentences; do not end mid-sentence."
