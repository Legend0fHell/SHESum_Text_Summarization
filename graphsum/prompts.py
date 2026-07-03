from __future__ import annotations


TOPIC_EXTRACT_SUPPORT_PROMPT = """You are a careful long-document summarizer. Write in {language_name}.

Task:
Summarize the topic using the source text. Use the source support to check factual accuracy, preserve important entities, factual phrases, numbers, dates, and contrasting claims. Do not invent facts.

Source text:
---
{source_text}
---

Source support, selected from the original source tri-sentence units:
---
{source_support}
---

Instructions:
- Cover the central events, actors, causes, consequences, and important context.
- Prefer claims grounded in the source support when the source text is ambiguous.
- Preserve contrast or uncertainty when sources disagree.
- Do not mention "source text", "support", "chunk", "topic", or "summary" in the output.
- Return only the summary.
"""


MERGE_EXTRACT_SUPPORT_PROMPT = """You are merging topic summaries into one coherent final summary. Write in {language_name}.

The gist must be based on the topic summaries. Use the supporting source-derived text only to check factual accuracy and avoid unsupported facts.

Topic summaries:
---
{topic_summaries}
---

Source-derived support:
---
{source_support}
---

Instructions:
- Merge repeated information once.
- Preserve the main information from all topic summaries.
- Keep factual details only when supported.
- Preserve major disagreements, caveats, or uncertainty.
- Do not mention "topic summaries", "source support", or the summarization process.
- Return only the final summary.
"""


DIRECT_SUMMARY_PROMPT = """You are a careful multi-document summarizer. Write in {language_name}.

Task:
Summarize the following source documents directly. Do not use external knowledge. Do not invent facts.

Source documents:
---
{source_text}
---

Instructions:
- Cover the central events, actors, causes, consequences, and important context.
- Preserve important entities, factual phrases, numbers, dates, and contrasting claims.
- Preserve major disagreements, caveats, or uncertainty when sources disagree.
- Do not mention "source documents", "prompt", or the summarization process.
- Return only the summary.
"""


def language_name(language: str) -> str:
    return "Vietnamese" if language == "vi" else "English"


def render_topic_prompt(source_text: str, source_support: str, language: str) -> str:
    return TOPIC_EXTRACT_SUPPORT_PROMPT.format(
        language_name=language_name(language),
        source_text=source_text.strip(),
        source_support=source_support.strip(),
    )


def render_merge_prompt(topic_summaries: str, source_support: str, language: str) -> str:
    return MERGE_EXTRACT_SUPPORT_PROMPT.format(
        language_name=language_name(language),
        topic_summaries=topic_summaries.strip(),
        source_support=source_support.strip(),
    )


def render_direct_prompt(source_text: str, language: str) -> str:
    return DIRECT_SUMMARY_PROMPT.format(
        language_name=language_name(language),
        source_text=source_text.strip(),
    )
