from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    documents: list[Document]
    references: list[str]
    language: str
    dataset: str


def load_samples(dataset: str, root: str | Path, limit: int | None = None) -> list[Sample]:
    root = Path(root)
    if dataset == "vn_mds":
        samples = list(_load_vn_mds(root / "VN-MDS" / "clusters"))
    elif dataset == "vims":
        samples = list(_load_vims(root / "ViMs"))
    elif dataset == "multi_news":
        samples = list(_load_multi_news(limit=limit))
        return samples
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    samples.sort(key=lambda item: item.sample_id)
    return samples[:limit] if limit else samples


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _load_vn_mds(root: Path) -> Iterable[Sample]:
    for cluster_dir in root.glob("cluster_*"):
        if not cluster_dir.is_dir():
            continue
        cluster_id = cluster_dir.name
        documents = []
        for body_path in sorted(cluster_dir.glob("*.body.txt")):
            doc_id = body_path.stem.replace(".body", "")
            info_path = cluster_dir / f"{doc_id}.info.txt"
            title = ""
            if info_path.exists():
                title = _read_text(info_path).splitlines()[0].strip() if _read_text(info_path) else ""
            documents.append(Document(doc_id=doc_id, title=title, text=_read_text(body_path)))
        references = []
        for ref_path in sorted(cluster_dir.glob(f"{cluster_id}.ref*.txt")):
            if ref_path.name.endswith(".tok.txt"):
                continue
            text = _read_text(ref_path)
            if text:
                references.append(text)
        if documents and references:
            yield Sample(cluster_id, documents, references, "vi", "vn_mds")


def _load_vims(root: Path) -> Iterable[Sample]:
    original_root = root / "original"
    summary_root = root / "summary"
    for cluster_dir in original_root.glob("Cluster_*"):
        if not cluster_dir.is_dir():
            continue
        sample_id = cluster_dir.name
        documents = []
        for doc_path in sorted((cluster_dir / "original").glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            raw = _read_text(doc_path)
            title = ""
            source = ""
            content_lines = []
            in_content = False
            for line in raw.splitlines():
                if line.startswith("Title:"):
                    title = line.replace("Title:", "", 1).strip()
                elif line.startswith("Source:"):
                    source = line.replace("Source:", "", 1).strip()
                elif line.startswith("Content:"):
                    in_content = True
                elif in_content:
                    content_lines.append(line)
            text = "\n".join(content_lines).strip() or raw
            documents.append(Document(doc_id=doc_path.stem, title=title, source=source, text=text))
        references = []
        for ref_path in sorted((summary_root / sample_id).glob("*.gold.txt")):
            text = _read_text(ref_path)
            if text:
                references.append(text)
        if documents and references:
            yield Sample(sample_id, documents, references, "vi", "vims")


def _load_multi_news(limit: int | None = None) -> Iterable[Sample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to load Multi-News from HuggingFace.") from exc
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    for idx, row in enumerate(dataset):
        document_text = row.get("document") or row.get("documents") or ""
        if isinstance(document_text, list):
            docs = [Document(doc_id=f"{idx}_{i}", text=str(text)) for i, text in enumerate(document_text)]
        else:
            parts = [part.strip() for part in str(document_text).split("|||||") if part.strip()]
            docs = [Document(doc_id=f"{idx}_{i}", text=text) for i, text in enumerate(parts or [str(document_text)])]
        summary = row.get("summary") or row.get("summaries") or ""
        refs = summary if isinstance(summary, list) else [str(summary)]
        yield Sample(str(idx), docs, [ref for ref in refs if ref.strip()], "en", "multi_news")
