"""Belge okuma + segmentasyon.

- read_document: pdf/txt/docx okur, OCR temizler.
- segment: metni parçalara böler (temizse BÖLÜM'e göre, değilse boyuta göre).
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import median

from models import Segment

_CHAPTER_RE = re.compile(r"B[ÖO]L[ÜU]M\s*0*(\d+)", re.IGNORECASE)


# ── OKUMA ────────────────────────────────────────────────────────────────────
def read_document(path: str | Path, start_page: int = 1) -> str:
    """Belgeyi metne çevirir. start_page: PDF'te hangi sayfadan başlansın (1-tabanlı)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path, start_page)
    elif suffix == ".docx":
        text = _read_docx(path)
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Desteklenmeyen dosya türü: {suffix}")
    return _clean(text)


def _read_pdf(path: Path, start_page: int = 1) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PDF için: pip install PyMuPDF") from e
    with fitz.open(path) as doc:
        pages = [page.get_text("text") for page in doc]
    if start_page > 1:                      # ilk sayfaları (önsöz/künye) atla
        pages = pages[start_page - 1:]
    return "\n".join(_strip_running_lines(pages))


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as e:
        raise RuntimeError("DOCX için: pip install python-docx") from e
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)


def _strip_running_lines(pages: list[str], min_ratio: float = 0.12) -> list[str]:
    """Sayfaların çoğunda tekrar eden kısa satırları (üstbilgi/altbilgi) siler."""
    from collections import Counter
    counter: Counter[str] = Counter()
    for page in pages:
        for ln in {x.strip() for x in page.splitlines() if x.strip()}:
            counter[ln] += 1
    threshold = max(3, int(len(pages) * min_ratio))
    boiler = {ln for ln, c in counter.items() if c >= threshold and len(ln) < 80}
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in boiler) for p in pages]


def _clean(text: str) -> str:
    """OCR temizliği: tireli bölünmeleri birleştir, fazla boşlukları kırp."""
    text = re.sub(r"([A-Za-zçğıöşüÇĞİÖŞÜ])[-­]\s*\n\s*", r"\1", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── SEGMENTASYON ─────────────────────────────────────────────────────────────
def segment(text: str, strategy: str = "auto", max_chars: int = 4000) -> list[Segment]:
    if strategy == "size":
        return _chunk_by_size(text, max_chars)
    if strategy == "chapters":
        return _by_chapters(text)
    chapters = _by_chapters(text)          # auto
    return chapters if _reliable(chapters) else _chunk_by_size(text, max_chars)


def _reliable(segments: list[Segment]) -> bool:
    if len(segments) < 5:
        return False
    tiny = sum(1 for s in segments if len(s.text) < 80)
    return tiny / len(segments) <= 0.15 and median(len(s.text) for s in segments) > 250


def _by_chapters(text: str) -> list[Segment]:
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return _chunk_by_size(text)
    bounds, last = [], 0
    for m in matches:                      # sadece monotonik artan numaralar
        no = int(m.group(1))
        if no > last:
            bounds.append((no, m.start()))
            last = no
    segments = []
    for i, (no, start) in enumerate(bounds):
        end = bounds[i + 1][1] if i + 1 < len(bounds) else len(text)
        segments.append(Segment(f"bolum-{no}", no, f"BÖLÜM {no}", text[start:end].strip()))
    return segments


def _chunk_by_size(text: str, max_chars: int = 4000, overlap: int = 200) -> list[Segment]:
    """Metni ~max_chars'lık örtüşmeli parçalara böler.
    step kadar ilerler (range garantili → sonsuz döngü imkânsız)."""
    step = max(1, max_chars - overlap)          # her parça bir öncekiyle 'overlap' kadar örtüşür
    segments = []
    for idx, pos in enumerate(range(0, len(text), step), start=1):
        chunk = text[pos:pos + max_chars].strip()
        if chunk:                               # boş parçayı ekleme
            segments.append(Segment(f"chunk-{idx}", idx, f"Parça {idx}", chunk))
    return segments
