"""JSON AYRIŞTIRMA (parsing) — pipeline'ın 2. aşaması.

NE İŞE YARAR: LLM'in döndürdüğü HAM METNİ, Python'un kullanabileceği DICT
LİSTESİNE çevirir. Yani "metin → veri".

PIPELINE'DAKİ YERİ:
    llm.py (AI'dan metin getir)  →  BU DOSYA (metinden JSON çıkar)  →  validator.py (şemaya uydur)

SORDUĞU SORU: "Bu dağınık metin geçerli JSON mu? Öyleyse temizini çıkar."
- Model JSON'un önüne/arkasına laf eklemiş olabilir → parantez sayarak {...}'i bulur.
- ```json ... ``` bloğuna sarmış olabilir → işaretleri temizler.
- Bozuksa çökmez, boş sonuç döner.

Benzetme: Gelen mektubun ZARFINI açıp içindeki kâğıdı çıkarmak.
"""
from __future__ import annotations

import json
import re


def _strip_fences(s: str) -> str:
    """Baştaki/sondaki ```json ... ``` işaretlerini temizler."""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json(s: str):
    """Metindeki ilk DENGELİ {...} objesini parantez sayarak bulur.
    Model JSON'un önüne/arkasına düz metin eklese bile çıkarır."""
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None                    # dengesiz parantez


def _safe_json(raw: str):
    if not raw:                    # None ya da boş cevap → boş sonuç (çökme yok)
        return {}
    s = _strip_fences(raw)
    try:
        return json.loads(s)       # 1) doğrudan dene
    except json.JSONDecodeError:
        pass
    obj = _extract_first_json(s)   # 2) araya metin karışmışsa ilk {...}'i çıkar
    if obj:
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return {}
    return {}


def _parse(raw: str) -> list[dict]:
    """LLM cevabından {"items":[...]} listesini çıkarır."""
    data = _safe_json(raw)
    if isinstance(data, dict):
        return [] if data.get("_dry_run") else data.get("items", [])
    return data if isinstance(data, list) else []


def _item_text(item: dict) -> str:
    """Öğenin string/list alanlarını tek metne toplar (ağırlık skoru için)."""
    parts = []
    for v in item.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
    return " ".join(parts)
