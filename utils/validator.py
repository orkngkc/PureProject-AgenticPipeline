"""DENETLEYİCİ (validation) — pipeline'ın 3. aşaması.

NE İŞE YARAR: json_utils'ten gelen DICT LİSTESİNİ bir ŞEMAYA göre kontrol edip
düzeltir. Yani "ham JSON → temiz, kurallı JSON". Böylece çıktı HER ZAMAN
beklenen yapıda olur.

PIPELINE'DAKİ YERİ:
    json_utils.py (metinden JSON çıkar)  →  BU DOSYA (şemaya uydur)  →  eşleyici (tipli modele çevir)

SORDUĞU SORU: "Bu dict'ler kurallarıma uyuyor mu? Uymayanı düzeltebilir miyim?"
- Zorunlu alan (ör. 'name') eksikse → öğeyi ATAR.
- Tür yanlışsa (confidence='0.9' metni) → doğru türe ÇEVİRİR (0.9 sayı).
- Sayı aralık dışıysa (confidence=5) → min/max'a KIRPAR (1.0).
- Eksik opsiyonel alana → VARSAYILAN koyar.
Sonuç: (geçerli_öğeler, hatalar). Hatalar orchestrator'da 'audit'e yazılır.

Benzetme: Kâğıttaki FORMU kontrol etmek (alanlar dolu mu, tarih formatı doğru mu → düzelt/reddet).
"""
from __future__ import annotations


def validate_items(items, schema) -> tuple[list[dict], list[str]]:
    """Öğe listesini şemaya göre doğrular -> (geçerli_öğeler, hatalar)."""
    if not isinstance(items, list):
        return [], ["Çıktı liste değil"]
    valid, errors = [], []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"[{idx}] öğe dict değil, atlandı")
            continue
        cleaned, drop = {}, False
        for field, spec in schema.items():
            if field not in item or item[field] is None:
                if spec.get("required"):
                    errors.append(f"[{idx}] zorunlu alan eksik: '{field}'")
                    drop = True
                    break
                cleaned[field] = spec.get("default")
                continue
            value, err = _coerce(item[field], spec["type"])
            if err:
                errors.append(f"[{idx}] '{field}': {err}")
            if value is None:
                if spec.get("required"):
                    drop = True
                    break
                value = spec.get("default")
            if spec["type"] in (int, float) and isinstance(value, (int, float)):
                value = max(spec.get("min", value), min(spec.get("max", value), value))
            cleaned[field] = value
        if not drop:
            valid.append(cleaned)
    return valid, errors


def _coerce(value, expected):
    """Değeri beklenen türe güvenle çevirir. (değer, hata|None)."""
    if expected is str:
        return (value.strip() if isinstance(value, str) else str(value).strip()), None
    if expected is list:
        if isinstance(value, list):
            return value, None
        if isinstance(value, str):
            return [value.strip()], "tek değer listeye sarıldı"
        return [], f"liste bekleniyordu, {type(value).__name__} geldi"
    if expected is float:
        try:
            return float(value), None
        except (TypeError, ValueError):
            return None, f"sayı bekleniyordu, '{value}'"
    if expected is int:
        try:
            return int(float(value)), None
        except (TypeError, ValueError):
            return None, f"tam sayı bekleniyordu, '{value}'"
    return value, None
