"""TIMELINE AGENT — olayları kronolojik sıraya dizer.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""

from __future__ import annotations


import config
from models import TimelineEntry
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.validator import validate_items


# Tüm agent'lara otomatik eklenen sabit kurallar
_TURKCE_KURALI = (
    "\nÖNEMLİ: Bütün çıktıyı ve tüm metin alanlarını HER ZAMAN TÜRKÇE yaz. "
    "Asla İngilizce yazma."
)
_KALITE_KURALI = (
    "\nKALİTE: Yalnızca metinde AÇIKÇA geçenleri çıkar. UYDURMA; emin değilsen "
    "ekleme. Anlamsız/OCR'dan bozuk parçaları atla. Emin olmadığına düşük confidence ver."
)



# ═════════════════════════════════════════════════════════════════════════════
# AGENT MOTORU
# -----------------------------------------------------------------------------
# Agent, bir LLM uzmanının ortak iskeletidir. Her uzman ona SADECE 3 şey verir:
# rol (görev tanımı), schema (denetleyici kuralları) ve weights (ağırlıklı
# kelimeler). Agent gerisini yapar: rolü kurar (ağırlık ipucu + Türkçe + kalite
# kurallarını otomatik ekler), LLM'e sorar, cevabı JSON'a ayrıştırır, şemaya
# göre DENETLER ve her çıkan öğeye ağırlık SKORU verir. set_model ile çalışma
# anında başka bir modele geçebilir.
# ═════════════════════════════════════════════════════════════════════════════
class Agent:
    def __init__(self, role, model=None, name="agent", schema=None, weights=None):
        self.name = name
        self.model = model or config.LLM_MODEL
        self.schema = schema
        self.weights = weights or {}
        self.role = role + self._weight_hint() + _TURKCE_KURALI + _KALITE_KURALI
        self.client = LLMClient(model=self.model)
        self.last_errors: list[str] = []

    def set_model(self, model: str) -> None:
        """Agent'ı çalışma anında herhangi bir modele geçirir (client'ı yeniler)."""
        if model:
            self.model = model
            self.client = LLMClient(model=model)

    def run(self, text: str, segment_id: str = "") -> list[dict]:
        """LLM'e sor -> JSON ayrıştır -> DENETLE -> ağırlık skoru ekle."""
        user = f"SEGMENT ID: {segment_id}\n\nMETİN:\n\"\"\"\n{text}\n\"\"\"\n\nYalnızca JSON döndür."
        raw = self.client.chat(self.role, user)
        items = _parse(raw)
        if self.schema:
            items, self.last_errors = validate_items(items, self.schema)
        else:
            self.last_errors = []
        for it in items:
            it["weight_score"] = self._score(_item_text(it))
        return items

    def _weight_hint(self) -> str:
        """Ağırlıklı kelimeleri prompt ipucuna çevirir (role'e otomatik eklenir)."""
        if not self.weights:
            return ""
        ordered = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)
        terms = ", ".join(f"{kw}({w:g})" for kw, w in ordered)
        return f"\nAĞIRLIKLI ANAHTAR KELİMELER (yüksek = güçlü sinyal): {terms}"

    def _score(self, text: str) -> float:
        """Metinde geçen ağırlıklı kelimelerin toplam skoru."""
        if not text or not self.weights:
            return 0.0
        low = text.lower()
        return round(sum(w for kw, w in self.weights.items() if kw.lower() in low), 2)

    def __repr__(self):
        return f"Agent(name={self.name!r}, model={self.model!r})"


# ═════════════════════════════════════════════════════════════════════════════
# BU UZMANIN TANIMI (şema + ağırlık + rol)
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA = {
    "order": {"type": int, "default": 0},
    "date_hint": {"type": str, "default": ""},
    "event_summary": {"type": str, "required": True},
    "actors": {"type": list, "default": []},
    "segment_id": {"type": str, "default": ""},
}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="timeline",
    model=MODEL,
    schema=SCHEMA,
    role=(
        "Sen tarihsel kronoloji ve olay dizilişi konusunda uzman bir tarihçisin. Sana numaralı OLAY listesi verilecek; bunları KRONOLOJİK sıraya diz. "
        "Tarih ipuçlarını ve anlatı sırasını kullan. Her olaya 0'dan başlayan 'order' ver.\n"
        'ÇIKTI (JSON): {"items":[{"order":0,"date_hint":"Osman Bey dönemi",'
        '"event_summary":"Karacahisar fethi","actors":["Osman Bey"],"segment_id":"bolum-15"}]}'
    ),
)



# ═════════════════════════════════════════════════════════════════════════════
# EŞLEYİCİ (to_models)
# -----------------------------------------------------------------------------
# to_models, LLM'in verdiği HAM SÖZLÜĞÜ (dict) bu uzmanın TİPLİ MODELİNE çevirir.
# Neden: LLM dict verir; ama orchestrator, kaydetme ve grafik hep tipli
# nesnelerle çalışır. Ayrıca eksik alanları güvenle doldurur (it.get) ve
# boş/geçersiz kayıtları atar.
# ═════════════════════════════════════════════════════════════════════════════
def to_models(items) -> list[TimelineEntry]:
    out = [TimelineEntry(int(it.get("order", i)), it.get("date_hint", ""),
                         it.get("event_summary", ""), it.get("actors", []),
                         it.get("segment_id", "")) for i, it in enumerate(items)]
    out.sort(key=lambda t: t.order)
    return out
