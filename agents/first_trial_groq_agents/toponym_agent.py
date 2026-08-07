"""TOPONYM AGENT — yer adlarını (toponimleri) çıkarır.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Toponym
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
    "name": {"type": str, "required": True},
    "modern_name": {"type": str, "default": ""},
    "place_type": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {  # yer sinyali veren kelimeler
    "kalesi": 2.5, "hisar": 2.5, "hisarı": 2.5, "kale": 2.0, "şehri": 2.0,
    "nehri": 2.0, "ırmak": 2.0, "boğaz": 2.0, "sancak": 2.0, "vilayet": 2.0,
    "ovası": 2.0, "dağı": 2.0, "şehir": 1.5, "ova": 1.5, "diyar": 1.5,
}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="toponym",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
        "Sen tarihî coğrafya ve toponimi (yer adı bilimi) konusunda uzman bir tarihçisin. Erken Osmanlı coğrafyasını (Söğüt, Bilecik, İnegöl, Karacahisar, Domaniç, Eskişehir çevresi...) ve yer adlarının eski/güncel karşılıklarını iyi bilir; şehir, kale, nehir, dağ, sancak gibi türleri ayırt edersin. Metindeki YER ADLARINI çıkar. "
        "Biliniyorsa güncel adı modern_name'e yaz (Kostantiniyye -> İstanbul). "
        "place_type: şehir, kale, nehir, bölge... confidence: 0-1. "
        "SADECE gerçek yer adlarını al; kişi adlarını, unvanları ve OCR'dan bozuk "
        "görünen anlamsız kelimeleri (ör. 'Beli', 'Yıkılınaya') ALMA.\n"
        'ÇIKTI (JSON): {"items":[{"name":"Kostantiniyye","modern_name":"İstanbul",'
        '"place_type":"şehir","confidence":0.9}]}  Yoksa: {"items":[]}'
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
def to_models(items, segment_id) -> list[Toponym]:
    out = []
    for it in items:
        name = str(it.get("name", "")).strip()
        if name:
            out.append(Toponym(name, it.get("modern_name", ""),
                              it.get("place_type", ""), [segment_id],
                              float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
