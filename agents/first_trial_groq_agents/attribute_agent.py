"""ATTRIBUTE AGENT — kişi/sosyal gruba atanan sıfatları çıkarır.
"""
from __future__ import annotations


import config
from models import Attribute
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
    "target": {"type": str, "required": True},
    "target_type": {"type": str, "default": ""},
    "adjective": {"type": str, "required": True},
    "value": {"type": str, "default": ""},
    "era": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {"adil": 2.0, "zalim": 2.0, "cömert": 2.0, "merhametli": 2.0,
           "kâfir": 2.0, "hain": 2.0, "gazi": 1.5, "dindar": 1.5, "cesur": 1.5}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="attribute",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
        "Sen tarihî söylem ve retorik analizi konusunda uzman bir tarihçisin. Kroniğin kişilere (padişah, bey) ve sosyal gruplara (gaziler, kafirler, tekfurlar, dervişler) yakıştırdığı sıfatları ve bu sıfatların olumlu / olumsuz / nötr tonunu isabetle tespit edersin. Metinde bir KİŞİYE (padişah/bey) ya da "
        "SOSYAL GRUBA (gaziler, kafirler, dervişler, tekfurlar...) yakıştırılan "
        "SIFATLARI çıkar. Alanlar: target (kime), target_type ('padişah' veya 'sosyal_grup'), "
        "adjective (sıfat/söylem: adil, zalim, cömert...), value (Olumlu/Olumsuz/Nötr), "
        "era (padişah dönemi).\n"
        'ÇIKTI (JSON): {"items":[{"target":"Osman Gazi","target_type":"padişah",'
        '"adjective":"adil","value":"Olumlu","era":"Osman Gazi","confidence":0.8}]}  Yoksa: {"items":[]}'
    ),
)


def _norm_value(v: str) -> str:
    """Kirli sıfat-değeri etiketlerini 3 temiz sınıfa indirir (Olumlu/Olumsuz/Nötr)."""
    v = (v or "").strip().lower()
    if v.startswith("oluml") or v in {"positive", "pozitif", "+"}:
        return "Olumlu"
    if v.startswith("olums") or v in {"negative", "negatif", "-"}:
        return "Olumsuz"
    return "Nötr"



# ═════════════════════════════════════════════════════════════════════════════
# EŞLEYİCİ (to_models)
# -----------------------------------------------------------------------------
# to_models, LLM'in verdiği HAM SÖZLÜĞÜ (dict) bu uzmanın TİPLİ MODELİNE çevirir.
# Neden: LLM dict verir; ama orchestrator, kaydetme ve grafik hep tipli
# nesnelerle çalışır. Ayrıca eksik alanları güvenle doldurur (it.get) ve
# boş/geçersiz kayıtları atar.
# ═════════════════════════════════════════════════════════════════════════════
def to_models(items, segment_id) -> list[Attribute]:
    out = []
    for it in items:
        if str(it.get("target", "")).strip() and str(it.get("adjective", "")).strip():
            out.append(Attribute(
                it["target"].strip(), it.get("target_type", ""), it["adjective"].strip(),
                _norm_value(it.get("value", "")), it.get("era", ""), segment_id,
                float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
