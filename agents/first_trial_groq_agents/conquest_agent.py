"""CONQUEST AGENT — fetihleri çıkarır (padişah – fethettiği yerler).

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Conquest
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
    "place": {"type": str, "required": True},
    "era": {"type": str, "default": ""},
    "year": {"type": str, "default": ""},
    "intent": {"type": str, "default": ""},
    "success": {"type": str, "default": ""},
    "ottoman_leader": {"type": str, "default": ""},
    "opponent_leader": {"type": str, "default": ""},
    "method": {"type": str, "default": ""},
    "opponent_muslim": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {"fetih": 3.0, "fethetti": 2.5, "aldı": 1.5, "hediye": 2.0,
           "kılıç": 2.0, "kuşatma": 2.5, "sefer": 2.0, "verdi": 1.5}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="conquest",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
        "Sen erken Osmanlı fetih ve genişleme tarihinde uzman bir tarihçisin. Bir yerin hangi padişah döneminde, kim tarafından ve hangi yöntemle (kılıç yoluyla, hediye, satın alma, ittifak) ele geçirildiğini; sonucun başarılı mı başarısız mı olduğunu metinden titizlikle çıkarırsın. Metindeki YER ALMA/FETİH olaylarını çıkar. "
        "Her kayıt bir yer içindir. Alanlar: place (yerin adı), era (hangi padişah dönemi), "
        "year (varsa tarih), intent (Alma/Verme), success (Başarılı/Başarısız), "
        "ottoman_leader (Osmanlı lideri), opponent_leader (karşı lider), "
        "method (hediye / kılıç yoluyla / satın alma...), opponent_muslim (evet/hayır/bilinmiyor).\n"
        'ÇIKTI (JSON): {"items":[{"place":"Karacahisar","era":"Osman Gazi","year":"1288",'
        '"intent":"Alma","success":"Başarılı","ottoman_leader":"Osman Gazi",'
        '"opponent_leader":"Karacahisar tekfuru","method":"kılıç yoluyla",'
        '"opponent_muslim":"hayır","confidence":0.85}]}  Yoksa: {"items":[]}'
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
def to_models(items, segment_id) -> list[Conquest]:
    out = []
    for it in items:
        if str(it.get("place", "")).strip():
            out.append(Conquest(
                it["place"].strip(), it.get("era", ""), it.get("year", ""),
                it.get("intent", ""), it.get("success", ""), it.get("ottoman_leader", ""),
                it.get("opponent_leader", ""), it.get("method", ""),
                it.get("opponent_muslim", ""), segment_id,
                float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
