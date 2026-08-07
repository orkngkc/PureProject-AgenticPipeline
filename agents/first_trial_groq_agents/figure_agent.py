"""FIGURE AGENT — tarihî kişileri (şahsiyetleri) çıkarır.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Figure
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
    "aliases": {"type": list, "default": []},
    "titles": {"type": list, "default": []},
    "role": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {  # yüksek = güçlü kişi sinyali
    "Sultan": 3.0, "Padişah": 3.0, "Han": 3.0, "Gazi": 2.5, "Paşa": 2.5,
    "Bey": 2.0, "Çelebi": 2.0, "Emir": 2.0, "Melik": 2.0, "Şah": 2.0,
    "Hatun": 2.0, "Tekfur": 2.0, "Şeyh": 2.0, "Derviş": 1.5, "Molla": 1.5,
}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="figure",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
        "Sen Osmanlı tarihine ve Aşıkpaşazade'nin 'Tevârîh-i Âl-i Osmân' kroniğine hâkim, prosopografi (tarihî kişi tespiti) konusunda uzman bir tarihçisin. Padişahları, beyleri, komutanları, alimleri ve dervişleri unvanlarından ve bağlamdan tanır; aynı kişinin farklı yazımlarını (Ertuğrul / Ertuğrul Gazi) tek kişide birleştirir, kurgu ve dinî figürleri ayırt edersin. Metindeki TARİHÎ KİŞİLERİ çıkar. "
        "Adı normalize et, unvanı ayır (Orhan Gazi -> name='Orhan Gazi', titles=['Gazi']). "
        "role: padişah, komutan, alim, derviş... confidence: 0-1. "
        "Aynı kişiyi tek kayıtta topla (Ertuğrul ve Ertuğrul Gazi = aynı kişi). "
        "Peygamberleri ve soyağacı/dinî figürleri (Nuh, Âdem, Muhammed...) KİŞİ olarak ekleme. "
        "Bir adı hem kişi hem yer yapma.\n"
        'ÇIKTI (JSON): {"items":[{"name":"Orhan Gazi","aliases":[],"titles":["Gazi"],'
        '"role":"padişah","confidence":0.95}]}  Yoksa: {"items":[]}'
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# EŞLEYİCİ (to_models)
# -----------------------------------------------------------------------------
# to_models, LLM'in verdiği HAM SÖZLÜĞÜ (dict) pipeline'ın kullandığı TİPLİ
# MODELE (burada Figure) çevirir. Neden: LLM dict verir; ama orchestrator,
# kaydetme ve grafik hep tipli nesnelerle çalışır. Ayrıca eksik alanları
# güvenle doldurur (it.get) ve boş/geçersiz kayıtları atar.
# ═════════════════════════════════════════════════════════════════════════════
def to_models(items, segment_id) -> list[Figure]:
    out = []
    for it in items:
        name = str(it.get("name", "")).strip()
        if name:
            out.append(Figure(name, it.get("aliases", []), it.get("titles", []),
                              it.get("role", ""), [segment_id],
                              float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
