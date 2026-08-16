"""OBJECT AGENT — eşyaların/eylemlerin padişah dönemine göre dağılımı.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import ObjectExchange
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.scoring import keyword_score
from utilities.source_fidelity import quote_supported, is_material_object
from utilities.validator import validate_items


# Tüm agent'lara otomatik eklenen sabit kurallar
_TURKCE_KURALI = (
    "\nÖNEMLİ: Bütün çıktıyı ve tüm metin alanlarını HER ZAMAN TÜRKÇE yaz. "
    "Asla İngilizce yazma."
)
_KALITE_KURALI = (
    "\nKAYNAK SADAKATİ: Modern tarihsel doğrulama yapma; yalnız verilen "
    "Aşıkpaşazade parçasının ne söylediğini çıkar. Dış bilgiyle düzeltme, "
    "boşluk doldurma veya söylenmeyen neden/rol/sonuç ekleme. "
    "\nDİKKATLİ TARAMA: Cevaptan önce metni sessizce iki kez tara: ilk geçişte "
    "agent kapsamındaki bütün açık adayları bul; ikinci geçişte her kaydın "
    "özne-nesne-yön-yer-tarih bağını metinden doğrula. Yinelenen, yalnız ima "
    "edilen veya OCR nedeniyle anlamsız kaydı çıkar. Her kayıt tek ve atomik "
    "olgu olsun. Confidence yalnız metinsel çıkarım açıklığını göstersin: açık "
    "ifade yüksek, zamir/bağ belirsizliği düşük; metinsel dayanak yoksa hiç ekleme."
    "\nBOŞ ÇIKTI GEÇERLİ BİR CEVAPTIR: Bu bölümde senin alanına giren hiçbir "
    "kayıt yoksa {\"items\":[]} döndür. Boş liste bir başarısızlık değildir; her "
    "bölümden en az bir kayıt çıkarma zorunluluğun YOKTUR. Kurallardan biri bir "
    "adayı eliyorsa onu zorlayarak kaydetme — elenen aday, yerine daha zayıf bir "
    "kayıt yazılması gerektiği anlamına gelmez. Aşağıdaki ÖRNEKLER bölümündeki her "
    "pasajın dolu bir çıktısı vardır; bu, her pasajın dolu çıktı vermesi gerektiği "
    "anlamına GELMEZ — örnekler yalnız biçimi ve ayrıntı düzeyini gösterir. Emin "
    "olmadığın bir kaydı yazmak, o kaydı hiç yazmamaktan daha kötüdür."
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
        items = [it for it in items if is_material_object(it, text)]
        # KANIT DENETİMİ — alıntı kaynak metinde birebir yoksa kaydı ele.
        # Güven skoru kalibre olmadığı için (model neredeyse her şeye 0.8-0.9 verir)
        # asıl eleme kapısı budur; elenenlerin gerekçesi denetim raporuna düşer.
        gecen = []
        for it in items:
            if quote_supported(it.get("evidence_quote", ""), text):
                gecen.append(it)
            else:
                self.last_errors.append(
                    f"kanıt alıntısı metinde bulunamadı, kayıt elendi: "
                    f"{str(it.get('evidence_quote', ''))[:60]!r}")
        items = gecen
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
        return keyword_score(text, self.weights)

    def __repr__(self):
        return f"Agent(name={self.name!r}, model={self.model!r})"


# ═════════════════════════════════════════════════════════════════════════════
# BU UZMANIN TANIMI (şema + ağırlık + rol)
# ═════════════════════════════════════════════════════════════════════════════
SCHEMA = {   # eşya + eylem + dönem (kim verdi/aldı YOK → o relation'da)
    "object_name": {"type": str, "required": True},
    "quality": {"type": str, "default": ""},
    "action": {"type": str, "default": ""},
    "era": {"type": str, "default": ""},
    "evidence_quote": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {"kılıç": 2.0, "at": 1.5, "kaftan": 2.0, "sancak": 2.0, "tuğ": 2.0,
           "mühür": 2.0, "para": 1.5, "hediye": 2.0, "armağan": 2.0}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="object",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
    "Sen Aşıkpaşazade metnindeki maddi kültür öğelerini çıkaran bir nesne tarihi "
        "uzmanısın.\n"
        "\n"
        "AMAÇ:\n"
        "Metinde sahiplik, hediye, miras, ticaret veya yağma ilişkisi içinde geçen "
        "fiziksel nesneleri çıkar ve her nesneyi standart bir tür ve eylem "
        "kategorisiyle sınıflandır.\n"
        "\n"
        "KURALLAR:\n"
        "0. BU BÖLÜM İÇİN her nesneyi BİR KEZ yaz. Aynı nesne bölüm içinde birkaç kez "
        "geçse de tek kayıt aç (kaç kez geçtiğini sayma). Nesnenin kim tarafından kime "
        "verildiği BU AGENTIN İŞİ DEĞİLDİR; onu relation agent ayrı ayrı tutar.\n"
        "\n"
        "1. Yalnızca metinde açıkça bulunan ve sahiplik/transfer/ekonomik ilişki içinde "
        "anlamlı bir rolü olan nesneleri çıkar. Her kullanım fiilini ayrı nesne eylemi "
        "sayma. Bir nesnenin taşınması, binilmesi, giyilmesi veya kullanılması onun "
        "temel ilişkisi sahiplikse action='Sahiplik' olarak sınıflandırılabilir.\n"
        "\n"
        "1b. TARAMA ALANLARI — nesne ararken yalnız göze çarpanı değil, şu alanların "
        "HEPSİNİ tara. Bir bölümde bu alanlardan birine giren somut bir şey sahiplik "
        "ya da aktarım ilişkisi içinde geçiyorsa kaydet:\n"
        "   - silah ve savaş takımı\n"
        "   - binek ve yük hayvanı\n"
        "   - sürü hayvanı\n"
        "   - yiyecek ve içecek\n"
        "   - dokuma, döşeme ve ev eşyası\n"
        "   - giyim kuşam ve kumaş\n"
        "   - kıymetli maden, sikke ve para\n"
        "   - alâmet ve tören eşyası (sancak, bayrak, çırağ, mühür türü)\n"
        "   - kap kacak ve mutfak eşyası\n"
        "   - ganimet olarak sayılan insanlar (bkz. 6. kural)\n"
        "   Bu alanların sonuncusu ile kıymetli maden ve giyim kuşam sık atlanır; "
        "onları ayrıca ara.\n"
        "\n"
        "2. action SERBEST METİN DEĞİLDİR. Yalnızca şu değerlerden birini kullan: "
        "'Sahiplik' (el değiştirme yok), 'İsâr' (karşılıksız verme: "
        "hediye/armağan/ihsan), 'Emanet' (geri alınmak üzere bırakma), 'Ganimet' "
        "(savaş/yağma yoluyla), 'Miras', 'Ticaret', 'Yağma'.\n"
        "   AYIRT EDİCİ ÖLÇÜTLER — bu yedi değerin hepsi gerçek adaydır, sık "
        "görülenlere kaçma:\n"
        "   - Sahiplik : el değiştirme YOK; nesne sahibinin elinde duruyor.\n"
        "   - İsâr     : karşılıksız ve İSTEYEREK verme; veren bir şey beklemez.\n"
        "   - Emanet   : geçici bırakma; nesnenin geri alınacağı bellidir.\n"
        "   - Ganimet  : savaş sonucu ZORLA el değiştirme; alan galip taraftır.\n"
        "   - Yağma    : baskın/talan yoluyla alma; savaş sonucu değil, akın sonucu.\n"
        "   - Miras    : ölüm ya da soy yoluyla geçme ('atamdan kalmıştır').\n"
        "   - Ticaret  : karşılığı olan alışveriş; bedel ya da bac söz konusudur.\n"
        "   Aşağıdaki ÖRNEKLER ağırlıklı olarak Sahiplik ve İsâr içerir; bu bir "
        "sınırlama DEĞİL, örnekleme eksikliğidir. Metin Ganimet, Miras, Ticaret ya "
        "da Yağma gösteriyorsa onu yaz. Bu sözlük relation "
        "agent'ın transfer_verb sözlüğüyle aynıdır; iki tablo birbirine bağlanabilsin "
        "diye. Metindeki yüzey fiilini doğrudan action alanına yazma. Örneğin 'atını "
        "suya sürdü' → 'Sahiplik'; 'silahları verdi' → 'Hediye'; 'malları yağmaladılar' "
        "→ 'Yağma'.\n"
        "\n"
        "3. quality nesnenin standart tür kategorisidir ve yalnızca şu değerlerden biri "
        "olabilir: 'Ağnam', 'Binek', 'Binek Aksesuarı', 'Binek Hayvan', 'Dini', "
        "'Elbise', 'Erzak', 'Ganimet', 'İktidar', 'Kamu Esbabı', 'Kâlâ', 'Kıyafet', "
        "'Resm', 'Seyfiye', 'Taşınılmaz / Toprak'. Metinde kategori adı açıkça yazmasa "
        "da nesnenin türünden güvenilir biçimde sınıflandır. Örnek: at/öküz → 'Binek "
        "Hayvan'; koyun/kuzu → 'Ağnam'; gemi/kayık/sal → 'Binek'; kılıç/yarak/kın → "
        "'Seyfiye'; peynir/kaymak gibi yiyecekler → 'Erzak'; halı/kilim/bardak/maşraba "
        "→ 'Kâlâ'; köy/toprak/yurt/yayla → 'Taşınılmaz / Toprak'; taht/sancak → "
        "'İktidar'; çan/kilise eşyası → 'Dini'; altın/gümüş/filori gibi ganimet "
        "unsurları → 'Ganimet'.\n"
        "\n"
        "4. Bir ifadede birden fazla nesne varsa her nesne için ayrı kayıt oluştur. "
        "'peynir, halı, kilim ve kuzular' tek kayıt değildir. Birbirinden bağımsız "
        "fiziksel parçaları da metinde ayrı nesne olarak anlam taşıyorsa ayrı çıkar; "
        "örneğin 'kılıç' ve 'kın' ayrı nesnelerdir.\n"
        "\n"
        "5. Açık transfer ilişkisi varsa transfer kategorisi sahipliğe göre "
        "önceliklidir: hediye → 'Hediye', miras bırakma/devralma → 'Miras', alışveriş → "
        "'Ticaret', yağmalama/ganimet alma → 'Yağma'. Bunlardan hiçbiri yok ancak "
        "nesnenin bir kişiye veya gruba ait olduğu açıksa → 'Sahiplik'. Veren ve alan "
        "kişiyi bu agentta ayrı alan olarak üretme; yalnız nesneyi ve eylem "
        "kategorisini çıkar.\n"
        "\n"
        "6. Mecazî olup gerçek veya nesneleştirilmiş bir maddi öğeyi temsil etmeyen "
        "ifadeleri çıkarma. İnsanları normalde nesne olarak çıkarma; ancak anlatıda "
        "açıkça ganimet veya mülkiyet unsurları arasında nesneleştirilerek sayılmışsa "
        "'Ganimet' kategorisinde değerlendirilebilir. Genel 'eşya' veya 'mal' sözcüğü "
        "yerine aynı bağlamda açıkça belirtilmiş daha özel nesneler varsa özel "
        "nesneleri tercih et.\n"
        "   HAK, GÖREV VE GELİR NESNE DEĞİLDİR: 'tımar', 'bac', 'haraç', 'kadılık', "
        "'hitabet', 'imamlık', 'beylik', 'vergi', 'izin' gibi devredilen ama elle "
        "tutulmayan şeyleri object_name yapma. Bunlar makam, hak ya da gelir "
        "kalemidir; nesne kaydı değil, olay kaydıdır.\n"
        "   YER NESNE DEĞİLDİR: toprak, köy, hisar, vilayet, dağ, yayla, kışlak ve "
        "her türlü yer adı object_name OLAMAZ — bunlar taşınmazdır ve conquest ile "
        "toponym agent'larının işidir. 'Köyceğiz verdi', 'Domaniç Dağı'nı yayla "
        "verdiler', 'Kulaca hisarını aldı' → nesne kaydı ÜRETME. Bir yerin mülk "
        "olarak el değiştirmesi bile onu nesne yapmaz.\n"
        "   TAŞINABİLİRLİK ÖLÇÜTÜ: bir sandığa konabiliyor, elde taşınabiliyor ya da "
        "bir hayvana yüklenebiliyorsa nesnedir. Taşınamıyorsa değildir.\n"
        "\n"
        "7. object_name metinde geçen nesneyi göstermeli; metinde olmayan nesne, "
        "özellik veya eylem ekleme. era bilgisini yalnızca verilen metin veya bölüm "
        "bağlamından güvenilir biçimde belirle. Emin değilsen spekülatif kayıt üretme. "
        "Şiir/nazım bölümlerinden nesne çıkarma; yalnız düzyazı anlatıdaki somut "
        "nesneleri kaydet. \"Himmet kılıcı\", \"din kılıcı\", \"İslam kılıcı\" gibi bir "
        "nesneyi doğrudan adlandırmayan mecaz tamlamaları da nesne sayma.\n"
        "\n"
        "8. evidence_quote kaydı doğrudan destekleyen, kaynak metinde BİREBİR geçen en "
        "kısa parçadır. Kendi cümleni kurma, özetleme, sadeleştirme veya imla düzeltme "
        "yapma — metindeki harfleri aynen kopyala (OCR bozuklukları dahil). En az üç "
        "sözcük olsun. Metinde birebir bulamıyorsan o kaydı hiç üretme.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"object_name\":\"...\",\"quality\":\"Agnam|Binek|Binek Aksesuarı|Binek Ha yvan|Elbise|Erzak|Ganimet|Kâla|Seyfiye\",\"action\":\"Sahiplik|İsâr|Emanet|Ganimet|Miras|Ticaret|Yağma\",\"era\":\"\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Uygun nesne yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Süleyman Şah atını suya sürdü.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"At\",\"quality\":\"Binek Hayvan\",\"action\":\"Sahiplik\",\"era\":\"Pre-Osman\",\"confidence\":0.9,\"evidence_quote\":\"Süleyman Şah atını suya sürdü.\"}]}\n"
        "\n"
        "METİN: \"Ertuğrul Gazi oğlanlarından Saru Yatı'yı Sultan Alaeddin' e gönderdi ve 'Bize de yurt versin, gidelim gaza edelim.' dedi. Saru Yatı babasının sözlerini Sultan Ala ed din' e getirdi. Sultan bunlar geldiği için çok sevindi. O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi. O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"\n"
        "ÇIKTI: {\"items\":[]}\n"
        "(Boş: yurt/yayla olarak verilen TOPRAK taşınmazdır (conquest agent'ın işi), haraç ise gelir kalemidir; bu pasajda nesne yoktur.)\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu ve 'Sizden dileğimiz, biz yaylaya gittiğimiz vakit eşyalarımızı sizde emanet koyalım.' dedi. O da kabul etti.\"\n"
        "ÇIKTI: {\"items\":[]}\n"
        "(Boş: 'eşya' genel toplayıcı addır ve pasajda hiçbir somut nesne sayılmamıştır; emanet ilişkisini relation agent tutar.)\n"
        "\n"
        "METİN: \"Ne zaman yaylaya gidecek olursa, Osman Gazi bütün eşyalarını öküzlere yükletir ve bir nice hatun kişiyle gönderir; onlar da varıp kalede emanet korlardı.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Öküz\",\"quality\":\"Binek Hayvan\",\"action\":\"Sahiplik\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi bütün eşyalarını öküzlere yükletir ve bir nice hatun kişiyle gönderir;\"}]}\n"
        "\n"
        "METİN: \"Yayladan döndükleri vakit de peynir, halı, kilim ve kuzular hediye getirip emanetlerini alıp giderlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Peynir\",\"quality\":\"Erzak\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Yayladan döndükleri vakit de peynir\"},{\"object_name\":\"Halı\",\"quality\":\"Kâlâ\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Yayladan döndükleri vakit de peynir, halı, kilim ve kuzular hediye getirip emanetlerini alıp giderlerdi.\"},{\"object_name\":\"Kilim\",\"quality\":\"Kâlâ\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"kilim ve kuzular hediye getirip emanetlerini alıp giderlerdi.\"},{\"object_name\":\"Kuzu\",\"quality\":\"Ağnam\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"kilim ve kuzular hediye getirip emanetlerini alıp giderlerdi.\"}]}\n"
        "\n"
        "METİN: \"Derviş, 'Şehirden vazgeçtik, bize şu köyceğiz yeter.' cevabını verdi. Osman Gazi bu isteği kabul etti.\"\n"
        "ÇIKTI: {\"items\":[]}\n"
        "(Boş: köy bir YERDİR, taşınamaz; mülk olarak el değiştirmesi onu nesne yapmaz.)\n"
        "\n"
        "METİN: \"Derviş bunun üzerine, 'Bize şimdi yazılı bir belge ver.' dedi. Osman Gazi de, 'Ben yazı yazmasını bilir miyim ki benden yazılı kağıt istersin.' dedi. Sonunda Osman Gazi 'İşte babamdan ve dedemden kalmış bir kılıcım, bir de maşrapam var. Bunları sana vereyim, elinde bulunsunlar ve bu nişanları saklasınlar.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Kılıç\",\"quality\":\"Seyfiye\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Derviş bunun üzerine, 'Bize şimdi yazılı bir belge ver.' dedi. Osman Gazi de, 'Ben yazı yazmasını bilir miyim ki benden yazılı kağıt istersin.' dedi. Sonunda Osman Gazi 'İşte babamdan ve dedemden kalmış bir kılıcım, bir de maşrapam var. Bunları sana vereyim, elinde bulunsunlar ve bu nişanları saklasınlar.\"},{\"object_name\":\"Maşraba\",\"quality\":\"Kâlâ\",\"action\":\"İsâr\",\"era\":\"Osman Gazi\",\"confidence\":0.9,\"evidence_quote\":\"bir de maşrapam var.\"}]}"
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
def to_models(items, segment_id) -> list[ObjectExchange]:
    out = []
    for it in items:
        if str(it.get("object_name", "")).strip():
            out.append(ObjectExchange(
                it["object_name"].strip(), it.get("quality", ""), it.get("action", ""),
                it.get("era", ""), segment_id,
                float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
