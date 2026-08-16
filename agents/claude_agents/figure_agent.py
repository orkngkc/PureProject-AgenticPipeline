"""FIGURE AGENT — tarihî kişileri (şahsiyetleri) çıkarır.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Figure
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.scoring import keyword_score
from utilities.source_fidelity import quote_supported
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

    def run(self, text: str, segment_id: str = "",
            known: list[str] | None = None) -> list[dict]:
        """LLM'e sor -> JSON ayrıştır -> DENETLE -> ağırlık skoru ekle."""
        user = f"SEGMENT ID: {segment_id}\n\nMETİN:\n\"\"\"\n{text}\n\"\"\"\n"
        if known:
            user += ("\nÖNCEKİ BÖLÜMLERDE GEÇEN KİŞİLER (yalnız yazım birliği ve "
                     "kimlik sürekliliği için; kanıt değildir): "
                     + ", ".join(known[:60]) + "\n")
        user += "\nYalnızca JSON döndür."
        raw = self.client.chat(self.role, user)
        items = _parse(raw)
        if self.schema:
            items, self.last_errors = validate_items(items, self.schema)
        else:
            self.last_errors = []
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
SCHEMA = {
    "name": {"type": str, "required": True},
    "aliases": {"type": list, "default": []},
    "titles": {"type": list, "default": []},
    "role": {"type": str, "default": ""},
    "social_group": {"type": str, "default": ""},
    "evidence_quote": {"type": str, "default": ""},
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
    "Sen Aşıkpaşazade'nin Tevârîh-i Âl-i Osmân anlatısındaki tarihî kişileri tespit "
        "eden bir prosopografi uzmanısın.\n"
        "\n"
        "AMAÇ:\n"
        "Metinde bağımsız tarihî aktör olarak geçen kişileri çıkar; adlarını, "
        "unvanlarını, alternatif yazımlarını ve kısa toplumsal/işlevsel rollerini ayır.\n"
        "\n"
        "KURALLAR:\n"
        "1. Yalnızca metinde kişi olarak geçen tarihî aktörleri çıkar. Padişah, bey, "
        "gazi, tekfur, şeyh, derviş, komutan, hatun ve adı bilinmeyen fakat belirli bir "
        "kişi olarak anılan aktörler kapsama girer. Peygamberleri, yalnız dinî referans "
        "olarak geçen figürleri ve sadece soy zinciri vermek amacıyla anılan kişileri "
        "çıkarma.\n"
        "\n"
        "2. Adı bilinen kişilerde ayrılabilir unvanları name alanından çıkarıp titles "
        "alanına koy. Örnek: 'Osman Gazi' → name='Osman', titles=['Gazi']; 'Sultan "
        "Alaeddin' → name='Alaeddin', titles=['Sultan']. Ancak kişinin şahsi adı "
        "bilinmiyor ve yalnız görev/yer ilişkisiyle tanımlanıyorsa 'Bilecik Tekfuru' "
        "gibi açıklayıcı kişi etiketini name alanında koru.\n"
        "\n"
        "2b. social_group kişinin metindeki TOPLUMSAL/İŞLEVSEL GRUBUDUR — tek sözcüklük "
        "(en fazla iki sözcük) bir etiket. Sık kullanılan etiketler: Gazi, Tekfur, "
        "Kâfir, Bey, Şeyh, Derviş, Hatun, Sultan, Martaloz, Leşker, Anlatıcı. Grubu "
        "metnin kendisinden belirle; listede karşılığı yoksa metnin kullandığı sözcüğü "
        "tekil ve yalın biçimde yaz. Biyografik açıklama veya cümle yazma. Grup "
        "güvenilir biçimde belirlenemiyorsa boş bırak.\n"
        "\n"
        "3. role kısa bir kategori olmalıdır; biyografik açıklama veya cümle yazma. "
        "Örnek roller: 'padişah', 'sultan', 'bey', 'gazi', 'tekfur', 'şeyh', 'derviş', "
        "'alim', 'komutan', 'hatun', 'lider'. Rol metinden güvenilir biçimde "
        "belirlenemiyorsa boş bırak.\n"
        "\n"
        "4. Aynı kişinin metinde geçen farklı yazımlarını tek kayıtta birleştir ve "
        "diğer biçimleri aliases alanına koy. Alias eşleşmesini yalnızca metin veya "
        "verilen bağlam aynı kişi olduğunu açıkça destekliyorsa yap; dış tarih "
        "bilgisinden hareketle iki adı aynı kişi varsayma.\n"
        "\n"
        "5. Yer adıyla tanımlanan anonim aktörleri kaçırma. 'Bilecik tekfuru', "
        "'Karacahisar tekfuru', 'İnegöl tekfuru' gibi ifadeler belirli bir kişiyi "
        "gösteriyorsa Figure olarak çıkar. Bir yer adı başka bağlamda gerçek lokasyon "
        "olarak da geçebilir; aynı kökün yer olarak kullanılması onun kişi kullanımını "
        "engellemez. Kişi mi yer mi olduğuna cümledeki işlevine göre karar ver. Aynı "
        "biçimde yalnız akrabalık veya bağlılık ilişkisiyle tanımlanan ADSIZ kişileri "
        "de çıkar: \"Köse Mihal'in kızı\", \"Taceddin-i Kürdi'nin kızı\", \"şeyhin "
        "öğrencisi\", \"Gül Falanozoğlu'nun beyi\" gibi ifadeler belirli bir kişiyi "
        "gösteriyorsa Figure say ve bu tanımlayıcı ifadeyi name alanına aynen yaz. "
        "SINIRLAR: (i) Tek başına unvan veya makam adı kişi kaydı DEĞİLDİR — \"Sultan\", "
        "\"Bey\", \"Paşa\", \"Tekfur\", \"Hatun\" gibi sözcükler ancak bir ada ya da ayırt "
        "edici bir tamlamaya bağlıysa kişi gösterir (\"Sultan\" ✗, \"Bilecik tekfuru\" ✓). "
        "(ii) Yalnızca bir YER ADININ içinde geçen kişi adını kişi olarak çıkarma; "
        "\"Hamza Bey köyü\", \"Harmankaya\" gibi yer adlarındaki şahıs adları o metinde "
        "aktör değildir. (iii) Aynı birey metinde hem adıyla hem tanımlayıcı bir "
        "ifadeyle anılıyorsa TEK kayıt aç, diğer biçimi aliases alanına koy (\"Köse "
        "Mihal'in kızı\" ile \"gelin\" aynı kişiyse tek kayıt).\n"
        "   (iv) ÇIPLAK KAVİM/HALK ADI KİŞİ DEĞİLDİR: \"Tatar\", \"Türk\", \"Acem\", "
        "\"Arap\", \"Rum\", \"kafirler\" gibi sözcükler tek başına Figure olamaz — bunlar "
        "bir topluluğun adıdır, belirli bir aktörü göstermez. Ancak bir YERE ya da bir "
        "KİŞİYE bağlanarak belirli bir grubu işaret ediyorsa çıkarılır: \"İnegöl "
        "kafirleri\" ✓, \"kafirler\" ✗; \"Germiyanoğlu\" ✓, \"Germiyanlılar\" ✗.\n"
        "\n"
        "6. Aynı kişi aynı metin parçasında tekrar geçiyorsa birden fazla Figure kaydı "
        "oluşturma. Tek kayıtta name, aliases, titles ve role bilgilerini birleştir.\n"
        "\n"
        "7. Metinde bulunmayan kişi adı, alias, unvan veya rol EKLEME. Zamir veya örtük "
        "kişi referansını yalnızca hangi kişiye ait olduğu açıkça çözülebiliyorsa "
        "mevcut kişiyle birleştir; emin değilsen yeni kişi üretme.\n"
        "\n"
        "8. Anlatının ÇERÇEVESİNİ kişi sayma. Şunlar kapsam dışıdır: (a) müellifin "
        "kendi sesi ve takma adı (\"Aşıki\", \"ben\", \"derviş\"); (b) eserin kimden "
        "işitildiğini, ne zaman derlendiğini veya kime sunulduğunu anlatan künye/icazet "
        "cümlelerinde geçen çağdaş hükümdarlar ve rivayet halkası (\"Bu hikayeyi ... "
        "zamanında ...'dan işittim\" kalıbı); (c) Bâb 1'deki Nuh-Oğuz soy zincirinde "
        "yalnız ata adı olarak sıralanan kişiler. Bunlar anlatının aktörü değil, eserin "
        "çerçevesidir. Aynı kişi başka bir yerde anlatının içinde gerçek aktör olarak "
        "geçiyorsa orada çıkarılır.\n"
        "\n"
        "10. evidence_quote kaydı doğrudan destekleyen, kaynak metinde BİREBİR geçen en "
        "kısa parçadır. Kendi cümleni kurma, özetleme, sadeleştirme veya imla düzeltme "
        "yapma — metindeki harfleri aynen kopyala (OCR bozuklukları dahil). En az üç "
        "sözcük olsun. Metinde birebir bulamıyorsan o kaydı hiç üretme.\n"
        "\n"
        "9. ÖNCEKİ BÖLÜMLERDEN GELEN ADLAR. Girdide 'ÖNCEKİ BÖLÜMLERDE GEÇEN KİŞİLER' "
        "listesi verilebilir. Bu liste ZORLAYICI DEĞİLDİR; yalnız yazım birliği ve "
        "kimlik sürekliliği içindir. İki işi vardır: (a) bu bölümde aynı varlık "
        "geçiyorsa listedeki yazımı kullan, farklı bir biçimle karşılaşırsan onu "
        "aliases'a koy; (b) bu bölümde zamirle ya da tanımlayıcı bir ifadeyle anılan "
        "varlık listedekilerden biriyse ('gelin' = önceki bölümdeki 'Yarhisar "
        "tekfurunun kızı') listedeki adı kullan. Listedeki bir varlığı BU bölümde "
        "geçmiyorsa yazma; liste kanıt değildir, yalnız hafızadır.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"name\":\"...\",\"aliases\":[],\"titles\":[],\"role\":\"...\",\"social_group\":\"\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Kişi yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Göçer evin önde gelenlerinden olan Süleyman Şah Gazi'yi ileri çektiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Süleyman Şah\",\"aliases\":[\"Süleyman Şah Gazi\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Göçer evin önde gelenlerinden olan Süleyman Şah Gazi'yi ileri çektiler.\"}]}\n"
        "\n"
        "METİN: \"Bunların biri Sunkur Tigin, biri Gündoğdu ve biri de Ertuğrul Gazi' dir.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Sunkur Tigin\",\"aliases\":[],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Bunların biri Sunkur Tigin\"},{\"name\":\"Gündoğdu\",\"aliases\":[],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"biri Gündoğdu ve biri de Ertuğrul Gazi' dir.\"},{\"name\":\"Ertuğrul Gazi\",\"aliases\":[\"Ertuğrul\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"biri Gündoğdu ve biri de Ertuğrul Gazi' dir.\"}]}\n"
        "\n"
        "METİN: \"Bir hayli zaman geçince Sultan Alaeddin Rum ülkesine yöneldi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Sultan Alaeddin\",\"aliases\":[\"Alaeddin\"],\"titles\":[\"Sultan\"],\"role\":\"sultan\",\"social_group\":\"Sultan\",\"confidence\":0.9,\"evidence_quote\":\"Bir hayli zaman geçince Sultan Alaeddin Rum ülkesine yöneldi.\"}]}\n"
        "\n"
        "METİN: \"Ertuğrul Gazi'nin, Osman, Gündüz ve Saru Yatı adında üç oğlu var idi. Saru Yatı'ya Savcı da derlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Osman\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"Ertuğrul Gazi'nin, Osman, Gündüz ve Saru Yatı adında üç oğlu var idi. Saru Yatı'ya Savcı da derlerdi.\"},{\"name\":\"Gündüz\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"Gündüz ve Saru Yatı adında üç oğlu var idi.\"},{\"name\":\"Saru Yatı\",\"aliases\":[\"Savcı\"],\"titles\":[\"Bey\"],\"role\":\"bey\",\"social_group\":\"Bey\",\"confidence\":0.9,\"evidence_quote\":\"Gündüz ve Saru Yatı adında üç oğlu var idi.\"}]}\n"
        "\n"
        "METİN: \"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Karacahisar tekfuru\",\"aliases\":[],\"titles\":[\"Tekfur\"],\"role\":\"tekfur\",\"social_group\":\"Tekfur\",\"confidence\":0.9,\"evidence_quote\":\"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"},{\"name\":\"Bilecik tekfuru\",\"aliases\":[],\"titles\":[\"Tekfur\"],\"role\":\"tekfur\",\"social_group\":\"Tekfur\",\"confidence\":0.9,\"evidence_quote\":\"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"}]}\n"
        "\n"
        "METİN: \"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı, ona Çavudur derlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Alışıra\",\"aliases\":[\"Çavudur\"],\"titles\":[\"Bey\"],\"role\":\"bey\",\"social_group\":\"Bey\",\"confidence\":0.9,\"evidence_quote\":\"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı\"}]}\n"
        "\n"
        "METİN: \"Ertuğrul Gazi vefat edince, Osman Gazi'yi babasının yerine koydular.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Ertuğrul Gazi\",\"aliases\":[\"Ertuğrul\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Ertuğrul Gazi vefat edince\"},{\"name\":\"Osman Gazi\",\"aliases\":[\"Osman\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi'yi babasının yerine koydular.\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi başa geçince komşu kafirlerle çok iyi geçindi, ancak Germiyanoğlu'yla düşmanlığa başladı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Osman Gazi\",\"aliases\":[\"Osman\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi başa geçince komşu kafirlerle çok iyi geçindi\"},{\"name\":\"Germiyanoğlu\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"ancak Germiyanoğlu'yla düşmanlığa başladı.\"}]}\n"
        "\n"
        "METİN: \"Aya Nikola adında bir kafir, İnegöl' de Osman yay la ya ve kışlaya gittikleri zamanda, bunların göçünü yağmalayıp zarar verirdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Aya Nikola\",\"aliases\":[],\"titles\":[],\"role\":\"kafir\",\"social_group\":\"Kâfir\",\"confidence\":0.9,\"evidence_quote\":\"Aya Nikola adında bir kafir\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Osman Gazi\",\"aliases\":[\"Osman\"],\"titles\":[\"Gazi\"],\"role\":\"gazi\",\"social_group\":\"Gazi\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu\"},{\"name\":\"Bilecik tekfuru\",\"aliases\":[\"Bilecik tekturu\"],\"titles\":[\"Tekfur\"],\"role\":\"tekfur\",\"social_group\":\"Tekfur\",\"confidence\":0.9,\"evidence_quote\":\"Bilecik tekturuna bundan şikayette bulundu\"}]}\n"
        "\n"
        "METİN: \"Ancak inegöl kafideri Osman Gazi' den çekinirler, onlar da bu kafiderden sakınırlardı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İnegöl kafirleri\",\"aliases\":[],\"titles\":[],\"role\":\"kafir\",\"social_group\":\"Kâfir\",\"confidence\":0.9,\"evidence_quote\":\"Ancak inegöl kafideri Osman Gazi' den çekinirler\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi'nin Artun adında bir adamı bu durumu gelip\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Artun\",\"aliases\":[],\"titles\":[],\"role\":\"martaloz\",\"social_group\":\"Martaloz\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi'nin Artun adında bir adamı bu durumu gelip\"}]}\n"
        "\n"
        "METİN: \"Osman'ın kardeşi Saru Yatı'nın oğlu Uyal Hoca şehit düştü;\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Saru Yatı\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"Osman'ın kardeşi Saru Yatı'nın oğlu Uyal Hoca şehit düştü;\"},{\"name\":\"Uyal Hoca\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"Osman'ın kardeşi Saru Yatı'nın oğlu Uyal Hoca şehit düştü;\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi'nin düşünü yoran Şeyh Edebalı, padişahlığı ona ve soyuna müjdeledi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Şeyh Edebalı\",\"aliases\":[\"Edebalı\",\"Şeyh\"],\"titles\":[\"Şeyh\"],\"role\":\"şeyh\",\"social_group\":\"Şeyh\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi'nin düşünü yoran Şeyh Edebalı\"}]}\n"
        "\n"
        "METİN: \"Ayrıca benim kızım Malhun, senin eşin olacak.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Malhun\",\"aliases\":[],\"titles\":[\"Hatun\"],\"role\":\"hatun\",\"social_group\":\"Hatun\",\"confidence\":0.9,\"evidence_quote\":\"Ayrıca benim kızım Malhun\"}]}\n"
        "\n"
        "METİN: \"Yanında şeyhin bir de öğrencisi vardı ve adına Derviş Tururoğlu derlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Derviş Tururoğlu\",\"aliases\":[\"Tururoğlu\"],\"titles\":[\"Derviş\"],\"role\":\"derviş\",\"social_group\":\"Derviş\",\"confidence\":0.9,\"evidence_quote\":\"Yanında şeyhin bir de öğrencisi vardı ve adına Derviş Tururoğlu derlerdi.\"}]}\n"
        "\n"
        "METİN: \"İhtiyarlığında aldığı hanım Taceddin-i Kürdl'nin kızı idi. Hayreddin bacanak oldu.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Taceddin-i Kürdi\",\"aliases\":[\"Taceddin-i Kürdl\"],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"İhtiyarlığında aldığı hanım Taceddin-i Kürdl'nin kızı idi.\"},{\"name\":\"Hayreddin\",\"aliases\":[],\"titles\":[],\"role\":\"\",\"social_group\":\"\",\"confidence\":0.9,\"evidence_quote\":\"Hayreddin bacanak oldu.\"}]}\n"
        "\n"
        "METİN: \"Köse Mihal düğün edip kızını Gül Falanozoğlu'nun beyine veriyor.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Köse Mihal\",\"aliases\":[],\"titles\":[],\"role\":\"bey\",\"social_group\":\"Bey\",\"confidence\":0.95,\"evidence_quote\":\"Köse Mihal düğün edip kızını Gül Falanozoğlu'nun beyine veriyor.\"},{\"name\":\"Köse Mihal'in kızı\",\"aliases\":[],\"titles\":[],\"role\":\"hatun\",\"social_group\":\"Hatun\",\"confidence\":0.9,\"evidence_quote\":\"Köse Mihal düğün edip kızını Gül Falanozoğlu'nun beyine veriyor.\"},{\"name\":\"Gül Falanozoğlu'nun beyi\",\"aliases\":[],\"titles\":[\"Bey\"],\"role\":\"bey\",\"social_group\":\"Bey\",\"confidence\":0.9,\"evidence_quote\":\"Köse Mihal düğün edip kızını Gül Falanozoğlu'nun beyine veriyor.\"}]}\n"
        "\n"
        "METİN: \"Sabah olunca o ilin kafideri toplanıp Karacahisar tekfuruna,\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Karacahisar tekfuru\",\"aliases\":[],\"titles\":[\"Tekfur\"],\"role\":\"tekfur\",\"social_group\":\"Tekfur\",\"confidence\":0.9,\"evidence_quote\":\"Sabah olunca o ilin kafideri toplanıp Karacahisar tekfuruna,\"}]}\n"
        "\n"
        "METİN: \"Onun Kalanoz adında bir de kardeşi vardı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Kalanoz\",\"aliases\":[\"Kalanoz kafiri\"],\"titles\":[],\"role\":\"kafir\",\"social_group\":\"Kâfir\",\"confidence\":0.9,\"evidence_quote\":\"Onun Kalanoz adında bir de kardeşi vardı.\"}]}"
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
                              it.get("role", ""), str(it.get("social_group", "")).strip(),
                              [segment_id],
                              float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
