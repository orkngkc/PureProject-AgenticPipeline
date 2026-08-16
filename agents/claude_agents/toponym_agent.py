"""TOPONYM AGENT — yer adlarını (toponimleri) çıkarır.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Toponym
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
            user += ("\nÖNCEKİ BÖLÜMLERDE GEÇEN YERLER (yalnız yazım birliği ve "
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
    "modern_name": {"type": str, "default": ""},
    "place_type": {"type": str, "default": ""},
    "evidence_quote": {"type": str, "default": ""},
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
    "Sen Aşıkpaşazade metnindeki yer adlarını çıkaran bir tarihî toponimi "
        "uzmanısın.\n"
        "\n"
        "AMAÇ:\n"
        "Metinde gerçek bir coğrafi konumu ifade eden açık yer adlarını çıkar ve yer "
        "türlerini tutarlı biçimde sınıflandır.\n"
        "\n"
        "KURALLAR:\n"
        "1. Şehir, köy, kale/hisar, yerleşim, bölge/vilayet, dağ, ova, nehir, geçit, "
        "höyük ve benzeri açık coğrafi yerleri çıkar. Bir sözcüğü yalnız yer adı "
        "biçimine benzediği için değil, cümlede gerçekten coğrafi konum olarak "
        "kullanılıyorsa kaydet.\n"
        "\n"
        "2. name alanında kaynakta geçen TAM tarihî yer ifadesini koru. 'kalesi', "
        "'hisarı', 'vilayeti', 'dağı', 'ovası', 'ırmağı', 'beli', 'köyü' gibi yer "
        "adının parçası olan tür ifadelerini name alanından silme; ayrıca place_type "
        "alanında sınıflandır. Örnek: 'Ca'ber Kalesi' → name='Ca'ber Kalesi', "
        "place_type='kale'; 'Rum vilayeti' → name='Rum vilayeti', place_type='bölge'.\n"
        "\n"
        "3. Bir kişi adı yer adının parçasıysa ifadeyi bölme. Örneğin 'Hamza Bey köyü' "
        "coğrafi bir yer adıysa name='Hamza Bey köyü' olarak çıkar; yalnız 'Hamza Bey' "
        "şeklinde kesme. Buna karşılık 'Bilecik tekfuru' gibi ifadelerde Bilecik "
        "yalnızca kişiyi tanımlayan bir unsur olarak kullanılıyorsa onu bu kullanımdan "
        "toponim olarak çıkarma.\n"
        "\n"
        "4. Hanedan, etnik grup, topluluk veya kişi adının içindeki yer kökünü otomatik "
        "olarak toponim sayma. Örneğin 'Şam Türkmenleri' ifadesinde Şam gerçek bir "
        "konum olarak kullanılmıyorsa kayıt üretme. Aynı sözcük başka bir cümlede "
        "gerçek yer olarak kullanılıyorsa o kullanım ayrıca çıkarılabilir.\n"
        "\n"
        "5. modern_name yalnızca verilen metin açıkça modern/eşdeğer adı bildiriyorsa "
        "doldur. Dış tarih veya coğrafya bilgisinden modern ad tahmin etme; metinde "
        "verilmiyorsa boş bırak. Farklı tarihî yazımları da dış bilgi kullanarak "
        "otomatik birleştirme.\n"
        "\n"
        "6. place_type kısa ve tutarlı olmalıdır. Mümkün olduğunda şu kategorileri "
        "kullan: 'şehir', 'köy', 'kale', 'yerleşim', 'bölge', 'dağ', 'ova', 'nehir', "
        "'geçit', 'höyük'. Tür metinden kesin belirlenemiyorsa en güvenli üst "
        "kategoriyi kullan; örneğin adı bilinen fakat türü açıklanmayan bir yer için "
        "'yerleşim'. Aynı yer ifadesi aynı bölümde tekrar geçiyorsa tek kayıt oluştur.\n"
        "\n"
        "7. OCR hatası olduğu açık olan bozuk kelimeleri, kişileri, unvanları ve "
        "yalnızca soy/topluluk bildiren ifadeleri yer adı olarak üretme. Metinde "
        "bulunmayan yer, modern ad veya yer türü ekleme; emin değilsen spekülatif kayıt "
        "üretme.\n"
        "\n"
        "9. evidence_quote kaydı doğrudan destekleyen, kaynak metinde BİREBİR geçen en "
        "kısa parçadır. Kendi cümleni kurma, özetleme, sadeleştirme veya imla düzeltme "
        "yapma — metindeki harfleri aynen kopyala (OCR bozuklukları dahil). En az üç "
        "sözcük olsun. Metinde birebir bulamıyorsan o kaydı hiç üretme.\n"
        "\n"
        "8. ÖNCEKİ BÖLÜMLERDEN GELEN ADLAR. Girdide 'ÖNCEKİ BÖLÜMLERDE GEÇEN YERLER' "
        "listesi verilebilir. Bu liste ZORLAYICI DEĞİLDİR; yalnız yazım birliği ve "
        "kimlik sürekliliği içindir. İki işi vardır: (a) bu bölümde aynı varlık "
        "geçiyorsa listedeki yazımı kullan, farklı bir biçimle karşılaşırsan onu "
        "aliases'a koy; (b) bu bölümde zamirle ya da tanımlayıcı bir ifadeyle anılan "
        "varlık listedekilerden biriyse ('gelin' = önceki bölümdeki 'Yarhisar "
        "tekfurunun kızı') listedeki adı kullan. Listedeki bir varlığı BU bölümde "
        "geçmiyorsa yazma; liste kanıt değildir, yalnız hafızadır.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"name\":\"...\",\"modern_name\":\"\",\"place_type\":\"şehir|köy|kale|yerleşim|bölge|dağ|ova|nehir|geçit|höyük\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Yer yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Osman Gazi' nin d ed esi olan Süleyman Şah Gazi en evvel Rum vilayetine gelmişti, işte gelmeleri için asıl sebep buydu.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Rum vilayeti\",\"modern_name\":\"Anadolu\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi' nin d ed esi olan Süleyman Şah Gazi en evvel Rum vilayetine gelmişti\"}]}\n"
        "\n"
        "METİN: \"Süleyman Şah bunu kabul edip önce Erzurum' a, sonra da Erzincan' a geldi. Erzincan' dan Rum vilayetine girdiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Erzurum\",\"modern_name\":\"Erzurum\",\"place_type\":\"şehir\",\"confidence\":0.9,\"evidence_quote\":\"Süleyman Şah bunu kabul edip önce Erzurum' a\"},{\"name\":\"Erzincan\",\"modern_name\":\"Erzincan\",\"place_type\":\"şehir\",\"confidence\":0.9,\"evidence_quote\":\"sonra da Erzincan' a geldi.\"},{\"name\":\"Rum vilayeti\",\"modern_name\":\"Anadolu\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Süleyman Şah bunu kabul edip önce Erzurum' a\"}]}\n"
        "\n"
        "METİN: \"Tekrar Türkistan'a yöneldiler. Geldikleri yola gitmeyip Halep vilayetine çıktılar. Ca'ber Kalesi'ne vardılar ve Fırat Irmağı önlerine geldiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Türkistan\",\"modern_name\":\"Türkistan\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Tekrar Türkistan'a yöneldiler.\"},{\"name\":\"Halep vilayeti\",\"modern_name\":\"Halep Eyaleti\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Geldikleri yola gitmeyip Halep vilayetine çıktılar.\"},{\"name\":\"Ca'ber Kalesi\",\"modern_name\":\"Caber Kalesi\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Ca'ber Kalesi'ne vardılar ve Fırat Irmağı önlerine geldiler.\"},{\"name\":\"Fırat Irmağı\",\"modern_name\":\"Fırat Nehri\",\"place_type\":\"nehir\",\"confidence\":0.9,\"evidence_quote\":\"Ca'ber Kalesi'ne vardılar ve Fırat Irmağı önlerine geldiler.\"}]}\n"
        "\n"
        "METİN: \"Sudan çıkarıp Ca'ber Kalesi'nin önüne defnettiler. Şimdiki zamanda oraya 'Türk Mezarı' derler. Ayrıca o kaleye de yine aynı soydan Dügerler adında bir topluluk, eskiden olduğu gibi şimdi de hükmetınektedir.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Ca'ber Kalesi\",\"modern_name\":\"Caber Kalesi\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Sudan çıkarıp Ca'ber Kalesi'nin önüne defnettiler.\"},{\"name\":\"Türk Mezarı\",\"modern_name\":\"Süleyman Şah Türbesi\",\"place_type\":\"türbe\",\"confidence\":0.9,\"evidence_quote\":\"Şimdiki zamanda oraya 'Türk Mezarı' derler.\"},{\"name\":\"o kale\",\"modern_name\":\"Caber Kalesi\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Şimdiki zamanda oraya 'Türk Mezarı' derler.\"}]}\n"
        "\n"
        "METİN: \"Bazısı çölden yana gitti. Şimdiki zamanda onlara Şam Türkmenleri derler. Bazısı yine Anadolu'ya döndü.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"çöl\",\"modern_name\":\"Suriye Çölü\",\"place_type\":\"çöl\",\"confidence\":0.9,\"evidence_quote\":\"Bazısı çölden yana gitti.\"},{\"name\":\"Anadolu\",\"modern_name\":\"Anadolu\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Bazısı yine Anadolu'ya döndü.\"}]}\n"
        "\n"
        "METİN: \"Bu üç kardeş geldikleri yola dönüp Fırat'ın başından yürüyüp Pasin Ovası ve Sürmelü Çukuru'na gıi.ı:üer.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Fırat\",\"modern_name\":\"Fırat Nehri\",\"place_type\":\"nehir\",\"confidence\":0.9,\"evidence_quote\":\"Bu üç kardeş geldikleri yola dönüp Fırat'ın başından yürüyüp Pasin Ovası ve Sürmelü Çukuru'na gıi.ı:üer.\"},{\"name\":\"Pasin Ovası\",\"modern_name\":\"Pasinler Ovası\",\"place_type\":\"ova\",\"confidence\":0.9,\"evidence_quote\":\"Bu üç kardeş geldikleri yola dönüp Fırat'ın başından yürüyüp Pasin Ovası ve Sürmelü Çukuru'na gıi.ı:üer.\"},{\"name\":\"Sürmelü Çukuru\",\"modern_name\":\"Sürmeli Çukuru\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Bu üç kardeş geldikleri yola dönüp Fırat'ın başından yürüyüp Pasin Ovası ve Sürmelü Çukuru'na gıi.ı:üer.\"}]}\n"
        "\n"
        "METİN: \"Bir hayli zaman geçince Sultan Alaeddin Rum ülkesine yöneldi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Rum ülkesi\",\"modern_name\":\"Anadolu\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Bir hayli zaman geçince Sultan Alaeddin Rum ülkesine yöneldi.\"}]}\n"
        "\n"
        "METİN: \"Bunlar da Anadolu'ya yöneldiler. Gelip hısn-ı Musul vilayetine ulaştılar.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Anadolu\",\"modern_name\":\"Anadolu\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Bunlar da Anadolu'ya yöneldiler.\"},{\"name\":\"hısn-ı Musul vilayeti\",\"modern_name\":\"Baştabya Kalesi\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Gelip hısn-ı Musul vilayetine ulaştılar.\"}]}\n"
        "\n"
        "METİN: \"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi. O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Karacahisar\",\"modern_name\":\"Karacahisar\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"},{\"name\":\"Bilecik\",\"modern_name\":\"Bilecik\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"},{\"name\":\"Söğüt vilayeti\",\"modern_name\":\"Söğüt\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler;\"},{\"name\":\"Domaniç Dağı\",\"modern_name\":\"Domaniç Dağları\",\"place_type\":\"dağ\",\"confidence\":0.9,\"evidence_quote\":\"ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"},{\"name\":\"İrıneni Bili'nin dağı\",\"modern_name\":\"Ahî Dağı\",\"place_type\":\"dağ\",\"confidence\":0.9,\"evidence_quote\":\"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"}]}\n"
        "\n"
        "METİN: \"Ertuğrul kabul edip yürüdü ve Ankara'ya geldiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Ankara\",\"modern_name\":\"Ankara\",\"place_type\":\"şehir\",\"confidence\":0.9,\"evidence_quote\":\"Ertuğrul kabul edip yürüdü ve Ankara'ya geldiler.\"}]}\n"
        "\n"
        "METİN: \"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı, ona Çavudur derlerdi. Ayrıca Karahisar'la Bilecik vilayetlerini zaman zaman vurup rahatsız eden Tatarlar vardı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Karahisar vilayeti\",\"modern_name\":\"Karacahisar\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı\"},{\"name\":\"Karahisar\",\"modern_name\":\"Karacahisar\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı\"},{\"name\":\"Bilecik vilayeti\",\"modern_name\":\"Bilecik\",\"place_type\":\"bölge\",\"confidence\":0.9,\"evidence_quote\":\"Ayrıca Karahisar'la Bilecik vilayetlerini zaman zaman vurup rahatsız eden Tatarlar vardı.\"}]}\n"
        "\n"
        "METİN: \"Aya Nikola adında bir kafir, İnegöl' de Osman yay la ya ve kışlaya gittikleri zamanda, bunların göçünü yağmalayıp zarar verirdi.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İnegöl\",\"modern_name\":\"İnegöl\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Aya Nikola adında bir kafir\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Bilecik\",\"modern_name\":\"Bilecik\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Bilecik tekturuna bundan şikayette bulundu\"}]}\n"
        "\n"
        "METİN: \"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İrıneni Beli\",\"modern_name\":\"Ermeni-Beli\",\"place_type\":\"geçit\",\"confidence\":0.9,\"evidence_quote\":\"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"},{\"name\":\"İnegöl\",\"modern_name\":\"İnegöl\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"}]}\n"
        "\n"
        "METİN: \"mezarı İrıneni Beli'nin bitiminde Hamza Bey köyünün bucağındadır.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İrıneni Beli\",\"modern_name\":\"Ermeni-Beli\",\"place_type\":\"geçit\",\"confidence\":0.9,\"evidence_quote\":\"mezarı İrıneni Beli'nin bitiminde Hamza Bey köyünün bucağındadır.\"},{\"name\":\"Hamza Bey köyü\",\"modern_name\":\"Hamzabey Köyü\",\"place_type\":\"köy\",\"confidence\":0.9,\"evidence_quote\":\"mezarı İrıneni Beli'nin bitiminde Hamza Bey köyünün bucağındadır.\"}]}\n"
        "\n"
        "METİN: \"Bir gece yürüyüp İnegöl'e vardı. Yanında Kulaca denen küçük bir hisar vardı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İnegöl\",\"modern_name\":\"İnegöl\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Bir gece yürüyüp İnegöl'e vardı.\"},{\"name\":\"Kulaca\",\"modern_name\":\"Kulaca\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Yanında Kulaca denen küçük bir hisar vardı.\"}]}\n"
        "\n"
        "METİN: \"Sabah olunca o ilin kafideri toplanıp Karacahisar tekfuruna,\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Karacahisar\",\"modern_name\":\"Karacahisar\",\"place_type\":\"kale\",\"confidence\":0.9,\"evidence_quote\":\"Sabah olunca o ilin kafideri toplanıp Karacahisar tekfuruna,\"}]}\n"
        "\n"
        "METİN: \"Emrine büyük bir ordu verdi, bunlar inegöl kafideriyle bir araya geldiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İnegöl\",\"modern_name\":\"İnegöl\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Emrine büyük bir ordu verdi\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi de gazilerini topladı ve İkizce'ye geldi. Torna59 60 lıç Beli'ni aştıkları yerde savaştılar.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İkizce\",\"modern_name\":\"İkizce\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi de gazilerini topladı ve İkizce'ye geldi.\"},{\"name\":\"Tornalıç Beli\",\"modern_name\":\"Domaniç\",\"place_type\":\"geçit\",\"confidence\":0.9,\"evidence_quote\":\"Torna59 60 lıç Beli'ni aştıkları yerde savaştılar.\"}]}\n"
        "\n"
        "METİN: \"O yerde bir çam ağacı var, şimdi ona Kandilli Çam derler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Kandilli Çam\",\"modern_name\":\"Kandilli Çam\",\"place_type\":\"mevki\",\"confidence\":0.9,\"evidence_quote\":\"şimdi ona Kandilli Çam derler.\"}]}\n"
        "\n"
        "METİN: \"Bu sebepten o yerin adı 'İteşeni' olarak kaldı.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"İteşeni\",\"modern_name\":\"İteşeni\",\"place_type\":\"mevki\",\"confidence\":0.9,\"evidence_quote\":\"Bu sebepten o yerin adı 'İteşeni' olarak kaldı.\"}]}\n"
        "\n"
        "METİN: \"Saru Yatı'yı da götürüp Söğüt'te babasının yanına defnettiler.\"\n"
        "ÇIKTI: {\"items\":[{\"name\":\"Söğüt\",\"modern_name\":\"Söğüt\",\"place_type\":\"yerleşim\",\"confidence\":0.9,\"evidence_quote\":\"Saru Yatı'yı da götürüp Söğüt'te babasının yanına defnettiler.\"}]}"
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
