"""ATTRIBUTE AGENT — kişi/sosyal gruba atanan sıfatları çıkarır.
"""
from __future__ import annotations


import config
from models import Attribute
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.scoring import keyword_score
from utilities.source_fidelity import attribute_has_source_support
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
            figures: list[dict] | None = None) -> list[dict]:
        """LLM'e sor -> JSON ayrıştır -> DENETLE -> HEDEF SÜZ -> ağırlık skoru ekle.

        figures = figure agent'ın AYNI bölümde bulduğu kayıtlar. Sıfatın hedefi yalnız
        bu listedeki kişilerden veya onların sosyal gruplarından seçilebilir; listede
        olmayan hedef için üretilen kayıt elenir.
        """
        figures = figures or []
        user = (f"SEGMENT ID: {segment_id}\n\nMETİN:\n\"\"\"\n{text}\n\"\"\"\n"
                + _figure_block(figures) + "\nYalnızca JSON döndür.")
        raw = self.client.chat(self.role, user)
        items = _parse(raw)
        if self.schema:
            items, self.last_errors = validate_items(items, self.schema)
        else:
            self.last_errors = []
        items = _keep_known_targets(items, figures, self.last_errors)
        items = [it for it in items if attribute_has_source_support(it, text)]
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
    "target": {"type": str, "required": True},
    "target_type": {"type": str, "default": ""},
    "social_group": {"type": str, "default": ""},
    "adjective": {"type": str, "required": True},
    "value": {"type": str, "default": ""},
    "era": {"type": str, "default": ""},
    # Kaynak denetiminde kullanılır; tipli Attribute çıktısına yazılmaz.
    "evidence_quote": {"type": str, "required": True},
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
    "Sen Aşıkpaşazade metnindeki kişi ve topluluklara yönelik tasvirleri çıkaran "
        "bir söylem analistisin.\n"
        "\n"
        "AMAÇ:\n"
        "Metinde bir kişi, padişah veya sosyal gruba atfedilen sıfatları, niteleyici "
        "ifadeleri ve açık değer yargılarını çıkar.\n"
        "\n"
        "KURALLAR:\n"
        "0. HEDEF LİSTESİ BAĞLAYICIDIR. Girdide sana o bölüm için bulunmuş KİŞİLER "
        "listesi (ad + sosyal grup) verilir. target alanı YALNIZCA şu ikisinden biri "
        "olabilir:\n"
        "   (a) listedeki bir kişinin adı — target_type='kişi', social_group o kişinin "
        "listedeki grubu;\n"
        "   (b) listede geçen bir sosyal grubun kendisi (Kâfir, Leşker, Tekfur, Bey...) "
        "— target_type='sosyal_grup', social_group=target.\n"
        "   Listede olmayan bir kişi veya grup için sıfat ÜRETME, kaydı tamamen atla. "
        "Metinde niteleme görsen bile hedef listede yoksa yazma. Adı listedekinin yazım "
        "varyantıysa (Bilecik tekfuru / Bilecük teküri) listedeki biçimi kullan.\n"
        "\n"
        "0a. KİŞİ ÖNCELİĞİ. Bir niteleme metinde ADI GEÇEN belirli bir kişiye "
        "bağlanabiliyorsa hedef O KİŞİdir; aynı nitelemeyi bir de topluluğa yazma. Grup "
        "hedefi yalnız niteleme topluluğun tamamına yöneltilmişse ve belirli bir kişiye "
        "bağlanamıyorsa kullanılır ('komşu kafirlerle iyi geçindi'). Grup hedefi "
        "seçerken metnin adlandırdığı EN BELİRLİ topluluğu al: 'İnegöl kafirleri' "
        "diyebiliyorken çıplak 'kafirler' yazma.\n"
        "   ÇIPLAK UNVAN GRUP HEDEFİ OLAMAZ: 'Gazi', 'Bey', 'Sultan', 'Hatun', "
        "'Paşa', 'Derviş' gibi UNVAN niteliğindeki etiketleri target yapma. Bunlar "
        "bir topluluk değil, kişilerin sıfatıdır ve yeri social_group alanıdır. "
        "'Gaziler gayretliydi' gibi bir cümlede nitelemenin sahibi neredeyse her "
        "zaman metinde adı geçen belirli bir kişidir — onu hedef al; hiçbir ada "
        "bağlanamıyorsa kaydı hiç yazma.\n"
        "   Bir topluluğa sıfat yığmak, çoğu kez "
        "nitelemenin asıl sahibi olan kişiyi kaçırdığının işaretidir: aynı gruba ikinci "
        "bir sıfat yazmadan önce, o nitelemenin metinde adı geçen belirli bir kişiye "
        "ait olup olmadığına bak.\n"
        "\n"
        "0b. KAPSAMA — GENİŞ TARA, AMA DOLDURMA. Bu bölümde eyleyen her kişi için metni "
        "ayrıca tara; yalnız karşı taraftaki topluluğa sıfat yazıp adı geçen yoldaşları "
        "(kılavuz, çavuş, fakıh, hatun, derviş) atlama. Bir kişi için metinde AÇIK bir "
        "niteleme yoksa onu atla — uydurma, doldurma yapma.\n"
        "\n"
        "0c. BAŞ KİŞİYE YIĞMA. Anlatının merkezindeki kişi (çoğunlukla hükümdar) her "
        "paragrafta bir şey yapar; yaptığı her iş bir kişilik niteliği DEĞİLDİR. "
        "Aynı kişiye üst üste sıfat yazdığını fark edersen her biri için ayrı ayrı "
        "duraksa ve şunu sor: elimdeki, metnin o kişiye YAKIŞTIRDIĞI bir vasıf mı, "
        "yoksa onun davranışından benim çıkardığım bir yorum mu? İkincisiyse yazma.\n"
        "   Bir kişiye kaç sıfat yazılabileceğinin sabit bir sayısı yoktur; ölçüt "
        "sayı değil, her kaydın metinde ayrı ve açık bir dayanağı olmasıdır.\n"
        "   AYRI DAYANAK TESTİ: ölçüt sıfatların kaç cümleye dağıldığı DEĞİLDİR. "
        "Metin bir kişiye aynı cümlede birden çok nitelik yakıştırıyorsa hepsini "
        "ayrı ayrı yaz — 'hem cömert hem adil idi' iki kayıttır, 'cimri ve hileci "
        "oldu' iki kayıttır. Ölçüt şudur: her sıfatın metinde KENDİ AÇIK dayanağı "
        "var mı?\n"
        "   - Metin sıfatı kendisi söylüyorsa (aynı cümlede bile olsa) → yaz.\n"
        "   - Tek bir olgudan senin çıkardığın birden çok yorumsa → yalnız en "
        "açığını yaz. 'Hile ile tekfuru yakaladı' cümlesinden 'hileci' çıkar; "
        "'kurnaz', 'cesur', 'zeki' TÜRETME — metin onları söylemiyor.\n"
        "\n"
        "1. Metinde doğrudan bulunan sıfat ve nitelemeleri çıkar. Sıfat biçiminde "
        "olmayan bir davranıştan yalnızca özellik açık ve doğrudan ifade ediliyorsa "
        "kısa bir attribute üretilebilir. Örnek: 'yağmalayıp zarar verirdi' → "
        "'yağmacı', 'adaleti ve güveni sağladılar' → 'adil'. Ancak 'savaştı' → 'cesur', "
        "'plan yaptı' → 'zeki' gibi yorum gerektiren özellikler TÜRETME.\n"
        "\n"
        "2. adjective alanı kısa ve normalize edilmiş bir nitelik olmalıdır. Tam cümle, "
        "uzun açıklama veya olay özeti yazma. Örnek: 'pek çok yiğitlikler gösterdi' → "
        "'yiğit', 'kutluluk belirtileri çoktur' → 'kutlu'.\n"
        "\n"
        "3. KİMLİK ETİKETİ SIFAT DEĞİLDİR. 'gazi', 'sultan', 'bey', 'tekfur', 'kafir', "
        "'müslüman', 'Türk', 'Tatar' gibi din, kavim, makam veya unvan bildiren "
        "sözcükleri adjective alanına YAZMA — bunların yeri social_group alanıdır. "
        "'İnegöl tekfuru → kafir' geçersiz bir kayıttır: kâfirlik onun kimliğidir, "
        "metnin ona yakıştırdığı bir vasıf değil. Ancak sözcük bağlamda açıkça bir "
        "değer yargısı taşıyorsa (bir kişiye hakaret ya da övgü olarak kullanılmışsa) "
        "değerlendirilebilir.\n"
        "   AKRABALIK BAĞI DA SIFAT DEĞİLDİR: 'kayınpeder', 'damat', 'kayın', "
        "'üvey', 'kardeş', 'oğul', 'gelin' gibi sözcükler iki kişi arasındaki "
        "İLİŞKİYİ gösterir, kişinin vasfını değil. 'Şeyh Edebalı → kayınpeder' "
        "geçersizdir; bu bilgi figure agent'ın akrabalık alanına aittir.\n"
        "\n"
        "3b. OLAYIN SONUCU SIFAT DEĞİLDİR. Kişinin BAŞINA GELEN şey onun niteliği "
        "değildir: 'şehit oldu', 'bozguna uğradı', 'parça parça edildi', 'öldürüldü', "
        "'yağmalandı', 'esir düştü' birer olaydır ve timeline agent'ın işidir. "
        "adjective yalnız kişinin NASIL BİRİ olduğunu söyleyen ifadeden gelir. "
        "Ölçüt şudur: sıfat, olay olmadan da o kişi için doğru olabiliyor mu? "
        "'cömert' olaysız da doğrudur, 'bozguna uğramış' değildir.\n"
        "   KALICILIK TESTİ — bu yasak SIFAT BİÇİMİNDEKİ sonuçları da kapsar. "
        "Sözcük sıfat gibi göründü diye geçirme; şunu sor: bu vasıf, bölümdeki olay "
        "bittikten sonra da o kişide DURUR MU?\n"
        "   - Durur  → kalıcı vasıftır, yaz: 'cömert', 'adil', 'yiğit', 'hilekâr'.\n"
        "   - Durmaz → o ana ya da o olaya aittir, YAZMA: 'başarılı' (o seferde "
        "başarılı oldu), 'sarhoş' (o gece sarhoştu), 'bağlılık gösteren' (o anda "
        "bağlılık gösterdi), 'yararlı' (o işte yararı dokundu).\n"
        "   DİKKAT — bu test bir kişinin SÜREKLİ hâlini anlatan nitelemeleri "
        "elemez. Metin birini 'her dem hazır', 'dâim yoldaş', 'hep uyanık' gibi "
        "süreklilik bildiren bir ifadeyle anıyorsa bu kalıcı vasıftır, YAZ.\n"
        "   Kısaca: kişinin O BÖLÜMDE yaptığı şeyin adı sıfat değildir; sıfat "
        "kişinin nasıl biri olduğunu söyler.\n"
        "\n"
        "4. Bir hedef için birden fazla bağımsız attribute varsa her birini ayrı kayıt "
        "oluştur. target niteliğin gerçekten yöneltildiği kişi veya topluluk olmalıdır. "
        "Zamir veya örtük özne yalnızca bağlam açık olduğunda çözülmelidir.\n"
        "\n"
        "5. target_type yalnızca 'kişi' veya 'sosyal_grup'; value yalnızca 'Olumlu', "
        "'Olumsuz' veya 'Nötr' olabilir. social_group değeri 0. kuraldaki listeden "
        "KOPYALANIR; kendin grup adı uydurma, listedeki kişinin grubu boşsa boş bırak. "
        "Bir niteleme belirli bir kişiye değil topluluğun tamamına yöneltiliyorsa "
        "('kafirler komşu idi' gibi) hedefi kişi değil GRUP yap.\n"
        "   value ÖLÇÜTÜ: değeri metnin TUTUMUNA göre ver, sözcüğün sözlük anlamına "
        "göre değil. Anlatıcı o niteliği överek mi, yererek mi, yoksa yalnızca "
        "bildirerek mi anıyor?\n"
        "   - 'Olumlu' : övgü, takdir, meşrulaştırma.\n"
        "   - 'Olumsuz': yergi, suçlama, aşağılama.\n"
        "   - 'Nötr'   : yalnızca tanıtan, sınıflandıran ya da bilgi veren niteleme "
        "(lakap, meslek, akrabalık bağı, fiziksel özellik). Anlatıcı bir değer "
        "yargısı taşımıyorsa Nötr'dür — her nitelemeyi Olumlu ya da Olumsuz'a "
        "zorlama. Nötr, örneklerde az geçse de tam yetkili bir değerdir.\n"
        "\n"
        "6. evidence_quote attribute kararını doğrudan destekleyen en kısa yeterli "
        "metin parçası olmalıdır. era bilgisini yalnızca verilen metin veya bölüm "
        "bağlamından güvenilir biçimde belirle; dışarıdan tarihsel bilgi ekleme.\n"
        "\n"
        "7. Metinde yeterli dayanak yoksa veya target/attribute eşleşmesinden emin "
        "değilsen spekülatif kayıt üretme; kaydı atla.\n"
        "\n"
        "7b. ŞİİR VE NAZIM DA KAPSAMDADIR. Eser, düzyazı anlatının arasına manzum "
        "parçalar serpiştirir (kafiyeli satırlar, beyitler, mersiye ve övgü "
        "şiirleri). Bu parçalardaki nitelemeler de sıfat kaydıdır — anlatıcı "
        "kişiyi orada da niteliyordur. Manzum diye ATLAMA.\n"
        "   Yalnız şu ikisi dışarıdadır: (a) genel hikmet ve atasözü niteliğinde, "
        "belirli bir kişiye yöneltilmemiş mısralar; (b) dua ve temenni kalıpları "
        "('ömrü uzun olsun' gibi) — bunlar dilek bildirir, kişinin vasfını değil.\n"
        "   Aynı kişi hem düzyazıda hem şiirde nitelenmişse İKİSİNİ de yaz; "
        "sıfatlar farklıysa ayrı kayıtlardır.\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"target\":\"...\",\"target_type\":\"kişi|sosyal_grup\",\"social_group\":\"\",\"adjective\":\"...\",\"value\":\"Olumlu|Olumsuz|Nötr\",\"era\":\"\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Uygun attribute yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Abbasi'lerden Süleyman Şah devrine kadar Celi (Arap) 53 54 sülalesi Y afesoğulları sülalesine üstün idi.\"\n"
        "ÇIKTI: {\"items\":[]}\n"
        "(Boş: soy/kavim adları KİŞİLER listesinde bulunmaz (figure agent bunları üretmez), dolayısıyla 0. kural gereği hedef olamazlar.)\n"
        "\n"
        "METİN: \"Acemler de bu göçer evli halktan çekinip tedbir aradılar.\"\n"
        "ÇIKTI: {\"items\":[]}\n"
        "(Boş: 'Acemler' bir kavim adıdır; figure agent çıplak kavim adı üretmez, bu yüzden hedef listesinde yer almaz.)\n"
        "\n"
        "METİN: \"Göçer evin önde gelenlerinden olan Süleyman Şah Gazi'yi ileri çektiler.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Süleyman Şah\",\"target_type\":\"kişi\",\"social_group\":\"Gazi\",\"adjective\":\"ulu (önde gelen)\",\"value\":\"Olumlu\",\"era\":\"Pre-Osman\",\"evidence_quote\":\"Göçer evin önde gelenlerinden olan\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Süleyman Şah Gazi pek çok yiğitlikler gösterdi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Süleyman Şah\",\"target_type\":\"kişi\",\"social_group\":\"Gazi\",\"adjective\":\"yiğit\",\"value\":\"Olumlu\",\"era\":\"Pre-Osman\",\"evidence_quote\":\"pek çok yiğitlikler gösterdi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Karacahisar tekfuru\",\"target_type\":\"kişi\",\"social_group\":\"Tekfur\",\"adjective\":\"itaatkâr\",\"value\":\"Olumlu\",\"era\":\"Pre-Osman\",\"evidence_quote\":\"Sultan'a itaat edip haraç verirlerdi\",\"confidence\":0.9},{\"target\":\"Bilecik tekfuru\",\"target_type\":\"kişi\",\"social_group\":\"Tekfur\",\"adjective\":\"itaatkâr\",\"value\":\"Olumlu\",\"era\":\"Pre-Osman\",\"evidence_quote\":\"Sultan'a itaat edip haraç verirlerdi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"O zaman Karahisar vilayetinde Germiyan babası Alışıra vardı, ona Çavudur derlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Alışıra\",\"target_type\":\"kişi\",\"social_group\":\"Bey\",\"adjective\":\"Çavudur\",\"value\":\"Nötr\",\"era\":\"Pre-Osman\",\"evidence_quote\":\"ona Çavudur derlerdi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Osman Gazi başa geçince komşu kafirlerle çok iyi geçindi, ancak Germiyanoğlu'yla düşmanlığa başladı.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"komşu kafirler\",\"target_type\":\"sosyal_grup\",\"social_group\":\"komşu kafirler\",\"adjective\":\"komşu\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"komşu kafirlerle çok iyi geçindi\",\"confidence\":0.9},{\"target\":\"Germiyanoğlu\",\"target_type\":\"kişi\",\"social_group\":\"Bey\",\"adjective\":\"düşman\",\"value\":\"Olumsuz\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"Germiyanoğlu'yla düşmanlığa başladı\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Muhammed' e inananların önderi Osman Mucizelerin gösterdiği kimsedir artık.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Osman Gazi\",\"target_type\":\"kişi\",\"social_group\":\"Gazi\",\"adjective\":\"önder\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"Muhammed'e inananların önderi Osman\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Aya Nikola adında bir kafir, İnegöl' de Osman yay la ya ve kışlaya gittikleri zamanda, bunların göçünü yağmalayıp zarar verirdi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Aya Nikola\",\"target_type\":\"kişi\",\"social_group\":\"Kâfir\",\"adjective\":\"yağmacı\",\"value\":\"Olumsuz\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"bunların göçünü yağmalayıp zarar verirdi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu ve 'Sizden dileğimiz, biz yaylaya gittiğimiz vakit eşyalarımızı sizde emanet koyalım.' dedi. O da kabul etti.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Bilecik tekfuru\",\"target_type\":\"kişi\",\"social_group\":\"Tekfur\",\"adjective\":\"itimat edilen\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"eşyalarımızı sizde emanet koyalım\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Ancak inegöl kafideri Osman Gazi' den çekinirler, onlar da bu kafiderden sakınırlardı.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"İnegöl kafirleri\",\"target_type\":\"sosyal_grup\",\"social_group\":\"İnegöl kafirleri\",\"adjective\":\"sakıngan (çekinen)\",\"value\":\"Olumsuz\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"Osman Gazi'den çekinirler\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Kendilerinin aralarında bir sevgili şeyhin bulunduğunu gördü. Onun pek çok kerameti görülmüştü ve bütün halk ona candan gönülden bağlı idiler.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Şeyh Edebalı\",\"target_type\":\"kişi\",\"social_group\":\"Şeyh\",\"adjective\":\"aziz (sevgili)\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"bir sevgili şeyhin bulunduğunu gördü\",\"confidence\":0.9},{\"target\":\"Şeyh Edebalı\",\"target_type\":\"kişi\",\"social_group\":\"Şeyh\",\"adjective\":\"keramet sahibi\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"Onun pek çok kerameti görülmüştü\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Ayrıca benim kızım Malhun, senin eşin olacak.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Malhun\",\"target_type\":\"kişi\",\"social_group\":\"Hatun\",\"adjective\":\"helal (eş)\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"benim kızım Malhun, senin eşin olacak\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Yanında şeyhin bir de öğrencisi vardı ve adına Derviş Tururoğlu derlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Derviş Tururoğlu\",\"target_type\":\"kişi\",\"social_group\":\"Derviş\",\"adjective\":\"mürid (öğrenci)\",\"value\":\"Olumlu\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"şeyhin bir de öğrencisi vardı\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Sonra o Kalanoz adlı kafir de vuruldu. Osman Gazi, 'Önce onun karnını yarın, sonra da eşip it gibi gömün.' dedi.\"\n"
        "ÇIKTI: {\"items\":[{\"target\":\"Kalanoz\",\"target_type\":\"kişi\",\"social_group\":\"Kâfir\",\"adjective\":\"it\",\"value\":\"Olumsuz\",\"era\":\"Osman Gazi\",\"evidence_quote\":\"eşip it gibi gömün\",\"confidence\":0.9}]}"
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


_SOCIAL_GROUP_MARKERS = {
    "halk", "halki", "kafirler", "kafirleri", "tatarlar", "turkmenler",
    "gaziler", "araplar", "acemler", "rumlar", "gocer ev", "cemaat",
    "topluluk", "taife", "ordu", "askerler", "soy", "soyu", "sulale",
    "padisahlari", "insanlar", "dervisler", "nesil", "nesli", "neslinden",
    "gelenler",
}


def _target_key(value: str) -> str:
    """Hedefleri Türkçe büyük/küçük harf ve aksan farkından bağımsız karşılaştır."""
    from utilities.source_fidelity import normalize_text
    return normalize_text(value)


def _looks_social_group(target: str) -> bool:
    key = _target_key(target)
    return any(marker in key for marker in _SOCIAL_GROUP_MARKERS)


# ═════════════════════════════════════════════════════════════════════════════
# HEDEF BAĞLAMA — attribute, figure agent'ın AYNI bölümde bulduğu kişilere bağlıdır.
# Aşağıdaki iki yardımcı o listeyi prompta yazar ve dönen kayıtları listeye göre
# süzer; böylece "figure'ün göremediği kişi için sıfat" üretilemez.
# ═════════════════════════════════════════════════════════════════════════════
def _figure_block(figures: list[dict]) -> str:
    """Hedef listesini prompta yazılabilir biçime çevirir."""
    if not figures:
        return ("\nBU BÖLÜMDE BULUNAN KİŞİLER: (liste boş)\n"
                "Liste boş olduğu için HİÇBİR sıfat kaydı üretme: {\"items\":[]}\n")
    satir, gruplar = [], []
    for f in figures:
        ad = str(f.get("name", "")).strip()
        if not ad:
            continue
        grup = str(f.get("social_group", "")).strip()
        diger = [str(a) for a in (f.get("aliases") or []) if str(a).strip()]
        ek = f"  (diğer biçimleri: {', '.join(diger)})" if diger else ""
        satir.append(f"- {ad} [sosyal grup: {grup or '—'}]{ek}")
        if grup and grup not in gruplar:
            gruplar.append(grup)
    return ("\nBU BÖLÜMDE BULUNAN KİŞİLER (target YALNIZ bunlardan veya bunların sosyal "
            "gruplarından seçilebilir):\n" + "\n".join(satir) + "\n"
            + (f"KULLANILABİLİR SOSYAL GRUPLAR: {', '.join(gruplar)}\n" if gruplar else "")
            + "Bu listede olmayan bir kişi/grup için kayıt üretme.\n")


def _keep_known_targets(items: list[dict], figures: list[dict],
                        errors: list[str]) -> list[dict]:
    """Hedefi figure listesinde olmayan kayıtları eler; kalanları listeye göre düzeltir.

    target_type ve social_group modelin sözüne bırakılmaz: hedef bir kişi adına
    eşleşiyorsa 'kişi' + o kişinin grubu, bir sosyal gruba eşleşiyorsa
    'sosyal_grup' + grubun kendisi yazılır. Elenenlerin sebebi audit'e düşer.
    """
    if not figures:
        for it in items:
            errors.append(f"figure listesi boş; '{it.get('target', '')}' sıfat kaydı elendi")
        return []
    ad_grubu: dict[str, tuple[str, str]] = {}
    gruplar: dict[str, str] = {}
    for f in figures:
        gorunen = str(f.get("name", "")).strip()
        grup = str(f.get("social_group", "")).strip()
        adlar = [gorunen] + [str(a) for a in (f.get("aliases") or [])]
        for ad in adlar:
            k = _target_key(ad)
            if k:
                ad_grubu.setdefault(k, (gorunen, grup))
        if grup:
            gruplar.setdefault(_target_key(grup), grup)
    out = []
    for it in items:
        hedef = _target_key(it.get("target", ""))
        if hedef in ad_grubu:
            gorunen, grup = ad_grubu[hedef]
            it["target"], it["target_type"], it["social_group"] = gorunen, "kişi", grup
            out.append(it)
        elif hedef in gruplar:
            it["target"] = it["social_group"] = gruplar[hedef]
            it["target_type"] = "sosyal_grup"
            out.append(it)
        else:
            errors.append(f"'{it.get('target', '')}' figure listesinde yok; sıfat kaydı elendi")
    return out


def _norm_target_type(value: str, target: str) -> str:
    """Hedef ya bir KİŞİ ya da bir SOSYAL GRUPtur; ara sınıf yok.

    Eski 'padişah' / 'ekstra_kişi' ayrımı kaldırıldı: padişahlık bir sosyal grup
    etiketidir, ayrı bir hedef türü değil. Kişinin grubu social_group alanına yazılır.
    """
    raw = _target_key(value).replace(" ", "_")
    if _looks_social_group(target):
        return "sosyal_grup"
    if raw in {"sosyal_grup", "sosyalgrup", "topluluk", "grup"}:
        return "sosyal_grup"
    return "kişi"


def canonicalize_targets(items: list[Attribute]) -> list[Attribute]:
    """Kafirler/kafirler gibi yazım varyantlarını tek görünen hedefte birleştir."""
    preferred: dict[str, str] = {}
    for item in items:
        key = _target_key(item.target)
        if key and key not in preferred:
            preferred[key] = item.target.strip()
    for item in items:
        item.target = preferred.get(_target_key(item.target), item.target)
    return items



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
                it["target"].strip(), _norm_target_type(it.get("target_type", ""), it["target"]),
                str(it.get("social_group", "")).strip(),
                it["adjective"].strip(),
                _norm_value(it.get("value", "")), it.get("era", ""), segment_id,
                float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
