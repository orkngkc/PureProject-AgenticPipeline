"""CONQUEST AGENT — fetihleri çıkarır (padişah – fethettiği yerler).

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Conquest
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.scoring import keyword_score
from utilities.source_fidelity import quote_supported, conquest_has_source_support
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
        items = [it for it in items if conquest_has_source_support(it, text)]
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
    "place": {"type": str, "required": True},
    "era": {"type": str, "default": ""},
    "year": {"type": str, "default": ""},
    "intent": {"type": str, "default": ""},
    "success": {"type": str, "default": ""},
    "ottoman_leader": {"type": str, "default": ""},
    "opponent_leader": {"type": str, "default": ""},
    "method": {"type": str, "default": ""},
    "opponent_muslim": {"type": str, "default": ""},
    "evidence_quote": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {
    "fetih": 3.0, "fethetti": 2.5, "aldı": 1.5, "hediye": 2.0,
    "kılıç": 2.0, "kuşatma": 2.5, "sefer": 2.0, "verdi": 1.5,
    "hücum": 2.5, "vurdu": 2.5, "yağma": 2.5, "ateşe verdi": 2.5,
    "savaş": 2.0, "itaat": 2.5, "boyun eğdi": 2.5, "ganimet": 1.5,
}

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
    "Sen Aşıkpaşazade anlatısındaki fetih, toprak kazanımı/kaybı ve bir yer "
        "üzerindeki askerî-siyasî kontrol değişimlerini çıkaran bir uzmansın.\n"
        "\n"
        "AMAÇ:\n"
        "Metinde açıkça adı verilen bir yer üzerindeki bağımsız ve anlamlı fetih veya "
        "kontrol değişimlerini çıkar. Her askerî hareketi ayrı kayıt yapma.\n"
        "\n"
        "KURALLAR:\n"
        "1. Fetih, zapt, başarılı/başarısız saldırı veya kuşatma, yağma-yakma, "
        "teslim/itaat, yurt-yayla-kışlak tahsisi, savunma ve yer kaybı kapsama girer. "
        "Ancak yalnızca bir yere gitmek, geçmek, konaklamak, asker toplamak, "
        "haberleşmek, casusluk veya hazırlık yapmak tek başına fetih kaydı değildir.\n"
        "\n"
        "2. Aynı hedef yere yönelik hazırlık, yürüyüş, saldırı, kuşatma, muharebe ve "
        "sonuç aynı harekâtın parçalarıysa TEK kayıt üret. Nihai veya ana askerî/toprak "
        "sonucunu temsil et; alt aşamaları ayrı kayıt yapma.\n"
        "\n"
        "3. place yalnızca eylemin doğrudan hedefi olan yer olmalıdır. Bir yer sadece "
        "olayın geçtiği konumsa fetih hedefi sayma. Metinde olmayan daha özel yer adı "
        "ekleme; önceki bağlamdan yalnızca kesin olarak çözülebilen yer referanslarını "
        "kullan.\n"
        "\n"
        "4. intent daima Osmanlı/Türk tarafının hedef yer üzerindeki durumunu gösterir "
        "ve yalnızca 'alma', 'verme', 'savunma' veya 'kayıp' olabilir. Bir yer "
        "Osmanlı/Türk tarafına fetih, teslim, hediye, yurt, yayla veya kışlak yoluyla "
        "geçiyorsa 'alma' yaz. Karşı taraf yeri dilbilgisel olarak 'vermiş' olsa bile "
        "Osmanlı/Türk tarafı alıyorsa intent='alma'dır. 'teslim', 'itaat', 'haraç' gibi "
        "ifadeler intent değil method alanına yazılır.\n"
        "\n"
        "4b. DÖRT AMACIN AYRIMI. Doğru değeri iki soruyla bul: (a) yer olaydan ÖNCE "
        "kimin elindeydi? (b) el değiştirme İSTEYEREK mi oldu, ZORLA mı?\n"
        "   - 'alma'    : yer önce KARŞI taraftaydı, bize geçiyor. (fetih, teslim alma, "
        "yurt/yayla/kışlak olarak verilme, hediye)\n"
        "   - 'verme'   : yer önce BİZDEYDİ, kendi irademizle başkasına geçiyor. "
        "(bağış, tımar/yurt olarak tahsis, anlaşma ya da evlilik yoluyla bırakma)\n"
        "   - 'savunma' : yer BİZDEYDİ, karşı taraf onu almak için saldırıyor; kayıt "
        "saldırının değil SAVUNMANIN kaydıdır.\n"
        "   - 'kayıp'   : yer BİZDEYDİ, ZORLA elimizden çıkıyor. (savaşta kaybetme, "
        "karşı tarafın geri alması)\n"
        "   Ayırt edici nokta: 'verme' ile 'kayıp' aynı yönde hareket eder, ikisini "
        "İRADE ayırır — biri bizim kararımızdır, öteki bize dayatılmıştır. 'savunma' "
        "ile 'kayıp' da karışabilir: yer elimizde kaldıysa 'savunma', çıktıysa 'kayıp'.\n"
        "   Bu bölümde yalnız 'alma' örneği görüyorsan bile diğer üçünü unutma; "
        "aşağıdaki ÖRNEKLER bölümü ağırlıklı olarak 'alma' içerir, bu bir sınırlama "
        "değil örnekleme eksikliğidir. Metin ne diyorsa onu yaz.\n"
        "\n"
        "5. success yalnızca 'başarılı' veya 'başarısız' olabilir. Yer ele geçirilmiş, "
        "teslim olmuş, verilmiş veya savunulmuşsa 'başarılı'; ele geçirme girişimi "
        "açıkça sonuçsuz kalmışsa 'başarısız' yaz. Sonuç güvenilir biçimde "
        "belirlenemiyorsa 'belirsiz' üretme; yeterli kanıt yoksa kaydı atla.\n"
        "\n"
        "6. ottoman_leader Osmanlı/Türk tarafındaki ana lideri, opponent_leader karşı "
        "tarafın açıkça belirtilen liderini gösterir. Yurt/yayla/kışlak tahsisinde yeri "
        "alan lider ottoman_leader, yeri veren hükümdar opponent_leader olmalıdır. Aynı "
        "fetihte birden fazla Osmanlı/Türk lider birlikte rol alıyorsa aynı kayıtta "
        "birleştir. Metinde olmayan lider adı ekleme.\n"
        "\n"
        "7. method sonucu doğuran ana yöntemi kısa biçimde yaz: 'fetih', 'akın', "
        "'kuşatma ve zapt', 'yağma ve yakma', 'teslim', 'itaat ve haraç', 'yurt olarak "
        "verilme', 'yayla olarak verilme', 'tekfur ele geçirmek' gibi. year yalnızca metinde açıkça verilmişse "
        "yaz; era ve opponent_muslim bilgilerini yalnızca metin veya verilen bölüm "
        "bağlamından güvenilir biçimde belirle. Dışarıdan tarihsel bilgi ekleme.\n"
        "\n"
        "7b. TEKFURUN ELE GEÇİRİLMESİ FETİH KAYDIDIR. Metin bir yerin tekfurunun "
        "(ya da beyinin, hâkiminin) yakalandığını, tutulduğunu, esir alındığını veya "
        "öldürüldüğünü söylüyorsa, bu o yer üzerindeki kontrolün el değiştirdiğini "
        "gösterir: O YER İÇİN KAYIT AÇ.\n"
        "   - place  = tekfurun yönettiği YER (kişinin adı değil).\n"
        "   - method = 'tekfur ele geçirmek'.\n"
        "   - opponent_leader = ele geçirilen tekfur.\n"
        "   Yer adını unvandan türet: 'Bilecik tekfurunu tuttu' → place='Bilecik'. "
        "Kişiyi place alanına YAZMA.\n"
        "   Aynı bölümde hem tekfurun ele geçirilmesi hem hisarın alınması "
        "anlatılıyorsa bu TEK olaydır (8. kural): tek kayıt aç ve sonucu doğuran "
        "belirleyici yöntemi yaz — tekfurun ele geçirilmesi hisarın düşmesini "
        "sağladıysa method='tekfur ele geçirmek'.\n"
        "   SINIR: Tekfur kendi yerinden UZAKTA, bir meydan savaşında ele "
        "geçirildiyse ve metin onun yerinin alındığını söylemiyorsa kayıt açma; "
        "bu bir fetih değil, savaş sonucudur.\n"
        "\n"
        "8. BİR YER BİR KEZ FETHEDİLİR. Aynı yer için aynı niyetle (intent) İKİNCİ bir "
        "'başarılı' kayıt açma. Bunun iki tipik kaynağı vardır, ikisinde de kayıt "
        "üretme:\n"
        "   (a) HATIRLATMA/ÖZET: Bölüm, fethin kendisini anlatmıyor; daha önce olmuş "
        "bir fethi anıyor. Belirtiler: 'aldığı hisarlar', 'fethettiği vilayetler', 'bu "
        "fethin tarihi ...', 'o vilayet ki almıştı' gibi ifadeler ya da bir yer listesi "
        "içinde geçen ad.\n"
        "   (b) AYNI OLAYIN PARÇALARI: Kuşatma, baskın, yağma ve nihai zapt aynı fethin "
        "aşamalarıdır (2. kural). Hepsini TEK kayıtta topla; en belirleyici yöntemi "
        "method alanına yaz.\n"
        "   İSTİSNALAR — bunlar ayrı kayıttır, ikisini de yaz:\n"
        "   - Önce BAŞARISIZ bir girişim, sonra BAŞARILI fetih (örn. İnegöl: önce gece "
        "akını sonuçsuz, sonra kuşatmayla alınıyor). Başarısızlıkları asla eleme.\n"
        "   - Yer elden çıkıp (intent='kayıp') yeniden alınıyorsa.\n"
        "   - Niyet farklıysa (bir bölümde 'alma', başkasında 'verme'/'savunma').\n"
        "\n"
        "9. success ile intent çelişmemelidir. intent='kayıp' ise yer Osmanlı/Türk "
        "tarafının elinden çıkmıştır; bu kaydın success değeri 'başarısız'tır. "
        "intent='savunma' için success, savunmanın tutup tutmadığını gösterir.\n"
        "\n"
        "Metinde yeterli dayanak yoksa spekülatif kayıt üretme.\n"
        "\n"
        "10. evidence_quote kaydı doğrudan destekleyen, kaynak metinde BİREBİR geçen en "
        "kısa parçadır. Kendi cümleni kurma, özetleme, sadeleştirme veya imla düzeltme "
        "yapma — metindeki harfleri aynen kopyala (OCR bozuklukları dahil). En az üç "
        "sözcük olsun. Metinde birebir bulamıyorsan o kaydı hiç üretme.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"place\":\"...\",\"era\":\"\",\"year\":\"\",\"intent\":\"alma|verme|savunma|kayıp\",\"success\":\"başarılı|başarısız\",\"ottoman_leader\":\"\",\"opponent_leader\":\"\",\"method\":\"\",\"opponent_muslim\":\"\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Uygun kayıt yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Acemler de bu göçer evli halktan çekinip tedbir aradılar. Göçer evin önde gelenlerinden olan Süleyman Şah Gazi'yi ileri çektiler. Elli bin kadar Türkmen ve Tatar evini emrine vererek, 'Haydi Anadolu'ya gidip Allah yolunda çarpışarak gaza edin.' dediler. Süleyman Şah bunu kabul edip önce Erzurum' a, sonra da Erzincan' a geldi. Erzincan' dan Rum vilayetine girdiler. Bir nice yıl ilerlediler ve etrafiarını fethettiler. \"\n"
        "ÇIKTI: {\"items\":[{\"place\":\"Rum vilayeti\",\"era\":\"Süleyman Şah\",\"year\":\"\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Süleyman Şah Gazi\",\"opponent_leader\":\"\",\"method\":\"gaza\",\"opponent_muslim\":\"hayır\",\"confidence\":0.9,\"evidence_quote\":\"Süleyman Şah bunu kabul edip önce Erzurum' a\"}]}\n"
        "\n"
        "METİN: \"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"\n"
        "ÇIKTI: {\"items\":[{\"place\":\"Söğüt vilayeti\",\"era\":\"Ertuğrul Gazi\",\"year\":\"\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Ertuğrul Gazi\",\"opponent_leader\":\"Sultan Alaeddin\",\"method\":\"hediye (kışlak/yurt olarak verilmesi)\",\"opponent_muslim\":\"evet\",\"confidence\":0.9,\"evidence_quote\":\"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler;\"},{\"place\":\"Domaniç Dağı\",\"era\":\"Ertuğrul Gazi\",\"year\":\"\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Ertuğrul Gazi\",\"opponent_leader\":\"Sultan Alaeddin\",\"method\":\"hediye (yayla olarak verilmesi)\",\"opponent_muslim\":\"evet\",\"confidence\":0.9,\"evidence_quote\":\"ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"},{\"place\":\"İrıneni Bili dağı\",\"era\":\"Ertuğrul Gazi\",\"year\":\"\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Ertuğrul Gazi\",\"opponent_leader\":\"Sultan Alaeddin\",\"method\":\"hediye (yayla olarak verilmesi)\",\"opponent_muslim\":\"evet\",\"confidence\":0.9,\"evidence_quote\":\"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler;\"}]}\n"
        "\n"
        "METİN: \"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış. Ancak kafirlerin casusu var imiş, onlara bildirmiş. Bunun üzerine kafider pusu kurdular. Osman Gazi'nin Artun adında bir adamı bu durumu gelip 'Bel tükendiği yerde pusu kurdular.' diye haber verdi. Gaziler de Tanrı'ya sığınarak doğru pusuya yürüdüler. Piyade olan gazilede kafider arasında savaş koptu. Kafider çok fazla idi, büyük muharebe oldu.\"\n"
        "ÇIKTI: {\"items\":[{\"place\":\"İnegöl\",\"era\":\"Osman Gazi\",\"year\":\"\",\"intent\":\"alma\",\"success\":\"başarısız\",\"ottoman_leader\":\"Osman Gazi\",\"opponent_leader\":\"\",\"method\":\"kılıç yoluyla (gece akını)\",\"opponent_muslim\":\"hayır\",\"confidence\":0.9,\"evidence_quote\":\"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi bu tabiri işitince himmet kılıcını gönül beline sıkıca bağladı. Bir gece yürüyüp İnegöl'e vardı. Yanında Kulaca denen küçük bir hisar vardı. Onu yağmaladı ve ateşe verdi, kafiderini o gece kırdı.\"\n"
        "ÇIKTI: {\"items\":[{\"place\":\"Kulaca\",\"era\":\"Osman Gazi\",\"year\":\"685 / M. 1286\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Osman Gazi\",\"opponent_leader\":\"Karacahisar tekfuru\",\"method\":\"kılıç yoluyla (yağma ve ateşe verme)\",\"opponent_muslim\":\"hayır\",\"confidence\":0.9,\"evidence_quote\":\"Yanında Kulaca denen küçük bir hisar vardı.\"}]}\n"
        "\n"
        "METİN: \"Onun Kalanoz adında bir de kardeşi vardı. Emrine büyük bir ordu verdi, bunlar inegöl kafideriyle bir araya geldiler. Osman Gazi de gazilerini topladı ve İkizce'ye geldi. Tornalıç Beli'ni aştıkları yerde savaştılar. Oldukça büyük savaş yapıldı. Osman Gazi'nin kardeşi Saru Yatı'yı orada şehit ettiler. O yerde bir çam ağacı var, şimdi ona Kandilli Çam derler. Vakit vakit onda bir ışık görürler. Sonra o Kalanoz adlı kafir de vuruldu.\"\n"
        "ÇIKTI: {\"items\":[{\"place\":\"İkizce\",\"era\":\"Osman Gazi\",\"year\":\"685 / M. 1286\",\"intent\":\"alma\",\"success\":\"başarılı\",\"ottoman_leader\":\"Osman Gazi\",\"opponent_leader\":\"Karacahisar tekfuru\",\"method\":\"kılıç yoluyla (meydan savaşı)\",\"opponent_muslim\":\"hayır\",\"confidence\":0.9,\"evidence_quote\":\"Onun Kalanoz adında bir de kardeşi vardı.\"}]}\n"
        "\n"
        ""
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
