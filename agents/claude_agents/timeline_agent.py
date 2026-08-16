"""TIMELINE (OLAY+KRONOLOJİ) AGENT — askerî / diplomatik olayları çıkarır.

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Event
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

    def run(self, text: str, segment_id: str = "") -> list[dict]:
        """LLM'e sor -> JSON ayrıştır -> DENETLE -> ağırlık skoru ekle."""
        user = f"SEGMENT ID: {segment_id}\n\nMETİN:\n\"\"\"\n{text}\n\"\"\"\n\nYalnızca JSON döndür."
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
    "summary": {"type": str, "required": True},
    # Olayın 2-6 kelimelik kısa adı. Uzun anlatı özeti karşılaştırma verisiyle
    # eşleştirilemiyordu; kısa etiket olayı adlandırır, summary anlatır.
    "action_label": {"type": str, "default": ""},
    "event_type": {"type": str, "default": ""},
    "actors": {"type": list, "default": []},
    "places": {"type": list, "default": []},
    "date_hint": {"type": str, "default": ""},
    "evidence_quote": {"type": str, "default": ""},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {  # olay sinyali veren kelimeler
    "fetih": 3.0, "fethetti": 2.5, "savaş": 2.5, "muharebe": 2.5, "cenk": 2.5,
    "kuşatma": 2.5, "kuşattı": 2.0, "antlaşma": 2.5, "sefer": 2.0, "akın": 2.0,
    "baskın": 2.0, "isyan": 2.0, "gaza": 2.0, "elçi": 2.0, "barış": 2.0,
}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="timeline",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,
    role=(
    "Sen Aşıkpaşazade anlatısındaki tarihsel faaliyetleri çıkaran ve metindeki "
        "kronolojik akışı koruyan bir tarih analistisin.\n"
        "\n"
        "AMAÇ:\n"
        "Metindeki her fiili değil, anlatının bağımsız ve tarihsel açıdan anlamlı ANA "
        "faaliyetlerini çıkar. Olayları metinde geçtikleri sırayla ver.\n"
        "\n"
        "KURALLAR:\n"
        "1. GRANÜLERLİK — BİR BÂB, BİRKAÇ KİLOMETRE TAŞI. Bir bölümde genellikle 1-3 "
        "kayda değer olay vardır. Hazırlık, yolculuk, haberleşme, görüşme, çatışma ve "
        "sonuç aynı olayın aşamalarıysa TEK kayıt yaz ve en belirleyici sonucu özetle: "
        "asker toplama, kaleye yürüme, kuşatma ve kaleyi alma aynı süreçse yalnız fetih "
        "olayı çıkar. Cümle cümle olay üretme — bir cümlede olay görmen o cümlenin ayrı "
        "bir kilometre taşı olduğu anlamına gelmez.\n"
        "\n"
        "2. Ayrı kayıt yalnız şunlar için açılır: sonucu birbirinden bağımsız olaylar "
        "(bir muharebe ile o muharebede önemli bir kişinin öldürülmesi anlatıda ayrı "
        "sonuçlar olarak vurgulanıyorsa iki kayıt), farklı taraflar arasında geçen "
        "olaylar, ve anlatının açıkça ayırdığı zaman sıçramaları. Birbiriyle ilgisiz "
        "sonuçları tek summary içinde birleştirme.\n"
        "\n"
        "3. Tek başına olay DEĞİLDİR: bir yere gitmek/gelmek, hazırlık yapmak, haber "
        "alıp göndermek, bir sözü aktarmak, geçmiş bir olayı yalnızca hatırlatan arka "
        "plan cümleleri ve salt tarih notları. Ancak göç, yerleşme, istişare veya "
        "emanet ilişkisi anlatıda başlı başına bir faaliyet olarak sunuluyorsa kayıt "
        "olur.\n"
        "\n"
        "4. event_type yalnızca şu değerlerden biri olmalıdır: 'Askeri', 'Diplomatik', "
        "'İdari', 'Dini', 'Ekonomik'. Fetih, savaş, hücum, pusu ve askerî kayıplar → "
        "'Askeri'; görüşme, istişare, ittifak, dostluk ve taraflar arası hediye/emanet "
        "ilişkileri → 'Diplomatik'; yerleşme, göç, tayin ve yönetim kararları → "
        "'İdari'; dinî uygulamalar → 'Dini'; pazar ve doğrudan ekonomik faaliyetler → "
        "'Ekonomik'. 'ölüm', 'fetih', 'savaş' gibi olay adlarını event_type olarak "
        "kullanma.\n"
        "\n"
        "5. summary kısa ve metne sadık olmalıdır; yorum, sebep-sonuç tahmini veya "
        "metinde olmayan tarihsel bilgi ekleme. Şiirsel/mecazî ifadeleri gerçek olay "
        "sayma. Ana faaliyet için gerekli olmayan ayrıntıları summary'ye doldurma.\n"
        "\n"
        "5a. BİR KAYIT = BİR EYLEM. summary tek bir eylemi anlatmalıdır. İçinde 've', "
        "';' ya da nokta ile bağlanmış ikinci bir eylem varsa o AYRI bir kayıttır. "
        "'Alâeddin, Aktemur'a ihsanlarda bulundu VE Osman'a sancak takımı, atlar, "
        "silahlar gönderdi' → bu iki olaydır, ikisini ayrı yaz. Zincirleme özet, "
        "kaydı hem eşleştirilemez hem sayılamaz hâle getirir.\n"
        "\n"
        "5b. action_label — OLAYIN KISA ADI. Her kayıt için, olayı adlandıran 2-6 "
        "kelimelik bir etiket yaz. Bu bir cümle değil, bir BAŞLIKTIR: fiilimsi ya da "
        "isim öbeği kullan, özneyi ve ayrıntıyı dışarıda bırak.\n"
        "   Biçim kalıpları (somut örnek değil, KALIP): «<yer>'in fethi», «<kişi> ile "
        "görüşme», «<nesne> göndermek», «<yapı> yaptırmak», «<yer>'e göç». Etiketi bu "
        "metinden kendin türet; hazır bir olay adı listesi verilmiyor.\n"
        "   summary ise aynı olayın bir cümlelik açıklamasıdır; ikisi birbirini "
        "tekrarlamaz, etiket adlandırır, summary anlatır.\n"
        "\n"
        "5c. SEÇİCİ OL. Metindeki her cümle bir olay değildir. Anlatının ilerlemesini "
        "sağlayan dönüm noktalarını al: fetih, ölüm, tayin, ittifak, anlaşma, göç, "
        "baskın, kurum kurma. Hazırlık, yol alma, konuşma ve betimleme cümlelerini "
        "ayrı olay yapma; ait oldukları ana olayın içinde bırak.\n"
        "\n"
        "6. actors yalnızca olayda doğrudan rol alan kişi ve grupları, places yalnızca "
        "olayla doğrudan ilişkili yerleri içerir. Metinde sadece geçerken anılanları "
        "ekleme; metinde bulunmayan ad üretme.\n"
        "\n"
        "7. date_hint yalnızca metinde açıkça desteklenen tarih/dönem bilgisidir. "
        "Bölümler zaten kronolojik işlendiği için olayları yeniden sıralama. Yeterli "
        "dayanak yoksa spekülatif kayıt üretme.\n"
        "\n"
        "8. evidence_quote kaydı doğrudan destekleyen, kaynak metinde BİREBİR geçen en "
        "kısa parçadır. Kendi cümleni kurma, özetleme, sadeleştirme veya imla düzeltme "
        "yapma — metindeki harfleri aynen kopyala (OCR bozuklukları dahil). En az üç "
        "sözcük olsun. Metinde birebir bulamıyorsan o kaydı hiç üretme.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"action_label\":\"...\",\"summary\":\"...\",\"event_type\":\"Askeri|Diplomatik|İdari|Dini|Ekonomik\",\"actors\":[],\"places\":[],\"date_hint\":\"\",\"evidence_quote\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Olay yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Süleyman Şah atını suya sürdü. Önü yar imiş, at sürçünce Süleyman Şah suya düştü. Eceli gelip Allah'ın rahmetine kavuştu.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Süleyman Şah'ın vefatı\",\"summary\":\"Süleyman Şah, Fırat Irmağı'nı atıyla geçmeye çalışırken suya düşüp vefat etti.\",\"event_type\":\"İdari\",\"actors\":[\"Süleyman Şah Gazi\"],\"places\":[\"Fırat Irmağı\"],\"date_hint\":\"Süleyman Şah dönemi\",\"confidence\":0.95,\"evidence_quote\":\"Süleyman Şah atını suya sürdü.\"}]}\n"
        "\n"
        "METİN: \"Bazısı Süleyman Şah'ın üç oğluna uydular. Bunların biri Sunkur Tigin, biri Gündoğdu ve biri de Ertuğrul Gazi' dir. Bu üç kardeş geldikleri yola dönüp Fırat'ın başından yürüyüp Pasin Ovası ve Sürmelü Çukuru'na gıi.ı:üer.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Fırat'tan göç\",\"summary\":\"Süleyman Şah'ın halkı ayrıldı; oğulları Sunkur Tigin, Gündoğdu ve Ertuğrul Gazi Fırat'ın başından yürüyüp Pasin Ovası ile Sürmelü Çukuru'na göçtü.\",\"event_type\":\"İdari\",\"actors\":[\"Sunkur Tigin\",\"Gündoğdu\",\"Ertuğrul Gazi\"],\"places\":[\"Fırat Irmağı\",\"Pasin Ovası\",\"Sürmelü Çukuru\"],\"date_hint\":\"Süleyman Şah'ın vefatından sonra\",\"confidence\":0.9,\"evidence_quote\":\"Bazısı Süleyman Şah'ın üç oğluna uydular.\"}]}\n"
        "\n"
        "METİN: \"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler. Saru Yatı babasına varınca bu haberi söyledi. Ertuğrul kabul edip yürüdü ve Ankara'ya geldiler. Yurtlarında yerleştiler.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Yurt gösterilen yere konmak\",\"summary\":\"Sultan Alaeddin'in yurt gösterdiği Söğüt vilayetine Ertuğrul Gazi'nin obası yerleşti.\",\"event_type\":\"İdari\",\"actors\":[\"Ertuğrul Gazi\",\"Sultan Alaeddin\"],\"places\":[\"Söğüt vilayeti\",\"Domaniç Dağı\",\"İrıneni Bili dağı\"],\"date_hint\":\"Ertuğrul Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler. Saru Yatı babasına varınca bu haberi söyledi. Ertuğrul kabul edip yürüdü ve Ankara'ya geldiler. Yurtlarında yerleştiler.\"}]}\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu ve 'Sizden dileğimiz, biz yaylaya gittiğimiz vakit eşyalarımızı sizde emanet koyalım.' dedi. O da kabul etti.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Emanet anlaşması\",\"summary\":\"Osman Gazi, Aya Nikola'nın yağmalarını Bilecik tekfuruna şikâyet etti ve eşyalarını kalede emanet bırakma izni aldı.\",\"event_type\":\"Diplomatik\",\"actors\":[\"Osman Gazi\",\"Bilecik tekfuru\"],\"places\":[\"Bilecik\"],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu ve 'Sizden dileğimiz, biz yaylaya gittiğimiz vakit eşyalarımızı sizde emanet koyalım.' dedi. O da kabul etti.\"}]}\n"
        "\n"
        "METİN: \"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"İnegöl'e akın\",\"summary\":\"Osman Gazi yetmiş kişiyle İrıneni Beli'nden gelip İnegöl'ü vurmak üzere harekete geçti.\",\"event_type\":\"Askeri\",\"actors\":[\"Osman Gazi\"],\"places\":[\"İrıneni Beli\",\"İnegöl\"],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"Bir gün Osman Gazi İrıneni Beli'nden yetmiş kişiyle gelip İnegöl'ü vuracakmış.\"}]}\n"
        "\n"
        "METİN: \"Ancak kafirlerin casusu var imiş, onlara bildirmiş. Bunun üzerine kafider pusu kurdular. Osman Gazi'nin Artun adında bir adamı bu durumu gelip 'Bel tükendiği yerde pusu kurdular.' diye haber verdi. Gaziler de Tanrı'ya sığınarak doğru pusuya yürüdüler. Piyade olan gazilede kafider arasında savaş koptu.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Casus haberiyle pusu kurmak\",\"summary\":\"İnegöl kâfirleri casus haberi üzerine gazileri pusuya düşürdü; Artun bunu Osman Gazi'ye bildirdi.\",\"event_type\":\"Askeri\",\"actors\":[\"İnegöl kâfirleri\",\"Artun\",\"Osman Gazi\"],\"places\":[\"İrıneni Beli\"],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"Ancak kafirlerin casusu var imiş\"},{\"action_label\":\"Pusu savaşı\",\"summary\":\"Gaziler pusunun üzerine yürüdü ve kâfirlerle savaş başladı.\",\"event_type\":\"Askeri\",\"actors\":[\"Osman Gazi\",\"gaziler\",\"İnegöl kâfirleri\"],\"places\":[\"İrıneni Beli\"],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"Gaziler de Tanrı'ya sığınarak doğru pusuya yürüdüler.\"}]}\n"
        "\n"
        "METİN: \"Şeyh, 'Oğul Osman Gazi, sana müjdeler olsun, yüce Tanrı sana ve nesline padişahlık verdi, kutlu olsun. Ayrıca benim kızım Malhun, senin eşin olacak.' dedi ve o anda nikahlayıp kızını Osman Gazi'ye verdi.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Rüya tabiri\",\"summary\":\"Şeyh Edebalı, Osman Gazi'nin rüyasını yorumlayıp ona ve nesline padişahlık müjdesi verdi.\",\"event_type\":\"Dini\",\"actors\":[\"Şeyh Edebalı\",\"Osman Gazi\"],\"places\":[],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"Şeyh, 'Oğul Osman Gazi, sana müjdeler olsun, yüce Tanrı sana ve nesline padişahlık verdi, kutlu olsun. Ayrıca benim kızım Malhun, senin eşin olacak.' dedi ve o anda nikahlayıp kızını Osman Gazi'ye verdi.\"},{\"action_label\":\"Malhun ile nikâh\",\"summary\":\"Şeyh Edebalı, kızı Malhun'u Osman Gazi ile nikâhladı.\",\"event_type\":\"Diplomatik\",\"actors\":[\"Şeyh Edebalı\",\"Malhun Hatun\",\"Osman Gazi\"],\"places\":[],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.95,\"evidence_quote\":\"Şeyh, 'Oğul Osman Gazi, sana müjdeler olsun, yüce Tanrı sana ve nesline padişahlık verdi, kutlu olsun. Ayrıca benim kızım Malhun, senin eşin olacak.' dedi ve o anda nikahlayıp kızını Osman Gazi'ye verdi.\"}]}\n"
        "\n"
        "METİN: \"O derviş, 'Ey Osman Gazi! Yüce Allah sana padişahlık verdi, bize de şükrane gerek.' dedi. Bunun üzerine Osman Gazi, 'Her ne zaman padişah olursam sana bir şehir vereyim.' deyince, Derviş, 'Şehirden vazgeçtik, bize şu köyceğiz yeter.' cevabını verdi. Osman Gazi bu isteği kabul etti.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Köy bağışlamak\",\"summary\":\"Osman Gazi, Derviş Tururoğlu'na şükrane olarak bir köy verdi.\",\"event_type\":\"İdari\",\"actors\":[\"Osman Gazi\",\"Derviş Tururoğlu\"],\"places\":[],\"date_hint\":\"Osman Gazi dönemi\",\"confidence\":0.9,\"evidence_quote\":\"'Ey Osman Gazi! Yüce Allah sana padişahlık verdi\"}]}\n"
        "\n"
        "METİN: \"Bir gece yürüyüp İnegöl'e vardı. Yanında Kulaca denen küçük bir hisar vardı. Onu yağmaladı ve ateşe verdi, kafiderini o gece kırdı.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"Hisara gece baskını\",\"summary\":\"Osman Gazi gece yürüyüp Kulaca hisarını yağmaladı ve ateşe verdi.\",\"event_type\":\"Askeri\",\"actors\":[\"Osman Gazi\"],\"places\":[\"Kulaca\",\"İnegöl\"],\"date_hint\":\"685 / M. 1286\",\"confidence\":0.9,\"evidence_quote\":\"Bir gece yürüyüp İnegöl'e vardı. Yanında Kulaca denen küçük bir hisar vardı. Onu yağmaladı ve ateşe verdi, kafiderini o gece kırdı.\"}]}\n"
        "\n"
        "METİN: \"Emrine büyük bir ordu verdi, bunlar inegöl kafideriyle bir araya geldiler. Osman Gazi de gazilerini topladı ve İkizce'ye geldi. Tornalıç Beli'ni aştıkları yerde savaştılar. Oldukça büyük savaş yapıldı. Osman Gazi'nin kardeşi Saru Yatı'yı orada şehit ettiler.\"\n"
        "ÇIKTI: {\"items\":[{\"action_label\":\"İkizce savaşı\",\"summary\":\"Osman Gazi'nin gazileri, Kalanoz komutasındaki ordu ve İnegöl kâfirleriyle İkizce yakınında büyük bir savaş yaptı.\",\"event_type\":\"Askeri\",\"actors\":[\"Osman Gazi\",\"Kalanoz\",\"İnegöl kâfirleri\"],\"places\":[\"İkizce\",\"Tornalıç Beli\"],\"date_hint\":\"685 / M. 1286\",\"confidence\":0.9,\"evidence_quote\":\"Osman Gazi de gazilerini topladı ve İkizce'ye geldi.\"},{\"action_label\":\"Saru Yatı'nın şehit düşmesi\",\"summary\":\"Osman Gazi'nin kardeşi Saru Yatı bu savaşta şehit edildi.\",\"event_type\":\"Askeri\",\"actors\":[\"Saru Yatı\"],\"places\":[\"İkizce\"],\"date_hint\":\"685 / M. 1286\",\"confidence\":0.95,\"evidence_quote\":\"Osman Gazi de gazilerini topladı ve İkizce'ye geldi.\"}]}"
),)




# ═════════════════════════════════════════════════════════════════════════════
# EŞLEYİCİ (to_models)
# -----------------------------------------------------------------------------
# to_models, LLM'in verdiği HAM SÖZLÜĞÜ (dict) bu uzmanın TİPLİ MODELİNE çevirir.
# Neden: LLM dict verir; ama orchestrator, kaydetme ve grafik hep tipli
# nesnelerle çalışır. Ayrıca eksik alanları güvenle doldurur (it.get) ve
# boş/geçersiz kayıtları atar.
# ═════════════════════════════════════════════════════════════════════════════
def to_models(items, segment_id) -> list[Event]:
    out = []
    for it in items:
        summary = str(it.get("summary", "")).strip()
        if summary:
            out.append(Event(
                summary=summary,
                action_label=str(it.get("action_label", "")).strip(),
                event_type=it.get("event_type", ""), actors=it.get("actors", []),
                places=it.get("places", []), date_hint=it.get("date_hint", ""),
                segment_id=segment_id,
                confidence=float(it.get("confidence", 0.7)),
                weight_score=float(it.get("weight_score", 0.0))))
    return out
