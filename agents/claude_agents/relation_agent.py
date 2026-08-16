"""RELATION AGENT — bir eşyayı kim verdi / kim aldı (alan-veren ağı).

Bu dosya KENDİ BAŞINA çalışır: agent motorunu (Agent sınıfı), bu uzmanın
tanımını (şema + ağırlık + rol) ve eşleyicisini (to_models) birlikte içerir.
"""
from __future__ import annotations


import config
from models import Relation
from utilities.json_utils import _item_text, _parse
from utilities.llm import LLMClient
from utilities.scoring import keyword_score
from utilities.source_fidelity import is_material_relation
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
# ═════════════════════════════════════════════════════════════════════════════
# NESNE BAĞLAMA — relation, object agent'ın AYNI bölümde bulduğu nesnelere bağlıdır
# (object–relation çifti, figure–attribute çiftinin eşleniğidir).
# ═════════════════════════════════════════════════════════════════════════════
def _obj_key(x: str) -> str:
    import re
    import unicodedata
    x = str(x or "").lower().replace("ı", "i").replace("â", "a")
    x = "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", x).strip()


def _object_block(objects: list[dict]) -> str:
    """Nesne listesini prompta yazılabilir biçime çevirir."""
    if not objects:
        return ("\nBU BÖLÜMDE BULUNAN NESNELER: (liste boş)\n"
                "Liste boş olduğu için HİÇBİR kayıt üretme: {\"items\":[]}\n")
    satir = [f"- {str(o.get('object_name', '')).strip()} "
             f"[tür: {str(o.get('quality', '')).strip() or '—'}; "
             f"eylem: {str(o.get('action', '')).strip() or '—'}]"
             for o in objects if str(o.get("object_name", "")).strip()]
    return ("\nBU BÖLÜMDE BULUNAN NESNELER (object_name YALNIZ bunlardan seçilebilir):\n"
            + "\n".join(satir) + "\n"
            "Listede olmayan bir eşya için kayıt üretme. Listedeki nesnelerin HER BİRİ için "
            "önce 'sahibi kim' sorusunu sor, sonra el değiştirip değiştirmediğine bak — "
            "sahibi de belirlenemeyen nesneyi atla. Sahip veya alan zamirle ya da bir önceki "
            "cümlede anılıyorsa ('ona', 'bunlara') bağlam açıksa çözüp yaz; yalnız gerçekten "
            "belirsizse kaydı bırak.\n")


def _figure_hint(figures: list[dict]) -> str:
    """Veren/alan adlarını figure çıktısıyla aynı yazımda tutmak için ipucu."""
    adlar = [str(f.get("name", "")).strip() for f in figures
             if str(f.get("name", "")).strip()]
    if not adlar:
        return ""
    return ("\nBU BÖLÜMDE BULUNAN KİŞİLER (giver/receiver yazımını bunlarla aynı tut): "
            + ", ".join(adlar) + "\n")


def _keep_known_objects(items: list[dict], objects: list[dict],
                        errors: list[str]) -> list[dict]:
    """object_name'i nesne listesine bağlar; listede olmayan kaydı eler."""
    if not objects:
        for it in items:
            errors.append(f"nesne listesi boş; '{it.get('object_name', '')}' ilişkisi elendi")
        return []
    bilinen = {_obj_key(o.get("object_name", "")): str(o.get("object_name", "")).strip()
               for o in objects if str(o.get("object_name", "")).strip()}
    out = []
    for it in items:
        k = _obj_key(it.get("object_name", ""))
        if k in bilinen:
            it["object_name"] = bilinen[k]          # yazımı listeyle birebir yap
            out.append(it)
        else:
            errors.append(
                f"'{it.get('object_name', '')}' nesne listesinde yok; ilişki kaydı elendi")
    return out


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

    def run(self, text: str, segment_id: str = "", objects: list[dict] | None = None,
            figures: list[dict] | None = None) -> list[dict]:
        """LLM'e sor -> JSON ayrıştır -> DENETLE -> NESNEYE BAĞLA -> skor ekle.

        objects = object agent'ın AYNI bölümde bulduğu nesneler. İlişkinin eşyası yalnız
        bu listeden seçilebilir. figures verilirse veren/alan yazımı da o listeye uydurulur.
        """
        objects = objects or []
        user = (f"SEGMENT ID: {segment_id}\n\nMETİN:\n\"\"\"\n{text}\n\"\"\"\n"
                + _object_block(objects) + _figure_hint(figures or [])
                + "\nYalnızca JSON döndür.")
        raw = self.client.chat(self.role, user)
        items = _parse(raw)
        if self.schema:
            items, self.last_errors = validate_items(items, self.schema)
        else:
            self.last_errors = []
        items = [it for it in items if is_material_relation(it, text)]
        items = _keep_known_objects(items, objects, self.last_errors)
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
SCHEMA = {   # eşyayı kim verdi/aldı + tür
    "object_name": {"type": str, "required": True},
    "giver": {"type": str, "default": ""},
    "receiver": {"type": str, "default": ""},
    "object_type": {"type": str, "default": ""},
    "era": {"type": str, "default": ""},
    # Bu iki alan doğrulama içindir; tipli Relation çıktısına yazılmaz.
    "transfer_verb": {"type": str, "required": True},
    "evidence_quote": {"type": str, "required": True},
    "confidence": {"type": float, "min": 0.0, "max": 1.0, "default": 0.7},
}

WEIGHTS = {"verdi": 2.5, "hediye": 2.5, "bağışladı": 2.0, "sundu": 2.0,
           "gönderdi": 2.0, "armağan": 2.0, "aldı": 2.0, "verildi": 2.0,
           "getirdi": 2.0, "haraç": 2.0, "ihsan": 2.0, "ısmarladı": 2.0}

# ── MODEL: bu agent hangi modeli kullansın? ──
# Boş bırakırsan config.LLM_MODEL (herkesin .env'indeki genel model) kullanılır.
# İstersen bu agent'a ÖZEL bir model yaz, ör: MODEL = "gpt-4o"
MODEL = "llama-3.3-70b-versatile"

agent = Agent(
    name="relation",
    model=MODEL,
    schema=SCHEMA,
    weights=WEIGHTS,

    role=(
    "Sen Aşıkpaşazade metnindeki maddi nesnelerin SAHİPLİK ve AKTARIM ilişkilerini "
        "çıkaran bir ilişki uzmanısın.\n"
        "\n"
        "AMAÇ:\n"
        "Her maddi nesne için 'bu nesnenin sahibi kim' sorusunu yanıtla; nesne el "
        "değiştiriyorsa ayrıca 'kime geçti' sorusunu da yanıtla. Sahiplik temel "
        "ilişkidir; aktarım (hediye/ihsan, emanet, ganimet) onun bir alt türüdür.\n"
        "\n"
        "İKİ KAYIT TÜRÜ:\n"
        "- SAHİPLİK — metin nesnenin sahibini bildiriyor ama el değiştirme anlatmıyor. "
        "giver = sahip, receiver = BOŞ BIRAK, transfer_verb = 'Sahiplik'. Örn. 'Osman "
        "Gazi bütün eşyalarını öküzlere yükletirdi' -> öküzün sahibi Osman Gazi'dir, "
        "kimseye geçmiyor.\n"
        "- AKTARIM — nesne bir taraftan diğerine geçiyor. giver = veren, receiver = "
        "alan, transfer_verb = aktarımın türü.\n"
        "\n"
        "Bir nesne hem sahiplenilip hem aktarılıyorsa YALNIZ aktarım kaydını yaz; aynı "
        "nesne için ikisini birden üretme.\n"
        "\n"
        "KURALLAR:\n"
        "0. NESNE LİSTESİ BAĞLAYICIDIR. Girdide o bölüm için bulunmuş NESNELER listesi "
        "verilir. object_name YALNIZ bu listedeki bir nesne olabilir; listede olmayan "
        "bir eşya, hak, görev veya soyut kavram ('ihsan', 'kadılık', 'hatiplik', "
        "'bağış') için kayıt ÜRETME. Listedeki nesnenin yazımını AYNEN kullan — çoğul "
        "veya çekim eki ekleme ('halılar' değil 'Halı'). Listedeki nesnelerin HER BİRİ "
        "için önce 'sahibi kim' sorusunu sor, sonra el değiştirip değiştirmediğine bak; "
        "sahibi de belirlenemeyen nesneyi atla.\n"
        "\n"
        "0a. HER AKTARIM AYRI KAYITTIR. Aynı nesne (örn. halı) bu bölümde farklı "
        "kişiler arasında birden çok kez el değiştirebilir; her veren-alan çifti için "
        "AYRI kayıt aç. Nesne adı aynı diye kayıtları birleştirme.\n"
        "\n"
        "1. AKTARIM kaydı için nesnenin giver'dan receiver'a geçtiği açık olmalıdır "
        "(verme, gönderme, getirme, hediye, emanet bırakma, teslim, devretme). SAHİPLİK "
        "kaydı için ise metin nesnenin kime ait olduğunu açıkça göstermelidir: iyelik "
        "eki ('kılıcım', 'Osman Gazi'nin atı'), açık bir bulundurma ('belinde kılıç "
        "vardı', 'misafirhanesi'), bir topluluğun elindeki eşya ('kafirlerin çanı'), ya "
        "da kişinin nesneyi kendi tasarrufunda kullanması/kullandırması ('eşyalarını "
        "öküzlere yükletirdi' -> öküzler onun tasarrufundadır). Nesnenin yalnızca "
        "metinde ANILMASI sahiplik değildir; sahibi belli değilse kayıt üretme.\n"
        "\n"
        "2. giver gerçekten nesnenin çıktığı kişi/grup, receiver gerçekten nesnenin "
        "ulaştığı kişi/grup olmalıdır. Aktarım fiilinin dilbilgisel öznesi genellikle "
        "giver'dır; yönü yalnız önceki cümlede geçen kişi sırasına göre belirleme.\n"
        "\n"
        "3. Örtük özneyi yalnızca aynı cümledeki bağlı yapılarda veya hemen sonraki "
        "cümlede özne kesin biçimde devam ediyorsa taşı. Yeni bir kişi/grup açıkça özne "
        "olduğunda önceki giver bilgisini DERHAL bırak. Paragraf boyunca önceki faili "
        "otomatik olarak giver kabul etme.\n"
        "\n"
        "4. object_name ve giver HER kayıtta metinden güvenilir biçimde "
        "belirlenebilmelidir; bu ikisinden biri belirsizse kayıt üretme. receiver "
        "yalnız AKTARIM kayıtlarında zorunludur — aktarım varken alanın kim olduğu "
        "belirsizse yön tahmin etme, kaydı atla. Sahiplik kaydında receiver'ı boş "
        "bırak; oraya 'yok', 'n/a', '-' gibi bir şey YAZMA. Metinde bulunmayan tarihî "
        "kişileri dış bilgiden ekleme.\n"
        "\n"
        "5. Bir aktarımda birden fazla nesne varsa her nesne için ayrı kayıt oluştur. "
        "'peynir, halı, kilim ve kuzular getirdiler' ifadesinde yön aynıysa dört ayrı "
        "kayıt üret. 'altın ve gümüş' gibi bağımsız maddi öğeleri de mümkün olduğunda "
        "ayrı çıkar.\n"
        "\n"
        "6. object_type nesnenin kısa ve normalize edilmiş türünü yazar. transfer_verb "
        "ilişkinin EYLEM TÜRÜdür; metnin gösterdiği türü seç: 'Sahiplik' (el değiştirme "
        "yok, nesne sahibinin elinde), 'İsâr' (karşılıksız verme: hediye, armağan, "
        "ihsan, bağış), 'Emanet' (geçici bırakma; nesne geri alınmak üzere verilir), "
        "'Ganimet' (savaş/yağma yoluyla zorla el değiştirme). Metindeki eylem bu "
        "dördünden hiçbirine uymuyorsa uydurma: metindeki fiili kısaca kendi sözcüğüyle "
        "yaz ('sattı', 'haraç verdi').\n"
        "   İSTEK ile ZORUNLULUK'u ayır: İsâr ile Ganimet aynı yönde hareket eder, "
        "ikisini verenin İRADESİ ayırır — biri bağıştır, öteki el koymadır. Emanet ile "
        "İsâr'ı ise GERİ DÖNÜŞ ayırır: nesne geri alınacaksa Emanet'tir.\n"
        "   Aşağıdaki ÖRNEKLER ağırlıklı olarak İsâr içerir; bu bir sınırlama DEĞİL, "
        "örnekleme eksikliğidir. Bölümde sahiplik ya da ganimet varsa onu yaz.\n"
        "   evidence_quote ilişkiyi doğrulamaya yeterli EN "
        "KISA metin parçası olsun: aktarımda tarafları ve fiili, sahiplikte nesneyi ve "
        "sahibi göstermelidir.\n"
        "\n"
        "7. era bilgisini yalnızca verilen metin veya bölüm bağlamından güvenilir "
        "biçimde belirle. Aynı nesne aktarımını aynı giver-receiver yönüyle tekrar "
        "kaydetme. Yön konusunda emin değilsen yanlış ilişki üretmek yerine kaydı atla. "
        "Şiir/nazım bölümlerinden aktarım çıkarma; yalnız düzyazı anlatıdaki gerçek "
        "aktarımları kaydet. Mecaz tamlamalarını ve dua/temenni ifadelerini (\"baht "
        "açıklığı sana verildi\" gibi) aktarım sayma.\n"
        "\n"
        "ÇIKTI BİÇİMİ (JSON):\n"
        "{\"items\":[{\"object_name\":\"...\",\"giver\":\"...\",\"receiver\":\"...\",\"object_type\":\"\",\"transfer_verb\":\"\",\"evidence_quote\":\"\",\"era\":\"\",\"confidence\":0.0-1.0}]}\n"
        "Sahiplik kayıtlarında receiver alanını boş dizge (\"\") olarak bırak.\n"
        "Uygun sahiplik ya da aktarım yoksa: {\"items\":[]}\n"
        "\n"
        "ÖRNEKLER (metin parçası → o parçadan çıkarılması gereken TÜM kayıtlar. Metni eksiksiz tara; örneklerdeki ayrıntı düzeyini koru.):\n"
        "METİN: \"Süleyman Şah atını suya sürdü. Önü yar imiş, at sürçünce Süleyman Şah suya düştü.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"At\",\"giver\":\"Süleyman Şah Gazi\",\"receiver\":\"\",\"object_type\":\"Binek Hayvan\",\"transfer_verb\":\"Sahiplik\",\"evidence_quote\":\"Süleyman Şah atını suya sürdü\",\"era\":\"Süleyman Şah\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Ertuğrul Gazi oğlanlarından Saru Yatı'yı Sultan Alaeddin' e gönderdi ve 'Bize de yurt versin, gidelim gaza edelim.' dedi. Saru Yatı babasının sözlerini Sultan Ala ed din' e getirdi. Sultan bunlar geldiği için çok sevindi. O zamanda Karacahisar tekfuruyla Bilecik tekfuru Sultan' a itaat edip haraç verirlerdi. O iki yerin arasında bulunan Söğüt vilayetini bunlara kışlamak için yurt gösterdiler; ayrıca Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Toprak (Domaniç ve İrmeni Bili)\",\"giver\":\"Sultan Alaeddin\",\"receiver\":\"Ertuğrul Gazi\",\"object_type\":\"Taşınılmaz / Toprak\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"Domaniç Dağı'nı ve İrıneni Bili'nin dağını bunlara yayla verdiler\",\"era\":\"Pre-Osman\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Osman Gazi, Bilecik tekturuna bundan şikayette bulundu ve 'Sizden dileğimiz, biz yaylaya gittiğimiz vakit eşyalarımızı sizde emanet koyalım.' dedi. O da kabul etti. Ne zaman yaylaya gidecek olursa, Osman Gazi bütün eşyalarını öküzlere yükletir ve bir nice hatun kişiyle gönderir; onlar da varıp kalede emanet korlardı. Yayladan döndükleri vakit de peynir, halı, kilim ve kuzular hediye getirip emanetlerini alıp giderlerdi.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Eşya\",\"giver\":\"Bilecik tekfuru\",\"receiver\":\"Osman Gazi\",\"object_type\":\"Kamu Esbabı\",\"transfer_verb\":\"Emanet\",\"evidence_quote\":\"emanetlerini alıp giderlerdi\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Öküz\",\"giver\":\"Osman Gazi\",\"receiver\":\"\",\"object_type\":\"Binek Hayvan\",\"transfer_verb\":\"Sahiplik\",\"evidence_quote\":\"bütün eşyalarını öküzlere yükletir\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Peynir\",\"giver\":\"Osman Gazi\",\"receiver\":\"Bilecik tekfuru\",\"object_type\":\"Erzak\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"peynir, halı, kilim ve kuzular hediye getirip\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Halı\",\"giver\":\"Osman Gazi\",\"receiver\":\"Bilecik tekfuru\",\"object_type\":\"Kâlâ\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"peynir, halı, kilim ve kuzular hediye getirip\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Kilim\",\"giver\":\"Osman Gazi\",\"receiver\":\"Bilecik tekfuru\",\"object_type\":\"Kâlâ\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"peynir, halı, kilim ve kuzular hediye getirip\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Kuzu\",\"giver\":\"Osman Gazi\",\"receiver\":\"Bilecik tekfuru\",\"object_type\":\"Ağnam\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"peynir, halı, kilim ve kuzular hediye getirip\",\"era\":\"Osman Gazi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Derviş, 'Şehirden vazgeçtik, bize şu köyceğiz yeter.' cevabını verdi. Osman Gazi bu isteği kabul etti.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Köyceğiz\",\"giver\":\"Osman Gazi\",\"receiver\":\"Derviş Tururoğlu\",\"object_type\":\"Taşınılmaz / Toprak\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"bize şu köyceğiz yeter\",\"era\":\"Osman Gazi\",\"confidence\":0.9}]}\n"
        "\n"
        "METİN: \"Derviş bunun üzerine, 'Bize şimdi yazılı bir belge ver.' dedi. Osman Gazi de, 'Ben yazı yazmasını bilir miyim ki benden yazılı kağıt istersin.' dedi. Sonunda Osman Gazi 'İşte babamdan ve dedemden kalmış bir kılıcım, bir de maşrapam var. Bunları sana vereyim, elinde bulunsunlar ve bu nişanları saklasınlar.\"\n"
        "ÇIKTI: {\"items\":[{\"object_name\":\"Kılıç\",\"giver\":\"Osman Gazi\",\"receiver\":\"Derviş Tururoğlu\",\"object_type\":\"Seyfiye\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"bir kılıcım, bir de maşrapam var. Bunları sana vereyim\",\"era\":\"Osman Gazi\",\"confidence\":0.9},{\"object_name\":\"Maşraba\",\"giver\":\"Osman Gazi\",\"receiver\":\"Derviş Tururoğlu\",\"object_type\":\"Kâlâ\",\"transfer_verb\":\"İsâr\",\"evidence_quote\":\"bir kılıcım, bir de maşrapam var. Bunları sana vereyim\",\"era\":\"Osman Gazi\",\"confidence\":0.9}]}"),

)



# ═════════════════════════════════════════════════════════════════════════════
# EŞLEYİCİ (to_models)
# -----------------------------------------------------------------------------
# to_models, LLM'in verdiği HAM SÖZLÜĞÜ (dict) bu uzmanın TİPLİ MODELİNE çevirir.
# Neden: LLM dict verir; ama orchestrator, kaydetme ve grafik hep tipli
# nesnelerle çalışır. Ayrıca eksik alanları güvenle doldurur (it.get) ve
# boş/geçersiz kayıtları atar.
# ═════════════════════════════════════════════════════════════════════════════
def to_models(items, segment_id) -> list[Relation]:
    out = []
    for it in items:
        if str(it.get("object_name", "")).strip():
            out.append(Relation(
                it["object_name"].strip(), it.get("giver", ""), it.get("receiver", ""),
                it.get("object_type", ""), it.get("transfer_verb", ""),
                it.get("era", ""), segment_id,
                float(it.get("confidence", 0.7)), float(it.get("weight_score", 0.0))))
    return out
