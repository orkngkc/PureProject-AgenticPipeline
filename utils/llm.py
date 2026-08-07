"""LLM İSTEMCİSİ (AI ile iletişim) — pipeline'ın 1. aşaması.

NE İŞE YARAR: AI sağlayıcısıyla (Groq / Gemini / OpenAI-uyumlu herhangi biri)
internet üzerinden konuşan TEK yer. 8 agent da AI'a bunun üzerinden ulaşır →
paylaşılan altyapı olduğu için 'utility'.

PIPELINE'DAKİ YERİ:
    BU DOSYA (AI'a sor, cevabı getir)  →  json_utils.py (JSON çıkar)  →  validator.py (denetle)

NE YAPAR:
1. Paketi hazırlar  — model + sıcaklık + system(rol)/user(metin) mesajları + anahtar.
2. Gönderir         — HTTP POST ile sağlayıcının chat/completions ucuna.
3. Dayanıklıdır     — 429 (hız limiti) alınca 3→6→12→24 sn bekleyip tekrar dener;
                      anahtar yoksa DRY-RUN (istek atmaz); boş cevapta çökmez; 120 sn timeout.
4. Cevabı çıkarır   — sağlayıcının yanıtından modelin ürettiği metni alır.

Sağlayıcı .env'deki LLM_BASE_URL + LLM_API_KEY ile belirlenir; kod aynı kalır.
Benzetme: AI'ı arayan TELEFON — agent'lar numarayı bilmez, sadece 'chat(...)' der.
"""
from __future__ import annotations

import json
import time

import requests

import config


class LLMClient:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.model = model or config.LLM_MODEL
        self.dry_run = not bool(self.api_key)

    def chat(self, system: str, user: str,
             temperature: float = config.DEFAULT_TEMPERATURE) -> str:
        """Sistem + kullanıcı mesajı gönderir, modelin metin cevabını döndürür."""
        if self.dry_run:
            return json.dumps({"_dry_run": True, "items": []}, ensure_ascii=False)

        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        data = self._post(config.LLM_BASE_URL, headers, payload)
        # Bazı (özellikle ücretsiz) modeller content=null döndürebilir → boş say
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""
        return content or ""

    def _post(self, url: str, headers: dict, payload: dict, retries: int = 4) -> dict:
        """HTTP POST. 429 (hız limiti) alınca 3→6→12→24 sn bekleyip tekrar dener."""
        for attempt in range(retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:                # hız limiti → bekle ve tekrar dene
                wait = 3 * (2 ** attempt)              # 3, 6, 12, 24 sn
                print(f"    [429 hız limiti] {wait}sn bekleniyor... "
                      f"(deneme {attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            resp.raise_for_status()                    # diğer HTTP hataları → istisna
            return resp.json()
        raise RuntimeError("429: hız limiti aşıldı, tekrar denemeler tükendi "
                           "(kısa süre bekle ya da güçlü/ücretli modele geç).")
