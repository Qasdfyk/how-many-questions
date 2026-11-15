"""Async client for a vLLM OpenAI-compatible server with a resumable on-disk cache.

Cache: one JSONL file per model in cache_dir; key = sha1 of everything that determines the output
(model, messages incl. image id, temperature, sample index, max_tokens). Identical prompts produced
by different (variant, subset) combinations are therefore computed only once.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image

from pbad.vlm.prompt import SYSTEM


@lru_cache(maxsize=64)
def encode_image(path: str, max_pixels: int) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Request:
    __slots__ = ("image", "text", "temperature", "sample", "tag")

    def __init__(self, image: str, text: str, temperature: float = 0.0, sample: int = 0, tag=None):
        self.image, self.text, self.temperature, self.sample, self.tag = image, text, temperature, sample, tag

    def key(self, model: str, max_tokens: int, max_pixels: int) -> str:
        img_id = Path(self.image).parent.name + "/" + Path(self.image).name
        blob = json.dumps([model, SYSTEM, img_id, max_pixels, self.text, round(self.temperature, 4),
                           self.sample if self.temperature > 0 else 0, max_tokens])
        return hashlib.sha1(blob.encode()).hexdigest()


class VLMClient:
    def __init__(self, vcfg: dict, cache_dir: Path):
        from openai import AsyncOpenAI
        self.cfg = vcfg
        self.model = vcfg["model"]
        self.api = AsyncOpenAI(base_url=vcfg["base_url"], api_key=vcfg.get("api_key", "EMPTY"),
                               timeout=vcfg["timeout_s"], max_retries=0)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"vlm_{self.model.replace('/', '__')}.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[r["key"]] = r
                    except json.JSONDecodeError:
                        pass  # truncated last line after a crash
        self._fh = open(self.cache_path, "a")
        self.failures = 0

    def key(self, req: Request) -> str:
        return req.key(self.model, self.cfg["max_tokens"], self.cfg["max_pixels"])

    def cached(self, req: Request) -> dict:
        return self.cache.get(self.key(req)) or {"text": "", "latency": float("nan"), "tokens_out": None,
                                                 "failed": True}

    async def _call(self, req: Request) -> dict:
        k = self.key(req)
        if k in self.cache:
            return self.cache[k]
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": encode_image(req.image, self.cfg["max_pixels"])}},
                {"type": "text", "text": req.text},
            ]},
        ]
        last_err = None
        for attempt in range(self.cfg["max_retries"]):
            try:
                t0 = time.perf_counter()
                extra = {"seed": req.sample} if req.temperature > 0 else {}
                r = await self.api.chat.completions.create(
                    model=self.model, messages=msgs, temperature=req.temperature,
                    max_tokens=self.cfg["max_tokens"], **extra)
                rec = {"key": k, "text": r.choices[0].message.content or "",
                       "latency": time.perf_counter() - t0,
                       "tokens_in": getattr(r.usage, "prompt_tokens", None),
                       "tokens_out": getattr(r.usage, "completion_tokens", None)}
                self.cache[k] = rec
                self._fh.write(json.dumps(rec) + "\n")
                self._fh.flush()
                return rec
            except Exception as e:  # network / server errors -> retry with backoff
                last_err = e
                await asyncio.sleep(min(60, 2 ** (attempt + 1)))
        # Not cached, so a rerun retries it; callers treat the empty text as a parse failure.
        self.failures += 1
        print(f"[vlm] call failed after {self.cfg['max_retries']} retries: {last_err!r}", flush=True)
        return {"key": k, "text": "", "failed": True}

    async def run(self, reqs: list[Request], progress: bool = True) -> list[dict]:
        sem = asyncio.Semaphore(self.cfg["concurrency"])
        todo = sum(self.key(r) not in self.cache for r in reqs)
        bar = None
        if progress and todo:
            from tqdm import tqdm
            bar = tqdm(total=todo, desc=f"VLM {self.model.split('/')[-1]}")

        async def one(r):
            hit = self.key(r) in self.cache
            async with sem:
                out = await self._call(r)
            if bar is not None and not hit:
                bar.update(1)
            return out

        try:
            return await asyncio.gather(*(one(r) for r in reqs))
        finally:
            if bar is not None:
                bar.close()

    def run_sync(self, reqs: list[Request], progress: bool = True) -> list[dict]:
        return asyncio.run(self.run(reqs, progress))

    def close(self):
        self._fh.close()
