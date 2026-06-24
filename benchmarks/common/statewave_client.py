"""
Statewave Client — drop-in for Mem0Client in this harness
=========================================================

Lets the mem0 benchmark suite run **Statewave** through the EXACT same pipeline
it uses for mem0 (same dataset parsing, same answer prompt, same judge, same
top-k cutoffs, same metrics). Only the memory backend changes. This is the
fairness gold standard: if Statewave wins here, it wins on mem0's own bench.

Exposes the same async interface the runners call on Mem0Client:
    async with StatewaveClient(host=...) as sw:
        await sw.add(messages, user_id, timestamp=epoch)
        results = await sw.search(query, user_id, top_k=200)
        await sw.delete_user(user_id)
        await sw.get_user_profile(user_id)   # -> None (no equivalent)

Backend: talks REST to a running Statewave server (default http://localhost:8100):
  - add      -> POST /v1/episodes/batch  (one episode per add call)
  - compile  -> POST /v1/memories/compile (async job) on first search per user,
               then polled to completion + embeddings-ready (lazy, so the
               runner's ingest-then-search flow needs no changes)
  - search   -> GET  /v1/memories/search (hybrid + entity-boost, product defaults)
  - delete   -> DELETE /v1/subjects/{subject}

Statewave's search response is already rank-ordered but doesn't expose a numeric
score; we synthesize a descending rank score so the harness's score-sort + top-k
cutoffs preserve Statewave's ranking.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


def _subject(user_id: str) -> str:
    return f"mh2h:{user_id}"


class StatewaveClient:
    def __init__(
        self,
        mode: str = "statewave",
        host: str | None = None,
        api_key: str | None = None,
        *,
        entity_weight: float | None = None,
        entity_max_distance: float | None = None,
        hybrid: bool = True,
        rerank: bool = False,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        timeout: float = 600.0,
        compile_timeout: float = 7200.0,
        embed_wait: float = 900.0,
        **_ignored: Any,
    ):
        self.host = (host or os.getenv("STATEWAVE_URL", "http://localhost:8100")).rstrip("/")
        self.api_key = api_key or os.getenv("STATEWAVE_API_KEY", "dev")
        # Product defaults (the shipped /v1/memories/search defaults).
        self.hybrid = hybrid
        self.entity_weight = (
            entity_weight if entity_weight is not None
            else float(os.getenv("SW_ENTITY_WEIGHT", "0.3"))
        )
        self.entity_max_distance = (
            entity_max_distance if entity_max_distance is not None
            else float(os.getenv("SW_ENTITY_MAX_DISTANCE", "0.3"))
        )
        self.rerank = rerank or os.getenv("SW_RERANK") == "1"
        # Reuse mode: subjects are already compiled in the DB from a prior run, so
        # skip delete+ingest+compile and go straight to search. Lets us A/B a
        # search-time lever (rerank pool) in minutes instead of a 3h recompile;
        # the compiled memories are identical across arms, so the delta is clean.
        self.reuse = os.getenv("SW_REUSE") == "1"
        # The runner suffixes user_id with a per-run hash (locomo_4_<run_id>), so
        # reuse must pin the run_id of the already-compiled subjects, else it
        # queries empty subjects. Set SW_REUSE_RUNID to the compiled run's id.
        self.reuse_runid = os.getenv("SW_REUSE_RUNID", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.compile_timeout = compile_timeout
        self.embed_wait = embed_wait
        self._session: aiohttp.ClientSession | None = None
        self._compiled: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────
    async def _sess(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-API-Key": self.api_key},
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "StatewaveClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    def _subj(self, user_id: str) -> str:
        """Subject for this user, repointing the run_id suffix when reusing a
        prior compile (locomo_4_<newrun> -> locomo_4_<reuse_runid>)."""
        if self.reuse and self.reuse_runid:
            base = user_id.rsplit("_", 1)[0]
            return f"mh2h:{base}_{self.reuse_runid}"
        return _subject(user_id)

    # ── ingest ──────────────────────────────────────────────────────────────
    async def add(
        self,
        messages: list[dict],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Ingest one session/chunk as a Statewave episode (no compile yet —
        compile happens lazily on first search for this user)."""
        subject = _subject(user_id)
        if self.reuse:
            # Subject already compiled from a prior run — skip ingest entirely.
            return {"results": []}
        # Reset the subject on the first add of a run so re-runs are clean.
        if user_id not in getattr(self, "_seen_users", set()):
            if not hasattr(self, "_seen_users"):
                self._seen_users: set[str] = set()
            self._seen_users.add(user_id)
            with_suppress = True
            try:
                await self.delete_user(user_id)
            except Exception:
                pass
            self._compiled.discard(user_id)

        payload_messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in messages
        ]
        episode: dict[str, Any] = {
            "subject_id": subject,
            "source": "mem0bench",
            "type": "chat",
            "payload": {"messages": payload_messages},
            "metadata": metadata or {},
        }
        if timestamp:
            from datetime import datetime, timezone

            iso = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
            episode["occurred_at"] = iso
            # CRITICAL for temporal: the compiler's reference date (episode_valid_from)
            # uses payload.event_time first, then messages[0].timestamp, then
            # created_at (= ingest date). The harness passes the real session date
            # only via `timestamp`, and its messages carry no per-message timestamp,
            # so without event_time the compiler grounds every relative date to
            # *today*, corrupting all temporal answers. This is the documented way
            # to feed Statewave historical data (mem0 gets the same date via its
            # own `timestamp` param — equivalent, each in its system's form).
            episode["payload"]["event_time"] = iso
        sess = await self._sess()
        for attempt in range(self.max_retries):
            try:
                async with sess.post(
                    f"{self.host}/v1/episodes/batch", json={"episodes": [episode]}
                ) as resp:
                    if resp.status >= 500:
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history, status=resp.status
                        )
                    resp.raise_for_status()
                    return {"results": []}
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Statewave add failed (user=%s): %s", user_id, str(exc)[:200])
                    return {"results": []}
        return {"results": []}

    # ── compile (lazy, once per user) ────────────────────────────────────────
    async def _ensure_compiled(self, user_id: str) -> None:
        if user_id in self._compiled:
            return
        if self.reuse:
            # Memories already present from a prior run; no compile needed.
            self._compiled.add(user_id)
            return
        async with self._lock(user_id):
            if user_id in self._compiled:
                return
            subject = _subject(user_id)
            sess = await self._sess()
            try:
                async with sess.post(
                    f"{self.host}/v1/memories/compile",
                    json={"subject_id": subject, "async": True},
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    job_id = data.get("job_id")
                # poll job
                if job_id:
                    deadline = asyncio.get_event_loop().time() + self.compile_timeout
                    while asyncio.get_event_loop().time() < deadline:
                        async with sess.get(
                            f"{self.host}/v1/memories/compile/{job_id}"
                        ) as jr:
                            jd = await jr.json()
                        if jd.get("status") in ("completed", "failed"):
                            break
                        await asyncio.sleep(2.0)
                # wait for embeddings to be queryable
                edeadline = asyncio.get_event_loop().time() + self.embed_wait
                while asyncio.get_event_loop().time() < edeadline:
                    probe = await self._raw_search(subject, "memory", 1)
                    if probe:
                        break
                    await asyncio.sleep(2.0)
            except Exception as exc:
                logger.error("Statewave compile failed (user=%s): %s", user_id, str(exc)[:200])
            self._compiled.add(user_id)

    # ── search ───────────────────────────────────────────────────────────────
    async def _raw_search(self, subject: str, query: str, top_k: int) -> list[dict]:
        sess = await self._sess()
        params: dict[str, Any] = {
            "subject_id": subject,
            "q": query,
            "semantic": "true",
            "hybrid": "true" if self.hybrid else "false",
            "entity_weight": str(self.entity_weight),
            "entity_max_distance": str(self.entity_max_distance),
            "limit": str(top_k),
        }
        if self.rerank:
            params["rerank"] = "true"
            # Generous pool so the reranker actually FIRES: the route only reranks
            # when pool > limit, and subjects here hold 320-900 memories. Default
            # ~2x limit lets the LLM surface the answer fact from mid-pack instead
            # of being capped at the hybrid top-`limit` (where rerank no-ops).
            params["rerank_pool"] = os.getenv("SW_RERANK_POOL", str(max(top_k * 2, 400)))
        async with sess.get(f"{self.host}/v1/memories/search", params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
        return data.get("memories", []) if isinstance(data, dict) else []

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        await self._ensure_compiled(user_id)
        subject = self._subj(user_id)
        for attempt in range(self.max_retries):
            try:
                mems = await self._raw_search(subject, query, top_k)
                break
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error("Statewave search failed (user=%s): %s", user_id, str(exc)[:200])
                    return []
        n = len(mems)
        out: list[dict] = []
        for i, m in enumerate(mems):
            content = m.get("content") or m.get("memory") or ""
            entry = {
                "memory": content,
                # synthesize a descending rank score so the harness's score-sort
                # + top-k cutoffs preserve Statewave's returned ranking.
                "score": (n - i) / n if n else 0.0,
                "id": m.get("id", ""),
            }
            vf = m.get("valid_from") or m.get("created_at")
            if vf:
                entry["created_at"] = vf
            out.append(entry)
        return out

    # ── misc interface parity ────────────────────────────────────────────────
    async def delete_user(self, user_id: str) -> bool:
        subject = _subject(user_id)
        sess = await self._sess()
        try:
            async with sess.delete(f"{self.host}/v1/subjects/{subject}") as resp:
                return resp.status < 400
        except Exception:
            return False

    async def get_user_profile(self, user_id: str) -> None:
        # Statewave has no per-user "profile" object in this harness; the runner
        # treats None as "no profile".
        return None
