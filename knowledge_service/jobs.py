"""
knowledge_service.jobs
========================

Фоновое выполнение задач загрузки и индексации: очередь и рабочие потоки
(`KB_WORKERS`). Сама работа — `KnowledgeService.run_job`; здесь только
очередь, запуск и ожидание (для тестов).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .service import KnowledgeService

log = logging.getLogger("knowledge_service.jobs")

TERMINAL = {"completed", "failed", "cancelled"}


class JobRunner:
    def __init__(self, service: "KnowledgeService", workers: int = 1):
        self.service = service
        self.workers = max(1, workers)
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for n in range(self.workers):
            thread = threading.Thread(target=self._loop, name=f"kb-worker-{n + 1}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()
        self._started = False

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self.service.run_job(job_id)
            except Exception:  # noqa: BLE001 — рабочий поток не должен падать
                log.exception("задача %s завершилась с необработанной ошибкой", job_id)

    def wait(self, job_id: str, timeout: float = 30.0) -> dict:
        """Дождаться завершения задачи (для тестов и отладки)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.service.get_job(job_id)
            if job["status"] in TERMINAL:
                return job
            time.sleep(0.02)
        raise TimeoutError(f"задача {job_id} не завершилась за {timeout} с")
