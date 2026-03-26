from __future__ import annotations

from redis import Redis
from rq import Queue

from backend.config import settings

redis_conn = Redis.from_url(settings.redis_url)
task_queue = Queue(settings.queue_name, connection=redis_conn)

