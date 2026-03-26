from __future__ import annotations

from rq import Worker

from backend.queue import redis_conn, task_queue


def main() -> None:
    worker = Worker([task_queue.name], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
