"""
Minimal asyncio walkthrough illustrating cooperative multitasking.

Run the file directly to see:
1. A sequential simulation of blocking I/O.
2. A concurrent simulation using asyncio tasks.
3. How `asyncio.create_task` schedules work ahead of the next `await`.
"""

import asyncio
import time
from typing import List


async def simulated_io(name: str, delay: float) -> str:
    """Pretend to perform I/O by awaiting asyncio.sleep."""
    print(f"[{elapsed():>5.2f}s] {name} -> start (sleeping {delay:.1f}s)")
    await asyncio.sleep(delay)
    result = f"{name} result after {delay:.1f}s"
    print(f"[{elapsed():>5.2f}s] {name} -> done")
    return result


async def run_sequentially() -> List[str]:
    print("\n--- Sequential awaits (tasks finish one after another) ---")
    outputs = []
    for idx, delay in enumerate((1.0, 0.5, 1.5), start=1):
        outputs.append(await simulated_io(f"job-{idx}", delay))
    return outputs


async def run_concurrently() -> List[str]:
    print("\n--- Concurrent tasks (event loop interleaves work) ---")
    tasks = []
    for idx, delay in enumerate((1.0, 0.5, 1.5), start=1):
        task = asyncio.create_task(simulated_io(f"job-{idx}", delay))
        print(f"[{elapsed():>5.2f}s] scheduled {task.get_name()}")
        tasks.append(task)

    results = []
    for task in tasks:
        results.append(await task)
    return results


async def immediate_feedback_demo() -> None:
    print("\n--- create_task schedules work immediately ---")
    task = asyncio.create_task(simulated_io("background-task", 0.4))
    print(f"[{elapsed():>5.2f}s] control returned to caller right after create_task")
    await asyncio.sleep(0.1)
    print(f"[{elapsed():>5.2f}s] caller doing other work before awaiting the task")
    await task


def elapsed() -> float:
    return time.perf_counter() - START_TIME


async def main() -> None:
    await run_sequentially()
    await run_concurrently()
    await immediate_feedback_demo()


if __name__ == "__main__":
    START_TIME = time.perf_counter()
    asyncio.run(main())


