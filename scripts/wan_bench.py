# -*- coding: utf-8 -*-
"""跨境链路吞吐压测：模拟官方 16 并发灌 Add，测真实 WAN 吞吐。

用法（本地 Windows，走公网直连 ECS）:
  python scripts/wan_bench.py --host http://43.106.60.85:8000 --token <AML_API_TOKEN> \
      --threads 16 --seconds 120
判读:
  - 吞吐 ≈10/s → 跨境链路瓶颈实锤（官方 3 次失败的根因）
  - 吞吐 100+/s → 瓶颈在官方发送侧，需联系官方
"""
from __future__ import annotations

import argparse
import json
import random
import string
import threading
import time
import urllib.request

URL = "/add"


def make_payload(i: int) -> dict:
    uid = f"perf_wan_{i % 64}"  # 64 个独立 user，模拟官方每题一库
    msgs = [
        {
            "role": "user",
            "content": f"pressure message {i} " + "x" * random.randint(50, 300),
            "timestamp": 1759000000000 + i,
        }
        for _ in range(random.randint(1, 5))
    ]
    return {
        "request_id": "perf-" + "".join(random.choices(string.hexdigits.lower(), k=16)),
        "user_id": uid,
        "session_id": "perf-sess",
        "messages": msgs,
    }


def worker(host: str, token: str, deadline: float, stat: dict, lock: threading.Lock) -> None:
    req = urllib.request.Request(
        host + URL,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
    )
    i = 0
    while time.time() < deadline:
        body = json.dumps(make_payload(i)).encode()
        r = urllib.request.Request(req.full_url, data=body, headers=req.headers, method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                resp.read()
                dt = time.perf_counter() - t0
            with lock:
                stat["ok"] += 1
                stat["lat"].append(dt)
        except Exception as e:
            with lock:
                stat["err"] += 1
                if len(stat["errs"]) < 5:
                    stat["errs"].append(str(e)[:120])
        i += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seconds", type=int, default=120)
    args = ap.parse_args()

    # 建连基线：单请求
    t0 = time.perf_counter()
    r = urllib.request.Request(
        args.host + "/health",
        headers={"Authorization": f"Token {args.token}"},
    )
    with urllib.request.urlopen(r, timeout=10) as resp:
        resp.read()
    print(f"基线 /health: {(time.perf_counter()-t0)*1000:.0f}ms")

    stat = {"ok": 0, "err": 0, "lat": [], "errs": []}
    lock = threading.Lock()
    deadline = time.time() + args.seconds
    ths = [threading.Thread(target=worker, args=(args.host, args.token, deadline, stat, lock), daemon=True)
           for _ in range(args.threads)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    el = time.time() - t0
    lat = sorted(stat["lat"])
    p = lambda q: lat[int(q * len(lat)) - 1] * 1000 if lat else -1
    print(f"\n{args.threads}并发 {el:.0f}s: ok={stat['ok']} err={stat['err']} "
          f"吞吐={stat['ok']/el:.1f}/s")
    print(f"延迟: P50={p(0.5):.0f}ms P90={p(0.9):.0f}ms P99={p(0.99):.0f}ms max={p(1.0):.0f}ms")
    if stat["errs"]:
        print("首错:", *stat["errs"], sep="\n  ")


if __name__ == "__main__":
    main()
