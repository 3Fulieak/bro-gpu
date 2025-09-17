"""
命令行工具：GPU PoW 挖矿（Numba/CuPy CUDA，仅 GPU）

用法 1（持续模式，默认）:
  python pow_cli.py
  运行后输入 txid / vout，其它使用默认参数并持续扫描；当 baseline >= ALERT_THRESHOLD 时持续提示音。

用法 2（阈值模式，一次性扫描，如需）:
  python pow_cli.py --threshold 20 --start 0 --count 5000000 --blocks 256 --tpb 256
"""
import time
import json
import argparse
import sys
import platform
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

# 优先使用 CuPy 后端；失败则回退 Numba 后端
from cupy_pow import mine_gpu



# ========= 直接在这里写报警阈值与响铃间隔 =========
ALERT_THRESHOLD = 43          # 当 baseline >= 该值时，开始持续播放提示音
ALERT_BEEP_INTERVAL = 1.0     # 持续响铃的间隔（秒）
# =================================================
OUTPUT_FILE = "1.json"

# 持续蜂鸣线程控制
_alert_event = threading.Event()
_alert_thread_started = threading.Event()

def beep_once():
    """播放一次提示音（非阻塞主程序：在独立线程中循环调用）"""
    try:
        if platform.system() == "Windows":
            import winsound
            # 1000 Hz, 500 ms；注意 winsound.Beep 本身是阻塞的，但我们在独立线程循环调用
            winsound.Beep(1000, 500)
        else:
            # 终端响铃（不同终端/系统可能静音）
            sys.stdout.write('\a')
            sys.stdout.flush()
            # 留一点时间让终端有机会发声
            time.sleep(0.2)
    except Exception:
        pass

def _beep_forever():
    """阈值命中后持续蜂鸣，直到进程结束（不清除事件）"""
    while _alert_event.is_set():
        beep_once()
        time.sleep(ALERT_BEEP_INTERVAL)

def start_alert_beeper():
    """首次命中阈值时启动持续蜂鸣线程（只启动一次）"""
    if not _alert_event.is_set():
        _alert_event.set()
    if not _alert_thread_started.is_set():
        t = threading.Thread(target=_beep_forever, name="alert-beeper", daemon=True)
        t.start()
        _alert_thread_started.set()


def main():
    p = argparse.ArgumentParser()
    # txid / vout 通过 input() 获取
    p.add_argument('--threshold', '-t', type=int, help='阈值模式：前导零位阈值（与 --stream 互斥）')
    p.add_argument('--stream', action='store_true', default=True,
                   help='持续模式（默认开启）：不设固定阈值，发现更优即打印一次')
    p.add_argument('--start', type=int, default=0, help='起始 nonce（stream/threshold 通用）')
    p.add_argument('--count', type=int, default=1_000_000, help='一次性扫描 nonce 数（仅阈值模式）')
    p.add_argument('--batch', type=int, default=1_000_000, help='每轮批大小（持续模式）')
    p.add_argument('--baseline', type=int, default=0, help='持续模式初始基线（前导零位）')
    p.add_argument('--blocks', type=int, default=256, help='CUDA blocks 数')
    p.add_argument('--tpb', type=int, default=256, help='每个 block 的线程数 (threads per block)')
    # HTTP 服务（保持原有）
    p.add_argument('--serve', action='store_true', help='启动HTTP服务')
    p.add_argument('--host', default='0.0.0.0', help='HTTP服务监听地址')
    p.add_argument('--port', type=int, default=8080, help='HTTP服务端口')
    args = p.parse_args()

    # 交互输入 txid / vout
    txid = input("请输入交易ID (txid): ").strip()
    vout = int(input("请输入输出索引 (vout): ").strip())

    # 若用户既没给 --stream 也没给 --threshold，则强制默认走 stream
    if not args.stream and args.threshold is None:
        args.stream = True

    # HTTP 服务模式（保持原样；该模式下 txid/vout 由 HTTP 请求体提供）
    if args.serve:
        default_blocks = args.blocks
        default_tpb = args.tpb
        default_start = args.start
        default_count = args.count

        cache_file = 'pow_results_cache.json'
        busy_lock = threading.Lock()
        state_lock = threading.Lock()
        current_key = {'value': None}
        cache = {}

        def make_key(txid_, vout_, threshold_):
            return f"{txid_}:{vout_}:{threshold_}"

        def load_cache():
            nonlocal cache
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        def save_cache():
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False)
            except Exception:
                pass

        load_cache()

        class Handler(BaseHTTPRequestHandler):
            def _json(self, code, obj):
                body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    raw = self.rfile.read(length) if length > 0 else b''
                    try:
                        data = json.loads(raw.decode('utf-8')) if raw else {}
                    except Exception:
                        return self._json(400, {'error': 'bad_request', 'message': 'invalid json'})

                    txid_ = (data.get('txid') or '').strip()
                    vout_ = data.get('vout')
                    threshold_ = data.get('threshold')
                    if not txid_ or not isinstance(vout_, int) or threshold_ is None:
                        return self._json(400, {'error': 'bad_request', 'message': 'required: txid(string), vout(int), threshold(int)'})

                    start = default_start
                    count = default_count
                    blocks = default_blocks
                    tpb = default_tpb

                    challenge_ = f"{txid_}:{vout_}"
                    key = make_key(txid_, vout_, int(threshold_))

                    if key in cache and isinstance(cache[key], dict) and 'result' in cache[key]:
                        return self._json(200, cache[key])

                    if not busy_lock.acquire(blocking=False):
                        with state_lock:
                            running_key = current_key.get('value')
                        if running_key == key:
                            return self._json(423, {
                                'status': 'running',
                                'message': 'the same job is currently running; please retry later or check cache',
                                'key': key,
                                'hint': 'server caches results by key (txid:vout:threshold)'
                            })
                        return self._json(423, {
                            'status': 'busy',
                            'message': 'another job is running; please retry later',
                            'current_key': running_key
                        })
                    try:
                        with state_lock:
                            current_key['value'] = key
                        current = start
                        batches_run = 0
                        print(f"[JOB START] key={key} challenge={challenge_} threshold={int(threshold_)} start={start} count={count} blocks={blocks} tpb={tpb}")
                        while True:
                            res = mine_gpu(
                                challenge=challenge_,
                                threshold_bits=int(threshold_),
                                start_nonce=current,
                                total_nonces=count,
                                blocks=blocks,
                                threads_per_block=tpb,
                            )
                            batches_run += 1
                            if res and res.get('leading_zero_bits', 0) >= int(threshold_):
                                try:
                                    print(f"[JOB DONE] key={key} batches_run={batches_run} nonce={res.get('nonce')} lz={res.get('leading_zero_bits')}")
                                except Exception:
                                    pass
                                cache[key] = {
                                    'status': 'done',
                                    'challenge': challenge_,
                                    'params': {'txid': txid_, 'vout': vout_, 'threshold': int(threshold_)},
                                    'result': res,
                                    'batches_run': batches_run
                                }
                                save_cache()
                                try:
                                    return self._json(200, cache[key])
                                except Exception:
                                    return
                            current += count
                    finally:
                        if busy_lock.locked():
                            busy_lock.release()
                        with state_lock:
                            current_key['value'] = None
                        try:
                            print(f"[JOB RELEASE] key={key}")
                        except Exception:
                            pass
                except Exception as e:
                    return self._json(500, {'error': 'internal', 'message': str(e)})

        httpd = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"HTTP server listening on http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()
        return

    # 非 HTTP：CLI 模式
    challenge = f"{txid}:{vout}"

    if args.stream:
        baseline = int(args.baseline)
        current = int(args.start)
        batch = int(args.batch)
        while True:
            res = mine_gpu(
                challenge=challenge,
                threshold_bits=baseline,
                start_nonce=current,
                total_nonces=batch,
                blocks=args.blocks,
                threads_per_block=args.tpb,
            )
            print(f"\r[SCAN] current nonce = {current}", end="", flush=True)

            if res and res.get('leading_zero_bits', 0) >= baseline:
                baseline = int(res['leading_zero_bits'])
                print()  # 换行打印结果
                result_obj = {
                    "nonce": res.get("nonce"),
                    "hash": res.get("hash_hex"),
                    "bestHash": res.get("hash_hex"),
                    "bestNonce": res.get("nonce"),
                    "bestLeadingZeros": baseline,
                    "challenge": challenge,
                    "timestamp": int(time.time() * 1000)
                }
                print(json.dumps(result_obj, ensure_ascii=False, indent=2), flush=True)

                # ⚡ 写入到 1.json
                try:
                    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                        json.dump(result_obj, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[WARN] 写入 {OUTPUT_FILE} 失败: {e}")

                if baseline >= ALERT_THRESHOLD:
                    start_alert_beeper()

            current += batch
    else:
        # 阈值模式
        if args.threshold is None:
            raise SystemExit('缺少 --threshold 或使用 --stream 模式')
        res = mine_gpu(
            challenge=challenge,
            threshold_bits=args.threshold,
            start_nonce=args.start,
            total_nonces=args.count,
            blocks=args.blocks,
            threads_per_block=args.tpb,
        )
        print(json.dumps({
            'mode': 'threshold',
            'challenge': challenge,
            'threshold': args.threshold,
            'result': res
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
