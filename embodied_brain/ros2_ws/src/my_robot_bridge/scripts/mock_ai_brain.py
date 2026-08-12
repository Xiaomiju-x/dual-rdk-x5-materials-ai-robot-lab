#!/usr/bin/env python3
"""mock_ai_brain.py — 模拟 AI 脑 dashboard:8888 的 4 个端点, 给 ai_brain_bridge 测试用.

不依赖任何 AI 脑代码, 只用 Python http.server stdlib.

启动:
    python3 mock_ai_brain.py --port 8888

提供端点:
    GET  /api/embodied/dispatch_queue   返一个 fetch_sample 任务 (一次)
    POST /api/embodied/report           接车端 telemetry/result, 打印到 stdout
    POST /api/embodied/alarm            接报警, 打印
    POST /api/qwen_vl_check             模拟 Qwen-VL, 返 {pv: 1350, sv: 1350, mv: 49.7}
    POST /api/say                       接 TTS, 打印 "[TTS] {text}"
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# 全局状态: 已经发过的 task 列表 (避免重复发同一个)
_dispatched_once = False


class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 用 stderr 写, 跟 stdout 数据隔离
        sys.stderr.write(f'[mock] {self.address_string()} {format % args}\n')

    def do_GET(self):
        global _dispatched_once
        if self.path == '/api/embodied/dispatch_queue':
            if not _dispatched_once:
                _dispatched_once = True
                tasks = [{
                    'task_id': 'mock-task-001',
                    'type': 'fetch_sample',
                    'args': {
                        'bottle_id': 'SYGO-1',
                        'from': 'shelf_1_slot_3',
                        'to': 'furnace_1',
                    },
                    'priority': 'high',
                }]
            else:
                tasks = []
            self._reply(200, tasks)
            return

        if self.path == '/api/health':
            self._reply(200, {'ok': True, 'mock': True})
            return

        self._reply(404, {'error': 'unknown GET'})

    def do_POST(self):
        n = int(self.headers.get('Content-Length', '0') or '0')
        body = self.rfile.read(n) if n > 0 else b''
        try:
            data = json.loads(body.decode('utf-8') or '{}')
        except Exception:
            data = {'raw': body.decode('utf-8', errors='replace')}

        # 路由
        if self.path == '/api/embodied/report':
            print(f'[mock] REPORT: {self._summarize_report(data)}')
            self._reply(200, {'ok': True})
            return

        if self.path == '/api/embodied/alarm':
            print(f'[mock] ALARM: source={data.get("source")} '
                  f'level={data.get("level")} '
                  f'title={data.get("title", "?")}')
            self._reply(200, {'ok': True})
            return

        if self.path == '/api/qwen_vl_check':
            # 模拟 Qwen-VL 看图返 1350
            print(f'[mock] QWEN-VL: 收到 snapshot ({len(data.get("snapshot_b64", ""))} chars), '
                  f'OCR 给的 PV={data.get("pv_ocr")}')
            self._reply(200, {'pv': 1350.0, 'sv': 1350.0, 'mv': 49.7,
                              'confidence': 0.99, 'model': 'mock'})
            return

        if self.path == '/api/say':
            print(f'[mock] TTS: {data.get("text", "?")} '
                  f'(priority={data.get("priority", "")})')
            self._reply(200, {'ok': True})
            return

        self._reply(404, {'error': 'unknown POST'})

    def _summarize_report(self, data):
        # 大致摘要
        if 'task_id' in data:
            return (f'task {data["task_id"]} success={data.get("success")} '
                    f'msg={data.get("message", "")[:60]}')
        if 'cpu_pct' in data:
            return (f'telemetry cpu={data["cpu_pct"]:.1f}% '
                    f'ram={data.get("ram_used_gb", 0):.2f}/{data.get("ram_total_gb", 0):.2f}GB '
                    f'slam={data.get("slam_active")} nav={data.get("nav2_active")}')
        return str(data)[:120]

    def _reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8888)
    args = ap.parse_args()

    server = ThreadingHTTPServer(('0.0.0.0', args.port), MockHandler)
    print(f'[mock] AI brain mock listening on 0.0.0.0:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('[mock] shutdown')


if __name__ == '__main__':
    main()
