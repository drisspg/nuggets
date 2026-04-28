import json
from urllib.request import Request, urlopen

MCP_URL = 'http://127.0.0.1:40404/mcp'


def call_mcp(tool_name, arguments, timeout=120):
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'tools/call',
        'params': {'name': tool_name, 'arguments': arguments},
    }
    req = Request(
        MCP_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode('utf-8')


def execute_td_python(code, timeout=120):
    return call_mcp('td_execute_python', {'code': code}, timeout=timeout)
