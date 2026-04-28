import argparse
from pathlib import Path

from mcp import execute_td_python


def main():
    parser = argparse.ArgumentParser(description='Build a TouchDesigner visual through twozero MCP.')
    parser.add_argument('script', type=Path, help='TouchDesigner Python builder script')
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()

    script = args.script.read_text()
    code = "ns = dict(globals())\nexec(" + repr(script) + ", ns, ns)"
    print(execute_td_python(code, timeout=args.timeout))


if __name__ == '__main__':
    main()
