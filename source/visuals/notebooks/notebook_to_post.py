#!/usr/bin/env python3
"""Convert an executed Jupyter notebook into Nuggets notebook-cell Markdown and a Colab copy.

Markdown cells are copied verbatim. Code cells become ```python cell fences and their
text outputs become ```output fences rendered by the NotebookCells transformer.
The Colab copy drops outputs and host-specific metadata and can prepend setup cells.
"""

import argparse
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Host-specific paths: interpreter prefixes and Jupyter's per-cell temp files.
HOST_PATHS = [
    (re.compile(r"\S*/site-packages/"), ""),
    (re.compile(r"/tmp/ipykernel_\d+/\d+\.py"), "<cell>"),
]


def text(source: str | list[str]) -> str:
    return source if isinstance(source, str) else "".join(source)


def cell_output(cell: dict, drop: list[re.Pattern[str]]) -> str:
    parts = []
    for output in cell.get("outputs", []):
        match output["output_type"]:
            case "stream":
                parts.append(text(output["text"]))
            case "execute_result" | "display_data":
                if plain := output.get("data", {}).get("text/plain"):
                    parts.append(text(plain) + "\n")
            case "error":
                parts.append("\n".join(output["traceback"]) + "\n")
    output = ANSI.sub("", "".join(parts))
    for pattern, replacement in HOST_PATHS:
        output = pattern.sub(replacement, output)
    lines = [line for line in output.split("\n") if not any(p.search(line) for p in drop)]
    return "\n".join(lines).strip("\n")


def fence(lang: str, body: str) -> str:
    ticks = "`" * max(3, max((len(run) for run in re.findall(r"`+", body)), default=0) + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}"


def to_markdown(notebook: dict, collapse_over: int, drop: list[re.Pattern[str]]) -> str:
    blocks = []
    for cell in notebook["cells"]:
        source = text(cell["source"]).strip("\n")
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            blocks.append(source)
            continue
        blocks.append(fence("python cell", source))
        if output := cell_output(cell, drop):
            collapsed = " collapsed" if output.count("\n") + 1 > collapse_over else ""
            blocks.append(fence(f"output{collapsed}", output))
    return "\n\n".join(blocks) + "\n"


def to_colab(notebook: dict, setup: list[dict]) -> dict:
    cells = []
    for cell in setup + notebook["cells"]:
        source = text(cell["source"])
        if not source.strip():
            continue
        clean = {"cell_type": cell["cell_type"], "metadata": {}, "source": source}
        if cell["cell_type"] == "code":
            clean |= {"execution_count": None, "outputs": []}
        cells.append(clean)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="executed .ipynb")
    parser.add_argument("--markdown", type=Path, help="write the Markdown body here")
    parser.add_argument("--colab", type=Path, help="write a clean Colab notebook here")
    parser.add_argument(
        "--colab-setup", type=Path, help="notebook whose cells are prepended to the Colab copy"
    )
    parser.add_argument(
        "--collapse-over", type=int, default=40, help="collapse outputs longer than this"
    )
    parser.add_argument(
        "--drop-line",
        action="append",
        default=[],
        metavar="REGEX",
        help="omit output lines matching this pattern (repeatable)",
    )
    args = parser.parse_args()

    notebook = json.loads(args.notebook.read_text())
    if args.markdown:
        drop = [re.compile(pattern) for pattern in args.drop_line]
        args.markdown.write_text(to_markdown(notebook, args.collapse_over, drop))
    if args.colab:
        setup = json.loads(args.colab_setup.read_text())["cells"] if args.colab_setup else []
        args.colab.write_text(json.dumps(to_colab(notebook, setup), indent=1) + "\n")


if __name__ == "__main__":
    main()
