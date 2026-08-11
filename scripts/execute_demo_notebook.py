"""Execute the artifact-only project demo without touching release evidence."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "project_demo.ipynb"
FORBIDDEN_OUTPUT_ROOT = ROOT / "release" / "v1.0-final"


def execute_notebook(input_path: Path, output_path: Path, working_directory: Path) -> Path:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    working_directory = working_directory.resolve()
    if output_path == FORBIDDEN_OUTPUT_ROOT or FORBIDDEN_OUTPUT_ROOT in output_path.parents:
        raise ValueError("Notebook output cannot modify frozen release evidence")
    notebook = nbformat.read(input_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
        allow_errors=False,
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    runtime_root = Path(tempfile.mkdtemp(prefix="notebook-runtime-", dir=output_path.parent))
    previous = {
        key: os.environ.get(key)
        for key in ("PYTHONDONTWRITEBYTECODE", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR")
    }
    os.environ.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "IPYTHONDIR": str(runtime_root / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(runtime_root / "jupyter"),
        }
    )
    try:
        client.execute()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(runtime_root, ignore_errors=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--working-directory", type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.output or args.input
    print(execute_notebook(args.input, output, args.working_directory))


if __name__ == "__main__":
    main()
