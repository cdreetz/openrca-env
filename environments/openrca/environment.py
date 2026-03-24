"""
OpenRCA StatefulToolEnv implementation.

Provides the OpenRCAEnv class with persistent Python execution and
file exploration tools for analyzing telemetry data.
"""

import ast
import asyncio
import os
import sys
import threading
from io import StringIO

import verifiers as vf

from .prompts import EXEC_TIMEOUT_SECONDS, MAX_OUTPUT_CHARS


class OpenRCAEnv(vf.StatefulToolEnv):
    """OpenRCA environment for root cause analysis of software failures.

    Provides a persistent Python execution environment with tools for
    analyzing telemetry data (metrics, traces, logs) from microservice
    systems to identify root causes of failures.
    """

    _exec_lock = threading.Lock()

    def __init__(self, data_dir: str, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.data_dir = os.path.abspath(data_dir)
        self.add_tool(self.execute_python, args_to_skip=["namespace"])
        self.add_tool(self.list_directory)

    async def setup_state(self, state: vf.State, **kwargs: object) -> vf.State:
        """Initialize a persistent Python namespace for each rollout."""
        namespace: dict[str, object] = {}
        setup_code = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "import os\n"
            "import json\n"
            "import re\n"
            "from datetime import datetime, timedelta\n"
            "import pytz\n"
            "\n"
            "pd.set_option('display.width', 427)\n"
            "pd.set_option('display.max_columns', 10)\n"
            "pd.set_option('display.max_rows', 50)\n"
            "\n"
            f"DATA_DIR = {self.data_dir!r}\n"
        )
        exec(setup_code, namespace)  # noqa: S102
        state["namespace"] = namespace
        return await super().setup_state(state, **kwargs)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, object],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: object,
    ) -> dict[str, object]:
        """Inject the persistent namespace into execute_python calls."""
        if tool_name == "execute_python":
            tool_args["namespace"] = state["namespace"]
        return tool_args

    async def execute_python(self, code: str, namespace: dict[str, object]) -> str:
        """Execute Python code in a persistent IPython-like environment.

        The environment has pandas, numpy, os, json, datetime, and pytz
        pre-imported. Variables persist across calls within the same rollout.
        The variable DATA_DIR contains the path to the telemetry data root.

        Args:
            code: Python code to execute. The last expression's value is
                  returned automatically (like in Jupyter/IPython).

        Returns:
            The execution output including printed text and the last
            expression's value, or an error message if execution fails.
        """

        def _run() -> str:
            with OpenRCAEnv._exec_lock:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                captured = StringIO()
                sys.stdout = captured
                sys.stderr = captured

                try:
                    tree = ast.parse(code)
                    last_value = None

                    if tree.body and isinstance(tree.body[-1], ast.Expr):
                        last_expr = tree.body.pop()
                        if tree.body:
                            module = ast.Module(
                                body=tree.body, type_ignores=[]
                            )
                            exec(  # noqa: S102
                                compile(module, "<code>", "exec"), namespace
                            )
                        last_value = eval(  # noqa: S307
                            compile(
                                ast.Expression(body=last_expr.value),
                                "<code>",
                                "eval",
                            ),
                            namespace,
                        )
                    else:
                        exec(  # noqa: S102
                            compile(tree, "<code>", "exec"), namespace
                        )

                    output = captured.getvalue()
                    if last_value is not None:
                        value_str = str(last_value)
                        if output:
                            output = output.rstrip("\n") + "\n" + value_str
                        else:
                            output = value_str

                    if not output or not output.strip():
                        return "Code executed successfully (no output)."

                    if len(output) > MAX_OUTPUT_CHARS:
                        output = (
                            output[:MAX_OUTPUT_CHARS]
                            + "\n\n... [Output truncated. Use .head() or more "
                            "specific queries to limit output.]"
                        )

                    return output.strip()
                except Exception as e:
                    output = captured.getvalue()
                    error_msg = f"{type(e).__name__}: {e}"
                    if output and output.strip():
                        return f"{output.rstrip()}\n\n{error_msg}"
                    return error_msg
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run),
                timeout=EXEC_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return (
                f"Error: Code execution timed out after "
                f"{EXEC_TIMEOUT_SECONDS} seconds. Simplify your query or "
                f"process data in smaller chunks."
            )

    async def list_directory(self, path: str) -> str:
        """List files and directories within the telemetry data directory.

        Args:
            path: Relative path from the data root directory
                  (e.g., "Bank/telemetry/2021_03_05/metric").

        Returns:
            A listing of files and directories at the specified path.
        """

        def _list() -> str:
            full_path = os.path.normpath(os.path.join(self.data_dir, path))

            if not full_path.startswith(self.data_dir):
                return "Error: Access denied. Path must be within the data directory."

            if not os.path.exists(full_path):
                return f"Error: Path does not exist: {path}"

            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                return f"File: {path} ({size:,} bytes)"

            entries = sorted(os.listdir(full_path))
            if not entries:
                return "Empty directory."

            result = []
            for entry in entries:
                entry_path = os.path.join(full_path, entry)
                if os.path.isdir(entry_path):
                    result.append(f"[DIR]  {entry}/")
                else:
                    size = os.path.getsize(entry_path)
                    result.append(f"[FILE] {entry} ({size:,} bytes)")

            return "\n".join(result)

        return await asyncio.to_thread(_list)
