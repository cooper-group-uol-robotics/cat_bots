import os
import subprocess
import sys
from pathlib import Path

import pytest

from catbots.paths import BENCHMARK_DIR

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not BENCHMARK_DIR.exists(), reason="benchmark data not available")
def test_quick_start_command_runs_on_shields():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / "run_pipeline.py"), "--benchmark", "shields_arylation"],
                         capture_output=True, text=True, timeout=600, check=True, env=env).stdout
    assert "Pipeline: 24 experiments" in out and "{'WP2A': 12, 'BO': 12}" in out
    assert all(f"  {s}" in out for s in ("pipeline", "random", "bo"))
