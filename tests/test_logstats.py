"""Smoke tests for logstats. Run:  python3 -m unittest discover -s tests -v"""

import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from logstats import parse_file, watch_items, to_json  # noqa: E402

SAMPLE = os.path.join(ROOT, "examples", "sample.log")


class TestParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = parse_file(SAMPLE)

    def test_models_and_counts(self):
        self.assertEqual(self.fs["models"], ["Qwen3.8-27B-Splash"])
        self.assertEqual(len(self.fs["requests"]), 3)
        self.assertEqual(self.fs["loads"], 1)
        self.assertGreaterEqual(self.fs["stops"], 1)

    def test_request_fields(self):
        first = self.fs["requests"][0]
        self.assertEqual((first["input"], first["cached"], first["output"]),
                         (6543, 0, 3131))
        self.assertAlmostEqual(first["ttft"], 6.8)
        self.assertAlmostEqual(first["tps"], 58.5)
        self.assertEqual(first["model"], "incoai/Qwen3.8-27B-Splash")

    def test_peak_context(self):
        peak = max(r["input"] + r["output"] for r in self.fs["requests"])
        self.assertEqual(peak, 23625 + 1636)

    def test_errors_are_engine_level_only(self):
        self.assertGreaterEqual(self.fs["errors"].get("gpu_oom", 0), 1)
        self.assertGreaterEqual(self.fs["errors"].get("engine_error", 0), 1)

    def test_cache_stats_and_watch(self):
        self.assertAlmostEqual(self.fs["cache"]["used_gib"], 114.0, delta=1)
        self.assertAlmostEqual(self.fs["cache"]["cap_gib"], 163.0, delta=1)
        self.assertAlmostEqual(self.fs["cache"]["restore_eff_pct"], 95.59, delta=0.01)
        watch = watch_items(self.fs)
        self.assertTrue(any("GPU OOM" in w for w in watch))
        self.assertTrue(any("prompt-cache disk at 70%" in w for w in watch))


class TestJson(unittest.TestCase):
    def test_json_shape(self):
        fs = parse_file(SAMPLE)
        out = json.loads(json.dumps(to_json(fs)))  # must be serializable
        for key in ("models", "requests", "output_tokens", "peak_context",
                    "per_model", "errors", "cache", "watch"):
            self.assertIn(key, out)
        self.assertEqual(out["requests"], 3)
        self.assertEqual(out["output_tokens"], 3131 + 11 + 1636)


class TestCli(unittest.TestCase):
    def test_main_with_model_filter(self):
        """--model must not crash on files with unattributed requests."""
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "logstats.py"),
             "--model", "Qwen3.8", SAMPLE],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("3 reqs", proc.stdout)

    def test_json_flag(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "logstats.py"),
             "--json", SAMPLE],
            capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        json.loads(proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
