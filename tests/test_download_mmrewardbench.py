import io
import json
import random
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "data/scripts/download_mmrewardbench.py"


class PaperIdSelectionTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)
        self.output_dir = self.directory / "export"
        self.rows = [
            {
                "ID": name, "Text": name, "Output1": "first", "Output2": "second",
                "Better": 1, "Image": None,
            }
            for name in ("alpha", "beta")
        ]

    def run_export(self, paper_ids=None, extra_args=()):
        argv = [str(SCRIPT), "--output_dir", str(self.output_dir), *extra_args]
        if paper_ids is not None:
            manifest = self.directory / "paper_ids.json"
            manifest.write_text(json.dumps(paper_ids), encoding="utf-8")
            argv.extend(["--paper_ids", str(manifest)])

        datasets = ModuleType("datasets")
        datasets.load_dataset = lambda *args, **kwargs: self.rows
        tqdm = ModuleType("tqdm")
        tqdm.tqdm = lambda rows, **kwargs: rows
        with patch.dict(sys.modules, {"datasets": datasets, "tqdm": tqdm}):
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                runpy.run_path(str(SCRIPT), run_name="__main__")

    def exported_rows(self):
        output = self.output_dir / "mmrewardbench_test.jsonl"
        return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    def test_missing_id_fails_without_an_export(self):
        with self.assertRaisesRegex(SystemExit, r"Cannot reproduce the exact paper subset: 1 .*missing"):
            self.run_export(["alpha", "missing"])
        self.assertEqual(list(self.output_dir.rglob("*.jsonl")), [])

    def test_complete_selection_preserves_manifest_order(self):
        self.run_export(["beta", "alpha"], extra_args=("--sample_size", "1"))
        self.assertEqual(self.exported_rows(), list(reversed(self.rows)))

    def test_normal_sampling_keeps_seeded_selection(self):
        self.run_export(extra_args=("--sample_size", "1", "--seed", "123"))
        self.assertEqual(self.exported_rows(), random.Random(123).sample(self.rows, 1))

    def test_entirely_missing_subset_limits_the_diagnostic_sample(self):
        with self.assertRaisesRegex(SystemExit, r"Cannot reproduce the exact paper subset: 6 ") as error:
            self.run_export([f"missing_{i}" for i in range(6)])
        self.assertIn("missing_4", str(error.exception))
        self.assertNotIn("missing_5", str(error.exception))
        self.assertEqual(list(self.output_dir.rglob("*.jsonl")), [])
