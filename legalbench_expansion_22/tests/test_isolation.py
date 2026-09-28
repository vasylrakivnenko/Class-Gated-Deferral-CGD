"""Check the boundaries that keep this cohort and its outputs separate."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("expansion_run", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.registry = runner.read_json(ROOT / "data/task_registry.json")
        self.selection = runner.read_json(ROOT / "data/selection.json")
        self.original = runner.read_json(ROOT / "data/original_37_task_names.json")

    def test_actual_cohorts_are_disjoint(self):
        runner.verify_registry(self.registry, self.selection, self.original)
        self.assertEqual(len(self.original), 37)

    def test_original_task_cannot_be_selected(self):
        with self.assertRaisesRegex(ValueError, "original-cohort"):
            runner.select_tasks(self.registry, self.original[0], False)

    def test_original_task_cannot_be_inserted_into_registry(self):
        self.registry[0]["task"] = self.original[0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            runner.verify_registry(self.registry, self.selection, self.original)

    def test_smoke_output_is_separate_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke = runner.reserve_output(tmp, "check", True)
            full = runner.reserve_output(tmp, "check", False)
            self.assertNotEqual(smoke, full)
            (full / "keep.txt").write_text("original results")
            with self.assertRaises(FileExistsError):
                runner.reserve_output(tmp, "check", False)
            self.assertEqual((full / "keep.txt").read_text(), "original results")

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("../legalbench_map/results", "/tmp/escape", "..", "a/b"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    runner.reserve_output(tmp, name, False)

    def test_symlink_to_other_results_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "expansion"
            old = Path(tmp) / "old_results"
            root.mkdir()
            old.mkdir()
            (root / "results").symlink_to(old, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "inside this experiment"):
                runner.reserve_output(root, "run", False)
            self.assertEqual(list(old.iterdir()), [])

    def test_exact_duplicates_cannot_cross_training_and_evaluation(self):
        entry = {"task": "example", "input_columns": ["text"]}
        splits = {
            "train": [{"text": "shared", "answer": "yes"}, {"text": "training", "answer": "no"}],
            "test": [{"text": "shared", "answer": "yes"}, {"text": "shared", "answer": "yes"},
                     {"text": "evaluation", "answer": "no"}],
        }
        effective, clean, note = runner.prepare_task(entry, splits)
        self.assertEqual([r["text"] for r in clean["train"]], ["training"])
        self.assertEqual(effective["n_test"], 2)
        self.assertEqual(note["removed_duplicate_test_rows"], 1)
        self.assertEqual(note["removed_train_rows"], 1)
        self.assertEqual(len(splits["train"]), 2)  # Sources are not mutated.
        self.assertEqual(len(splits["test"]), 3)

    def test_conflicting_duplicate_labels_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            runner.prepare_task(
                {"task": "example", "input_columns": ["text"]},
                {"train": [], "test": [{"text": "same", "answer": "yes"},
                                      {"text": "same", "answer": "no"}]},
            )


if __name__ == "__main__":
    unittest.main()
