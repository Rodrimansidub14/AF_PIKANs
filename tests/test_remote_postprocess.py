import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


class ResultPostprocessTests(unittest.TestCase):
    def test_results_archive_keeps_offline_wandb_run_payload_and_hashes_it(self):
        from src.research_pipeline.remote.postprocess import create_results_archive

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_file = root / "runs/seed-101/wandb/offline-run-abc/run-abc.wandb"
            run_file.parent.mkdir(parents=True)
            payload = b"offline W&B history and artifact references"
            run_file.write_bytes(payload)
            (root / "analysis").mkdir()
            (root / "analysis/quantitative-signals.csv").write_text("signal,value\nrmse,0.1\n")
            (root / "analysis/artifact-hashes.json").write_text('{"stale":true}')
            archive_path = root / "heat-burgers-t4x2-results.zip"
            archive_path.write_bytes(b"older archive must not archive itself")

            create_results_archive(root, root / "analysis", archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                self.assertIn("runs/seed-101/wandb/offline-run-abc/run-abc.wandb", names)
                self.assertIn("analysis/quantitative-signals.csv", names)
                self.assertNotIn("heat-burgers-t4x2-results.zip", names)
                hashes = json.loads(archive.read("analysis/artifact-hashes.json"))
                self.assertEqual(
                    hashes["runs/seed-101/wandb/offline-run-abc/run-abc.wandb"],
                    hashlib.sha256(payload).hexdigest(),
                )

    def test_result_archive_extraction_rejects_path_traversal(self):
        from src.research_pipeline.remote.postprocess import extract_results_archive

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "remote-output.zip"
            destination = root / "extracted"
            escaped = root / "outside.txt"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("analysis/safe.txt", "safe")
                archive.writestr("../outside.txt", "outside")

            with self.assertRaises(ValueError):
                extract_results_archive(archive_path, destination)

            self.assertFalse(escaped.exists())
            self.assertFalse(destination.exists())

    def test_result_hash_verification_rejects_modified_measured_file(self):
        from src.research_pipeline.remote.postprocess import verify_results_hashes

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            analysis.mkdir()
            (root / "analysis/quantitative-signals.csv").write_text("signal,value\nrmse,0.1\n")
            (analysis / "artifact-hashes.json").write_text(
                json.dumps({"analysis/quantitative-signals.csv": "0" * 64})
            )

            with self.assertRaises(ValueError):
                verify_results_hashes(root)

    def test_offline_wandb_runs_sync_with_memory_only_credentials(self):
        from types import SimpleNamespace
        from src.research_pipeline.remote.postprocess import sync_offline_wandb

        with tempfile.TemporaryDirectory() as temporary:
            seed_root = Path(temporary) / "seed-101"
            run_dir = seed_root / "wandb/wandb/offline-run-abc"
            run_dir.mkdir(parents=True)
            (run_dir / "run-abc.wandb").write_bytes(b"offline data")
            for subdir in ("wandb", "wandb-cache", "wandb-config", "wandb-data"):
                (seed_root / subdir).mkdir(exist_ok=True)
            api_key = "secret-must-not-be-written-or-printed"
            environment = {
                "WANDB_API_KEY": api_key,
                "WANDB_ENTITY": "man22611-uvg",
                "WANDB_PROJECT": "AF_PIKANS",
            }
            with patch.dict("os.environ", {"USERNAME": "runner"}), patch(
                "src.research_pipeline.remote.postprocess.subprocess.run",
                return_value=SimpleNamespace(returncode=0, stdout="synced", stderr=""),
            ) as run_command:
                result = sync_offline_wandb(temporary, environment, python_executable="python-test")

            self.assertEqual(result["status"], "synced")
            self.assertEqual(result["run_count"], 1)
            command, kwargs = run_command.call_args
            self.assertEqual(command[0], [
                "python-test", "-m", "wandb", "sync", "--yes",
                "--entity", "man22611-uvg", "--project", "AF_PIKANS", str(run_dir),
            ])
            self.assertEqual(kwargs["env"]["WANDB_API_KEY"], api_key)
            self.assertEqual(kwargs["env"]["USERNAME"], "runner")
            self.assertEqual(kwargs["env"]["WANDB_DATA_DIR"], str(seed_root / "wandb-data"))
            self.assertNotIn("KAGGLE_API_TOKEN", kwargs["env"])
            self.assertNotIn(api_key, json.dumps(result))

    def test_offline_wandb_sync_defers_without_api_key(self):
        from src.research_pipeline.remote.postprocess import sync_offline_wandb

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "seed-101/wandb/wandb/offline-run-abc"
            run_dir.mkdir(parents=True)
            (run_dir / "run-abc.wandb").write_bytes(b"offline data")
            with patch("src.research_pipeline.remote.postprocess.subprocess.run") as run_command:
                result = sync_offline_wandb(temporary, {"WANDB_ENTITY": "man22611-uvg"})

            self.assertEqual(result, {"status": "deferred", "reason": "missing_api_key", "run_count": 1})
            run_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
