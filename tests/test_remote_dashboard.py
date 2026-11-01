import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class RunMonitorTests(unittest.TestCase):
    def test_monitor_reads_configured_campaign_bundle_and_wandb_sync(self):
        from src.research_pipeline.remote import dashboard

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            bundle = campaign / "bundle-final"
            bundle.mkdir(parents=True)
            coordinator = root / "run_campaign.py"
            (bundle / "manifest.json").write_text(json.dumps({
                "training": {"seeds": [101], "timeout_sec_per_seed": 2100},
                "problem": {"sequence": ["heat", "burgers"]},
                "tracking": {"mode": "offline"},
                "remote": {"setup_timeout_sec": 900},
            }))
            (bundle / "submission.json").write_text(json.dumps({
                "kernel": "researcher/heat-burgers-current", "status": "submitted",
            }))
            runs = campaign / "download-extracted" / "runs"
            seed = runs / "seed-101"
            seed.mkdir(parents=True)
            (seed / "run_summary.json").write_text(json.dumps({
                "seed": 101,
                "result": {"status": "completed", "elapsed_sec": 120,
                            "metrics": {"heat/field_relative_l2": 0.0001}},
            }))
            (campaign / "wandb-sync.json").write_text(json.dumps({
                "status": "synced", "run_count": 2, "synced_count": 2,
            }))
            env = {
                "AF_PIKAN_CAMPAIGN": str(campaign),
                "AF_PIKAN_BUNDLE": str(bundle),
                "AF_PIKAN_STATUS_SCRIPT": str(coordinator),
            }
            response = SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"provider_status": "KernelWorkerStatus.COMPLETE"}),
                stderr="",
            )
            with patch.object(dashboard, "ROOT", root), \
                 patch.dict("os.environ", env, clear=False), \
                 patch.object(dashboard, "_CACHE", {"at": 0.0, "value": None}), \
                 patch.object(dashboard.subprocess, "run", return_value=response) as run_command:
                result = dashboard.snapshot()

            self.assertEqual(result["provider_state"], "KernelWorkerStatus.COMPLETE")
            self.assertEqual(result["runs"][0]["metrics"]["heat/field_relative_l2"], 0.0001)
            self.assertEqual(result["wandb_sync_status"], "synced")
            self.assertEqual(result["submission_state"], "submitted")
            self.assertEqual(run_command.call_args.args[0][1], str(coordinator))


if __name__ == "__main__":
    unittest.main()
