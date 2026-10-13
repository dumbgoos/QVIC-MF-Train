"""Regression for representation/structure summaries and recovery without GPU runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_htm_event_ablation import variant_settings


def main():
    script = Path(__file__).with_name("check_htm_event_ablation.py")
    with tempfile.TemporaryDirectory() as temp:
        for ablation in ("representation", "structure"):
            out = Path(temp) / ablation
            out.mkdir()
            key = "abcdefghijk_1"
            (out / "selection.json").write_text(json.dumps({"seed": 42, "selected": [{"key": key, "answer": "A"}]}))
            for name, (representation, budget, layout) in variant_settings(ablation).items():
                group = out / name
                group.mkdir()
                merges = 0 if layout == "frames" else 2
                report = {"passed": True, "exit_code": 0, "checkpoint": "/example/checkpoint-10", "frames": 64,
                          "feedback": "on", "event_representation": representation, "memory_tokens": budget,
                          "memory_layout": layout, "relevance_memory_entries": 256,
                          "score": {"n": 1, "correct": 1, "overall_pct": 100.0}}
                (group / "report.json").write_text(json.dumps(report))
                (group / "predictions.json").write_text(json.dumps({"key": key, "pred": "A", "gt": "A"}) + "\n")
                (group / "inference.log").write_text(f"[HTM τ] key={key} merge_rate=0.5000 n_merges={merges}/64 |M|=64 L={budget} tau_s=1.0000 tau_d=2.0000 ready=True\n")
            command = [sys.executable, str(script), "--outdir", str(out), "--ablation", ablation, "--summarize-only"]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            combined = json.loads((out / "ablation_report.json").read_text())
            assert combined["complete"] and combined["merge_counts_equal"]
            assert set(combined["summary"]) == set(variant_settings(ablation))
            assert set(combined["per_question"]) == {key}
            print(f"PASS {ablation} summary recovery without model or source dataset")
            first = next(iter(variant_settings(ablation)))
            report_path = out / first / "report.json"
            report = json.loads(report_path.read_text())
            report["passed"] = False
            report_path.write_text(json.dumps(report))
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 2 and "did not pass" in result.stdout
        print("PASS incomplete inference results rejected")


if __name__ == "__main__":
    main()
