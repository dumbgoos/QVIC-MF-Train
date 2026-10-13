"""Verify auxiliary metrics do not change Stage A/B loss or gradient weights."""
from pathlib import Path
import ast
import textwrap
import types

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    source = (root / "qvic/model/language_model/llava_qwen.py").read_text()
    start = source.index("            # #@HTM — Stage A: L_pred")
    end = source.index("            return outputs", start) + len("            return outputs")
    body = textwrap.dedent(source[start:end])
    ns = {"torch": torch}
    exec("def apply_loss(self, outputs):\n" + textwrap.indent(body, "    "), ns)

    class Output(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value

    for stage_a, qa_weight in ((False, 1.0), (True, 0.0), (True, 0.25)):
        qa = torch.tensor(2., requires_grad=True)
        pred = torch.tensor(3., requires_grad=True)
        model = types.SimpleNamespace(training=True, use_htm=True, htm_stage_a=stage_a,
                                      htm_pred_loss=pred, htm_lambda_lm=qa_weight, htm_pred_weight=0.1,
                                      _htm_last_eval_stats={"final_M": 64, "merge_rate": 0.5})
        output = ns["apply_loss"](model, Output(loss=qa, logits=torch.zeros(1, 2)))
        pred_weight = 1.0 if stage_a else 0.1
        assert torch.allclose(output.loss, torch.tensor(2 * qa_weight + 3 * pred_weight))
        assert output["htm_qa_loss"] == 2 and output["htm_prediction_loss"] == 3
        assert not output["htm_qa_loss"].requires_grad
        assert output["htm_final_M"] == 64
        output.loss.backward()
        assert torch.allclose(pred.grad, torch.tensor(pred_weight))
        if qa_weight == 0:
            assert qa.grad is None
        else:
            assert torch.allclose(qa.grad, torch.tensor(qa_weight))
    print("PASS QA/prediction metrics preserve Stage A/B joint losses and gradients")
    trainer_tree = ast.parse((root / "qvic/train/qvic_trainer.py").read_text())
    trainer_cls = next(n for n in trainer_tree.body if isinstance(n, ast.ClassDef) and n.name == "QViCTrainer")
    methods = [n for n in trainer_cls.body if isinstance(n, ast.FunctionDef) and n.name in ("compute_loss", "log")]

    class ParentTrainer:
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return model.outputs.loss, model.outputs

        def log(self, logs):
            return logs

    cls = ast.ClassDef(name="MetricsTrainer", bases=[ast.Name(id="ParentTrainer", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"ParentTrainer": ParentTrainer}
    exec(compile(tree, "qvic_trainer.py", "exec"), namespace)
    trainer = namespace["MetricsTrainer"]()
    trainer._htm_metric_sums = {}
    trainer._htm_metric_counts = {}
    model = types.SimpleNamespace(training=True, outputs=output)
    for _ in range(2):
        trainer.compute_loss(model, {})
    logged = trainer.log({"loss": float(output.loss.detach())})
    assert logged["htm_qa_loss_local"] == 2 and logged["htm_prediction_loss_local"] == 3
    assert logged["htm_final_M_local"] == 64 and logged["htm_merge_rate_local"] == 0.5
    assert not trainer._htm_metric_sums and not trainer._htm_metric_counts
    print("PASS Trainer reports microbatch component means and clears logging accumulators")


if __name__ == "__main__":
    main()
