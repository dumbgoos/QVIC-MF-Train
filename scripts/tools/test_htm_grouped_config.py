"""Check real HTM initialization/config reload and grouped decoder memory on CPU."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from htm_diagnostic_utils import load_htm


def main():
    root = Path(__file__).resolve().parents[2]
    htm = load_htm()
    tree = ast.parse((root / "qvic/model/qvic_meta_mixin.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QViCMetaMixin")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_htm")
    namespace = {"HTMPredictor": htm.HTMPredictor, "HTMMemoryController": htm.HTMMemoryController, "rank0_print": lambda *a: None}
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(root / "qvic/model/qvic_meta_mixin.py"), "exec"), namespace)

    class Model:
        use_htm = False
        compress_with_relevance = True
        context_memory_length = 256
        context_embed_tokens = 4

        def __init__(self, config):
            self.config = config

        def get_model(self):
            return types.SimpleNamespace(config=self.config)

        def set_use_htm(self, value):
            self.use_htm = value

        def set_htm_stage_a(self, value):
            self.htm_stage_a = value

    Model.initialize_htm = namespace["initialize_htm"]
    config = types.SimpleNamespace(hidden_size=8, context_embed_tokens=4, context_memory_length=256)
    model = Model(config)
    model.initialize_htm(hidden_size=4, token_budget=64, event_representation="tokens", memory_layout="chronological", group_tokens=True, reset_thresholds_per_video=True)
    saved_config = json.loads(json.dumps(vars(config)))
    restored = Model(types.SimpleNamespace(**saved_config))
    restored.initialize_htm(hidden_size=saved_config["htm_hidden_size"], recent_window_W=saved_config["htm_recent_window"], quantile=saved_config["htm_quantile"])
    restored.htm_predictor.load_state_dict(model.htm_predictor.state_dict(), strict=True)
    assert restored.htm_reset_thresholds_per_video

    mm = types.ModuleType("qvic.mm_utils")
    mm.get_model_name_from_path = lambda p: p
    utils = types.ModuleType("qvic.utils")
    utils.rank0_print = lambda *a: None
    sys.modules["qvic.mm_utils"] = mm
    sys.modules["qvic.utils"] = utils
    spec = importlib.util.spec_from_file_location("grouped_eval", root / "qvic/eval/htm_eval_utils.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    restored.htm_controller.L = 256  # Evaluator sets relevance/frame capacity first.
    assert runtime.apply_htm_eval_runtime(restored, context_memory_length=256)
    ctrl = restored.htm_controller
    assert ctrl.L == 64 and ctrl.group_tokens and ctrl.event_representation == "tokens"
    assert ctrl.memory_layout == "chronological"
    ctrl.reset(1, reset_thresholds=True)
    features = torch.randn(1, 1, 4, 8, requires_grad=True)
    for _ in range(6):
        ctrl.write_batch(features, restored.htm_predictor, True, retain_memory_grad=True)
    memory = ctrl.materialize_context_memory(features)
    assert memory.shape[2] == 4 and memory.shape[1] * 4 <= 64
    assert torch.autograd.grad(memory.square().sum(), features)[0].norm() > 0

    # Exercise the actual grid newline method, without VLM dependency imports.
    arch = ast.parse((root / "qvic/model/llava_arch.py").read_text())
    grid = next(n for n in ast.walk(arch) if isinstance(n, ast.FunctionDef) and n.name == "add_token_per_grid")
    import math
    ns = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[grid], type_ignores=[]), "llava_arch.py", "exec"), ns)
    target = types.SimpleNamespace(model=types.SimpleNamespace(image_newline=torch.zeros(8)), config=types.SimpleNamespace(add_faster_video=False))
    decoded = ns["add_token_per_grid"](target, memory[0])
    assert decoded.shape == (memory.shape[1] * 6, 8)  # C=4: 2x2 grid plus 2 newline tokens.
    print("PASS saved config restores predictor, budget, token grouping and per-video reset")
    print("PASS grouped memory gradients and native decoder grid newline count")


if __name__ == "__main__":
    main()
