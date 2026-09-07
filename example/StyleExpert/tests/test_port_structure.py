import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_runtime_is_loaded_before_model_stack():
    source = (ROOT / "infer_core.py").read_text(encoding="utf-8")
    runtime_pos = source.index("from jittor_runtime import")
    transformers_pos = source.index("from transformers import")
    assert runtime_pos < transformers_pos


def test_no_native_torch_dependency():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "\ntorch==" not in requirements
    installer = (ROOT / "install.py").read_text(encoding="utf-8")
    assert "JittorRepos/jtorch" in installer
    assert 'pip_install("--no-deps"' in installer
    assert (ROOT / "compat" / "torch_stub" / "pyproject.toml").is_file()


def test_official_moe_config_is_shipped():
    text = (ROOT / "configs" / "config_moe.yaml").read_text(encoding="utf-8")
    assert 'train_modules_csv: "configs/lora_modules.csv"' in text
    assert "num_experts: 16" in text
    assert (ROOT / "configs" / "lora_modules.csv").is_file()


def test_sdpa_fallback_is_registered():
    source = (ROOT / "jittor_runtime.py").read_text(encoding="utf-8")
    assert 'functional.scaled_dot_product_attention = scaled_dot_product_attention' in source


def test_device_selection_respects_jittor_flag():
    source = (ROOT / "infer_core.py").read_text(encoding="utf-8")
    assert '"cuda" if jt.flags.use_cuda else "cpu"' in source


def test_jtorch_semantic_compatibility_patches_are_present():
    source = (ROOT / "jittor_runtime.py").read_text(encoding="utf-8")
    assert "torch.equal = tensors_equal" in source
    assert "torch.is_grad_enabled = lambda: False" in source
    assert "generator_cls.device = property" in source


def test_current_upstream_interfaces_are_preserved():
    source = (ROOT / "infer_core.py").read_text(encoding="utf-8")
    assert "def run_inference_with_bundle(" in source
    assert "def get_pipeline_bundle(" in source
    assert 'parser.add_argument("--prompt"' in (ROOT / "infer.py").read_text(encoding="utf-8")


def test_official_showcase_assets_are_included():
    assert (ROOT / "assets" / "figures" / "teaser.png").is_file()
    assert (ROOT / "assets" / "figures" / "compare.png").is_file()
