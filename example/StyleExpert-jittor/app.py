import tempfile
from pathlib import Path

import gradio as gr

from infer_core import inference

BASE_DIR = Path(__file__).resolve().parent
CONFIG_CHOICES = {
    "MoE (config_moe.yaml)": str(BASE_DIR / "configs" / "config_moe.yaml"),
}


def run_demo(content_image, style_image, config_name, seed):
    if content_image is None or style_image is None:
        return None
    config_path = CONFIG_CHOICES[config_name]
    with tempfile.TemporaryDirectory() as tmpdir:
        content_path = Path(tmpdir) / "content.png"
        style_path = Path(tmpdir) / "style.png"
        content_image.save(content_path)
        style_image.save(style_path)
        return inference(str(content_path), str(style_path), config_path, seed=seed)


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# StyleExpert Inference Demo")
        with gr.Row():
            content = gr.Image(label="Content Image", type="pil")
            style = gr.Image(label="Style Image", type="pil")
        with gr.Row():
            config = gr.Dropdown(
                choices=list(CONFIG_CHOICES.keys()),
                value="MoE (config_moe.yaml)",
                label="Config",
            )
            seed = gr.Slider(0, 9999, value=42, step=1, label="Seed")
        run_btn = gr.Button("Run")
        output = gr.Image(label="Output")

        run_btn.click(run_demo, inputs=[content, style, config, seed], outputs=output)
    return demo


if __name__ == "__main__":
    build_ui().launch()
