"""MediSyn research demo: only measured results and trained checkpoints."""
import json
import os
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
import torch
from src.gan.inference import load_generator, generate_images, png_grid

st.set_page_config(page_title="MediSyn | AURORA", layout="wide")


def main():
    st.title("MediSyn | AURORA")
    st.caption("Synthetic chest X-ray augmentation research")
    st.warning("Research prototype. Generated images are synthetic and are not validated for diagnosis or clinical use.")
    page = st.sidebar.radio("Navigation", ["Overview", "Generator", "Results", "Study workflow"])
    if page == "Overview":
        st.markdown("AURORA investigates whether synthetic TB-manifestation-labelled training images improve a classifier under controlled data scarcity.")
        st.markdown("Generators learn from real training images. Synthetic output does not by itself guarantee privacy or preserve diagnostic findings.")
        st.info("The current NLM research protocol has unresolved duplicate review. Earlier mixed-source smoke results are not final paper evidence.")
        st.markdown("[Project repository](https://github.com/xt67/medimage-synth)")
    elif page == "Generator":
        checkpoint = Path(os.environ.get("AURORA_CHECKPOINT", str(ROOT / "outputs/gan/models/generator_latest.pth")))
        if not checkpoint.is_file():
            st.info("No trained checkpoint is installed. Generation will be available after training and model review.")
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, config = load_generator(checkpoint, device)
        except (ValueError, RuntimeError, KeyError, OSError, EOFError, pickle.UnpicklingError) as exc:
            st.error(f"Unable to load the configured model: {exc}")
            return
        count = st.slider("Synthetic images", 1, 8, 4)
        seed = st.number_input("Seed", min_value=0, max_value=2147483647, value=42)
        if st.button("Generate synthetic images"):
            images = generate_images(model, count, config["z_dim"], int(seed))
            st.session_state["png"] = png_grid(images)
        if "png" in st.session_state:
            st.image(st.session_state["png"], caption="Synthetic research images", use_column_width=True)
            st.download_button("Download synthetic PNG grid", st.session_state["png"],
                               file_name="aurora_synthetic_grid.png", mime="image/png")
    elif page == "Study workflow":
        st.subheader("From reviewed data to measured results")
        st.markdown("1. Review candidate duplicates and label conflicts locally.\n2. Approve a canonical dataset and frozen split.\n3. Approve the experiment protocol and first training run.\n4. Run matched comparisons and inspect measured results.")
        st.info("Publication training is blocked until human review and researcher approval are recorded. This public demo does not accept medical-image uploads or start training jobs.")
        st.markdown("[Review workflow](https://github.com/xt67/medimage-synth/blob/aurora-refinement/docs/HUMAN_REVIEW.md) · [Experiment protocol](https://github.com/xt67/medimage-synth/blob/aurora-refinement/docs/EXPERIMENT_PROTOCOL.md)")
    else:
        path = Path(os.environ.get("AURORA_RESULTS", str(ROOT / "outputs/paper_results.json")))
        if not path.is_file():
            st.info("No verified experiment results have been published. No performance improvement is claimed.")
            return
        try:
            report = json.loads(path.read_text())
            if report.get("status") != "verified" or not report.get("protocol") or not report.get("experiments"):
                raise ValueError("Results require verified status, protocol, and measured experiments")
        except (ValueError, OSError, AttributeError) as exc:
            st.error(f"Results cannot be displayed: {exc}")
            return
        st.caption(f"Protocol: {report['protocol']}")
        st.json(report)


if __name__ == "__main__":
    main()
