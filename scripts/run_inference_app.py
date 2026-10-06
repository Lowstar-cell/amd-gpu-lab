import streamlit as st
import torch
import sys
import os
from pathlib import Path
import json

# Ensure repo root is on path so we can import model.py and dataset_utils.py
repo_root = Path(__file__).parents[1] # Go up two levels from scripts/run_inference_app.py to the root
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model import LSTMClassifier, Encoder
from dataset_utils import AMDLogDataset # Needed for vocab if not in checkpoint

def load_ckpt(path: str):
    ck = torch.load(path, map_location="cpu")
    return ck

def fix_state_dict(sd: dict):
    new = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("module."):
            nk = k[len("module."):]
        new[nk] = v
    return new

def parse_sample(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [int(p) for p in parts]

def get_model_and_vocab(checkpoint_path, vocab_size_fallback=1000, vocab_size_override=None):
    """Loads model and vocab from a checkpoint."""
    ck = torch.load(checkpoint_path, map_location="cpu")
    
    vocab = None
    vocab_size = vocab_size_fallback

    if vocab_size_override:
        vocab_size = vocab_size_override
        st.write(f"Using user-provided vocab size: {vocab_size}")
    elif isinstance(ck, dict) and 'vocab' in ck:
        try:
            vocab = ck['vocab']
            if isinstance(vocab, dict):
                vocab_size = len(vocab)
            else:
                vocab_size = int(vocab)
            st.write(f"Inferred vocab size from checkpoint: {vocab_size}")
        except Exception:
            vocab = None # Reset vocab if parsing fails
            st.write(f"Using fallback vocab size: {vocab_size}")
    else:
        st.write(f"Using fallback vocab size: {vocab_size}")

    # Determine model type
    keys = set(ck.keys()) if isinstance(ck, dict) else set()
    model_type = 'auto'
    
    if 'model_state_dict' in keys or 'model_state' in keys:
        model_type = 'classifier'
    elif 'encoder_state_dict' in keys or 'encoder_state' in keys or ('encoder_state_dict' in keys and 'proj_state_dict' in keys):
        model_type = 'encoder'
    else:
        if any(k.startswith('embedding.') or 'embedding.weight' in k for k in keys):
            model_type = 'classifier'
        else:
            model_type = 'classifier'

    st.write(f"Detected model type: {model_type}")

    model = None
    if model_type == 'classifier':
        model = LSTMClassifier(vocab_size=vocab_size)
        sd = ck.get('model_state_dict', ck.get('model_state', ck))
    elif model_type == 'encoder':
        model = Encoder(vocab_size=vocab_size)
        sd = ck.get('encoder_state_dict', ck)

    if model:
        sd = fix_state_dict(sd)
        try:
            model.load_state_dict(sd)
        except Exception as e:
            st.warning(f"Failed to load state_dict directly: {e}. Attempting partial load.")
            model_state = model.state_dict()
            new_state = {k: v for k, v in sd.items() if k in model_state}
            model_state.update(new_state)
            model.load_state_dict(model_state)
        model.eval()
    
    return model, vocab, vocab_size


st.set_page_config(page_title="AMD GPU Model Inference", layout="wide")
st.title("AMD GPU Event Sequence Model Inference")

st.markdown("""
Upload your trained PyTorch model (`.pt` file) and provide a comma-separated sequence
of GPU event token IDs to get a prediction from the model.
""")

# Model Upload
st.subheader("1. Upload Trained Model")
uploaded_file = st.file_uploader("Choose a .pt model file", type="pt")

# Vocab size override
st.subheader("Optional: Override Vocabulary Size")
vocab_size_override = st.number_input(
    "If your model's vocab size is not in the checkpoint, specify it here:",
    min_value=1,
    value=1000,
    step=1,
    help="Set this to the vocabulary size of your model if you get 'index out of range' errors."
)

model = None
vocab = None
vocab_size = vocab_size_override

if uploaded_file is not None:
    # Save uploaded file to a temporary location to load with torch.load
    with open("temp_model.pt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        model, vocab, vocab_size = get_model_and_vocab("temp_model.pt", vocab_size_fallback=1000, vocab_size_override=vocab_size_override)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    finally:
        if Path("temp_model.pt").exists():
            os.remove("temp_model.pt")

if model is not None:
    st.subheader("2. Enter GPU Event Token IDs")
    # Example input adjusted for the 'GPU data' clarification
    sample_input = st.text_area(
        "Enter a comma-separated sequence of integer token IDs (e.g., 1,5,12,3,7):",
        value="1,2,3,4", # Default example
        height=100
    )

    st.subheader("3. Run Inference")
    if st.button("Get Prediction"):
        if sample_input:
            try:
                sample_ids = parse_sample(sample_input)
                
                # Validate token IDs against vocab size
                invalid_ids = [id for id in sample_ids if id >= vocab_size]
                if invalid_ids:
                    st.error(f"Error: The following token IDs are out of range for vocab size {vocab_size}: {invalid_ids}")
                elif not sample_ids:
                    st.warning("Please enter some token IDs.")
                else:
                    # Prepare input tensor
                    x = torch.tensor([sample_ids], dtype=torch.long, device='cpu')

                    with torch.no_grad():
                        prediction = model(x)
                    
                    st.success("Prediction Result:")
                    st.write(prediction.cpu().numpy())

                    if model.__class__.__name__ == 'LSTMClassifier':
                        # For binary classifier, often useful to show sigmoid output
                        st.markdown(f"**Sigmoid Output (Probability):** `{torch.sigmoid(prediction).item():.4f}`")
                        if torch.sigmoid(prediction).item() > 0.5:
                            st.write("Interpretation: Model predicts positive class.")
                        else:
                            st.write("Interpretation: Model predicts negative class.")
                    elif model.__class__.__name__ == 'Encoder':
                        st.write("Interpretation: This is an embedding/feature vector from the encoder.")

            except ValueError:
                st.error("Invalid input. Please ensure all IDs are integers separated by commas.")
            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
        else:
            st.warning("Please enter a sequence of GPU event token IDs.")
else:
    st.info("Please upload a model file to proceed with inference.")
