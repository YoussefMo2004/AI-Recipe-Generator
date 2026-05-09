# AI Recipe Generator

A small Streamlit web app that generates recipe ideas from user-provided ingredients using the Hugging Face model `google/flan-t5-base` and the `transformers` library.

Quick start

1. Create and activate a Python environment (recommended).

```bash
# create venv
python -m venv .venv
# activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# or cmd: .\.venv\Scripts\activate.bat
```

2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run streamlit_app.py
```

Notes
- The first run will download the `google/flan-t5-base` model (~1GB). Ensure you have enough disk space and a decent network connection.
- If you have a GPU and `torch` is installed with CUDA support, the app will use it automatically.
- If you prefer to call the Hugging Face Inference API instead of running locally, modify the generator code to use `huggingface_hub.InferenceApi` and set `HF_API_TOKEN` environment variable.# AI-Recipe-Generator