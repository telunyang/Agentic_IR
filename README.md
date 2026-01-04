[![DOI](https://zenodo.org/badge/1067618260.svg)](https://doi.org/10.5281/zenodo.17240909)

# AgentKR
- temp

---

## Conda Installation
- [Anaconda - Download Now](https://www.anaconda.com/download/success)
```bash
conda create -n agentic_ir python=3.11
conda activate agentic_ir
```

---

## AG2
- [https://docs.ag2.ai/docs/home/home](https://docs.ag2.ai/docs/home/home)

### Packages
```bash
pip install -U ag2[openai,gemini,ollama]
```

---

## Package Installation
```bash
pip install -r requirements.txt
```
Note: we recommend using `cuda 12.1` for better performance with PyTorch and Transformers.
```bash
# $ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

---


## Google Gemini API
- [Google AI Studio - API keys](https://aistudio.google.com/api-keys)
Note: 
- In this project, we use `GOOGLE_API_KEY_01/02/03/04/05` as the environment variable name for Gemini API. Please see `.env-example` for more details and copy it to `.env`.
- After applying for the API, store the API Key in th `.env` file, named as `GOOGLE_API_KEY_{num}=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

---

## ollama

### Installation
- [Ollama GitHub](https://github.com/ollama/ollama)
```bash
# ubuntu
curl -fsSL https://ollama.com/install.sh | sh

# windows
# URL: https://ollama.com/download
```

### Optional: set ollama models path

#### Edit ollama.service
```bash
sudo vi /etc/systemd/system/ollama.service
```

#### Set ollama models path
```
[Service]
Environment="OLLAMA_MODELS=/your-path/ollama_models"
```
Note: You can use default path or set your own path.

#### Set keepalive
```
[Service]
Environment="OLLAMA_KEEP_ALIVE=-1"
```

#### Set Host IP and Port
```
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
```

#### Restart ollama
```bash
# Reload system services
sudo systemctl daemon-reload
sudo systemctl restart ollama.service

# Check ollama status
sudo systemctl status ollama.service
```

### Download ollama models
- [Ollama Search - Models](https://ollama.com/search)
```bash
# e.g.
ollama pull mistral-small3.1:24b
ollama pull qwen3:32b
ollama pull llama3.3:70b
ollama pull llama4:16x17b
ollama pull gpt-oss:20b
ollama pull gpt-oss:120b
```
Note: 
- You can change the model names based on your needs.
- The ollama version often updates, please refer to the [Ollama Docs](https://ollama.com/docs) for more details. Otherwise, you may encounter issues when running the ollama-related code or inappropriate model loading.

---

## Playwright Installation
```bash
# All browsers installation
playwright install

# or install specific browsers
playwright install chromium
playwright install firefox
playwright install webkit
```

---

## Customa
- [Custom Search JSON API](https://developers.google.com/custom-search/v1/overview)
Note: You need to create a Custom Search Engine and get the API key and Search Engine ID (CX).
- After applying for the API, store the API Key and CX in th `.env` file, named as `SEARCH_API_KEY=XXXXXXXXXXXXXXXXXXXXXX` and `SEARCH_ENGINE_ID=XXXXXXXXXXXXXXXXXXXXXX`

---

## How to run our pipeline

### 1. Launch web api for reranking service
```bash
python web_api_rerank.py
```
Note: Before you run the `run.py` script, please make sure the reranking web api service is running.

### 2. Launch the main pipeline
```bash
python run.py
```