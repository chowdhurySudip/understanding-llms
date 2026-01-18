# Understanding LLMs

A hands-on guide to understanding Large Language Models by implementing every building block from scratch. This project implements a complete transformer-based language model including tokenization, model architecture, training, and text generation - all from first principles.

## 🎯 Project Overview

This repository provides educational implementations of:
- **Byte-Pair Encoding (BPE) Tokenizer** with multiprocessing support
- **Transformer Architecture** with modern components (RoPE, RMSNorm, SwiGLU)
- **Training Pipeline** with AdamW optimizer and learning rate scheduling
- **Text Generation** with nucleus (top-p) sampling

All implementations are done from scratch using PyTorch, with detailed documentation to help understand how LLMs work under the hood.

## 📁 Project Structure

```
understanding-llms/
├── src/
│   ├── llm_intro/
│   │   ├── tokenizers/          # BPE tokenization
│   │   │   ├── bpe.py           # BPE training implementation
│   │   │   ├── encoder.py       # Tokenizer encode/decode
│   │   │   └── create_chunks.py # Multiprocessing utilities
│   │   ├── model/               # Transformer model
│   │   │   ├── transformer.py   # Main model classes
│   │   │   ├── attention.py     # Multi-head attention + RoPE
│   │   │   └── layers.py        # Building blocks (Linear, SwiGLU, RMSNorm)
│   │   ├── training/            # Training utilities
│   │   │   ├── trainer.py       # Main training loop
│   │   │   ├── data.py          # Data loading
│   │   │   └── optimizer.py     # AdamW optimizer
│   │   ├── utils/               # Helper utilities
│   │   │   ├── checkpointing.py # Model checkpointing
│   │   │   └── loss.py          # Loss functions
│   │   └── generation.py        # Text generation
│   └── configs/
│       └── config_llm_intro.yaml # Training configuration
├── scripts/
│   └── download_data.py         # Data download utility
├── data/                        # Training datasets
├── artifacts/                   # Model outputs
└── pyproject.toml              # Project dependencies
```

## 🚀 Quick Start

### 1. Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

**Dependencies:**
- torch >= 2.0.0
- numpy >= 1.24.0
- pyyaml >= 6.0
- regex >= 2023.0.0
- tqdm >= 4.65.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

### 2. Download Data

Download the training datasets (TinyStories and OpenWebText samples):

```bash
uv run python scripts/download_data.py
```

This downloads:
- `TinyStoriesV2-GPT4-train.txt` and `TinyStoriesV2-GPT4-valid.txt`
- `owt_train.txt` and `owt_valid.txt`

### 3. Train BPE Tokenizer

Train a Byte-Pair Encoding tokenizer on your dataset:

```bash
uv run python -m src.llm_intro.tokenizers.bpe
```

**What this does:**
- Trains a BPE tokenizer with 10,000 vocabulary size
- Uses multiprocessing for efficient training on large datasets
- Saves vocabulary to `artifacts/llm_intro/tokenizer/TinyStoriesV2-GPT4-train-vocab.json`
- Saves merges to `artifacts/llm_intro/tokenizer/TinyStoriesV2-GPT4-train-merges.txt`

**Configuration:**
- Default input: `data/TinyStoriesV2-GPT4-train.txt`
- Vocab size: 10,000 (256 base bytes + special tokens + learned merges)
- Special tokens: `<|endoftext|>`
- Processes: 10 (for parallel processing)

### 4. Prepare Training Data

Encode your text data using the trained tokenizer. The `encoder.py` file includes a memory-efficient `tokenize_file_to_npy` function that uses streaming and memory-mapped files:

```bash
uv run python -m src.llm_intro.tokenizers.encoder
```

**What this does:**
- Tokenizes the training text file using the trained BPE tokenizer
- Uses streaming to handle large files without loading into memory
- Saves encoded tokens to `artifacts/llm_intro/data/TinyStoriesV2-GPT4-train-encoded.npy`
- Outputs progress and timing information

**For custom files:**

```python
from llm_intro.tokenizers.encoder import tokenize_file_to_npy

tokenize_file_to_npy(
    input_path="data/your-file.txt",
    vocab_path="artifacts/llm_intro/tokenizer/TinyStoriesV2-GPT4-train-vocab.json",
    merges_path="artifacts/llm_intro/tokenizer/TinyStoriesV2-GPT4-train-merges.txt",
    output_npy_path="artifacts/llm_intro/data/your-file-encoded.npy",
    special_tokens=["<|endoftext|>"]
)
```

**Note:** Repeat this process for validation data with the validation text file.

### 5. Train the Model

Configure training parameters in `src/configs/config_llm_intro.yaml` and run:

```bash
uv run python -m src.llm_intro.training.trainer --config src/configs/config_llm_intro.yaml
```

**Key hyperparameters:**
- Model: 64-dim embeddings, 2 layers, 2 heads
- Training: Batch size 8, context length 256
- Optimizer: AdamW with learning rate 6e-4
- Scheduler: Warmup + cosine annealing

### 6. Generate Text

Generate text using a trained model checkpoint:

```bash
uv run python -m src.llm_intro.generation --config src/configs/config_llm_intro.yaml
```

**Generation features:**
- Nucleus (top-p) sampling
- Temperature control
- Customizable prompts

## 🧠 Technical Details

### Tokenization: Byte-Pair Encoding (BPE)

The BPE implementation follows the original algorithm:

1. **Pre-tokenization:** Split text using regex pattern into base units
2. **Byte encoding:** Convert to UTF-8 bytes (256 base tokens)
3. **Iterative merging:** Repeatedly merge most frequent byte pairs
4. **Multiprocessing:** Parallel processing for large datasets

**Key features:**
- Special token handling (`<|endoftext|>`)
- Deterministic merging with tie-breaking
- Efficient pair counting and updates
- Vocabulary serialization (JSON format)

**Encoder (`encoder.py`):**
- **Fast encoding/decoding** with merge rank lookup
- **Caching mechanism** to avoid re-calculating BPE for repeated tokens
- **Special token pattern matching** with proper handling of overlaps
- **Streaming tokenization** for memory-efficient processing of large files
- **Memory-mapped output** via `tokenize_file_to_npy` function
- **Iterable encoding** with `encode_iterable` for chunk-by-chunk processing

### Model Architecture: Transformer Language Model

Modern decoder-only transformer with:

**Components:**
- **Embeddings:** Token embeddings (no learned positional embeddings)
- **RoPE:** Rotary Position Embeddings for positional information
- **Multi-head Self-Attention:** Scaled dot-product attention with causal masking
- **RMSNorm:** Root Mean Square Layer Normalization
- **SwiGLU:** Gated Linear Unit with SiLU activation
- **Pre-norm architecture:** LayerNorm before each sub-layer

**Architecture flow:**
```
Input Tokens
    ↓
Token Embeddings
    ↓
[TransformerBlock × N]
    ├─ RMSNorm → Multi-Head Attention (+ RoPE) → Residual
    └─ RMSNorm → SwiGLU FFN → Residual
    ↓
RMSNorm
    ↓
Linear (to vocab)
```

### Training

**Optimizer:** AdamW with:
- Weight decay: 0.01
- Betas: (0.9, 0.999)
- Gradient clipping: max norm 1.0

**Learning Rate Schedule:**
1. **Warmup phase** (10% of iterations): Linear increase from 0 to max_lr
2. **Cosine annealing** (90% of iterations): Cosine decay from max_lr to min_lr

**Data loading:**
- Memory-mapped numpy arrays for efficient large dataset handling
- Random batch sampling with context windows

### Text Generation

**Sampling strategies:**
- **Nucleus (top-p) sampling:** Sample from smallest set of tokens with cumulative probability ≥ p
- **Temperature:** Control randomness (higher = more random, lower = more deterministic)

## 📊 Configuration

Edit `src/configs/config_llm_intro.yaml` to customize:

```yaml
data:
  train_path: "artifacts/llm_intro/data/TinyStoriesV2-GPT4-train-encoded.npy"
  val_path: "artifacts/llm_intro/data/TinyStoriesV2-GPT4-valid-encoded.npy"

training:
  batch_size: 8
  context_length: 256
  device: "mps"  # or "cuda", "cpu"
  max_iteration: 100

model:
  vocab_size: 10000
  d_model: 64
  num_layers: 2
  num_heads: 2
  rope_theta: 10000

optimizer:
  max_lr: 0.0006
  min_lr: 0.00006
  weight_decay: 0.01

generation:
  temperature: 0.8
  top_p: 0.9
  max_tokens: 50
```

## 🎓 Learning Resources

This implementation follows modern LLM architectures like:
- **GPT:** Decoder-only transformer
- **LLaMA:** RoPE, RMSNorm, SwiGLU
- **Transformer:** Attention mechanism, residual connections

Key papers:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
- "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)

## 📝 License

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

This is an educational project. Feel free to fork and experiment with different architectures and training strategies!

## 🔍 Repository

**Repository:** [understanding-llms](https://github.com/chowdhurySudip/understanding-llms)  
**Owner:** chowdhurySudip  
**Branch:** main
