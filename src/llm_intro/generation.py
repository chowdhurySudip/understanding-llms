"""
Text generation module for Transformer Language Model.

This module provides functionality for:
- Loading trained transformer models from checkpoints
- Performing nucleus (top-p) sampling for text generation
- Generating text sequences with temperature control
"""

import torch
import yaml
from typing import Dict, Any, Optional
from llm_intro.model.transformer import TransformerLM
from llm_intro.model.layers import softmax
from llm_intro.tokenizers.encoder import Tokenizer


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_model(config: Dict[str, Any]) -> TransformerLM:
    """
    Initialize a Transformer Language Model from configuration.
    
    Args:
        config: Configuration dictionary containing model and training parameters.
                Expected keys:
                - model: dict with vocab_size, d_model, num_layers, num_heads, 
                        d_ff, rope_theta
                - training: dict with context_length, device
    
    Returns:
        Initialized TransformerLM model on the specified device.
    """
    model_config = config['model']
    training_config = config['training']
    
    model = TransformerLM(
        vocab_size=model_config['vocab_size'],
        context_length=training_config['context_length'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        d_ff=model_config['d_ff'],
        rope_theta=model_config['rope_theta'],
        device=training_config['device']
    )
    
    model = model.to(training_config['device'])
    print(f"Initialized model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def load_checkpoint(model: TransformerLM, checkpoint_path: str) -> TransformerLM:
    """
    Load model weights from a checkpoint file.
    
    Args:
        model: TransformerLM model instance to load weights into.
        checkpoint_path: Path to the checkpoint file (.pth).
    
    Returns:
        Model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def load_tokenizer(vocab_path: str, merges_path: str, 
                   special_tokens: list) -> Tokenizer:
    """
    Load BPE tokenizer from vocabulary and merges files.
    
    Args:
        vocab_path: Path to the vocabulary JSON file.
        merges_path: Path to the BPE merges text file.
        special_tokens: List of special tokens (e.g., ["<|endoftext|>"]).
    
    Returns:
        Initialized Tokenizer instance.
    """
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    print(f"Loaded tokenizer with vocabulary size: {len(tokenizer.vocab)}")
    return tokenizer


def top_p_sampling(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Perform nucleus (top-p) sampling on logits.
    
    Nucleus sampling selects from the smallest set of tokens whose cumulative
    probability exceeds the threshold p. This provides a dynamic vocabulary
    size based on the model's confidence, balancing diversity and quality.
    
    Args:
        logits: Raw logits tensor of shape (vocab_size,) or (1, vocab_size).
        p: Cumulative probability threshold (0 < p <= 1.0). 
           Smaller values make output more focused, larger values more diverse.
    
    Returns:
        Sampled token index as a tensor of shape (1,).
    """
    # Ensure logits is 1D
    if logits.dim() > 1:
        logits = logits.squeeze()
    
    # Sort probabilities in descending order
    val, ind = torch.sort(softmax(logits, -1), descending=True)
    
    # Create mask for tokens beyond cumulative probability threshold
    mask = torch.cumsum(val, dim=0) > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    
    # Filter out low-probability tokens
    removed_tokens = ind[mask]
    logits_filtered = logits.clone()
    logits_filtered[removed_tokens] = -float("inf")
    
    # Sample from filtered distribution
    prob = softmax(logits_filtered, -1)
    return torch.multinomial(prob, num_samples=1)


def generate_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    eos_token_id: Optional[int] = None
) -> str:
    """
    Generate text continuation from a prompt using the language model.
    
    Args:
        model: Trained TransformerLM model.
        tokenizer: Tokenizer for encoding/decoding text.
        prompt: Initial text to condition generation on.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values (>1) increase randomness,
                    lower values (<1) make output more deterministic.
        top_p: Nucleus sampling threshold (0 < p <= 1.0).
        device: Device to run generation on ("cpu", "cuda", "mps").
        eos_token_id: End-of-sequence token ID. Generation stops if this is sampled.
                     If None, uses tokenizer's vocab size as default.
    
    Returns:
        Generated text as a string.
    """
    model.eval()
    
    # Set default EOS token (256 is <|endoftext|> in the vocab)
    if eos_token_id is None:
        eos_token_id = 256
    
    # Encode prompt
    input_ids = torch.tensor(
        tokenizer.encode(prompt), 
        device=device, 
        dtype=torch.long
    ).unsqueeze(0)
    
    # Generate tokens autoregressively
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits for next token
            logits = model(input_ids)
            logits = logits / (temperature + 1e-8)
            
            # Sample next token
            next_token_id = top_p_sampling(logits[:, -1, :], p=top_p)
            next_token_id = next_token_id.unsqueeze(0)
            
            # Check for end-of-sequence
            if next_token_id.item() == eos_token_id:
                break
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
    
    # Decode and return text
    return tokenizer.decode(input_ids[0].cpu().tolist())


def main(config_path: str = "configs/config.yaml", 
         prompt: Optional[str] = None):
    """
    Main function to run text generation.
    
    Args:
        config_path: Path to configuration YAML file.
        prompt: Text prompt for generation. If None, uses default from config.
    """
    # Load configuration
    config = load_config(config_path)
    gen_config = config['generation']
    training_config = config['training']
    
    # Setup model
    print("Setting up model...")
    model = setup_model(config)
    model = torch.compile(model, backend="aot_eager")
    
    # Load checkpoint
    model = load_checkpoint(model, gen_config['checkpoint_path'])
    
    # Load tokenizer
    tokenizer = load_tokenizer(
        gen_config['vocab_path'],
        gen_config['merges_path'],
        gen_config['special_tokens']
    )
    
    # Get generation parameters
    prompt_text = prompt if prompt is not None else gen_config['default_prompt']
    max_tokens = gen_config['max_tokens']
    temperature = gen_config['temperature']
    top_p = gen_config['top_p']
    device = training_config['device']
    
    # Generate text
    print(f"\nPrompt: {prompt_text}")
    print(f"Generating up to {max_tokens} tokens...\n")
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print("Generated text:")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate text with Transformer LM")
    parser.add_argument("--config", type=str, default="src/configs/config_llm_intro.yaml", help="Path to config file")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text")
    
    args = parser.parse_args()
    
    main(config_path=args.config, prompt=args.prompt)