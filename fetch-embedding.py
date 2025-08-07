#! /usr/bin/env python3

import functools
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import transformers

# We fetch the model and tokenizer from Hugging Face only on Colab,
# since they're too big for me to download at home quickly.  We then
# save just the relevant stuff and download it.
MODEL = "mistralai/Mistral-7B-v0.1"
SAVED_PT = Path("mistral-embedding.pt")

def float32_to_bf16_uint16(float32_array):
    """Convert float32 array to BF16 stored as uint16 array
    
    BF16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits
    We can get BF16 by taking the top 16 bits of float32
    """
    # Convert to bytes, take every other 2 bytes (the high bytes)
    float32_bytes = float32_array.astype(np.float32).tobytes()
    # Reinterpret as uint32, then shift right by 16 bits to get high 16 bits
    uint32_view = np.frombuffer(float32_bytes, dtype=np.uint32)
    bf16_uint16 = (uint32_view >> 16).astype(np.uint16)
    return bf16_uint16

def bf16_uint16_to_float32(bf16_uint16_array):
    """Convert BF16 stored as uint16 array back to float32
    
    This is the inverse of float32_to_bf16_uint16
    """
    # Shift left by 16 bits and pad with zeros to make float32
    uint32_expanded = bf16_uint16_array.astype(np.uint32) << 16
    # Reinterpret as float32
    float32_result = np.frombuffer(uint32_expanded.tobytes(), dtype=np.float32)
    return float32_result

@functools.cache
def get_model():
    print("Loading Mistral-7B model...")
    return transformers.AutoModelForCausalLM.from_pretrained(MODEL)

@functools.cache
def get_tokenizer():
    print("Loading Mistral-7B tokenizer...")
    return transformers.AutoTokenizer.from_pretrained(MODEL)

@functools.cache
def get_embedding():
    """Tensor of shape (32000, 4096)"""
    model = get_model()
    embedding = model.model.embed_tokens.weight
    return embedding

@functools.cache
def get_vocab():
    """Dict of string -> number mappings

    Example:
    {'ATTR': 10096,
     'folg': 17090,
     '▁som': 1113,
     'asa': 13937,
     'AD': 1841,
     'records': 28257,
     '▁desire': 8646,
     'É': 28901,
     '▁Budd': 13772,
     'eron': 13618}
    """
    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab()
    return vocab

@functools.cache
def get_special_tokens():
    """A dict mapping role -> string

    Example: {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}
    """
    tokenizer = get_tokenizer()
    special_tokens = tokenizer.special_tokens_map
    return special_tokens

@functools.cache
def get_all():
    if SAVED_PT.exists():
         return torch.load(SAVED_PT)
    rv = {
        "embedding": get_embedding(),
        "vocab": get_vocab(),
        "special_tokens": get_special_tokens(),
        # Mistral uses "▁" (LOWER ONE EIGHTH BLOCK); it inherits
        # this from the Google SentencePiece text tokenizer.
        "space_string": "\u2581",
    }
    torch.save(rv, SAVED_PT)
    return rv

def main():
    print("Starting embedding extraction process...")
    all_dict = get_all()
    
    # Extract data
    print("Extracting embedding data...")
    embedding = all_dict["embedding"]
    vocab = all_dict["vocab"]
    special_tokens = all_dict["special_tokens"]
    space_string = all_dict["space_string"]
    
    # Convert embedding to numpy float32 for filtering
    print("Converting embeddings to float32...")
    embedding_np = embedding.detach().cpu().numpy().astype(np.float32)
    
    # Create reverse vocab mapping (token_id -> string)
    reverse_vocab = {v: k for k, v in vocab.items()}
    
    # Filter out some special tokens that might be confusing for the demo
    # Keep only "normal" tokens that could be useful for analogies
    print("Filtering vocabulary...")
    filtered_vocab = {}
    filtered_embeddings = []
    filtered_ids = []
    
    for token_id, token_str in reverse_vocab.items():
        # Skip tokens that are clearly not words
        cleaned_token = token_str.strip(space_string)
        if (not token_str.startswith('<') and  # Skip <s>, </s>, <unk>, etc.
            len(cleaned_token) > 0 and  # Skip empty/whitespace
            token_id < len(embedding_np) and  # Ensure we have embedding
            # Additional filtering for better web demo
            len(cleaned_token) <= 20 and  # Skip very long tokens
            not any(char.isdigit() for char in cleaned_token[:3]) and  # Skip tokens starting with numbers
            cleaned_token.replace('_', '').replace('-', '').replace('.', '').isalpha()  # Prefer alphabetic tokens
            ):
            filtered_vocab[token_str] = len(filtered_ids)  # Reassign sequential IDs
            filtered_embeddings.append(embedding_np[token_id])
            filtered_ids.append(token_id)
    
    # Convert to numpy array (float32)
    print("Creating final embedding matrix...")
    filtered_embeddings = np.array(filtered_embeddings, dtype=np.float32)
    
    # Now convert to BF16 format for storage
    print("Converting to BF16 format...")
    filtered_embeddings_bf16 = float32_to_bf16_uint16(filtered_embeddings.flatten())
    
    print(f"Original vocab size: {len(vocab)}")
    print(f"Filtered vocab size: {len(filtered_vocab)}")
    print(f"Embedding shape: {filtered_embeddings.shape}")
    print(f"Data type: BF16 (stored as uint16)")
    
    # Save embeddings as binary file (can be loaded as ArrayBuffer in JS)
    print("Saving embedding files...")
    embeddings_path = Path("public/embeddings.bin")
    embeddings_path.parent.mkdir(exist_ok=True)
    with open(embeddings_path, "wb") as f:
        f.write(filtered_embeddings_bf16.tobytes())
    
    # Save metadata as JSON
    metadata = {
        "vocab": filtered_vocab,
        "vocab_size": len(filtered_vocab),
        "embedding_dim": filtered_embeddings.shape[1],
        "dtype": "bf16",  # Update to reflect BF16 format
        "space_string": space_string,
        "special_tokens": special_tokens,
        "original_model": MODEL
    }
    
    metadata_path = Path("public/metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved embeddings to {embeddings_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Total size: {embeddings_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("Embedding extraction complete!")


if __name__ == "__main__":
    main()
