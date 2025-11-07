#!/usr/bin/env python3
"""
Test script for Multi-Head Latent Attention (MLA) implementation
"""

import sys
sys.path.insert(0, 'src')

import torch
from config.config import ModelConfig
from model.factory import build_model


def test_mla_basic():
    """Test basic MLA functionality"""
    print("=" * 70)
    print("Test 1: Basic MLA Initialization and Forward Pass")
    print("=" * 70)

    # Create a small config with MLA
    config = ModelConfig(
        model_architecture="transformer",
        attention_type="mla",
        d_model=256,
        n_heads=8,
        n_layers=2,
        max_seq_len=128,
        vocab_size=1000,
        d_ff=1024,
        dropout=0.0,
        positional_encoding="rope",
        d_latent=64,  # d_model // 4
        d_rope_latent=32,  # d_k
    )

    print(f"\nConfig:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_k: {config.d_k}")
    print(f"  d_latent: {config.d_latent}")
    print(f"  d_rope_latent: {config.d_rope_latent}")
    print(f"  attention_type: {config.attention_type}")

    # Create model
    model = build_model(config)
    print(f"\n✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(x)

    print(f"\n✓ Forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Output shape mismatch"
    print(f"\n✓ Output shape correct")


def test_mla_default_params():
    """Test MLA with default d_latent and d_rope_latent"""
    print("\n" + "=" * 70)
    print("Test 2: MLA with Auto-Computed Default Parameters")
    print("=" * 70)

    # Create config without specifying d_latent and d_rope_latent
    config = ModelConfig(
        model_architecture="transformer",
        attention_type="mla",
        d_model=512,
        n_heads=8,
        n_layers=2,
        max_seq_len=128,
        vocab_size=1000,
        d_ff=2048,
    )

    print(f"\nAuto-computed parameters:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_k: {config.d_k}")
    print(f"  d_latent: {config.d_latent} (auto: d_model // 4 = {config.d_model // 4})")
    print(f"  d_rope_latent: {config.d_rope_latent} (auto: d_k = {config.d_k})")

    # Create model
    model = build_model(config)
    print(f"\n✓ Model created with default parameters")

    # Test forward pass
    x = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, 16, config.vocab_size)
    print(f"✓ Forward pass successful with defaults")


def test_mla_vs_other_attention():
    """Compare parameter counts: MLA vs MHA vs GQA"""
    print("\n" + "=" * 70)
    print("Test 3: Parameter Comparison (MLA vs MHA vs GQA)")
    print("=" * 70)

    base_config = {
        "model_architecture": "transformer",
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 4,
        "max_seq_len": 512,
        "vocab_size": 50000,
        "d_ff": 4096,
        "dropout": 0.0,
    }

    # MHA
    config_mha = ModelConfig(**base_config, attention_type="mha")
    model_mha = build_model(config_mha)
    params_mha = sum(p.numel() for p in model_mha.parameters())

    # GQA
    config_gqa = ModelConfig(**base_config, attention_type="gqa", n_kv_heads=2)
    model_gqa = build_model(config_gqa)
    params_gqa = sum(p.numel() for p in model_gqa.parameters())

    # MLA
    config_mla = ModelConfig(**base_config, attention_type="mla", d_latent=256)
    model_mla = build_model(config_mla)
    params_mla = sum(p.numel() for p in model_mla.parameters())

    print(f"\nParameter counts (d_model={base_config['d_model']}, n_layers={base_config['n_layers']}):")
    print(f"  MHA: {params_mha:,} parameters (100%)")
    print(f"  GQA: {params_gqa:,} parameters ({100 * params_gqa / params_mha:.1f}%)")
    print(f"  MLA: {params_mla:,} parameters ({100 * params_mla / params_mha:.1f}%)")

    print(f"\nParameter savings vs MHA:")
    print(f"  GQA saves: {params_mha - params_gqa:,} parameters ({100 * (1 - params_gqa / params_mha):.1f}%)")
    print(f"  MLA saves: {params_mha - params_mla:,} parameters ({100 * (1 - params_mla / params_mha):.1f}%)")


def test_mla_kv_cache_size():
    """Estimate KV cache size for MLA vs MHA"""
    print("\n" + "=" * 70)
    print("Test 4: KV Cache Size Analysis")
    print("=" * 70)

    d_model = 2048
    n_heads = 32
    d_k = d_model // n_heads  # 64
    d_latent = d_model // 4  # 512
    seq_len = 2048
    n_layers = 24
    batch_size = 1

    # MHA: stores full K and V for each head
    # K: (batch, n_layers, n_heads, seq_len, d_k)
    # V: (batch, n_layers, n_heads, seq_len, d_k)
    kv_cache_mha = 2 * batch_size * n_layers * n_heads * seq_len * d_k

    # MLA: stores compressed latent representation
    # Latent: (batch, n_layers, seq_len, d_latent)
    kv_cache_mla = batch_size * n_layers * seq_len * d_latent

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k: {d_k}")
    print(f"  d_latent: {d_latent}")
    print(f"  seq_len: {seq_len}")
    print(f"  n_layers: {n_layers}")

    print(f"\nKV cache size (elements):")
    print(f"  MHA: {kv_cache_mha:,} elements")
    print(f"  MLA: {kv_cache_mla:,} elements")
    print(f"  Compression ratio: {kv_cache_mha / kv_cache_mla:.2f}x")

    # In GB (assuming float16)
    bytes_per_element = 2  # float16
    gb_mha = kv_cache_mha * bytes_per_element / (1024 ** 3)
    gb_mla = kv_cache_mla * bytes_per_element / (1024 ** 3)

    print(f"\nKV cache size (FP16):")
    print(f"  MHA: {gb_mha:.2f} GB")
    print(f"  MLA: {gb_mla:.2f} GB")
    print(f"  Memory saved: {gb_mha - gb_mla:.2f} GB ({100 * (1 - gb_mla / gb_mha):.1f}%)")


def test_mla_with_different_positional_encodings():
    """Test MLA with different positional encoding types"""
    print("\n" + "=" * 70)
    print("Test 5: MLA with Different Positional Encodings")
    print("=" * 70)

    base_config = {
        "model_architecture": "transformer",
        "attention_type": "mla",
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 2,
        "max_seq_len": 128,
        "vocab_size": 1000,
        "d_ff": 1024,
        "d_latent": 64,
    }

    encodings = ["rope", "yarn", "sinusoidal", "alibi"]

    for encoding in encodings:
        config = ModelConfig(**base_config, positional_encoding=encoding)
        model = build_model(config)

        x = torch.randint(0, config.vocab_size, (1, 32))
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (1, 32, config.vocab_size)
        print(f"  ✓ {encoding.upper()}: Forward pass successful")


def main():
    print("\n" + "=" * 70)
    print("MULTI-HEAD LATENT ATTENTION (MLA) TEST SUITE")
    print("=" * 70)

    try:
        test_mla_basic()
        test_mla_default_params()
        test_mla_vs_other_attention()
        test_mla_kv_cache_size()
        test_mla_with_different_positional_encodings()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print("\nMLA is now ready to use!")
        print("\nTo use MLA in your config:")
        print('  "attention_type": "mla"')
        print('  "d_latent": 512  # Optional, defaults to d_model // 4')
        print('  "d_rope_latent": 64  # Optional, defaults to d_k')
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
