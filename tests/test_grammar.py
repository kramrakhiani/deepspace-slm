"""
Unit Tests for Grammar Masker and Bit-Packing
================================================
Tests SchemaMasker logit mask generation, constrained JSON decoding,
and 4-bit tensor packing/unpacking lossless recovery.
"""

import pytest
import torch
from data.tokenizer import HabitatTokenizer
from inference.grammar import SchemaMasker
from quantization.export import pack_int4_to_uint8, unpack_uint8_to_int4


class TestGrammarMasker:
    def test_schema_masker_initialization(self):
        tokenizer = HabitatTokenizer()
        masker = SchemaMasker(tokenizer)
        assert len(masker.allowed_actions) > 0

    def test_schema_masker_valid_token_ids(self):
        tokenizer = HabitatTokenizer()
        masker = SchemaMasker(tokenizer)

        # Unconstrained text -> None
        valid_ids = masker.get_valid_token_ids("check stock o2")
        assert valid_ids is None

        # Partial JSON -> restricted IDs
        valid_json_ids = masker.get_valid_token_ids('{"action"')
        assert valid_json_ids is not None
        assert len(valid_json_ids) > 0

    def test_apply_mask(self):
        tokenizer = HabitatTokenizer()
        masker = SchemaMasker(tokenizer)

        logits = torch.zeros(1, tokenizer.vocab_size)
        masked = masker.apply_mask(logits, '{"action"')
        assert masked.shape == logits.shape


class TestBitPacking:
    def test_int4_packing_roundtrip(self):
        """Verify 4-bit packing and unpacking recovers exact original values."""
        # Create int4 test values in [-7, 7]
        original = torch.tensor([-7, -3, 0, 4, 7, -1, 2, 5], dtype=torch.int8)

        # Pack into uint8 bytes (4 bytes for 8 elements)
        packed_bytes = pack_int4_to_uint8(original)
        assert len(packed_bytes) == 4

        # Unpack back
        unpacked = unpack_uint8_to_int4(packed_bytes, num_elements=len(original))

        # Check equality
        torch.testing.assert_close(unpacked, original)
