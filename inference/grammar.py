import json
import torch
from typing import List, Dict, Set, Optional, Any
from data.tokenizer import HabitatTokenizer


class SchemaMasker:
    def __init__(
        self,
        tokenizer: HabitatTokenizer,
        allowed_actions: Optional[List[str]] = None,
        allowed_items: Optional[List[str]] = None,
    ):
        self.tokenizer = tokenizer
        self.allowed_actions = allowed_actions or [
            "check_stock", "locate", "forecast", "get_alerts", "log_maintenance", "free_query"
        ]
        self.allowed_items = allowed_items or [
            "o2 canister", "n2 tank", "co2 scrubber", "air filter", "water filter",
            "food ration", "medical kit", "antibiotic pack", "wrench set", "eva suit patch kit"
        ]
        self.vocab_size = tokenizer.vocab_size

    def get_valid_token_ids(self, partial_text: str) -> Optional[List[int]]:
        if not partial_text.strip().startswith("{"):
            return None

        try:
            json.loads(partial_text)
            eos_tokens = [self.tokenizer.eos_id]
            if "\n" in self.tokenizer.token_to_id:
                eos_tokens.append(self.tokenizer.token_to_id["\n"])
            return eos_tokens
        except json.JSONDecodeError:
            pass

        valid_ids = set()
        for char in ['"', ':', ',', '}', ' ', '{', '[', ']']:
            if char in self.tokenizer.token_to_id:
                valid_ids.add(self.tokenizer.token_to_id[char])

        if '"action"' in partial_text:
            for action in self.allowed_actions:
                valid_ids.update(self.tokenizer.encode(action, add_bos=False, add_eos=False))

        if '"item"' in partial_text:
            for item in self.allowed_items:
                valid_ids.update(self.tokenizer.encode(item, add_bos=False, add_eos=False))

        for d in "0123456789":
            if d in self.tokenizer.token_to_id:
                valid_ids.add(self.tokenizer.token_to_id[d])

        return list(valid_ids) if valid_ids else None

    def apply_mask(self, logits: torch.Tensor, partial_text: str) -> torch.Tensor:
        valid_ids = self.get_valid_token_ids(partial_text)
        if valid_ids is None or len(valid_ids) == 0:
            return logits

        masked_logits = logits.clone()
        mask = torch.full_like(masked_logits, float("-inf"))
        for tid in valid_ids:
            if tid < masked_logits.size(-1):
                mask[..., tid] = 0.0

        return masked_logits + mask
