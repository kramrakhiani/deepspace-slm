import math
import time
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Any, Generator

from model.transformer import DeepSpaceSLM
from data.tokenizer import HabitatTokenizer


class InferenceEngine:
    def __init__(
        self,
        model: DeepSpaceSLM,
        tokenizer: HabitatTokenizer,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_seq_len = model.config.max_seq_len

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        stop_tokens: Optional[List[int]] = None,
        schema_masker: Optional[Any] = None,
        speculative: bool = False,
    ) -> str:
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_id]

        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        if speculative and getattr(self.model, "medusa", None) is not None:
            generated_ids = self._generate_speculative_tokens(
                input_tensor,
                max_new_tokens=max_new_tokens,
                stop_tokens=stop_tokens,
            )
        else:
            generated_ids = self._generate_tokens(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
                stop_tokens=stop_tokens,
                schema_masker=schema_masker,
            )

        return self.tokenizer.decode(generated_ids, skip_special=True)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
        stop_tokens: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        if stop_tokens is None:
            stop_tokens = [self.tokenizer.eos_id]

        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        batch_size, seq_len = input_tensor.shape
        generated = []
        kv_caches = None

        logits, kv_caches = self.model(input_tensor, kv_caches=None, start_pos=0)
        next_token_logits = logits[:, -1, :]

        for i in range(max_new_tokens):
            next_token = self._sample(
                next_token_logits, temperature, top_k, top_p, greedy
            )
            token_id = next_token.item()
            generated.append(token_id)

            decoded_text = self.tokenizer.decode([token_id], skip_special=True)
            if decoded_text:
                yield decoded_text

            if token_id in stop_tokens or (seq_len + len(generated) >= self.max_seq_len):
                break

            next_input = next_token.view(1, 1)
            start_pos = seq_len + len(generated) - 1
            logits, kv_caches = self.model(
                next_input, kv_caches=kv_caches, start_pos=start_pos
            )
            next_token_logits = logits[:, -1, :]

    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        greedy: bool,
        stop_tokens: List[int],
        schema_masker: Optional[Any] = None,
    ) -> List[int]:
        batch_size, seq_len = input_ids.shape
        generated = []
        kv_caches = None

        logits, kv_caches = self.model(input_ids, kv_caches=None, start_pos=0)
        next_token_logits = logits[:, -1, :]

        for i in range(max_new_tokens):
            if len(generated) >= max_new_tokens:
                break
            if schema_masker:
                current_text = self.tokenizer.decode(generated, skip_special=False)
                next_token_logits = schema_masker.apply_mask(next_token_logits, current_text)

            next_token = self._sample(
                next_token_logits, temperature, top_k, top_p, greedy
            )
            generated.append(next_token.item())

            if next_token.item() in stop_tokens:
                break

            if seq_len + len(generated) >= self.max_seq_len:
                break

            next_input = next_token.view(1, 1)
            start_pos = seq_len + len(generated) - 1
            logits, kv_caches = self.model(
                next_input, kv_caches=kv_caches, start_pos=start_pos
            )
            next_token_logits = logits[:, -1, :]

        return generated

    def _generate_speculative_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        stop_tokens: List[int],
    ) -> List[int]:
        batch_size, seq_len = input_ids.shape
        generated = []
        kv_caches = None

        curr_input = input_ids
        curr_pos = 0

        while len(generated) < max_new_tokens:
            logits, kv_caches, medusa_logits = self.model(
                curr_input, kv_caches=kv_caches, start_pos=curr_pos, return_medusa=True
            )
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated.append(next_token.item())

            if next_token.item() in stop_tokens or seq_len + len(generated) >= self.max_seq_len:
                break

            draft_tokens = [head[:, -1, :].argmax(dim=-1) for head in medusa_logits]
            accepted_drafts = []
            for draft in draft_tokens:
                if len(generated) + len(accepted_drafts) >= max_new_tokens:
                    break
                accepted_drafts.append(draft.item())
                if draft.item() in stop_tokens:
                    break

            generated.extend(accepted_drafts)
            if generated[-1] in stop_tokens or seq_len + len(generated) >= self.max_seq_len:
                break

            step_accepted = [next_token.item()] + accepted_drafts
            curr_pos = seq_len + len(generated) - len(step_accepted)
            curr_input = torch.tensor([step_accepted], dtype=torch.long, device=self.device)

        return generated

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        greedy: bool,
    ) -> torch.Tensor:
        logits = logits.clone()

        if greedy:
            return logits.argmax(dim=-1)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def compute_perplexity(self, text: str) -> float:
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=self.device)
        target_ids = torch.tensor([token_ids[1:]], dtype=torch.long, device=self.device)

        logits, _ = self.model(input_ids)
        log_probs = F.log_softmax(logits, dim=-1)

        target_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)

        avg_nll = -target_log_probs.mean().item()
        return math.exp(avg_nll)
