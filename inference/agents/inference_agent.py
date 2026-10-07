"""
Inference Agent
================
Handles free-form SLM neural text generation and
perplexity computation via the InferenceEngine.
"""

import logging
from typing import List, Optional

from inference.agents.base import (
    BaseAgent,
    AgentMessage,
    AgentResponse,
    AgentCapability,
    MessagePriority,
)
from inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class InferenceAgent(BaseAgent):
    """Domain agent for pure neural SLM text generation.

    Owns:
        - Free-form text generation (free_query)
        - Perplexity computation (compute_perplexity)
        - Streaming generation (generate_stream)
    """

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    # ── BaseAgent Protocol ────────────────────────────────────────────

    @property
    def agent_id(self) -> str:
        return "inference"

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                intent="free_query",
                description="Generate text using the SLM neural network",
            ),
            AgentCapability(
                intent="compute_perplexity",
                description="Compute perplexity of a given text sequence",
            ),
            AgentCapability(
                intent="generate_stream",
                description="Stream tokens from the SLM neural network",
            ),
        ]

    def handle_message(self, message: AgentMessage) -> AgentResponse:
        intent = message.intent
        payload = message.payload

        if intent == "free_query":
            return self._free_query(
                query=payload.get("query", ""),
                max_new_tokens=payload.get("max_new_tokens", 128),
                temperature=payload.get("temperature", 1.0),
                top_k=payload.get("top_k"),
                top_p=payload.get("top_p"),
                greedy=payload.get("greedy", True),
            )

        elif intent == "compute_perplexity":
            return self._compute_perplexity(payload.get("text", ""))

        elif intent == "generate_stream":
            # Streaming must be handled differently — return the generator reference
            return self._prepare_stream(
                query=payload.get("query", ""),
                max_new_tokens=payload.get("max_new_tokens", 128),
                greedy=payload.get("greedy", True),
            )

        return AgentResponse(
            agent_id=self.agent_id,
            status="not_handled",
            messages=[f"Inference agent does not handle intent '{intent}'"],
        )

    # ── Domain Logic ──────────────────────────────────────────────────

    def _free_query(
        self,
        query: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = True,
    ) -> AgentResponse:
        prompt = f"<QUERY> {query}"
        output = self.engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={"generated_text": output, "prompt": prompt},
            messages=[output],
        )

    def _compute_perplexity(self, text: str) -> AgentResponse:
        ppl = self.engine.compute_perplexity(text)

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={"perplexity": ppl, "text": text},
            messages=[f"Perplexity: {ppl:.2f}"],
        )

    def _prepare_stream(
        self,
        query: str,
        max_new_tokens: int = 128,
        greedy: bool = True,
    ) -> AgentResponse:
        """Prepare a streaming generator — the caller iterates the generator from data['stream']."""
        prompt = f"<QUERY> {query}"
        stream = self.engine.generate_stream(
            prompt, max_new_tokens=max_new_tokens, greedy=greedy,
        )

        return AgentResponse(
            agent_id=self.agent_id,
            status="success",
            data={"stream": stream, "prompt": prompt},
            messages=["Streaming generation prepared"],
        )
