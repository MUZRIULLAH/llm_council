import asyncio
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .config import get_config

logger = logging.getLogger(__name__)


class Response:
    def __init__(self, model: str, content: str, raw: Any = None):
        self.model = model
        self.content = content
        self.raw = raw


class CouncilMember:
    def __init__(self, model_name: str, client: AsyncOpenAI):
        self.model_name = model_name
        self.client = client

    async def get_response(
        self, messages: List[Dict[str, str]], timeout: float
    ) -> Optional[Response]:
        try:
            # The client retries with backoff (max_retries); timeout applies per attempt.
            response = await self.client.chat.completions.create(
                model=self.model_name, messages=messages, timeout=timeout
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty response")
            return Response(model=self.model_name, content=content, raw=response)
        except Exception as e:
            logger.error(f"Error getting response from {self.model_name}: {e}")
            return None


class Council:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.judge_client = AsyncOpenAI(
            api_key=self.config.openrouter_api_key or "none",  # local servers ignore it
            base_url=self.config.api_base,
            max_retries=self.config.max_retries,
        )
        self.members = [
            CouncilMember(m, self.judge_client) for m in self.config.council_models
        ]

    async def query(
        self, prompt: str, system_prompt: str = "You are a helpful assistant."
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        tasks = [
            member.get_response(messages, self.config.timeout)
            for member in self.members
        ]
        responses = await asyncio.gather(*tasks)

        valid_responses = [r for r in responses if r is not None]

        if not valid_responses:
            return {"error": "No valid responses received from council members."}

        if self.config.council_strategy == "synthesis":
            final_answer = await self._synthesize(prompt, valid_responses)
        else:  # "voting"; config rejects anything else
            final_answer = self._vote(valid_responses)

        return {
            "final_answer": final_answer,
            "individual_responses": [
                {"model": r.model, "content": r.content} for r in valid_responses
            ],
            "strategy": self.config.council_strategy,
        }

    async def _synthesize(self, original_prompt: str, responses: List[Response]) -> str:
        synthesis_prompt = f"Original Query: {original_prompt}\n\n"
        synthesis_prompt += "Different AI models provided the following responses:\n\n"
        for i, r in enumerate(responses):
            synthesis_prompt += f"--- Model {i+1} ({r.model}) ---\n{r.content}\n\n"

        synthesis_prompt += (
            "Based on the responses above, provide a single, comprehensive, and accurate final answer. "
            "Resolve any contradictions and combine the best insights from all models."
        )

        try:
            # A judge outside the council avoids grading its own answer
            judge_model = self.config.judge_model or self.config.council_models[0]
            resp = await self.judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a lead orchestrator who synthesizes multiple perspectives into a consensus.",
                    },
                    {"role": "user", "content": synthesis_prompt},
                ],
                timeout=self.config.timeout,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return responses[0].content  # Fallback to first valid response

    def _vote(self, responses: List[Response]) -> str:
        # Normalise case, spacing and trailing punctuation so "Paris." and "paris"
        # count as one vote. shortcut: still exact-match, so it only suits short
        # answers (facts, labels, multiple choice); long prose should use synthesis.
        def key(text: str) -> str:
            return " ".join(text.lower().split()).rstrip(".!")

        counts = Counter(key(r.content) for r in responses)
        top = counts.most_common(1)[0][0]
        return next(r.content.strip() for r in responses if key(r.content) == top)
