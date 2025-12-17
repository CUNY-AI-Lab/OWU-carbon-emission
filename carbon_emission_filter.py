"""
title: Remote LLM Energy Estimator
author: CUNY AI Lab
version: 0.1.0
description: Estimate energy use and CO₂e per exchange for models called via OpenRouter and AWS Bedrock using per-token emission factors.
license: MIT
"""

import json
import time
from typing import Optional, Tuple

from pydantic import BaseModel, Field


# Default per-token CO2 emission factors (mg CO2 per token)
DEFAULT_MODEL_CO2_MG_PER_TOKEN = {
    "gpt-oss-120b": 0.40,       # Large MoE model
    "gpt-4.1-mini": 0.15,       # Smaller efficient model
    "gpt-4.1": 0.35,            # Standard GPT-4.1
    "gpt-4o": 0.30,             # GPT-4o
    "gpt-4": 0.35,              # GPT-4 variants
    "gpt-3.5": 0.10,            # GPT-3.5 variants
    "claude-3-5-sonnet": 0.25,  # Claude 3.5 Sonnet
    "claude-3-opus": 0.45,      # Claude 3 Opus (larger)
    "claude-3-sonnet": 0.25,    # Claude 3 Sonnet
    "claude-3-haiku": 0.08,     # Claude 3 Haiku (efficient)
    "claude-2": 0.30,           # Claude 2
    "titan-text": 0.10,         # Amazon Titan Text
    "llama-3": 0.20,            # Llama 3 variants
    "llama-2": 0.15,            # Llama 2 variants
    "mistral": 0.15,            # Mistral variants
    "mixtral": 0.25,            # Mixtral MoE
    "gemini": 0.25,             # Google Gemini
    "__default__": 0.30,        # Fallback for unknown models
}


class Filter:
    """
    OpenWebUI Filter that estimates energy use and CO2 emissions for remote LLM calls.

    This filter appends a footer to assistant responses showing estimated emissions
    based on token usage and per-token emission factors.
    """

    class Valves(BaseModel):
        """Global configuration for the energy estimator filter."""
        enabled: bool = Field(
            default=True,
            description="Master on/off switch for this filter."
        )
        grid_emission_factor_kg_per_kwh: float = Field(
            default=0.40,
            description="Grid factor in kg CO2e per kWh (e.g. ~0.40 for typical US grid)."
        )
        show_energy_kwh: bool = Field(
            default=True,
            description="If true, show estimated energy in kWh as well as CO2."
        )
        show_per_1k_tokens: bool = Field(
            default=True,
            description="If true, show normalized estimates per 1,000 tokens."
        )
        debug_logging: bool = Field(
            default=False,
            description="If true, log intermediate calculations to the server console."
        )
        model_co2_mg_per_token_json: str = Field(
            default="",
            description="Optional JSON dict of model substring -> mg CO2/token. "
                        "If empty, use built-in defaults."
        )

    class UserValves(BaseModel):
        """Per-user configuration for the energy estimator filter."""
        enabled: bool = Field(
            default=True,
            description="Per-user toggle for showing energy estimates."
        )

    def __init__(self):
        """Initialize the filter with default settings."""
        self.type = "filter"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self._model_factors = DEFAULT_MODEL_CO2_MG_PER_TOKEN.copy()

    def _load_custom_factors(self) -> dict:
        """
        Load and merge custom model factors from JSON configuration.

        Returns:
            dict: Merged model factors (defaults + custom overrides)
        """
        factors = DEFAULT_MODEL_CO2_MG_PER_TOKEN.copy()

        if self.valves.model_co2_mg_per_token_json:
            try:
                custom = json.loads(self.valves.model_co2_mg_per_token_json)
                if isinstance(custom, dict):
                    factors.update(custom)
            except json.JSONDecodeError:
                if self.valves.debug_logging:
                    print("[EnergyEstimator] Invalid JSON in model_co2_mg_per_token_json, using defaults")

        return factors

    def _get_model_id(self, body: dict) -> str:
        """
        Extract the model identifier from the request body.

        Args:
            body: The request/response body dict

        Returns:
            str: The model identifier or "__unknown_model__" if not found
        """
        # Try common locations for model ID
        model_id = body.get("model")

        if not model_id:
            # Try metadata
            metadata = body.get("metadata", {})
            model_id = metadata.get("model")

        if not model_id:
            # Try nested locations
            model_id = body.get("model_id")

        return model_id if model_id else "__unknown_model__"

    def _get_co2_factor_mg_per_token(self, model_id: str) -> float:
        """
        Look up the CO2 emission factor for a given model.

        Uses substring matching to find the best match from configured factors.

        Args:
            model_id: The model identifier string

        Returns:
            float: CO2 emissions in mg per token
        """
        factors = self._load_custom_factors()
        model_id_lower = model_id.lower()

        # Try to find a matching key (substring match)
        for key, value in factors.items():
            if key == "__default__":
                continue
            if key.lower() in model_id_lower:
                return value

        # No match found, use default
        return factors.get("__default__", 0.30)

    def _get_token_counts(self, body: dict) -> Tuple[int, int, int]:
        """
        Extract or estimate token counts from the request body.

        Prefers actual usage data if available, otherwise estimates from text length.

        Args:
            body: The request/response body dict

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens)
        """
        # Try to get actual usage data
        usage = body.get("usage", {})

        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            if total_tokens > 0:
                return (int(prompt_tokens), int(completion_tokens), int(total_tokens))

        # Fallback: estimate from message content
        messages = body.get("messages", [])

        user_text = ""
        assistant_text = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                if role == "user":
                    user_text = content  # Use latest user message
                elif role == "assistant":
                    assistant_text = content  # Use latest assistant message

        # Estimate tokens: ~4 characters per token
        prompt_chars = len(user_text)
        completion_chars = len(assistant_text)

        prompt_tokens = max(1, prompt_chars // 4)
        completion_tokens = max(1, completion_chars // 4)
        total_tokens = prompt_tokens + completion_tokens

        return (prompt_tokens, completion_tokens, total_tokens)

    def _format_footer(
        self,
        co2_g: float,
        energy_kwh: Optional[float],
        total_tokens: int,
        model_id: str
    ) -> str:
        """
        Format the emissions footer as a Markdown blockquote.

        Args:
            co2_g: Total CO2 emissions in grams
            energy_kwh: Energy consumption in kWh (or None if not showing)
            total_tokens: Total tokens used
            model_id: The model identifier

        Returns:
            str: Formatted Markdown footer
        """
        lines = []

        # Main emissions line
        if energy_kwh is not None and self.valves.show_energy_kwh:
            main_line = f"> **Estimated data-center emissions for this exchange:** {energy_kwh:.6f} kWh (~{co2_g:.3f} g CO2e)"
        else:
            main_line = f"> **Estimated data-center emissions for this exchange:** ~{co2_g:.3f} g CO2e"

        lines.append(main_line)

        # Per-1k tokens line (if enabled and we have tokens)
        if self.valves.show_per_1k_tokens and total_tokens > 0:
            factor = 1000.0 / total_tokens
            co2_per_1k = co2_g * factor

            if energy_kwh is not None and self.valves.show_energy_kwh:
                energy_per_1k = energy_kwh * factor
                per_1k_line = f"> Approx. per 1,000 tokens: {energy_per_1k:.6f} kWh, {co2_per_1k:.3f} g CO2e"
            else:
                per_1k_line = f"> Approx. per 1,000 tokens: {co2_per_1k:.3f} g CO2e"

            lines.append(per_1k_line)

        # Disclaimer line
        lines.append("> _Remote LLM call via OpenRouter/Bedrock; highly approximate and for awareness only._")

        return "\n".join(lines)

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Process incoming request (before model call).

        Records a timestamp for potential latency tracking.

        Args:
            body: The request body
            __user__: User information dict

        Returns:
            dict: The (potentially modified) request body
        """
        try:
            if not self.valves.enabled:
                return body

            # Record start timestamp for potential latency tracking
            if "metadata" not in body:
                body["metadata"] = {}
            body["metadata"]["_energy_start_ts"] = time.time()

        except Exception as e:
            if self.valves.debug_logging:
                print(f"[EnergyEstimator] inlet error: {e}")

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        """
        Process outgoing response (after model call).

        Computes emissions estimate and appends footer to the assistant response.

        Args:
            body: The response body
            __user__: User information dict

        Returns:
            dict: The (potentially modified) response body
        """
        try:
            # Check if filter is enabled (global and per-user)
            if not self.valves.enabled:
                return body

            if hasattr(self, 'user_valves') and not self.user_valves.enabled:
                return body

            # Get messages
            messages = body.get("messages")
            if not messages:
                return body

            # Get model ID
            model_id = self._get_model_id(body)

            if self.valves.debug_logging:
                print(f"[EnergyEstimator] Model ID: {model_id}")

            # Get token counts
            prompt_tokens, completion_tokens, total_tokens = self._get_token_counts(body)

            if self.valves.debug_logging:
                print(f"[EnergyEstimator] Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

            # Get emission factor
            per_token_mg = self._get_co2_factor_mg_per_token(model_id)

            if self.valves.debug_logging:
                print(f"[EnergyEstimator] CO2 factor: {per_token_mg} mg/token")

            # Calculate emissions
            co2_mg = total_tokens * per_token_mg
            co2_g = co2_mg / 1000.0

            # Calculate energy if configured
            energy_kwh = None
            if self.valves.show_energy_kwh:
                co2_kg = co2_g / 1000.0
                grid_factor = self.valves.grid_emission_factor_kg_per_kwh
                if grid_factor > 0:
                    energy_kwh = co2_kg / grid_factor

            if self.valves.debug_logging:
                print(f"[EnergyEstimator] CO2: {co2_g:.4f} g, Energy: {energy_kwh}")

            # Format the footer
            footer = self._format_footer(co2_g, energy_kwh, total_tokens, model_id)

            # Append footer to the last assistant message
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    current_content = msg.get("content") or ""
                    msg["content"] = current_content + "\n\n" + footer
                    break

            body["messages"] = messages

        except Exception as e:
            if self.valves.debug_logging:
                print(f"[EnergyEstimator] outlet error: {e}")

        return body


# Simple test cases for validation
if __name__ == "__main__":
    print("Running basic tests...\n")

    # Create filter instance
    f = Filter()

    # Test 1: OpenRouter case with usage data
    print("Test 1: OpenRouter with usage data")
    body1 = {
        "model": "openrouter/nvidia/gpt-oss-120b-Eagle3-v2",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        "messages": [
            {"role": "user", "content": "Explain minimal computing in DH."},
            {"role": "assistant", "content": "Minimal computing is an approach in digital humanities..."},
        ],
        "metadata": {}
    }
    result1 = f.outlet(body1)
    assistant_msg1 = [m for m in result1["messages"] if m["role"] == "assistant"][-1]
    print(f"Footer appended: {'CO2e' in assistant_msg1['content']}")
    print(f"Content preview: ...{assistant_msg1['content'][-200:]}\n")

    # Test 2: Bedrock case without usage data
    print("Test 2: Bedrock without usage data")
    body2 = {
        "model": "bedrock/anthropic.claude-3-5-sonnet-v2:0",
        "messages": [
            {"role": "user", "content": "Summarize this text about climate change and its effects on coastal regions."},
            {"role": "assistant", "content": "Here is a summary of the key points about climate change impacts on coastal areas..."},
        ],
    }
    result2 = f.outlet(body2)
    assistant_msg2 = [m for m in result2["messages"] if m["role"] == "assistant"][-1]
    print(f"Footer appended: {'CO2e' in assistant_msg2['content']}")
    print(f"Content preview: ...{assistant_msg2['content'][-200:]}\n")

    # Test 3: Filter disabled
    print("Test 3: Filter disabled")
    f_disabled = Filter()
    f_disabled.valves.enabled = False
    body3 = {
        "model": "openrouter/openai/gpt-4.1-mini",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    }
    result3 = f_disabled.outlet(body3)
    assistant_msg3 = [m for m in result3["messages"] if m["role"] == "assistant"][-1]
    print(f"Footer NOT appended (disabled): {'CO2e' not in assistant_msg3['content']}")
    print(f"Content: {assistant_msg3['content']}\n")

    print("All tests completed!")
