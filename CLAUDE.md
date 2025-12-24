# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **OpenWebUI Filter Function** that estimates energy use and CO2 emissions for remote LLM API calls (OpenRouter, AWS Bedrock). It appends a footer to assistant responses showing estimated environmental impact with real-world comparisons.

Target platform: **OpenWebUI v0.6.41+**

## Running Tests

```bash
python carbon_emission_filter.py
```

Validates: OpenRouter/Bedrock models, token estimation fallback, filter toggle, comparison tiers, and footer stripping.

## Architecture

Single-file OpenWebUI Function following the Filter pattern:

- **`Filter` class**: Main entry point with `inlet()` and `outlet()` methods
- **`Valves` (Pydantic)**: Admin settings (grid factor, display options, custom emission factors via JSON)
- **`UserValves` (Pydantic)**: Per-user toggle
- **`DEFAULT_MODEL_CO2_MG_PER_TOKEN`**: Model substring → mg CO2/token lookup
- **`ENERGY_COMPARISONS`**: Tiered real-world comparisons (LED seconds → phone % → laptop min → TV min)
- **`CO2_COMPARISONS`**: Fallback comparisons (breaths → walking → driving)

### Data Flow

1. `inlet()`:
   - **Strips emission footers** from conversation history (prevents LLM from learning/mimicking the pattern)
   - Records timestamp for latency tracking
2. Model processes request (outside this filter)
3. `outlet()`:
   - Extracts model ID, gets token counts (from `usage` or estimates chars ÷ 4)
   - Looks up emission factor (case-insensitive substring match)
   - Calculates CO2 and energy, selects appropriate real-world comparison
   - Appends markdown footer to last assistant message

### Key Implementation Details

- **Footer stripping**: `_strip_emission_footers()` uses regex to remove footers from history before LLM sees them, preventing in-context learning mimicry
- Model matching: case-insensitive substring against model ID
- Comparison selection: iterates thresholds to pick appropriate tier based on magnitude
- All exceptions caught silently (never breaks chat), optionally logged with `debug_logging`
- Grid emission factor default: 0.40 kg CO2/kWh (US average)

## OpenWebUI Installation

1. Add as Function in Workspace → Functions
2. Enable globally via "..." menu → Global toggle, OR assign to specific models in Workspace → Models

## Documentation

See `estimate-guidelines.md` for calculation methodology and emission factor sources.
