Data Sources & Assumptions

This extension does not claim to report exact energy or CO₂ use for any individual request. Instead, it provides order-of-magnitude estimates based on:

Published lifecycle analyses from AI vendors (Google, Mistral, NVIDIA).

Cloud-provider sustainability disclosures (especially AWS).

Hardware-level specs (e.g., GPU TDPs) used as sanity checks.

Secondary syntheses that summarize typical per-query footprints for LLMs.

The goal is awareness and comparison, not precise accounting.

1. Model-specific anchor: GPT-OSS-120B-Eagle3

For GPT-OSS-120B, we rely on NVIDIA’s model card on Hugging Face:

The model card for nvidia/gpt-oss-120b-Eagle3 reports:

Cumulative compute: 4.8×10²⁰ FLOPs

Estimated training energy: 2,500 kWh

Estimated training emissions: 0.8075 tCO₂e (≈ 808 kg CO₂e)

We use this only to understand the scale of the model and to distinguish it from small models; we do not derive inference-time mg/token figures directly from this training data.

Reference

NVIDIA, gpt-oss-120b-Eagle3 model card on Hugging Face.

2. Per-prompt & per-token inference anchors

To get a reasonable band for “CO₂ per LLM query” and “CO₂ per token,” we draw on:

2.1 Google Gemini Apps

Google’s technical post on Gemini inference reports that, using their methodology:

The median Gemini Apps text prompt uses about 0.24 Wh of energy,

emits 0.03 g CO₂e,

and uses 0.26 mL of water per prompt.

If you assume a typical Gemini interaction is on the order of a few hundred tokens (≈ 400 is a reasonable ballpark), this implies:

0.24 Wh / 400 ≈ 0.0006 Wh per token

0.03 g / 400 ≈ 0.075 mg CO₂ per token

These are on a relatively efficient infrastructure stack and use Google’s carbon-accounting choices (market-based factors, median prompts), which tend to understate emissions compared with more conservative methods.

2.2 Mistral Large 2 (“Le Chat”)

Mistral’s lifecycle analysis for Large 2, conducted with ADEME and Carbone 4, provides a contrasting data point:

A 400-token response from their Le Chat assistant emits 1.14 g CO₂e and uses 45 mL of water, excluding user devices.

Normalized:

1.14 g / 400 ≈ 2.85 mg CO₂ per token

This is closer to a “high end” marginal inference cost, especially for large, dense models under a stricter lifecycle boundary.

2.3 Aggregate range from secondary analyses

A recent synthesis of public disclosures and studies (including Google’s Gemini data, Mistral’s LCA, and estimates for ChatGPT-like systems) summarizes the current landscape as:

~0.03–1.14 g CO₂e per LLM query,
depending on provider, model size, accounting method, and prompt size.

That article also notes:

Google / Gemini: ~0.24 Wh and 0.03 g CO₂ per query (efficient lower bound);

Mistral / Large 2: ~1.14 g CO₂ per ~400-token query (upper bound for large dense models).

How we use this

From these anchors, we treat per-token emissions as plausibly living in a band of:

Roughly 0.07–3 mg CO₂ per token, depending on model, provider, and accounting.

For the extension defaults:

Large models (GPT-4-class, GPT-OSS-120B-class) are assigned ~0.3–0.5 mg CO₂/token.

Smaller / more efficient models are assigned lower constants.

These values sit comfortably inside the published range above and are meant as conservative mid-range figures, not exact measurements.

3. Cloud vs on-premises carbon intensity (Bedrock)

For AWS Bedrock–hosted models, we apply a slightly lower per-token factor than for a generic “unknown data center,” based on AWS’s public sustainability studies:

An AWS–commissioned 451 Research study indicates that moving workloads from typical on-premises data centers to AWS can reduce carbon emissions by roughly 88% (i.e., AWS workloads emit ~12% of typical on-prem).

AWS and third-party blogs summarize this as “up to 80%–93% lower carbon emissions” for some workloads, due to higher energy efficiency and cleaner power mix.

We do not directly multiply our per-token constants by 0.12 or similar; instead:

These numbers justify treating Bedrock as “lower than generic”

and motivate future calibration using the AWS Customer Carbon Footprint Tool (account-level monthly KG CO₂ divided by Bedrock token counts, if available).

4. Hardware-level sanity checks (GPU power)

To make sure the per-token constants are physically plausible, we compare them against what you’d get from realistic GPU power draws and throughput:

NVIDIA’s H100 Tensor Core GPU datasheets list:

SXM5 configuration: up to 700 W TDP

PCIe configurations at lower TDP.

If you assume (for example):

~700–1,000 W effective draw per GPU under load,

a few hundred tokens/second per GPU for a large model,

you get sub-milligram CO₂ per token on a moderately clean grid, which is consistent with the 0.3–0.5 mg/token defaults derived from the vendor per-query data above. (We don’t rely on a specific paper here; we just use TDPs + typical token/sec to check that our constants are not wildly off.)

5. Methodological background

For the overall design pattern (“tokens × factor”) and lifecycle framing, we lean on:

Mistral’s lifecycle analysis of Large 2, which separates training and marginal inference impacts and reports explicit per-query figures.

Hugging Face’s review of environmental impact disclosures across major AI vendors, which highlights Mistral’s 1.14 g CO₂ / 400-token disclosure and similar numbers.

These sources support treating:

Training emissions as a large, mostly fixed cost amortized over many tokens, and

Inference as dominated (for a single query) by operational energy use (what our per-token factors try to capture).

6. How the extension uses these numbers

In code, the extension currently:

Estimates token counts for each exchange (using exact counts from the provider if available; otherwise approximating from character length).

Chooses a per-token factor (mg CO₂/token) based on model name and provider (OpenRouter vs Bedrock), falling back to a configurable default rooted in the ranges above.

Computes:

co2_mg = total_tokens * mg_per_token

co2_g = co2_mg / 1000

Optionally: energy_kwh = (co2_g / 1000) / grid_emission_factor_kg_per_kwh

Displays this in a short, clearly labeled footer that states it is:

an estimate,

based on remote data-center operations, not local hardware,

and meant for awareness, not audit-grade reporting.

All constants are exposed via configuration so they can be updated as:

Providers publish more detailed disclosures, or

You calibrate against real account-level data (e.g., AWS’s carbon footprint tool + token logs).

7. Suggested further reading / potential sources to link in the repo

You might want to link these directly in your repo’s docs:

NVIDIA model card: gpt-oss-120b-Eagle3 (training energy & emissions).

Google Cloud blog: Measuring the environmental impact of AI inference (Gemini prompt numbers).

Mistral blog: Our contribution to a global environmental standard for AI (Large 2 LCA & 400-token response figures).

Hugging Face blog: What kind of environmental impacts are AI companies disclosing? (summary of vendor disclosures, including Mistral’s 1.14 g CO₂ / 400 tokens).

Arbor / Sustainability by Numbers / similar syntheses on “0.03–1.14 g CO₂ per AI query” ranges.

AWS sustainability posts and 451 Research study summary on 80–88% lower emissions vs typical on-prem.

NVIDIA H100 datasheet and architecture whitepaper for TDPs and system power context.