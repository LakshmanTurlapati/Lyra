# Feature Landscape

**Domain:** Dataset curation and small LLM fine-tuning (synthetic data distillation)
**Project:** Lyra -- Opus-quality training data for SmolLM2-1.7B
**Researched:** 2026-04-20
**Mode:** Ecosystem

## Table Stakes

Features users expect from an open-source dataset + fine-tuned model release. Missing any of these and the project feels incomplete or unusable.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ShareGPT-format dataset files | Industry standard for conversation fine-tuning; native HuggingFace/Unsloth/Axolotl support | Low | Must follow strict role ordering: human/gpt alternate, with function_call and observation roles for tool calls |
| Dataset card (README) on HuggingFace | 86% of top-100 downloaded datasets have complete cards; users need to evaluate fitness before downloading | Low | Include: description, structure, creation methodology, intended uses, limitations, license, statistics |
| Model card on HuggingFace | Required for discoverability, reproducibility, trust; HuggingFace ranks models with complete cards higher | Low | Include: metadata YAML (pipeline_tag, library_name, language, license, datasets, base_model), usage examples, training params, benchmarks |
| Model weights in safetensors format | Safer and faster than pickle; community standard since 2024; required for HuggingFace inference widgets | Low | Use safetensors, not .bin or .pth; HuggingFace provides conversion tools if needed |
| Reproduction scripts | Users expect to verify claims and adapt to their use cases; "trust but verify" culture in open-source ML | Medium | End-to-end: data generation, preprocessing, training, evaluation; must run without modification on standard hardware |
| MIT license on all artifacts | Stated project constraint; maximizes adoption; no commercial restrictions | Low | Apply to datasets, model weights, scripts, and eval code consistently |
| Data quality filtering pipeline | Synthetic data from any source contains noise, duplicates, and failures; unfiltered datasets are considered amateur | Medium | Must include: deduplication (MinHash or exact), format validation, response quality checks, conversation coherence verification |
| Evaluation results with standard benchmarks | Users compare models on common leaderboards; results without benchmarks are unverifiable marketing | Medium | Minimum: MMLU (general), HumanEval or MBPP (code), BFCL or custom (tool calling); report pass@1 |
| Train/validation/test splits | Basic ML hygiene; required for reproducible evaluation and overfitting detection | Low | Standard 90/5/5 or 80/10/10 split; stratified across the three focus areas |
| Conversation format correctness | ShareGPT has strict turn ordering; malformed data silently degrades training or crashes trainers | Medium | Tool call conversations require: human -> gpt (function_call) -> observation (tool output) -> gpt ordering; validate programmatically |

## Differentiators

Features that set Lyra apart from the dozens of synthetic datasets on HuggingFace. Not expected, but create real value.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Adaptive output style (terse code, detailed reasoning) | Most synthetic datasets use uniform verbosity; real-world utility requires context-appropriate responses; 1.7B token budget demands efficiency | Medium | Encode style in system prompts during generation; code responses should be concise, reasoning should show chain-of-thought; train the model to detect which style is needed |
| Tool calling with structured JSON function calls | Very few small models can do reliable function calling; fine-tuned 1.5B models achieve 77.5% on ToolBench (per Microsoft research); huge gap to fill at small model scale | High | Must cover: single function call, multiple functions, multi-turn with tool results, parallel execution, missing information handling; use OpenAI-compatible tool call format within ShareGPT |
| MCP-style tool use patterns | Model Context Protocol is emerging as the standard for tool integration; no existing 1.7B model targets this | High | Include MCP server discovery, tool listing, tool invocation, and result handling patterns; forward-looking differentiator |
| Custom eval benchmarks released alongside model | Most projects use only standard benchmarks; custom evals for tool-call accuracy and code quality specific to the model's strengths build trust and enable community iteration | Medium | Create benchmarks that test: tool call format compliance, argument extraction accuracy, code correctness for utility-scale tasks, reasoning coherence |
| Versioned dataset releases with iteration history | Most dataset releases are one-shot; iterative releases with documented improvements show rigor and enable research on data scaling | Low | Use HuggingFace dataset versioning or Git tags; document what changed between v1, v2, etc.; publish metrics at each version |
| Generation pipeline as a reusable tool | Most projects release data but not the generation pipeline; releasing the Opus-powered pipeline lets others generate domain-specific data | Medium | Must be configurable: prompt templates, topic distributions, quality thresholds, output format; parameterize the Opus API calls |
| Prompt template library for data generation | The templates used to prompt Opus are themselves valuable; they encode what makes "good" training data | Low | Organize by category (tool call, code, general); include system prompts, user prompt templates, and quality criteria |
| GGUF quantized model variants | Enables local deployment on consumer hardware without Python; LM Studio / llama.cpp compatible; dramatically broadens user base | Medium | Release Q4_K_M and Q8_0 at minimum; use separate HuggingFace repos with base_model metadata linking back to full model |
| Interactive demo Space on HuggingFace | Lets users try the model without any setup; significantly boosts visibility and HuggingFace trending ranking | Medium | Gradio or Streamlit Space; show tool calling, code generation, and general knowledge capabilities side by side |
| Per-category quality metrics | Report separate metrics for tool calls, code, and general knowledge instead of aggregate-only; reveals true capability profile | Low | Users need to know if the model is good at tool calls but weak at reasoning, or vice versa; honest reporting builds trust |

## Anti-Features

Features to explicitly NOT build in v1. These are scope traps that delay shipping.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-model support (other base models) | Splits focus; each model has different tokenizer, chat template, context limits; doubles testing surface | Target SmolLM2-1.7B only; document architecture so community can adapt to other models |
| RLHF/DPO alignment training | Adds massive complexity (reward model, preference data collection, training instability); SFT is sufficient for v1 capability validation | Pure supervised fine-tuning; if output quality is insufficient, add DPO in v2 with preference pairs from the SFT model's outputs |
| Human annotation pipeline | Expensive, slow, and unnecessary when Opus generates high-quality data; introduces coordination overhead | Use Opus for all data generation; add automated quality filtering (LLM-as-judge with a second model or rule-based) instead of human review |
| Long-context training data | SmolLM2-1.7B has 8192 token context (trained at 2048 sequence length); long-context data wastes tokens and degrades short-context performance | Keep all training conversations under 2048 tokens; this matches the model's training sequence length and optimizes for the short, practical interactions Lyra targets |
| Image/multimodal data | SmolLM2-1.7B is text-only; multimodal requires different architecture (SmolVLM exists separately) | Stay text-only; if multimodal is needed later, it is a separate project |
| Web UI for dataset curation | Building a UI is a massive time sink; CLI scripts serve the same purpose for the target audience (ML practitioners) | CLI-first pipeline with clear documentation; users who want a UI can use Argilla or HuggingFace's data viewer |
| Real-time API serving infrastructure | Deployment infrastructure is a separate concern; model weights are the deliverable, not a hosted service | Release weights + inference examples; users deploy on their own infrastructure (Ollama, LM Studio, vLLM, llama.cpp) |
| Automated continuous training pipeline | CI/CD for model training is over-engineering for an iterative research project | Manual training runs with documented configs; automate data generation and quality filtering, not the training loop |
| Prompt engineering optimization framework | Opus prompt optimization is research, not infrastructure; building a framework around it delays data generation | Document effective prompts in the template library; iterate manually; let the community optimize further |

## Feature Dependencies

```
Data Generation Pipeline
  |
  v
ShareGPT Format Dataset -----> Data Quality Filtering
  |                                    |
  |                                    v
  |                             Train/Val/Test Splits
  |                                    |
  v                                    v
Tool Call Schema Design         Fine-tuning Scripts
  |                                    |
  v                                    v
Conversation Format Validation  Model Weights (safetensors)
                                       |
                                       v
                                Evaluation Benchmarks
                                  |           |
                                  v           v
                           Standard Evals  Custom Evals
                                  |           |
                                  v           v
                               Model Card + Dataset Card
                                       |
                                       v
                                HuggingFace Release
                                  |           |
                                  v           v
                           GGUF Variants   Demo Space
```

Key dependency chains:

- Data generation MUST precede quality filtering (nothing to filter without data)
- Tool call schema design MUST precede dataset generation (format must be defined before Opus generates examples)
- Quality filtering MUST precede splitting (filter first, then split clean data)
- Training scripts MUST precede model weights (obvious, but scripts are a deliverable too)
- Evaluation benchmarks MUST precede model card (can't report results without benchmarks)
- Standard and custom evals are independent of each other but both feed the model card
- GGUF variants and demo Space depend on final model weights but are independent of each other

## MVP Recommendation

**Prioritize (ship with v0.1):**

1. **Tool call schema design and format specification** -- Everything downstream depends on this; get the ShareGPT + tool calling format right first. Define the exact JSON structure for function definitions, function calls, and tool results within ShareGPT conversations.

2. **Data generation pipeline (all three categories)** -- The core deliverable is the dataset. Generate ~5K samples split evenly across tool calling, code generation, and general knowledge. Use Opus with documented prompt templates.

3. **Data quality filtering** -- Deduplication, format validation, response quality scoring. Without filtering, the dataset has no credibility. Use exact-match dedup + structural validation + basic quality heuristics.

4. **Fine-tuning scripts with documented config** -- End-to-end training using TRL's SFTTrainer + LoRA (PEFT). Target consumer GPU (8GB+ VRAM). Document hyperparameters, hardware requirements, and expected training time.

5. **Evaluation with standard benchmarks** -- Run MMLU (general), MBPP (code), and a tool-calling accuracy eval. Report base model vs fine-tuned model comparison. This is the proof that the project works.

6. **HuggingFace release with complete cards** -- Dataset card, model card, safetensors weights, MIT license. Follow the HuggingFace model release checklist.

**Defer to v0.2:**

- **GGUF quantized variants**: Valuable for adoption but not needed to validate the approach. Add once the base model is proven.
- **Interactive demo Space**: Nice for visibility but not needed for technical validation.
- **Custom eval benchmarks**: Standard benchmarks suffice for v0.1; custom evals matter more at scale.
- **MCP-style tool patterns**: Start with standard function calling; MCP patterns are a v0.2 differentiator.
- **Generation pipeline as reusable tool**: In v0.1 the pipeline exists but is project-specific; making it reusable is a packaging exercise for v0.2.
- **Versioned dataset releases**: Starts mattering once there is a v0.2 dataset to compare against.

**Defer to v0.3+:**

- **Adaptive output style encoding**: Requires careful prompt engineering and evaluation methodology; adds complexity to data generation.
- **Per-category quality metrics reporting**: Useful once there is enough evaluation infrastructure to support granular reporting.

## Sources

- [Microsoft: Fine-Tuning Small Language Models for Function Calling](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/fine-tuning-small-language-models-for-function-calling-a-comprehensive-guide/4362539)
- [Small Language Models for Efficient Agentic Tool Calling (arXiv)](https://arxiv.org/abs/2512.15943)
- [HuggingFace: Model Release Checklist](https://huggingface.co/docs/hub/en/model-release-checklist)
- [HuggingFace: Dataset Cards](https://huggingface.co/docs/hub/en/datasets-cards)
- [SmolLM2-1.7B Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- [SmolLM2 Fine-tuning on Custom Synthetic Data](https://huggingface.co/blog/prithivMLmods/smollm2-ft)
- [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Axolotl: ShareGPT Conversation Format](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [Unsloth: Datasets Guide for Fine-Tuning](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide)
- [Cosmopedia: Large-Scale Synthetic Data Pipeline](https://huggingface.co/blog/cosmopedia)
- [HuggingFace Synthetic Data Generator (distilabel)](https://huggingface.co/blog/synthetic-data-generator)
- [Optimizing Function Calling with Small Language Models (Microsoft)](https://medium.com/data-science-at-microsoft/optimizing-function-calling-with-small-language-models-data-quality-quantity-and-practical-353be49b7a00)
- [NVIDIA: Mastering LLM Data Processing](https://developer.nvidia.com/blog/mastering-llm-techniques-data-preprocessing/)
- [SmolLM2-1.7B-Instruct-16k](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-16k)
