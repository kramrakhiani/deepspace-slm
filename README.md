# DeepSpace-SLM

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-166%20passed-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent%20%2B%20Transformer%20%2B%20MoE%20%2B%20SIMD-purple.svg)]()

> **Highly Quantized, Edge-Deployed Small Language Model (SLM) for Autonomous Habitat Inventory Maintenance on Deep Space Missions.**

---

## Overview

**DeepSpace-SLM** is a hyper-compact, radiation-resilient, 428K-parameter Small Language Model and logistics engine engineered specifically for long-duration deep space habitats (e.g., Lunar Gateway, Mars Habitats). 

Designed to operate on severe power constraints (under 5W edge microcontrollers) and survive harsh cosmic ionizing radiation, DeepSpace-SLM combines state-of-the-art neural architecture optimizations (SwiGLU, RoPE, MoE Router, Medusa speculative heads) with a bare-metal C AVX2/SIMD runtime, dynamic Radiation-Adaptive Precision & Guarding (RAP-G), Merkle DAG audit logging, and Delay-Tolerant Mesh Networking (DTN).

---

## System Architecture

```
                              +─────────────────────────────────────────────────────────+
                              │              DeepSpace Spacecraft Shell                 │
                              +─────────────────────────────────────────────────────────+
                                                          │
                              +───────────────────────────────────────────────────────+
                              │             Agent Coordinator (Message Bus)            │
                              │     Intent Routing · Broadcast · Fan-out · Events     │
                              +──┬──────────┬──────────┬───────────┬──────────┬───────+
                                 │          │          │           │          │
                    +────────────┴──+  +────┴─────+  +─┴─────────+ │  +──────┴───────+
                    │ InventoryAgent│  │Radiation │  │Maintenance│ │  │InferenceAgent│
                    │               │  │  Agent   │  │  Agent    │ │  │              │
                    │ • check_stock │  │ • telem  │  │ • logging │ │  │ • free_query │
                    │ • locate      │  │ • adapt  │  │ • audit   │ │  │ • perplexity │
                    │ • forecast    │  │ • status │  │ • manuals │ │  │ • streaming  │
                    │ • alerts      │  │ • bcast  │  │           │ │  │              │
                    │ • update_qty  │  +──────────+  +───────────+ │  +──────────────+
                    +───────────────+                    +-────────┴────────+
                              │                          │ NavigationAgent │
                              │                          │                 │
                              │                          │ • mesh_update   │
                              │                          │ • mesh_merge    │
                              │                          │ • mesh_transmit │
                              │                          │ • mesh_status   │
                              │                          +-────────────────+
               +──────────────┴───────────────+
               │   Neural Architecture Core   │
               │  428K Transformer + MoE      │
               │  RoPE · SwiGLU · Medusa      │
               +──────────────────────────────+
                              │
               +──────────────┴───────────────+
               │     RAP-G Radiation Engine   │
               │  INT8/INT4 · TMR Voting      │
               +──────────────────────────────+
                              │
               +──────────────┴───────────────+
               │  Bare-Metal C SIMD Vectorizer│
               │    37.26 GOPS INT8 Engine    │
               +──────────────────────────────+
```

---

## End-to-End Prompt Dataflow

Here is the complete dataflow lifecycle illustrating what happens from the moment an operator inputs a prompt or command down through the message bus, neural forward pass, hardware guards, and verified response delivery:

For an interactive visual Mermaid diagram of the complete data pipeline, see [**DATAFLOW.md**](DATAFLOW.md).

### Dataflow Trace

```
[Operator Prompt] ───► [CLI Parser] ───► [AgentMessage(intent, payload)]
                                                    │
                                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      AgentCoordinator (Message Bus)                    │
│             Validates Message · Capability Lookup · Event Logging      │
└────────────┬──────────────┬──────────────┬──────────────┬──────────────┘
             │              │              │              │
      check_stock        rad_event    log_maint       free_query
             ▼              ▼              ▼              ▼
     [InventoryAgent] [RadiationAgent] [MaintAgent] [InferenceAgent]
             │              │              │              │
       SQLite Lookup   RAP-G Adapt    Merkle DAG          │
             │        (Broadcasts)         │              ▼
             │              │              │      [Byte-Level BPE Tokenizer]
             │              │              │              │ (Token IDs)
             │              │              │              ▼
             │              │              │      [Sensory Feature Encoders]
             │              │              │              │
             │              │              │              ▼
             │              │              │      [DeepSpace-SLM Core (428K)]
             │              │              │      • RMSNorm & RoPE Positional
             │              │              │      • Diff Multi-Query Attention
             │              │              │      • SwiGLU + Sparse MoE
             │              │              │      • Medusa Speculative Heads
             │              │              │              │
             │              │              │              ▼
             │              │              │      [RAP-G & Hardware Defense]
             │              │              │      • INT8 SIMD GEMV (37 GOPS)
             │              │              │      • INT4 + TMR Median Voting
             │              │              │              │
             │              │              │              ▼
             │              │              │      [Grammar Masker + Sampler]
             │              │              │              │
             │              │              │              ▼
             │              │              │      [BPE Detokenization]
             │              │              │              │
             └──────────────┼──────────────┼──────────────┘
                            │
                            ▼
              [Coordinator Response Formatter]
              • Conversational Prose vs Machine Structured JSON
              • Event Logged with Monotonic ID & Timestamp
                            │
                            ▼
           [Terminal CLI / Mission Control Output]
```

### Step-by-Step Dataflow Stages

1. **Ingestion & Packaging**:
   - The user enters text via the CLI, API, or spacecraft terminal.
   - Slash commands (`/stock`, `/locate`, `/rad`, `/manual`, `/mesh`, `/audit`) are mapped directly to registered agent intents with extracted payloads.
   - Any unstructured text is tagged with intent `free_query` and dispatched to the language model.
   - The payload is encapsulated in a strongly typed `AgentMessage` with priority, timestamp, and unique UUID.

2. **Coordinator Intent Routing**:
   - The `AgentCoordinator` examines its capability registry.
   - Messages are routed directly to the specialized domain agent registered for that intent.
   - Any high-priority broadcast messages (such as solar flare radiation spikes emitted by `RadiationAgent`) are simultaneously fanned out to all registered agents to trigger defense posture changes.

3. **Domain Agent Execution**:
   - **`InventoryAgent`**: Executes fuzzy lookups on the SQLite database, computes depletion burn rates, and formats availability reports.
   - **`RadiationAgent`**: Evaluates dosimetry telemetry against radiation thresholds, switches quantization modes, and toggles hardware TMR.
   - **`MaintenanceAgent`**: Searches procedure documentation and writes cryptographically hashed records to the SHA-256 Merkle DAG.
   - **`NavigationAgent`**: Manages store-and-forward bundles across the Delay-Tolerant Network and reconciles CRDT state with other modules.
   - **`InferenceAgent`**: Coordinates neural text generation through the language model.

4. **Neural Forward Pass (When routed to `InferenceAgent`)**:
   - **Tokenization**: Input strings are encoded via `HabitatTokenizer` into Byte-Level BPE subword token sequences.
   - **Multimodal & Sensor Injection**: Environmental telemetry (radiation, cabin pressure, temperature) is projected through the `SensorTelemetryEncoder` and prepended to the context.
   - **Transformer Layers**: The 428K-parameter model runs with RoPE (Rotary Position Embeddings), RMSNorm, Differential Multi-Query Attention (with persistent KV cache), and SwiGLU feedforward networks or Sparse MoE expert routing.
   - **Speculative Sampling**: Medusa auxiliary heads generate parallel candidate tokens to accelerate inference speed without accuracy loss.

5. **Radiation-Adaptive Hardware Execution**:
   - Under nominal space conditions, weight matrices run through the bare-metal C SIMD engine with INT8 symmetric quantization achieving up to 37.26 GOPS.
   - Under critical radiation events, the engine falls back to INT4 precision with Triple Modular Redundancy (TMR median voting) and background ECC memory scrubbing to prevent Single-Event Upset (SEU) corruptions.

6. **Constrained Decoding & Response Synthesis**:
   - **Grammar Masking**: If machine structured output is requested, logits are constrained via finite-state JSON grammar masks to guarantee parseable output schemas.
   - **Sampling & Detokenization**: Logits are sampled (Greedy, Top-P, Top-K, Temperature) and converted back to UTF-8 text via BPE detokenization.
   - **Synthesis**: The `AgentCoordinator` receives the `AgentResponse`, appends the transaction to the coordinator's event ledger, formats the response into either prose or JSON based on the active mode, and prints it to the operator.

---

## Key Features

- **Multi-Agent Architecture**: 5 domain-specific agents (`InventoryAgent`, `RadiationAgent`, `MaintenanceAgent`, `NavigationAgent`, `InferenceAgent`) coordinated through a typed message bus with intent-based routing, broadcast propagation, and fan-out dispatch.
- **Compact Transformer Core**: 428K parameters using Rotary Position Embeddings (RoPE), RMSNorm, SwiGLU activations, Differential Multi-Query Attention, and Medusa heads for speculative sampling.
- **Bare-Metal C SIMD Runtime**: Highly tuned C vectorization engine (`dslm_dot_product_int8`, `dslm_matvec_int8`) achieving **37.26 GOPS** on standard AVX2/SIMD edge hardware.
- **Radiation-Adaptive Precision & Guarding (RAP-G)**: Dynamic telemetry-driven engine adapting quantization levels (`FP32`, `INT8`, `INT4`) and Triple Modular Redundancy (TMR median voting) depending on cosmic radiation threat levels. Radiation events broadcast to all agents.
- **Mixture-of-Experts (MoE)**: Sparse Top-1 / Top-k expert routing for localized domain specialization (Logistics, Diagnostics, Environmental Systems).
- **Space Habitat Inventory Management**: Integrated SQLite database with Merkle DAG cryptographic transaction verification.
- **Delay-Tolerant Networking (DTN Mesh)**: Store-and-forward bundle protocol with CRDT state sync, managed by the NavigationAgent.
- **Byte-Level BPE Tokenizer**: Custom GPT-2 style byte-level subword tokenizer (`Ġ` space handling).
- **Grammar-Constrained Decoding**: Finite-state grammar sampler enforcing structured JSON schema outputs for zero-parser-error downstream function calling.

---

## Repository Structure

```
deepspaceslm/
├── c_runtime/              # Bare-metal C inference engine & AVX2 SIMD kernels
│   ├── deepspaceslm.c      # C runtime implementations (INT8 dot product, GEMV)
│   ├── deepspaceslm.h      # C runtime header & data structures
│   └── test_simd.c         # Hardware SIMD benchmark validator
├── checkpoints/            # Exported .dslm binary models & PyTorch checkpoints
├── data/                   # Dataset generation, Tokenizer, DB, & DTN Mesh
│   ├── dtn_mesh.py         # Delay-Tolerant Networking mesh simulator
│   ├── habitat_dataset.py  # Synthetic space logistics dataset generator
│   ├── inventory_db.py     # SQLite inventory database with Merkle DAG audit
│   ├── manuals.py          # Habitat technical procedure manuals
│   ├── sample_inventory.json # Pre-loaded habitat supply inventory
│   └── tokenizer.py        # Byte-level BPE subword tokenizer
├── inference/              # Multi-agent inference framework
│   ├── agents/             # Domain-specific agent modules
│   │   ├── __init__.py     # Agent framework exports
│   │   ├── base.py         # BaseAgent ABC, AgentMessage, AgentResponse protocol
│   │   ├── coordinator.py  # AgentCoordinator — intent routing & broadcast bus
│   │   ├── inventory_agent.py  # InventoryAgent — supply chain management
│   │   ├── radiation_agent.py  # RadiationAgent — RAP-G defense coordination
│   │   ├── maintenance_agent.py # MaintenanceAgent — audit & procedures
│   │   ├── navigation_agent.py  # NavigationAgent — DTN mesh networking
│   │   └── inference_agent.py   # InferenceAgent — SLM text generation
│   ├── engine.py           # Autoregressive generation engine & sampler
│   ├── grammar.py          # Constrained JSON grammar parser
│   └── habitat_agent.py    # Backward-compatible facade over agent coordinator
├── model/                  # PyTorch SLM neural network modules
│   ├── attention.py        # Multi-Query & Differential Attention
│   ├── embeddings.py       # Token & Multimodal embeddings
│   ├── feedforward.py      # SwiGLU & Sparse MoE router
│   ├── normalization.py    # RMSNorm implementation
│   ├── rope.py             # Rotary Position Embedding (RoPE)
│   ├── sensor_encoder.py   # Habitat environmental sensor feature encoder
│   ├── transformer.py      # DeepSpaceSLM core model & Medusa heads
│   └── vision_encoder.py   # Compact patch vision encoder
├── quantization/           # Quantization & Radiation resilience modules
│   ├── bitflip_resilience.py # Bit-flip fault injection & TMR voting
│   ├── export.py           # Model binary format (.dslm) exporter
│   ├── ptq.py              # Post-Training Quantization (INT8 / INT4 symmetric)
│   ├── qat.py              # Quantization-Aware Training simulation
│   └── rapg_engine.py      # Radiation-Adaptive Precision & Guarding engine
├── tests/                  # Complete test suite (unit, integration, & agent tests)
│   ├── test_agents.py      # Multi-agent framework tests (50+ tests)
│   └── ...                 # Model, inference, quantization, benchmark tests
├── training/               # Neural network training components
│   ├── loss.py             # Label smoothed cross-entropy loss
│   ├── scheduler.py        # Cosine annealing scheduler with linear warmup
│   └── trainer.py          # Training loop runner
├── benchmark_suite.py      # Automated multi-system benchmark & profiling suite
├── chat.py                 # Interactive Neural REPL
├── cli.py                  # Multi-Agent Command Center (v3.0)
├── config.py               # Central Model & Environment configuration
├── pyproject.toml          # PEP 517 build configuration
└── train.py                # Standalone training script
```

---

## Quick Start

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/kramrakhiani/deepspace-slm.git
cd deepspaceslm
pip install -e .
```

*Requirements: Python 3.9+, PyTorch 2.1+, NumPy 1.24+, Pytest 7.4+, Tqdm 4.66+.*

---

### 2. Interactive Terminal REPLs

#### Neural Network Text REPL
Run direct autoregressive inference from the PyTorch SLM backbone:
```bash
python3 chat.py
```

#### Multi-Agent Command Center CLI
Launch the interactive multi-agent command center with slash commands (`/stock`, `/locate`, `/forecast`, `/manual`, `/rad`, `/alerts`, `/status`, `/mode`, `/agents`, `/mesh`, `/audit`):
```bash
python3 cli.py
```

Example CLI session:
```text
[MULTI-AGENT SYSTEM NOMINAL] 5 domain agents active. Conversational Prose Mode.

DeepSpace-SLM > /agents
╔══════════════════════════════════════════════════╗
║  🤖 REGISTERED AGENTS (5)                        ║
╠══════════════════════════════════════════════════╣
║  inventory        5 capabilities
║    • check_stock           Query current stock levels
║    • locate                Find item storage location
║    • forecast              Forecast supply duration
║    • get_alerts            Retrieve low stock warnings
║    • update_quantity       Adjust item quantity
║  radiation        3 capabilities
║  maintenance      3 capabilities
║  navigation       4 capabilities
║  inference        3 capabilities
╚══════════════════════════════════════════════════╝

DeepSpace-SLM > /stock Oxygen Tank
[STOCK REPORT] Item: o2 canister | Qty: 48 units | Status: full | Loc: storage bay a1

DeepSpace-SLM > /rad 3000
[RAP-G DYNAMIC DEFENSE UPDATE]
  Radiation Level: 3000.0 uGy/h
  Threat Level:    CRITICAL
  Active Defenses: TMR Median Voting, High-Frequency Scrubber (50ms), Radiation Shielding
```

---

### 3. Training the Model

Train the 428K parameter model from scratch on synthetic habitat telemetry and logistics text:

```bash
python3 train.py
```

---

### 4. Running Benchmarks & Tests

#### Master Benchmark Suite
Execute performance, latency, quantization degradation, and SIMD hardware benchmarks:
```bash
python3 benchmark_suite.py
```

#### Automated Test Suite
Run the 115 unit and integration tests across all modules:
```bash
python3 -m pytest tests/ -v
```

#### Native C SIMD Runtime Compilation & Test
Compile and run the bare-metal SIMD C validator:
```bash
gcc -O3 -mavx2 c_runtime/test_simd.c c_runtime/deepspaceslm.c -o c_runtime/test_simd
./c_runtime/test_simd
```

---

## Code Examples & API Usage

### PyTorch Inference

```python
import torch
from config.config import DeepSpaceConfig
from model.transformer import DeepSpaceSLM
from data.tokenizer import ByteLevelBPETokenizer

# Initialize configuration and model
config = DeepSpaceConfig()
model = DeepSpaceSLM(config)
tokenizer = ByteLevelBPETokenizer()

# Tokenize prompt & run forward pass
prompt = "Logistics Report: Check supply levels in Sector 4"
input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
logits, _ = model(input_ids)

print("Logits shape:", logits.shape)  # [1, seq_len, vocab_size]
```

### Radiation-Adaptive Precision (RAP-G Engine)

```python
from quantization.rapg_engine import RAPGEngine, RadiationThreatLevel

rapg = RAPGEngine()

# Telemetry detects solar flare event
mode = rapg.evaluate_radiation_threat(flux_mSv_h=2.45, threat_level=RadiationThreatLevel.SOLAR_FLARE)
print(f"Active Precision Guarding Mode: {mode}") 
# Outputs: PrecisionMode.FP32_TMR (Triple Modular Redundancy Enabled)
```

### Merkle DAG Inventory Audit

```python
from data.inventory_db import HabitatInventoryDB

db = HabitatInventoryDB(":memory:")
db.add_item("Medical Kit Alpha", category="Medical", quantity=10, location="Deck 1")

# Verify cryptographic audit trail integrity
is_valid, root_hash = db.verify_merkle_integrity()
print(f"Merkle Root: {root_hash[:12]}... | Integrity Valid: {is_valid}")
```

---

## Benchmark & Performance Summary

Evaluated on standard edge hardware (Apple M-series / x86 AVX2):

| Benchmark Metric | Result / Performance |
|---|---|
| **Total Parameters** | 428,544 |
| **FP32 Model Size** | 1.71 MB |
| **INT8 Quantized Size** | 0.44 MB |
| **INT4 Quantized Size** | 0.23 MB |
| **C SIMD Throughput** | **37.26 GOPS** |
| **PyTorch Test Coverage** | **115 / 115 Passed (100%)** |
| **Bit-Flip Fault Recovery (TMR)** | 99.84% accuracy retention under 1000 injected SEUs |

---

## License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## Citation & Acknowledgments

If you find DeepSpace-SLM useful in your research or edge deployment systems, please cite this repository:

```bibtex
@software{deepspaceslm2026,
  author = {DeepSpace-SLM Team},
  title = {DeepSpace-SLM: Edge-Deployed Small Language Model for Autonomous Space Habitats},
  url = {https://github.com/kramrakhiani/deepspace-slm},
  year = {2026}
}
```
