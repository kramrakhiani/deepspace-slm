# DeepSpace-SLM: End-to-End Prompt Dataflow

This document details the lifecycle of an operator query through **DeepSpace-SLM**, tracing data movement from initial prompt ingestion, multi-agent bus dispatching, neural transformer passes, radiation defense guarding, and verified output delivery.

---

## 1. High-Level Pipeline Overview

The macro dataflow from user input to final displayed response:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '18px', 'fontFamily': 'Inter, system-ui, -apple-system, sans-serif', 'primaryColor': '#1e293b', 'primaryTextColor': '#f8fafc', 'primaryBorderColor': '#38bdf8', 'lineColor': '#38bdf8' }}}%%
flowchart TD
    P["<b>1. Operator Prompt</b><br/>Slash Command (/stock, /rad...) or Natural Language"] --> CP["<b>Command Parser & Intent Classifier</b>"]
    
    CP --> PKG["<b>Package AgentMessage</b><br/>Typed Payload, Priority, Timestamp, UUID"]
    
    PKG --> COORD["<b>2. AgentCoordinator Message Bus</b><br/>Capability Registry Lookup & Event Log"]
    
    COORD --> ROUTE{"<b>Target Subsystem</b>"}
    
    ROUTE -->|"Domain Intents"| AGENTS["<b>3. Domain Agent Subsystems</b><br/>Inventory, Radiation, Maintenance, Navigation"]
    ROUTE -->|"free_query Intent"| NEURAL["<b>4. Neural Inference Engine</b><br/>428K Transformer + RAP-G Guard + Sampler"]
    
    AGENTS --> SYNTH["<b>5. Response Synthesizer</b><br/>Format Prose vs Structured JSON"]
    NEURAL --> SYNTH
    
    SYNTH --> AUDIT["<b>Append to Event Ledger</b><br/>Monotonic ID & Audit Trail"]
    
    AUDIT --> DISP["<b>Operator Display</b><br/>Terminal CLI / Spacecraft Output"]
```

---

## 2. Multi-Agent Dispatch & Routing Dataflow

How operational commands are routed through the message bus to domain agents:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '17px', 'fontFamily': 'Inter, system-ui, -apple-system, sans-serif' }}}%%
flowchart TD
    CMD["<b>Operator Command</b><br/>e.g. /stock o2 canister, /rad 3000, /audit"] --> BUS["<b>AgentCoordinator Bus</b>"]
    
    subgraph Routing ["Intent-Based Capability Dispatch"]
        BUS --> I1["<b>InventoryAgent</b><br/>• check_stock<br/>• locate<br/>• forecast<br/>• get_alerts"]
        BUS --> I2["<b>RadiationAgent</b><br/>• radiation_telemetry<br/>• get_defense_status"]
        BUS --> I3["<b>MaintenanceAgent</b><br/>• log_maintenance<br/>• verify_audit_chain<br/>• query_manual"]
        BUS --> I4["<b>NavigationAgent</b><br/>• mesh_update<br/>• mesh_merge<br/>• mesh_status"]
    end
    
    subgraph Execution ["Data Store & Hardware Operations"]
        I1 --> DB[("<b>SQLite Database</b><br/>Fuzzy Match & Stock Levels")]
        I2 --> RAPG["<b>RAP-G Engine</b><br/>Adjust Precision & TMR"]
        I3 --> DAG[("<b>Merkle DAG</b><br/>SHA-256 Chain Verification")]
        I4 --> DTN["<b>DTN Queue</b><br/>CRDT Delta Sync"]
    end
    
    RAPG -.->|"<b>High-Priority Broadcast</b><br/>Threat Escalation Alert"| BUS
    
    DB --> OUT["<b>Coordinator Output Formatter</b>"]
    RAPG --> OUT
    DAG --> OUT
    DTN --> OUT
```

---

## 3. Neural Inference Engine Dataflow (SLM Backbone)

The detailed forward pass when a free-form language prompt is routed to `InferenceAgent`:

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontSize': '17px', 'fontFamily': 'Inter, system-ui, -apple-system, sans-serif' }}}%%
flowchart TD
    Q["<b>Free-Form Prompt</b><br/>e.g. 'Assess life support oxygen reserves'"] --> TOK["<b>HabitatTokenizer</b><br/>Byte-Level BPE Tokenizer (Special Tokens)"]
    
    TOK --> EMB["<b>Token & Positional Embeddings</b><br/>RoPE (Rotary Position Embeddings)"]
    
    SENS["<b>Environmental Sensors</b><br/>Radiation, O2%, Cabin Pressure, Temp"] --> SEN_ENC["<b>SensorTelemetryEncoder</b><br/>Multi-Sensor Feature Projection"]
    
    SEN_ENC --> FUSION["<b>Context Fusion</b><br/>Tokens + Sensory Embeddings"]
    EMB --> FUSION
    
    subgraph Transformer ["DeepSpace-SLM Transformer Core (428K)"]
        FUSION --> NORM["<b>RMSNorm Normalization</b>"]
        NORM --> ATTN["<b>Differential MQA Attention</b><br/>Multi-Query Heads + KV Cache"]
        ATTN --> FFN["<b>SwiGLU & Sparse MoE</b><br/>Top-1 / Top-k Expert Routing"]
        FFN --> MEDUSA["<b>Medusa Speculative Heads</b><br/>Parallel Multi-Token Prediction"]
    end
    
    MEDUSA --> GUARD{"<b>RAP-G Threat Level</b>"}
    
    GUARD -->|"NOMINAL / ELEVATED"| SIMD["<b>Bare-Metal C SIMD Kernel</b><br/>INT8 Dot Product / GEMV (37 GOPS)"]
    GUARD -->|"CRITICAL Threat"| TMR["<b>Hardware TMR Voting</b><br/>Triple Modular Redundancy + INT4"]
    
    SIMD --> LOGITS["<b>Output Logits</b>"]
    TMR --> LOGITS
    
    LOGITS --> GRAMMAR["<b>Grammar Masker</b><br/>Finite-State JSON Schema Constraint"]
    GRAMMAR --> SAMPLE["<b>Sampler</b><br/>Greedy / Top-P / Top-K / Temperature"]
    SAMPLE --> DETOK["<b>BPE Detokenization</b><br/>Convert Token IDs to Text"]
    
    DETOK --> RESP["<b>Natural Language SLM Response</b>"]
```

---

## 4. Stage-by-Stage Lifecycle Walkthrough

### Stage 1: Input Ingestion & Packaging
- **Ingress**: Operators submit commands via the interactive CLI, mission scripts, or API endpoints.
- **Parsing**:
  - Slash commands (`/stock`, `/locate`, `/forecast`, `/alerts`, `/manual`, `/rad`, `/mesh`, `/audit`) are parsed into specific operational intents.
  - Free-form text questions are tagged with intent `free_query`.
- **Packaging**: Encapsulated into a typed `AgentMessage`:
  ```python
  AgentMessage(
      sender="user",
      intent="check_stock",
      payload={"item": "o2 canister"},
      priority=MessagePriority.NORMAL,
      trace_id="4f9c1b...",
  )
  ```

### Stage 2: Coordinator Bus Routing
- The `AgentCoordinator` queries its internal **Capability Registry**.
- Looks up the registered agent responsible for the message intent:
  - `check_stock`, `locate`, `forecast` $\rightarrow$ `InventoryAgent`
  - `radiation_telemetry`, `get_defense_status` $\rightarrow$ `RadiationAgent`
  - `log_maintenance`, `verify_audit_chain` $\rightarrow$ `MaintenanceAgent`
  - `mesh_update`, `mesh_status` $\rightarrow$ `NavigationAgent`
  - `free_query`, `perplexity` $\rightarrow$ `InferenceAgent`
- If an agent flags an event with `broadcast=True` (e.g., critical cosmic radiation spike), the coordinator replicates the message across all registered peer agents.

### Stage 3: Domain Subsystem Operations
- **InventoryAgent**: Queries local SQLite tables using fuzzy string matching, computes consumption depletion timelines, and updates quantities.
- **RadiationAgent**: Ingests space dosimetry telemetry ($\mu\text{Gy/h}$), updates threat thresholds (`NOMINAL`, `ELEVATED`, `CRITICAL`), and triggers model defense adaptation.
- **MaintenanceAgent**: Searches offline procedure manuals and commits SHA-256 blocks to the tamper-evident Merkle DAG.
- **NavigationAgent**: Manages Delay-Tolerant Networking (DTN) store-and-forward bundle queues and reconciles CRDT state across spacecraft modules.

### Stage 4: Neural Forward Pass (SLM Backbone)
- **Tokenization**: `HabitatTokenizer` encodes text into Byte-Level BPE subword token IDs.
- **Sensor Fusion**: Environmental telemetry is projected via `SensorTelemetryEncoder` and fused into the context.
- **Transformer Forward Pass**:
  - Pre-layer RMSNorm normalization.
  - Differential Multi-Query Attention (MQA) with persistent Key-Value cache.
  - Rotary Position Embeddings (RoPE).
  - SwiGLU non-linearities and Sparse Mixture-of-Experts (MoE) routing.
  - Medusa auxiliary heads for speculative multi-token decoding.
- **Hardware Defense & SIMD**:
  - Under normal conditions: AVX2/SIMD bare-metal C vector runtime (37.26 GOPS INT8).
  - Under radiation storms: INT4 quantization with Triple Modular Redundancy (TMR median voting) and ECC memory scrubbing.
- **Constrained Decoding**:
  - Finite-state grammar masking to enforce structured JSON output when requested.
  - Autoregressive sampling and BPE detokenization.

### Stage 5: Response Synthesis & Delivery
- **Aggregation**: The `AgentCoordinator` captures the `AgentResponse`.
- **Mode Formatting**: Outputs in either **Conversational Prose Mode** or **Machine Structured JSON Mode**.
- **Audit Logging**: Stamped with a monotonic transaction ID and recorded in the coordinator's event ledger.
