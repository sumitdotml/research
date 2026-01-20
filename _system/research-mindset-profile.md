# Research Mindset Profile

## About
Software Engineer (Tokyo). Targeting ML research positions.
Building researcher intuition through deliberate practice.

## Current Weaknesses (self-identified, 2026-01)
- **Retention:** Forget concepts after learning them
- **Articulation:** Can use things (e.g., RMSNorm) but can't explain why they exist
- **Connecting dots:** Struggle to link concepts across papers
- **Intuition:** Lack the "feel" for why choices are made

## Growth Focus
- Transform "I can use X" → "I can explain why X exists"
- Build hypothesis-test-update thinking patterns
- Practice explicit uncertainty: always articulate what I don't know

## Learning Workflow (7-pass)
1. Skim
2. Annotate (color-coded by understanding level)
3. Deep read
4. LLM discussion (probe understanding)
5. Jupyter visualization
6. Replication in Python
7. Document (blog/work log)

---

## Concept Bank

### RMSNorm
- **Source:** Used in LLaMA, originally from "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Used in implementations but can't articulate why it exists vs LayerNorm. What does removing the mean centering actually do?

### LayerNorm
- **Source:** "Layer Normalization" (Ba et al., 2016), used in original Transformer
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Know it normalizes across features. Can't explain why this specific normalization helps training or why it's placed where it is.

### KV Cache
- **Source:** Standard transformer inference optimization
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Know it speeds up autoregressive generation. Can't explain the memory tradeoffs or when it breaks down.

### Attention Sink
- **Source:** "Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023)
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Heard the term, don't understand why first tokens accumulate attention or how the fix works.

### FFN Dimension Expansion
- **Source:** Original Transformer (Vaswani et al., 2017)
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Know FFN goes d_model → 4*d_model → d_model. Can't explain WHY this expansion helps or what the network learns in that expanded space.

### Transformer Time Complexity
- **Source:** Attention mechanism analysis
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Know it's O(n²) for attention. Can't break down full complexity or explain where the bottlenecks actually are in practice.

### Weight Tying
- **Source:** "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
- **Date added:** 2026-01-21
- **Last reviewed:** —
- **Status:** cannot-explain-yet
- **Notes:** Know embedding and output projection can share weights. Can't explain why this works or what the tradeoffs are.

---

## Learning Log

<!--
Format for each entry:
### Paper/Concept Title (YYYY-MM-DD)
- **Checkpoint:** A/B/C
- **Result:** Summary of interrogation
- **Confidence:** cannot-explain-yet | can-explain | can-teach
- **Gaps identified:** What needs more work
- **Concepts added to bank:** List
-->

(papers/concepts studied with dates and checkpoint results)
