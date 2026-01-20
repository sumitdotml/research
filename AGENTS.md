# AGENTS.md

This file provides guidance to agents (claude code, codex, etc.) when working in this repository.

Here, "you" = the user (Sumit).
"Claude" = agent (not just Claude in general, fyi).

## Repository Overview

This is a **research learning repository** for cultivating a researcher mindset. It serves two purposes:

1. **Research work:** Papers, implementations, notes, and experiments
2. **Learning system:** Structured workflow and accountability tools that transform surface-level understanding into deep, explainable knowledge

The goal is to close the gap between "I can use X" and "I can explain why X exists, its tradeoffs, and how it connects to Y."

## Key Files

| File | Purpose |
|------|---------|
| `_system/research-mindset-profile.md` | Living profile: weaknesses, concept bank, learning log |
| `_system/interrogation-protocols.md` | Standalone self-interrogation questions for each checkpoint |
| `.claude/skills/research-checkpoint.md` | Skill: checkpoint interrogation after reading/implementing |
| `.claude/skills/concept-review.md` | Skill: periodic review of old concepts |

## The Learning Workflow

### 7-Pass Paper Reading

1. **Skim** - Get the structure, identify what kind of paper this is
2. **Annotate** - Color-coded by understanding level (green=clear, yellow=fuzzy, red=lost)
3. **Deep read** - Work through the hard parts
4. **LLM discussion** - Probe understanding, ask "why" questions
5. **Jupyter visualization** - Visualize key concepts, attention patterns, etc.
6. **Replication in Python** - Implement core ideas from scratch
7. **Document** - Blog post or work log entry

### Checkpoints with Interrogation

Three checkpoints gate progress through the workflow:

**Checkpoint A (after passes 1-3): Post-Reading**
- Did you understand the problem space?
- What existed before? What's the core insight? What are you uncertain about?

**Checkpoint B (after passes 5-6): Post-Implementation**
- Do you understand WHY, not just WHAT?
- Why each design choice? What would break if you changed X?

**Checkpoint C (before pass 7): Pre-Writing**
- Can you explain clearly to others?
- 30-second version, limitations, what a skeptic would challenge

Use `/research-checkpoint` to run these interrogations with follow-up probing.

---

## Workflow Demo: Complete Example

### Scenario: Studying the RLM (Recursive Language Models) Paper

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DAY 1: Reading Phase                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  You: [Complete passes 1-3: skim, annotate, deep read]                      │
│                                                                             │
│  You: /research-checkpoint                                                  │
│                                                                             │
│  Claude: "Which checkpoint? What paper?"                                    │
│  You: "Checkpoint A, RLM paper"                                             │
│                                                                             │
│  Claude: [Reads your profile, sees your weaknesses]                         │
│          "What problem does RLM solve? What existed before?"                │
│  You: "It handles long contexts by breaking them into chunks"               │
│  Claude: "That's WHAT it does. WHY is that approach better than just        │
│           extending context length? What's the tradeoff space here?"        │
│  You: "Extending context is expensive, O(n²) attention..."                  │
│  Claude: "What existed before RLM for long contexts?"                       │
│  You: "RAG, sliding window attention, sparse attention..."                  │
│  Claude: "How does RLM differ from RAG specifically?"                       │
│  [...continues probing until satisfied or gaps exposed...]                  │
│                                                                             │
│  Claude: "Assessment: You understood the problem space but couldn't         │
│           articulate how recursive self-calls differ from retrieval.        │
│           Status: cannot-explain-yet.                                       │
│           Added 'RLM/Recursive Language Models' to your concept bank."      │
│                                                                             │
│  Claude: "You added a new concept. Want to review an older one?"            │
│  You: "Sure"                                                                │
│  Claude: [Picks an old concept from your bank, e.g., KV Cache]              │
│          "Explain KV Cache. What is it, why does it exist, tradeoffs?"      │
│  [...mini review session...]                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DAY 2-3: Implementation Phase                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  You: [Complete passes 4-6: LLM discussion, Jupyter viz, Python impl]       │
│                                                                             │
│  You: /research-checkpoint                                                  │
│                                                                             │
│  Claude: "Which checkpoint? What paper?"                                    │
│  You: "Checkpoint B, RLM paper"                                             │
│                                                                             │
│  Claude: "Why did you implement the recursive decomposition that way?"      │
│  You: "The paper splits the prompt and has the model call itself..."        │
│  Claude: "That's what the paper said. Why does recursion help here?         │
│           What would break if you just chunked without recursion?"          │
│  You: "The model needs to aggregate information across chunks..."           │
│  Claude: "How does it aggregate? What's the mechanism?"                     │
│  [...continues with Checkpoint B questions...]                              │
│                                                                             │
│  Claude: "Assessment: You implemented it but couldn't explain why           │
│           recursive aggregation beats single-pass retrieval. Still          │
│           cannot-explain-yet. Added note: 'unclear on aggregation.'"        │
│                                                                             │
│  Claude: "Review an old concept?"                                           │
│  You: "No, I'll do it tomorrow"                                             │
│  Claude: "OK."                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DAY 4: Pre-Writing Phase                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  You: /research-checkpoint                                                  │
│                                                                             │
│  Claude: "Which checkpoint? What paper?"                                    │
│  You: "Checkpoint C, RLM paper"                                             │
│                                                                             │
│  Claude: "Explain RLM in 30 seconds."                                       │
│  You: [Gives explanation]                                                   │
│  Claude: "Now in 2 minutes, with more depth."                               │
│  You: [Gives longer explanation]                                            │
│  Claude: "What limitations should you proactively mention?"                 │
│  You: "Latency from multiple calls, depends on good decomposition..."       │
│  Claude: "What would a skeptical reviewer push back on?"                    │
│  You: "Whether the recursive overhead is worth it vs. sparse attention..."  │
│  Claude: "Does this connect to anything you've studied before?"             │
│  You: "It's related to how KV cache works - if you're calling yourself      │
│        recursively, you might be able to cache across calls..."             │
│  Claude: "Interesting. Does RLM actually do that? Or is that your idea?"    │
│  [...probes the connection...]                                              │
│                                                                             │
│  Claude: "Assessment: Clear explanations, identified limitations,           │
│           made connection to KV Cache (though speculative).                 │
│           Updating 'RLM' to can-explain."                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Periodic Review (Not Tied to Any Paper)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ANYTIME: Standalone Review                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  You: /concept-review                                                       │
│                                                                             │
│  Claude: [Reads your concept bank, prioritizes by age and status]           │
│          "Reviewing 2 concepts today:                                       │
│           1. RMSNorm (added 2026-01-21, never reviewed, cannot-explain-yet) │
│           2. Weight Tying (added 2026-01-21, never reviewed)                │
│                                                                             │
│           Starting with RMSNorm. Explain it without looking it up.          │
│           What is it, why does it exist, what are the tradeoffs?"           │
│                                                                             │
│  You: "It's like LayerNorm but without the mean centering..."               │
│  Claude: "Why remove the mean centering? What's the benefit?"               │
│  You: "I think it's faster..."                                              │
│  Claude: "You said 'I think'. Do you know or are you guessing?"             │
│  You: "Guessing."                                                           │
│  Claude: "Noted. What else?"                                                │
│  You: "I don't remember why it works as well without the mean."             │
│  Claude: "Status remains cannot-explain-yet. You knew the 'what'            │
│           but not the 'why'. Updated last-reviewed date.                    │
│                                                                             │
│           Next: Weight Tying..."                                            │
│  [...continues...]                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Reference: When to Use What

| Situation | What to do |
|-----------|------------|
| Finished reading a paper (passes 1-3) | `/research-checkpoint` → Checkpoint A |
| Finished implementing (passes 5-6) | `/research-checkpoint` → Checkpoint B |
| About to write/present (before pass 7) | `/research-checkpoint` → Checkpoint C |
| Claude says "Review an old concept?" | Say yes or no (your choice) |
| Want to test retention on old concepts | `/concept-review` |
| Haven't reviewed anything in a while | `/concept-review` |
| Preparing for an interview | `/concept-review` (multiple sessions) |

### When Claude Prompts You

1. **After every `/research-checkpoint`** that adds/updates a concept → "Want to review an old concept?"
2. **During `/concept-review`** → Claude picks concepts for you based on age/status
3. **Never unprompted** → Claude doesn't randomly remind you; you initiate

---

### Concept Bank

Concepts you've learned are tracked in `_system/research-mindset-profile.md` with:
- Source paper/book
- Date added and last reviewed
- Status: `cannot-explain-yet` | `can-explain` | `can-teach`
- Notes on what you know and don't know

Use `/concept-review` to periodically test retention and catch knowledge decay.

### Confidence Levels (Strict Assessment)

Status is assessed with **skeptical defaults** - `cannot-explain-yet` is the baseline, upgrades must be earned:

- **cannot-explain-yet** (DEFAULT) - Any hesitation, jargon without intuition, couldn't answer "why" or survive "what if" probes, facts without tradeoffs
- **can-explain** (requires ALL: no jargon OR jargon explained, confident on "why it exists", mentioned tradeoff unprompted, survived follow-ups)
- **can-teach** (rare - all above PLUS multi-level explanation, proactive limitations, connected to other concepts unprompted)

**Anti-delusion principle:** When in doubt, assign the lower level. The goal is to catch gaps, not validate effort. Downgrades during review are features, not failures.

## Working with This Repository

### When starting a new paper/concept:
1. Create a directory for the paper if needed
2. Work through the 7-pass workflow
3. Run `/research-checkpoint` at each checkpoint (A, B, C)
4. Concept bank is updated automatically after interrogation

### When reviewing:
1. Run `/concept-review` to test old concepts
2. Be honest about decay - it's natural
3. Profile is updated automatically based on actual performance

### When Claude assists:
- Claude reads `_system/research-mindset-profile.md` to understand your current weaknesses and concept bank
- During interrogation, Claude acts as a skeptical but fair reviewer, not a cheerleader
- Weak answers get follow-up probing; vague answers get pushed for specificity
- Claude assesses confidence honestly and **automatically updates** the concept bank (no manual bookkeeping needed)

## Directory Structure

```
research/
├── AGENTS.md                    # This file (repo guidance)
├── CLAUDE.md                    # Symlink to AGENTS.md
├── _system/                     # Learning infrastructure (separate from research work)
│   ├── research-mindset-profile.md   # Your profile and concept bank
│   └── interrogation-protocols.md    # Standalone reference
├── .claude/
│   └── skills/
│       ├── research-checkpoint.md
│       └── concept-review.md
└── [papers-and-concepts]/       # Your actual research work
```

## Philosophy

**The friction is the point.** This system is designed to prevent the most dangerous learning failure mode: *believing you understand something when you don't.*

Core principles:
- Uncomfortable interrogation surfaces gaps before interviews do
- Explicit uncertainty is a feature, not a bug
- `cannot-explain-yet` is the default state - upgrades must be earned
- Downgrades during review are valuable information, not failures
- "Almost" and "close enough" don't exist - either you can explain it or you can't
- Sounding confident is not the same as understanding

The goal isn't to feel good about learning. It's to actually learn.