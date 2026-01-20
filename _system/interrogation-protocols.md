# Interrogation Protocols

Self-interrogation questions for each checkpoint in the 7-pass reading workflow. Use these to test your understanding before moving forward.

---

## Checkpoint A: Post-Reading (after passes 1-3)

**Focus:** Did you understand the problem space?

### Core Questions
1. **Problem:** What problem does this paper solve? What existed before?
2. **Insight:** What's the core hypothesis or insight?
3. **Unknowns:** What are you uncertain about? List your explicit unknowns.
4. **Risk:** What's the riskiest assumption the authors made?

### Red Flags (you're not ready to proceed if...)
- You can only describe WHAT, not WHY
- You can't articulate what existed before this paper
- You have no explicit unknowns (everyone has unknowns after first read)
- You can't identify any assumptions the authors made

---

## Checkpoint B: Post-Implementation (after passes 5-6)

**Focus:** Do you understand WHY, not just WHAT?

### Core Questions
1. **Design choices:** Why did you make each design choice? ("The paper said so" is NOT acceptable)
2. **Fragility:** What would break if you changed X? (Pick specific components)
3. **Minimum test:** What's the minimum experiment that would test your understanding?
4. **Surprises:** What did you learn that surprised you?

### Red Flags (you're not ready to proceed if...)
- You copied code without understanding each line
- You can't explain why specific hyperparameters/choices were made
- Changing one component would confuse you about what else breaks
- You had no surprises (means you weren't paying attention or just confirmed priors)

### Useful Probes
- "If I removed [component], what would happen?"
- "Why this loss function and not [alternative]?"
- "What's the computational bottleneck?"

---

## Checkpoint C: Pre-Writing/Pre-Presentation (before pass 7)

**Focus:** Can you explain clearly to others?

### Core Questions
1. **30-second version:** Explain this in 30 seconds.
2. **2-minute version:** Now explain it in 2 minutes.
3. **Limitations:** What are the limitations you should proactively mention?
4. **Skeptic's view:** What would a skeptical reviewer push back on?
5. **Connections:** Does this connect to anything you've studied before?

### Red Flags (you're not ready to proceed if...)
- Your 30-second version is just jargon
- You can't scale explanation from short to long
- You can't think of any limitations (everything has limitations)
- You don't know what a skeptic would challenge

### The Llion Jones Test
Imagine explaining this to someone deeply technical who will ask hard follow-ups:
- What if they ask "why not just..."?
- What if they point out a weakness?
- Can you defend the approach without getting defensive?

---

## Follow-Up Protocol (Self-Application)

After answering each question, check yourself:

| If your answer... | Ask yourself... |
|-------------------|-----------------|
| Is vague | "Can I be more specific? Give a concrete example?" |
| Sounds like recitation | "Why does this matter? What if it were different?" |
| Lacks uncertainty | "What am I uncertain about? What could go wrong?" |
| References "the paper says" | "But WHY does that make sense? Could it be wrong?" |

---

## Confidence Levels

Honestly assess yourself:

- **cannot-explain-yet:** Struggled with core questions, shallow answers, can't go beyond surface
- **can-explain:** Articulated core ideas and tradeoffs, handled follow-ups reasonably
- **can-teach:** Explained clearly, mentioned limitations proactively, connected to other concepts

---

## Using This Without Claude

1. Before each checkpoint, open this file
2. Go through each question, writing your answers
3. Apply the follow-up protocol to each answer
4. Note red flags honestly
5. Assign yourself a confidence level
6. If `cannot-explain-yet`, don't proceed - go back and study more

The friction is the point. If it feels uncomfortable, it's working.
