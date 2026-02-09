# LLM-Based Code Review Pipeline (RAG + Checklist)

## High-Level Flow

```text
START
  |
  V
Receive (Code File, Checklist File)
  |
  V
Initialize empty list: `all_individual_reviews`
  |
  V
[FOR EACH item IN Checklist] ---------------------------+
  |                                                     |
  +--> [1. RAG Query]                                   |
  |      - Use checklist item text to search            |
  |        Vector DB for relevant book passages.        |
  |                                                     |
  +--> [2. Construct Focused Prompt]                    |
  |      - System Prompt: "Review this code for         |
  |        [checklist item]"                            |
  |      - Context: Retrieved book passages             |
  |      - Data: The *entire* code file                 |
  |                                                     |
  +--> [3. Call LLM]                                    |
  |      - Get a focused review on ONE aspect           |
  |                                                     |
  +--> [4. Store Result]                                |
  |      - Append the LLM's review to                   |
  |        `all_individual_reviews`                     |
  |                                                     |
[LOOP ENDS] <-------------------------------------------+
  |
  V
[5. Final Consolidation]
  |
  +--> [A. Construct Consolidation Prompt]
  |      - System Prompt: "You are a lead reviewer..."
  |      - Context: Full list of all individual reviews
  |
  +--> [B. Final LLM Call]
  |      - Ask the LLM to synthesize, de-duplicate,
  |        and merge all feedback into one report
  |
  V
[6. Return Final Comprehensive Review]
  |
  V
END
