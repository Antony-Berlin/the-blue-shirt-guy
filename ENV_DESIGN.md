# Genesis Environment — Design Document

A technical walkthrough of how the Genesis environment works: how an episode runs, how tools get graded, how rewards are computed, and how the system decides which tools to evolve.

---

## Episode Flow

The environment follows a standard gym-style loop.

The agent starts an episode by calling `reset` — it receives a task description, starter code, and difficulty level. From there it works locally: calls tools, builds a solution, and when ready, submits the final code along with its complete tool call log.

On submission, the environment runs three things in parallel:

1. **Runs the hidden test suite** against the submitted code
2. **Grades every tool call** in the log
3. **Scores the agent's reasoning** via an LLM judge

All three feed into the final reward. The environment then returns the reward, a natural language critique, and the current EMA weight snapshot for each tool.

There's also an optional mid-episode call — the agent can submit individual tool calls for live grade feedback without waiting until the end. This doesn't affect the final evaluation; it just gives the agent signal while it's still working.

---

## How Tool Calls Get Graded

Every tool call in the log gets a grade between 0 and 1. The grading is content-aware — it looks at what the tool actually returned and picks the right grader for that output type.

**Content type detection** happens first. The system inspects the result string and classifies it as one of: code, test result, error traceback, search results, documentation, or plain text. That classification determines which grader runs.

**The graders:**

- **Code** → `CodeSyntaxGrader` + `CodeStyleGrader` (averaged). Syntax checks for parse errors, placeholder bodies, deep nesting, unimplemented stubs. Style checks cyclomatic complexity and variable naming.
- **Test result** → `TestResultGrader`. Parses PASSED/FAILED output — score is the fraction of tests passed. A failure with useful context scores slightly above zero; a bare failure scores zero.
- **Error traceback** → `ErrorExplanationGrader`. Graded on whether it identifies the error type, references the line number, and suggests a fix. More specific = higher score.
- **Search results** → `SearchRelevanceGrader`. Graded on result count and Jaccard similarity to the original query. No results scores zero.
- **Documentation** → `DocQualityGrader`. Graded on whether it includes a function signature, examples, and sufficient detail. Truncated output is penalised.
- **Plain text** → `ReasoningGrader`. Graded on information density and relevance to the task context.

If a tool raised an exception, the grade is capped at 0.2 regardless of content.

**Sequence-level adjustments** — applied by `RedundancyGrader` and `ErrorPropagationGrader` on top of per-call grades:

- **−0.3** for redundant calls — calling the same tool with the same args within a 3-call window
- **+0.2** for calling `explain_error` after a failure — the agent noticed the error and acted on it
- **−0.4** for ignoring an error and retrying the same thing — the agent didn't learn from the failure

Final grade per call is clamped to `[0.0, 1.0]`.

---

## Reward Formula

```
reward = (tests_passed / tests_total) × 0.6
       + mean(tool_call_grades)        × 0.2
       + reasoning_score               × 0.2
```

Test pass rate carries the most weight at 60% — did the code actually solve the problem? The remaining 40% splits evenly between tool quality and reasoning quality.

The tool score is the mean of all per-call grades across the episode. The reasoning score comes from the LLM judge — HuggingFace router first, Anthropic as fallback, a lightweight heuristic if neither is available.

Alongside the numeric reward, the environment returns **natural language feedback** from the judge — what went wrong, what was missing, what the agent should have done differently. This is the text the Tool Architect reads when deciding how to rewrite a failing tool.

---

## EMA Tool Tracking

Each tool maintains an exponential moving average of its performance, initialised at 0.5.

After every episode:

```
ema_weight = 0.3 × attributed_reward + 0.7 × previous_ema_weight
```

**Attribution:** Each tool's attributed reward is its last per-call grade that episode — a direct quality signal tied to that tool's actual output. If per-call grades aren't available, the episode reward is split proportionally by how many times each tool was called.

**Flagging thresholds:**

| EMA Weight | Status | Action |
|---|---|---|
| > 0.7 | KEEP | No action |
| 0.4 – 0.7 | REVIEW | Candidate for improvement |
| < 0.4 | REPLACE | Tool Architect target |

The EMA smoothing means a single bad episode won't trigger a rewrite — the system waits for a consistent downward trend before flagging a tool for replacement.

---

*For the high-level project overview, see [README.md](README.md).*
*For the reward formula rationale and grader weight derivation, see [reward_mechanism.md](reward_mechanism.md).*
