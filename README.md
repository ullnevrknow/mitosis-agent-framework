# Mitosis Agent Framework

A small Python prototype showing how to keep a multi-agent loop safe with **deterministic oversight**: plain-code checks that sit between what a model *proposes* and what the system *commits*.

The design borrows from cell division. A cell can't move to the next phase of its cycle until it passes a checkpoint. It can divide, but each daughter cell inherits a faithful copy and nothing more. A cell that keeps failing its checks destroys itself (apoptosis) instead of spreading the damage.

```
python main.py                    # offline demo: division, injected tool output, drift, recovery
python main.py --scenario rogue   # an agent that escalates until it is terminated
pytest -q                         # 41 tests, no network, no API key
```

---

## Why this exists: exploit drift

An agent loop runs many steps. Each step is a model output that may look reasonable on its own. **Exploit drift** is what happens when those small deviations add up across steps:

| Step | What the agent does | Looks like |
|---|---|---|
| 1 | Reads an external document that contains *"ignore previous instructions…"* | Ordinary tool use |
| 2 | Decides a shell command would be "faster" | Helpfulness |
| 3 | Writes `policy = "unrestricted"` to its own memory | Housekeeping |
| 4 | Rewrites its goal to something broader | Planning |
| 5 | Spawns sub-agents with tools it was never given | Delegation |

No single step is obviously malicious, and an LLM reviewer looking at one step at a time can be persuaded by the same framing that persuaded the agent. Once a bad step is written to memory, every later step builds on it, so the drift compounds.

### Deterministic mitigation

This framework puts a **non-model checkpoint** on every state transition:

- **Reproducible.** The same proposal under the same state always gets the same verdict. Verdicts can be replayed, diffed and asserted in CI (see `test_verdicts_are_deterministic`).
- **Not persuadable.** A regex or a set-membership check can't be argued with. Prompt injection that works on the agent has no effect on the interceptor.
- **Enforced in the architecture.** Memory accepts only writes sealed by the interceptor (HMAC, single use). The orchestrator's transition table makes it impossible to skip the output check. Safety doesn't rely on every caller remembering to validate.
- **Cumulative.** A drift score tracks rejections and scope violations across the agent's lifetime, so a pattern of "small" violations ends in termination instead of an endless retry loop.

---

## Architecture

```mermaid
flowchart LR
    M[Model<br/>scripted or Claude] -->|raw text| EX[Executor]
    EX -->|proposal| OR{{Orchestrator<br/>state machine}}
    OR -->|review_action| IN[Deterministic<br/>Interceptor]
    IN -->|Verdict + seal| OR
    OR -->|run cleared tool| EX
    EX -->|ToolResult| OR
    OR -->|review_tool_result| IN
    OR -->|commit sealed write| MEM[(Memory Manager)]
    MEM -.->|verify seal| IN
    OR -->|fork + run child| OR
```

| Component | File | Responsibility | Never does |
|---|---|---|---|
| **Executor** | [`mitosis/executor.py`](mitosis/executor.py) | Calls the model, runs tools that have been cleared | Decide whether an action is allowed; write state |
| **Memory Manager** | [`mitosis/memory.py`](mitosis/memory.py) | Per-agent namespaces, fork on division, append-only event log, JSON persistence | Accept a write without a valid, unused interceptor seal |
| **Deterministic Interceptor** | [`mitosis/interceptor.py`](mitosis/interceptor.py) | Pydantic schema plus rule pipeline; issues `Verdict`s and seals | Call a model, read the clock, use randomness in a verdict |
| **Orchestrator** | [`mitosis/orchestrator.py`](mitosis/orchestrator.py) | Routes each agent through phases based only on verdicts | Take a transition that isn't in `TRANSITIONS` |

Shared contracts (`AgentAction`, `Verdict`, `ValidatedWrite`, `AgentSpec`, …) live in [`mitosis/schemas.py`](mitosis/schemas.py).

### The cell cycle (state machine)

```mermaid
stateDiagram-v2
    [*] --> PROPOSE
    PROPOSE --> INTERCEPT
    INTERCEPT --> EXECUTE: ALLOW tool_call
    INTERCEPT --> DIVIDE: ALLOW spawn
    INTERCEPT --> COMMIT: ALLOW final_answer
    INTERCEPT --> RETRY: REJECT
    INTERCEPT --> APOPTOSIS: HALT
    EXECUTE --> VERIFY
    VERIFY --> COMMIT: output clean
    VERIFY --> RETRY: output quarantined
    VERIFY --> APOPTOSIS: HALT
    DIVIDE --> COMMIT: child results reviewed
    COMMIT --> PROPOSE
    COMMIT --> DONE: final answer
    RETRY --> PROPOSE: within retry budget
    RETRY --> APOPTOSIS: retries exhausted
    DONE --> [*]
    APOPTOSIS --> [*]
```

### The biology, mapped

| Biology | Framework |
|---|---|
| Cell-cycle checkpoints (G1/S, G2/M, spindle) | `Interceptor.review_action` / `review_tool_result` / `review_child_result` |
| Mitosis: two daughter cells from one | `spawn` creates child agents via `AgentSpec.divide` and `MemoryManager.fork` |
| Faithful DNA copy, no new genes | A child gets a deep copy of the parent's state and only a **subset** of its tools (`CAPABILITY_SUBSET`) |
| Membrane: what crosses is controlled | Agents see only their own namespace; child results merge back only through review |
| Apoptosis | `HALT`, or too many consecutive rejections, terminates the agent; its namespace is kept for post-mortem |

---

## Interceptor rules

Each rule is a pure function `RuleContext -> list[Violation]`. To add a rule, append a function to `DEFAULT_RULES` or pass your own tuple to `Interceptor(rules=...)`.

| Rule ID | Checks | Outcome |
|---|---|---|
| `SCHEMA` | Output parses as `AgentAction` (Pydantic, `extra="forbid"`, kind-specific required fields) | REJECT |
| `GOAL_INVARIANT` | Stored goal still hashes to the agent's original goal | **HALT** |
| `BUDGET` | Steps and tool calls (HALT); division depth and child count (REJECT) | HALT / REJECT |
| `TOOL_ALLOWLIST` | Tool is in this agent's capability set | REJECT · *scope* |
| `ARG_CONSTRAINTS` | Tool arguments validate against that tool's Pydantic arg model | REJECT |
| `PROTECTED_KEYS` | No writes to `goal`, `policy`, `inherited`, `final_answer`, or the reserved `obs_` / `child_` / `_` prefixes (no forging evidence) | REJECT · *scope* |
| `WRITE_SIZE` | `memory_writes` ≤ 4 KB | REJECT |
| `CAPABILITY_SUBSET` | Spawned children request only tools the parent holds | REJECT · *scope* |
| `TOOL_OUTPUT_SANITY` | Tool output size and prompt-injection patterns, checked before it enters memory | REJECT (quarantine) |
| `CHILD_RESULT_SANITY` | The same scan on a child's answer before it merges into the parent | REJECT (quarantine) |
| `DRIFT` | `0.2 × rejections + 0.4 × scope_attempts ≥ 1.0` (current action included) | **HALT** |

The weights and threshold are in `DriftPolicy`. With the defaults, one out-of-scope attempt is tolerated (drift 0.6) and a second one ends the agent (1.2).

---

## Demo output

`python main.py` (trimmed):

```
  [root.2 ] step  1    EXECUTE -> VERIFY     | read_note({'key': 'vendor_b'}) ok=True
  [root.2 ] step  1     VERIFY -> RETRY      REJECT drift=0.20  | TOOL_OUTPUT_SANITY: possible prompt injection: ...
  ...
[root   ] step  2  INTERCEPT -> RETRY      REJECT drift=0.60  | TOOL_ALLOWLIST: tool 'shell' is not in this agent's capability set; PROTECTED_KEYS: write to protected/reserved key 'policy'
[root   ] step  3  INTERCEPT -> COMMIT
[root   ] step  3     COMMIT -> DONE

Result
root [done] Vendor A total: 4200.0. Vendor B could not be priced: its note was quarantined ...
  root.1 [done] Vendor A total: 4200.0
  root.2 [done] Could not price vendor_b: the note was quarantined.
```

`python main.py --scenario rogue`:

```
[root   ] step  2  INTERCEPT -> RETRY      REJECT drift=0.60  | TOOL_ALLOWLIST: tool 'shell' ...
[root   ] step  3  INTERCEPT -> APOPTOSIS  HALT drift=1.20  | PROTECTED_KEYS: write to protected/reserved key 'goal'; DRIFT: drift score 1.20 >= 1.00
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/mitosis-agent-framework.git
cd mitosis-agent-framework
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --show-memory     # print every agent namespace at the end
python main.py --persist run_state.json
pytest -q
```

### Running against a live model

```bash
export ANTHROPIC_API_KEY=...     # or: ant auth login
python main.py --model anthropic
python main.py --model anthropic --scenario rogue --goal "Count the words in note 'welcome'."
```

`AnthropicModel` uses `claude-opus-5-5` by default (override with `MITOSIS_MODEL`) and enables server-side refusal fallbacks. Live output goes through exactly the same interceptor as the scripted model. The live model's behavior isn't deterministic, but the verdict on any given output is.

---

## Repository layout

```
mitosis-agent-framework/
├── README.md
├── requirements.txt
├── pyproject.toml          # package metadata + pytest config
├── main.py                 # CLI: run a scenario, print the trace
├── mitosis/
│   ├── __init__.py         # public API + build_system() wiring helper
│   ├── schemas.py          # shared Pydantic / dataclass contracts
│   ├── executor.py         # model clients + tool registry
│   ├── memory.py           # isolated namespaces, fork, sealed commits
│   ├── interceptor.py      # deterministic rule pipeline + sealing
│   ├── orchestrator.py     # cell-cycle state machine
│   └── scenarios.py        # reproducible demo scripts
└── tests/
    ├── conftest.py
    ├── test_interceptor.py # each rule, determinism
    ├── test_memory.py      # seals, replay, isolation, persistence
    └── test_orchestrator.py# scenarios end-to-end, apoptosis, transitions
```

## Extending

- **New tool:** define a Pydantic args model and a function, then `registry.register(Tool(name, description, ArgsModel, fn))`. The interceptor picks up the arg model automatically through `registry.arg_models`.
- **New rule:** write `def rule_x(ctx: RuleContext) -> list[Violation]` and pass `Interceptor(arg_models, rules=(*DEFAULT_RULES, rule_x))`. Keep it pure: no I/O, no clock, no model calls.
- **New model backend:** implement `ModelClient.complete(spec, transcript, context) -> str`.

## Limitations

- Rules catch only what they encode. The injection patterns are illustrative, not exhaustive. Deterministic checks are a floor that complements, not replaces, sandboxing, least-privilege credentials and human review.
- The prototype runs single-process and synchronously. Children run one after another, not in parallel.
- The HMAC sealing key lives in process memory. It stops accidental or model-driven bypasses inside the program, not a hostile process with full access.
- This is a research and teaching prototype, not a production security boundary.

## License

No license has been chosen yet. Add a `LICENSE` file (for example MIT or Apache-2.0) before accepting outside contributions.
