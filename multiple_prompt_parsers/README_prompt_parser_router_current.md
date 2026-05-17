# Prompt Parser Router / Graph Pipeline README

Status: current work-in-progress snapshot

This README describes the current prompt parser routing and graph pipeline files as they exist in this working set. It is intended to document what is supported today, what is only partially supported, what prompts may fail, and what the future direction should be.

## Attribution

This project currently combines three related parser layers:

1. `prompt_parser.py`
   - This is the modified Prompt Architect parser from Kittensx/Prompt-Architect-for-A1111, not the original Automatic1111 `prompt_parser.py`.
   - Project reference: `https://github.com/Kittensx/Prompt-Architect-for-A1111`

2. `prompt_parser_21.py`
   - The `_21` advanced backend parser is attributed to Konpr.
   - Author profile: `https://github.com/Konpr`
   - Referenced source location: `https://github.com/Konpr/whats-/tree/main/new`

3. New router / graph files
   - These files form the experimental compatibility and future graph execution layer around the modified regular parser and `_21` backend parser.

## Current file roles

### `prompt_parser_router.py`

This is the current canonical routing layer.

Responsibilities:

- Load `prompt_symbols.yaml` if available.
- Fall back to `PromptSymbolConfig.default()` if the YAML load fails.
- Convert user aliases into canonical backend syntax.
- Apply lightweight backend normalization.
- Detect advanced backend operators such as `BLEND`, `CHUNK`, `MORPH`, `ASSEMBLE`, `BIND`, and `POOL`.
- Route prompts between the regular/stable parser path and the `_21` advanced parser path.

Current limitation:

- It is still mostly a compatibility router, not a full graph compiler entry point.
- The future graph parser / normalizer / lowering / executor stack exists, but is not yet fully wired as the default execution path.

### `prompt_dispatcher.py`

This is now a backward-compatible wrapper around `prompt_parser_router.py`.

Purpose:

- Preserve old imports that expect `modules.prompt_dispatcher`.
- Re-export the router API.

New integrations should prefer importing from `modules.prompt_parser_router` directly.

### `prompt_combined_dispatcher.py`

This is also now a compatibility wrapper.

Purpose:

- Preserve old imports that expect `modules.prompt_combined_dispatcher`.
- Re-export everything from `modules.prompt_parser_router`.

### `prompt_symbol_defaults.py`

This contains the built-in default symbol config.

Current defaults include:

```text
BLEND    <+>
CHUNK    &&
MORPH    >>
POOL     $$
BIND     =>
ASSEMBLE @@
```

It also reserves:

```text
semantic_prompt: %%
```

Important note:

- `%%` is reserved for the separate `semantic_prompt` project and should not be reused by `BLEND` or any other prompt parser symbol.

### `prompt_symbols.yaml`

This is the user-editable symbol configuration file.

Purpose:

- Allow user-facing symbols and aliases to be changed without changing Python source.
- Normalize shorthand symbols into canonical backend operator names.

Important behavior:

- User YAML should merge with defaults.
- User aliases should take priority over default aliases where possible.
- Reserved symbols should not be reused by backend operators.

Known caution:

- Alias collisions can still make the YAML fail to load if two operators claim the same alias and collision handling is not enabled or correctly resolved.

### `prompt_symbol_interpreter.py`

This file loads and validates symbol configuration, then converts aliases into canonical syntax.

Responsibilities:

- Parse operator sections from default config and YAML.
- Validate alias collisions.
- Convert supported aliases into canonical forms like `BLEND{...}`.

Current limitation:

- Symbol interpretation is mostly surface-level canonicalization. It does not itself solve deep graph ambiguity or nested backend execution.

### `prompt_backend_normalizer.py`

This file performs lightweight backend lifting before routing.

Currently supported proof-normalization examples:

```text
{portrait BLEND{photo realism*0.9 | oil painting*1.5}}
```

becomes:

```text
BLEND{portrait photo realism*0.9 | portrait oil painting*1.5}
```

Similarly, simple wrapped `CHUNK` cases may be lifted.

Current limitations:

- This is intentionally lightweight.
- It only handles simple wrapped top-level backend blocks.
- It is not a complete nested backend graph normalizer.
- It should not be treated as full support for arbitrary nested `BLEND`, `CHUNK`, `MORPH`, `POOL`, `BIND`, or `ASSEMBLE` blocks.

### `prompt_graph.py`

This is the shared graph / AST schema layer.

It defines node types such as:

- `TextNode`
- `SequenceNode`
- `GroupNode`
- `AlternateNode`
- `ScheduleNode`
- `WeightNode`
- `BlendNode`
- `ChunkNode`
- `MorphNode`
- `PoolNode`
- `BindNode`
- `AssembleNode`
- `AndNode`

Current status:

- The schema is future-facing and broadly correct as a shared representation.
- It does not execute anything by itself.

### `prompt_graph_parser.py`

This parses canonical prompt text into a `PromptGraph`.

Responsibilities:

- Tokenize structural syntax.
- Recognize canonical backend keywords.
- Build graph nodes for text, groups, weights, alternates, schedules, and backend operators.

Current limitations:

- This is still experimental.
- It should be run after symbol canonicalization, not directly on arbitrary shorthand input.
- It may parse more structures than the runtime can safely execute.
- Deep nested backend syntax should still be treated as future support, not guaranteed current support.

### `prompt_graph_normalizer.py`

This rewrites graph structures into cleaner semantic forms.

Responsibilities:

- Remove empty text nodes.
- Flatten simple sequences.
- Collapse redundant groups.
- Normalize backend nodes.
- Lift simple backend nodes out of groups or sequences where the meaning is clear.

Current supported direction:

```text
{portrait BLEND{realism | oil}}
```

should normalize conceptually toward:

```text
BLEND{portrait realism | portrait oil}
```

Current limitations:

- Simple backend lifting is not the same as arbitrary nesting support.
- Some backend combinations are semantically ambiguous and need graph lowering instead of text rewriting.

### `prompt_graph_validator.py`

This validates a graph before lowering or execution.

Responsibilities:

- Check empty branches.
- Check invalid weights.
- Check invalid blend modes.
- Check invalid channel targets.
- Check invalid morph curves or boundaries.
- Warn or error on unsupported backend nesting depending on strictness flags.

Current limitation:

- Validator support is ahead of actual runtime support in some areas.
- Passing validation does not necessarily mean every advanced graph path is fully executable in A1111 today.

### `prompt_graph_lowering.py`

This converts a normalized graph into a `PromptExecutionPlan`.

Responsibilities:

- Lower simple text into prompt calls.
- Lower backend nodes into backend calls or merge operations.
- Convert complex graphs into a sequence of calls plus operations.

Current limitations:

- Lowering is future-facing.
- It depends on parser, validator, executor, and conditioning ops being complete.
- Nested backend lowering is the intended future solution, but should not yet be advertised as fully stable.

### `prompt_execution_plan.py`

This defines the execution plan schema.

Important structures:

- `PromptCall`
- `PlanOperation`
- `PlanReference`
- `ConditioningOutput`
- `ExecutionResult`
- `ExecutionTrace`
- `PromptExecutionPlan`

Current limitation:

- `MergeMode` currently covers core backend-style operations, but graph-level `ALTERNATE` and `SCHEDULE` are not yet first-class merge modes unless added later.

### `prompt_graph_executor.py`

This executes lowered plans.

Responsibilities:

- Call a backend adapter for primitive conditioning calls.
- Resolve operation dependencies.
- Dispatch merge operations.
- Cache results.
- Record execution trace events.

Current limitations:

- Requires a real backend adapter to bridge into A1111 / `_21` conditioning.
- Merge operations depend on `prompt_conditioning_ops.py` being correct for real conditioning payloads.
- Graph-level alternates and schedules need explicit execution-plan support before they can be considered complete.

### `prompt_conditioning_ops.py`

This owns tensor / conditioning merge behavior.

Responsibilities:

- Blend conditioning outputs.
- Chunk / concatenate conditioning outputs.
- Morph between outputs.
- Pool, bind, assemble, AND, sequence concat, and weight operations.

Current limitations:

- These operations are still the riskiest runtime layer.
- Actual A1111 and `_21` conditioning objects may require additional adapter handling.
- Tensor shape compatibility, SDXL channel routing, pooled output behavior, and metadata preservation need practical testing.

### `prompt_graph_serializer.py`

This serializes graph nodes for debugging or canonical output.

Responsibilities:

- Convert a graph to debug text.
- Convert graph nodes back to canonical prompt text.
- Support JSON-like inspection.

Current limitation:

- Serialization is useful for debugging, but a serialized graph is not automatically guaranteed to be safe for `_21` if it contains unsupported nested backend structures.

## Current supported prompt categories

### Regular parser prompts

These should continue to work through the modified regular parser path:

```text
portrait of a knight, cinematic lighting
```

```text
fantasy landscape with [mountain:lake:0.25]
```

```text
{red dress | blue dress | black armor}
```

```text
character:::outfit::red dress!, accessories::diamond necklace!!, dark background
```

### Advanced backend top-level prompts

These should route to `_21` after canonicalization:

```text
BLEND{photo realism*0.9 | oil painting*1.5}
```

```text
<+>(photo realism*0.9 | oil painting*1.5)
```

```text
CHUNK{face detail | body detail}
```

```text
MORPH{young face @0 | older face @1}
```

```text
POOL{cinematic lighting}
```

```text
BIND{character => red hair, green eyes}
```

```text
ASSEMBLE{enc1=short text; enc2=long descriptive text; pooled=style anchor}
```

Support for exact syntax depends on what `_21` accepts for that backend operator.

### Simple proof-normalized wrapped backend prompts

These are the main current bridge between user-friendly grouping and `_21` top-level backend restrictions:

```text
{portrait BLEND{photo realism | oil painting}}
```

Expected normalized form:

```text
BLEND{portrait photo realism | portrait oil painting}
```

```text
{portrait CHUNK{face detail | body detail}}
```

Expected normalized form:

```text
CHUNK{portrait face detail | portrait body detail}
```

## Prompts that may not work today

The following should be considered unsupported or risky in the current version.

### Nested backend inside backend

```text
BLEND{portrait CHUNK{face | body} | landscape}
```

Why it may fail:

- `_21` generally expects backend operators to appear as top-level branch roots.
- Nested backend operators need graph lowering into multiple calls and merge operations.

### Nested BLEND inside BLEND

```text
BLEND{portrait BLEND{photo | oil} | sketch}
```

Why it may fail:

- `_21` v1-style extraction rejects or cannot safely reconstruct nested backend blocks.
- The current lightweight normalizer is not intended to resolve recursive backend graphs.

### Backend inside schedule

```text
[BLEND{photo | oil}:sketch:0.5]
```

Why it may fail:

- Scheduling and backend conditioning have different ownership rules.
- The execution plan needs graph-level schedule support to decide which conditioning branch is active at each step.

### Schedule inside backend branch with backend-aware expectations

```text
BLEND{portrait [photo:oil:0.5] | sketch}
```

This may work if `_21` treats the schedule as ordinary branch text and delegates it safely, but it should not be assumed to support graph-level scheduled blending yet.

### Alternate containing backend branches

```text
[BLEND{photo | oil} | CHUNK{face | body}]
```

Why it may fail:

- Alternates need graph-level selection semantics.
- Current `MergeMode` does not yet have first-class `ALTERNATE` support.

### Multiple independent backend blocks in one prompt

```text
BLEND{photo | oil}, CHUNK{face | body}
```

Why it may fail:

- `_21` may reject multiple top-level backend blocks.
- The graph path should eventually lower this into separate operations, but that is not guaranteed today.

### Ambiguous grouped backend context

```text
{portrait, background BLEND{day | night}}
```

Why it may fail or behave unexpectedly:

- It is unclear whether `portrait` should apply to all branches, only the group, or remain separate from the backend block.
- These cases need explicit graph semantics.

## Practical current guidance

For the current version, prefer one of these styles:

### Safe regular prompt

```text
portrait of a knight, cinematic lighting, [day:night:0.5]
```

### Safe top-level backend prompt

```text
BLEND{portrait photo realism | portrait oil painting}
```

### Safe alias form

```text
<+>(portrait photo realism | portrait oil painting)
```

### Safe wrapped proof-normalization form

```text
{portrait BLEND{photo realism | oil painting}}
```

Avoid combining multiple backend operators together until the graph execution path is complete.

## Current design principle

The current system should be described as:

```text
regular modified prompt_parser support + _21 top-level backend support + early graph pipeline scaffolding
```

It should not yet be described as:

```text
full arbitrary nested backend parser
```

or:

```text
complete support for every regular and advanced prompt_parser feature inside every backend branch
```

## Future goals

### 1. Full graph-first routing

Future routing should become:

```text
raw prompt
-> symbol canonicalization
-> graph parser
-> graph normalizer
-> graph validator
-> graph lowering
-> graph executor
-> A1111 / _21 backend adapter
```

The router should decide when to use the legacy direct path and when to use the graph path.

### 2. Support all modified regular `prompt_parser.py` behavior

The graph pipeline should eventually preserve all regular parser features from the Kittensx modified parser, including:

- Grouping with `{...}`.
- Alternates with `|`.
- Scheduled prompts with `[...]` and step / percentage boundaries.
- Weighted attention syntax.
- Sequence syntax using `::`, `:::`, `!`, and `!!`.
- Nested sequence handling.
- Escaped literals.
- Existing prompt schedule behavior.

### 3. Support all `_21` advanced backend operators

The graph pipeline should eventually support all `_21` backend operators, including:

- `BLEND`
- `CHUNK`
- `MORPH`
- `POOL`
- `BIND`
- `ASSEMBLE`

The goal is not only top-level support, but nested graph-safe support.

### 4. Full nested backend support

Future versions should support prompts like:

```text
BLEND{portrait CHUNK{face | body} | landscape}
```

by lowering to a graph plan rather than trying to force `_21` to parse the entire nested expression directly.

Expected future strategy:

```text
1. Condition primitive branches separately.
2. Execute inner backend operations first.
3. Feed inner operation outputs into outer operations.
4. Preserve metadata and schedule context.
```

### 5. Graph-level schedule support

Future versions should make schedules first-class execution concepts.

Needed work:

- Add schedule-aware plan operations.
- Track active step / timestep context.
- Support scheduled conditioning outputs.
- Define how schedules interact with backend merges.

### 6. Graph-level alternate support

Future versions should make alternates first-class execution concepts.

Needed work:

- Add `ALTERNATE` or equivalent plan operation.
- Support deterministic cycle mode.
- Support seeded random mode.
- Decide whether alternates resolve before conditioning or per step.

### 7. Better backend adapter layer

A real backend adapter should make the graph executor independent from the underlying parser implementation.

Potential adapters:

- A1111 regular parser adapter.
- Kittensx modified parser adapter.
- `_21` adapter.
- Future ComfyUI / SDNext style adapters.

### 8. SDXL and multi-channel safety

Future conditioning ops should explicitly test and document:

- Cross-attention shape compatibility.
- Pooled output behavior.
- `enc1` / `enc2` routing.
- `ASSEMBLE` behavior.
- What happens when branches have incompatible shapes.

### 9. Better diagnostics

Future error messages should explain:

- Which operator failed.
- Whether the failure is parser-level, graph-level, validation-level, or backend-level.
- Which part of the source prompt caused the error.
- Suggested rewrites for currently unsupported nested forms.

Example:

```text
Nested backend block is not supported by the direct _21 path.
Try rewriting:

BLEND{portrait CHUNK{face | body} | landscape}

as separate top-level graph operations, or wait for graph lowering support.
```

### 10. Public compatibility table

Future README versions should include a table like:

| Feature | Regular parser | `_21` direct path | Graph path target |
| --- | --- | --- | --- |
| Basic text | yes | yes | yes |
| Attention weights | yes | yes | yes |
| Schedules | yes | partial | yes |
| Alternates | yes | partial | yes |
| Sequences `::` / `:::` | yes | partial | yes |
| Top-level `BLEND` | no | yes | yes |
| Nested `BLEND` | no | no | yes |
| `BLEND` inside `CHUNK` | no | no | yes |
| `CHUNK` inside schedule | no | no | yes |
| SDXL `ASSEMBLE` | no | partial | yes |

## Recommended wording for release notes

Use this wording for the current version:

```text
This release introduces a compatibility router and early graph pipeline for combining the modified Prompt Architect parser with Konpr's `_21` advanced backend parser. Top-level advanced backend operators are routed to `_21`, while regular prompts continue through the modified regular parser. Simple wrapped backend cases can be normalized into `_21`-safe top-level forms. Full arbitrary nested backend execution is planned but not yet complete.
```

Avoid this wording for now:

```text
This release fully supports nested BLEND, CHUNK, MORPH, POOL, BIND, and ASSEMBLE anywhere in any prompt.
```

## Short version

Current version:

```text
Works best for regular modified parser prompts and top-level `_21` backend prompts.
```

Current bridge:

```text
Simple wrapped backend prompts can be lifted into top-level backend prompts.
```

Not complete yet:

```text
Arbitrary nested backend operators, backend-aware schedules, backend-aware alternates, and full graph execution.
```

Future target:

```text
A full graph-first parser/executor that supports both the modified regular prompt_parser behavior and all `_21` advanced backend behavior, including safe nesting.
```
