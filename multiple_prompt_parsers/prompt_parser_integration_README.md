# Prompt Parser 21 + Combined Dispatcher Integration for A1111

## Credits

`prompt_parser_21.py` included in this project was contributed by GitHub user **Konpr**.

Repository:
https://github.com/Konpr/whats-new

---

# Overview

This project adds:

- `prompt_parser_21.py`
- `prompt_dispatcher.py`
- `prompt_combined_dispatcher.py`
- `prompt_symbol_interpreter.py`
- `prompt_symbols.yaml`

to Automatic1111 (A1111) Stable Diffusion WebUI.

The system allows:

- Advanced backend prompt syntax
- Symbol remapping through YAML
- Canonical prompt translation
- Combined parser support
- Compatibility routing between:
  - A1111 compatible parser
  - Advanced Parser 21 backend system

---

# Important Concept

You have TWO possible setups:

## Combined Dispatcher 

Use:

```python
prompt_combined_dispatcher.py
```

This allows:

- Original A1111 parser features
- Prompt Parser 21 backend syntax
- Symbol translation system
- Automatic routing between parsers

This is the recommended setup.

---

# Step 1 — Add Files to A1111

Place the following files into:

```text
stable-diffusion-webui/modules/
```

Files:

```text
prompt_parser_21.py
prompt_dispatcher.py
prompt_combined_dispatcher.py
prompt_symbol_interpreter.py
prompt_symbols.yaml
...
*all files in this folder*
```

---

# Step 2 — Update Imports

You must modify imports inside the following files:

```text
modules/ui.py
modules/sd_hijack_clip.py
modules/sd_samplers_cfg_denoiser.py
modules/processing.py
```


# sd_hijack_clip.py

Find the existing import:

```python
from modules import prompt_parser, 
```

Replace it with:

```python
from modules import prompt_combined_dispatcher as prompt_parser
from modules import devices, sd_hijack, sd_emphasis
```

---

# sd_samplers_cfg_denoiser.py

Find the existing import:

```python
from modules import prompt_parser,
```

Replace it with:

```python
from modules import prompt_combined_dispatcher as prompt_parser
from modules import sd_samplers_common
```

---

# ui.py

Find the existing import:

```python
from modules import prompt_parser
```

Replace it with:

```python
from modules import prompt_combined_dispatcher as prompt_parser
```
---

# processing.py
---
Find the existing import:

```python
from modules import prompt_parser, 
```
Replace it with:

```python
from modules import prompt_combined_dispatcher as prompt_parser
```
---
---

# Step 3 — Restart A1111

Completely restart the WebUI after replacing the imports.

Do NOT hot reload.

---

# How the System Works

---

## prompt_symbol_interpreter.py

This file:

- Loads `prompt_symbols.yaml`
- Converts user symbols into canonical backend syntax
- Does NOT modify parser internals
- Acts as a preprocessing translation layer

Example:

User Input:

```text
forest &&(wolves | moonlight)
```

Canonical Translation:

```text
forest CHUNK{wolves | moonlight}
```

---


## prompt_combined_dispatcher.py

Extended dispatcher that:

- Supports both parser systems simultaneously
- Applies symbol translation
- Preserves compatibility with existing A1111 calls
- Automatically selects backend logic


---

# prompt_symbols.yaml

This file defines custom user-facing symbols.

Example:

```yaml
reserved_symbols:
  semantic_prompt: "%%"

backend_symbols:
  chunk: "&&"
  blend: "<+>"
  morph: ">>"
  assemble: "@@"
  bind: "=>"
  pool: "$$"

sequence_symbols:
  group_open: "{"
  group_close: "}"
  sequence: "::"
  deep_sequence: ":::"
  close: "!"
  top_close: "!!"

backend_wrappers:
  open: "("
  close: ")"
```

---

# Example Syntax

---

## CHUNK

User Syntax:

```text
&&(wolf | moonlight)
```

Canonical:

```text
CHUNK{wolf | moonlight}
```

---

## BLEND

User Syntax:

```text
<+>(photo realism | oil painting)
```

Canonical:

```text
BLEND{photo realism | oil painting}
```

---

## MORPH

User Syntax:

```text
>>^1.3(human => cyborg)
```

Canonical:

```text
MORPH^1.3{human => cyborg}
```

---

## BIND

User Syntax:

```text
=>(girl: red eyes, silver hair)
```

Canonical:

```text
BIND{girl => red eyes, silver hair}
```

---

## POOL

User Syntax:

```text
$$(cold atmosphere)
```

Canonical:

```text
POOL{cold atmosphere}
```

---

# Notes

---

## Reserved Symbols

The following symbol is reserved:

```text
%%
```

This is reserved for the separate `semantic_prompt` project.

Do NOT reuse it for BLEND or other operators.

---

## Escaping Symbols

You can escape symbols using:

```text
\&&
\>>
\<+>
```

---

## YAML Errors

If:

```text
prompt_symbols.yaml
```

contains duplicate symbols or invalid configuration, startup validation may fail.

---

# Troubleshooting

---

## A1111 Fails to Start

Check:

- Import replacement spelling
- File placement inside `/modules`
- YAML formatting
- Python syntax errors

---

## Backend Syntax Not Activating

Verify:

- `prompt_combined_dispatcher.py` is being imported
- Canonical backend syntax appears in logs
- Prompt symbols are correctly mapped

---

## Prompt Syntax Ignored

Check:

- Escaping issues
- Incorrect wrappers
- Missing braces/parentheses
- Symbol collisions inside YAML

---

# Recommended Setup

Recommended import replacement:

```python
from modules import prompt_combined_dispatcher as prompt_parser
```

Recommended dispatcher:

```text
prompt_combined_dispatcher.py
```

Recommended symbol configuration:

```text
prompt_symbols.yaml
```

---

# Compatibility Notes

The combined dispatcher attempts to preserve compatibility with:

- Existing A1111 scheduling
- Prompt attention
- CLIP token emphasis
- Multicond conditioning
- Existing parser APIs

However:

- Some advanced scheduling features may flatten during preprocessing
- Certain Parser 21 backend structures may bypass legacy scheduling logic

---

# Final Notes

This system is intentionally modular.

You can:

- Replace symbols without editing parser code
- Add new backend operators
- Swap dispatchers
- Use only canonical syntax if desired
- Extend parser routing logic later

The goal is to avoid directly modifying core parser internals whenever possible.
