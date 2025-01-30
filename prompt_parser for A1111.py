from __future__ import annotations

import re
from collections import namedtuple
import lark
import random

# "fantasy landscape with a [mountain:lake:0.25] and {an oak | a christmas tree} [ in foreground::0.6][: in background:0.25] {shoddy | masterful}"
# On Step #25: "fantasy landscape with a mountain and an oak in foreground shoddy"
# On Step #50: "fantasy landscape with a lake and an oak in foreground shoddy"
# On Step #60: "fantasy landscape with a lake and an oak in foreground masterful"
#On Step #75: "fantasy landscape with a lake and a christmas tree in background masterful"
#On Step #100: "fantasy landscape with a lake and a christmas tree in background masterful"
# Step-by-Step Resolution
#[mountain:lake:0.25]
#
#"mountain" for 75% of the steps (0–75 steps).
#"lake" for the remaining 25% (75–100 steps).
#{an oak | a christmas tree}
#
#Randomly picks "an oak" or "a christmas tree" for each resolution.
#[ in foreground::0.6][: in background:0.25]
#
#"in foreground" for 60% of the steps (0–60 steps).
#Adds "in background" for 25% of the steps (75–100 steps).
#{shoddy | masterful}
#
#Randomly resolves to either "shoddy" or "masterful" for each step.
#
# The schedule_parser parses prompts with complex rules for weighting, alternates, grouping, and emphasis.
# Here’s how the key rules work:
#
# 1. **scheduled**:
#    - Syntax: `[prompt1 : prompt2 : ...] : weight`
#    - Specifies prompts to transition between at specific steps or weights.
#    - Example: `[mountain:lake:0.25]` means:
#        - Use "mountain" for 75% of the steps.
#        - Transition to "lake" for the remaining 25%.
#
# 2. **alternate**:
#    - Syntax: `[option1 | option2 | ...]`
#    - Randomly chooses one of the options for the final prompt.
#    - Example: `[cloudy | sunny]` randomly picks either "cloudy" or "sunny".
#
# 3. **alternate1**:
#    - Syntax: `option1 | option2`
#    - Works similarly to `alternate`, but without requiring brackets.
#    - Recognized as a standalone or within nested structures (like `grouped`).
#    - Example:
#        - `cloudy | sunny` randomly picks "cloudy" or "sunny".
#        - `{cloudy | sunny, rainy | overcast}` resolves alternates within groups.
#
# 4. **grouped**:
#    - Syntax: `{item1, item2, item3, ...}`
#    - Allows grouping multiple items separated by `,` or `|` for alternates.
#    - `,` defines a fixed list of items.
#    - `|` introduces alternates that are randomly resolved.
#    - Example:
#        - `{cloudy, sunny}` lists "cloudy" and "sunny".
#        - `{cloudy | sunny, rainy | overcast}` mixes grouping and alternates.
#
# Additional Rules:
# - **emphasized**:
#    - Increases or decreases emphasis on parts of the prompt using `()`, `[]`, or `:`.
#    - Example: `(important:2.0)` doubles the emphasis on "important".
# - **plain**:
#    - Captures plain text without any special formatting or operators.
# - **WHITESPACE**:
#    - Recognizes spaces and other whitespace for parsing clarity.
#
# Example Prompt:
# "fantasy landscape with a [mountain:lake:0.25] and {cloudy | sunny, stormclouds}"
# - Resolves into prompts that vary based on weights, alternates, and groups:
#   - "fantasy landscape with a mountain and cloudy"
#   - "fantasy landscape with a lake and sunny"
#   - "fantasy landscape with a lake and stormclouds"
#
# This flexible system enables dynamic generation of diverse and varied prompts for creative tasks.
# Sequence example:   
#  person:: white hair:: long:: slightly wavy; green eyes:: scelera shaped like a heart;_;
# Resolved output: person: white hair: long: slightly wavy; green eyes: scelera shaped like a heart
# nested sequences must be implicitly closed with "~" and the last "!" closes the entire sequence. sequence starts with "::" and ends with "~" or "!"
# A sequence starts with "::" and ends with "!" 
# It may contain nested sequences or regular prompts.
# A nested sequence starts with "::" and ends with "~" or "!"

schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | grouped | alternate | alternate1 | alternate2 |  top_level_sequence | compound | numbered | and_rule | plain | WHITESPACE)*

!emphasized: "(" prompt ")" 
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt (":" prompt)+] "]" ":" NUMBER (step_range_list | reverse_flag | step_range_list reverse_flag)?
reverse_flag: "reverse" | "r"
step_range_list: step_range ("," step_range)*
step_range: NUMBER "-" NUMBER | NUMBER "%" "-" NUMBER "%"
   
alternate: "[" prompt ("|" [prompt])+ "]"
alternate1: (prompt) "|" (prompt)+
alternate2: (plain | compound) ("|" (plain | compound))+
grouped: "{" ((NUMBER_Q | prompt | sequence | grouped) (","| "|")?)+ "}"

top_level_sequence: prompt "::" sequence ("::" sequence)*  "!!"
sequence: prompt "::" prompt ("," | WHITESPACE)* nested_sequence* "!"  
nested_sequence: "::" prompt ("," | WHITESPACE)* ("~" | "!")


compound: /[a-zA-Z0-9]+(_[a-zA-Z0-9]+)+/
numbered: NUMBER_Q ("!")? (grouped | sequence | compound | and_rule | plain)
and_rule: (plain | compound) ("&" (plain | compound))+
WHITESPACE: /\s+/
plain: /([^\\\[\]()]|\\.)+/

%import common.SIGNED_NUMBER -> NUMBER // For weights and general numbers
%import common.INT -> NUMBER_Q // For quantities

""")


def get_learned_conditioning_prompt_schedules(prompts, base_steps, hires_steps=None, use_old_scheduling=False):
    """
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10)[0]
    >>> g("test")
    [[10, 'test']]
    >>> g("a [b:3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [b: 3]")
    [[3, 'a '], [10, 'a b']]
    >>> g("a [[[b]]:2]")
    [[2, 'a '], [10, 'a [[b]]']]
    >>> g("[(a:2):3]")
    [[3, ''], [10, '(a:2)']]
    >>> g("a [b : c : 1] d")
    [[1, 'a b  d'], [10, 'a  c  d']]
    >>> g("a[b:[c:d:2]:1]e")
    [[1, 'abe'], [2, 'ace'], [10, 'ade']]
    >>> g("a [unbalanced")
    [[10, 'a [unbalanced']]
    >>> g("a [b:.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    >>> g("a [{b|d{:.5] c")  # not handling this right now
    [[5, 'a  c'], [10, 'a {b|d{ c']]
    >>> g("((a][:b:c [d:3]")
    [[3, '((a][:b:c '], [10, '((a][:b:c d']]
    >>> g("[a|(b:1.1)]")
    [[1, 'a'], [2, '(b:1.1)'], [3, 'a'], [4, '(b:1.1)'], [5, 'a'], [6, '(b:1.1)'], [7, 'a'], [8, '(b:1.1)'], [9, 'a'], [10, '(b:1.1)']]
    >>> g("[fe|]male")
    [[1, 'female'], [2, 'male'], [3, 'female'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'female'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g("[fe|||]male")
    [[1, 'female'], [2, 'male'], [3, 'male'], [4, 'male'], [5, 'female'], [6, 'male'], [7, 'male'], [8, 'male'], [9, 'female'], [10, 'male']]
    >>> g = lambda p: get_learned_conditioning_prompt_schedules([p], 10, 10)[0]
    >>> g("a [b:.5] c")
    [[10, 'a b c']]
    >>> g("a [b:1.5] c")
    [[5, 'a  c'], [10, 'a b c']]
    """

    if hires_steps is None or use_old_scheduling:
        int_offset = 0
        flt_offset = 0
        steps = base_steps
    else:
        int_offset = base_steps
        flt_offset = 1.0
        steps = hires_steps

    def collect_steps(steps, tree):
        #if not tree or not hasattr(tree, 'children') or not tree.children: #debugs
            #print("Invalid tree structure:", tree)
            #return []
        res = [steps]  # Always include the final step
        def resolve_tree(tree):
            """Recursively resolve a tree node to its final string representation."""
            if isinstance(tree, lark.Tree):
                # Recursively resolve each child
                return "".join(resolve_tree(child) for child in tree.children)
            return str(tree)

        class CollectSteps(lark.Visitor):
            def alternate2(self, tree):
                # Resolve all alternates
                options = []
                for child in tree.children:
                    if isinstance(child, lark.Tree):
                        options.append(resolve_tree(child))
                    else:
                        options.append(str(child).strip())

                # Combine options with `_` where applicable
                combined_options = []
                for option in options:
                    if "_" in option:
                        prefix, suffix = option.split("_", 1)
                        combined_options.append(f"{prefix}_{suffix}")
                    else:
                        combined_options.append(option)

                # Add combined alternates to results
                res.append("|".join(combined_options))
            
            def compound(self, tree):
                # Treat compound phrases as a single unit
                res.append("".join(tree.children))  

            def top_level_sequence(self, tree):
                """
                Handles top-level sequences by ensuring a defined owner (prompt) before '::'.
                Example: woman:::hair_style::ponytail!, eye_color::blue!
                """
                owner = resolve_tree(tree.children[0])  # Get the owner prompt (e.g., "woman")
                sequences = []

                for child in tree.children[1:]:
                    if isinstance(child, lark.Tree) and child.data == "sequence":
                        sequences.append(self.sequence(child, owner))
                    elif isinstance(child, str):
                        if child.strip() == "!!":  # Ensure closure is respected
                            break
                        sequences.append(child.strip(" ~!")) # Strip trailing markers                       

                # Store the structured sequence
                res.append(f"{owner} -> {', '.join(sequences)}")

            def sequence(self, tree, parent=None):
                """
                Handles individual sequences. Ensures sequences have an owner.
                Example: hair_style::ponytail~hair_color::blonde!
                """
                if parent is None:
                    # If this is a root sequence, the first child is the owner.
                    owner = resolve_tree(tree.children[0])
                    children = tree.children[1:]
                else:
                    # This is a child sequence under a parent (top-level sequence).
                    owner = parent
                    children = tree.children

                descriptors = []
                for child in children:
                    if isinstance(child, lark.Tree):
                        if child.data == "nested_sequence":  # Handle nested sequences separately
                            descriptors.append(self.nested_sequence(child))
                        else:
                            descriptors.append(resolve_tree(child))  # Recursively resolve other structures
                    elif isinstance(child, str):
                        descriptors.append(child.strip(" ~!"))  # Strip trailing "~" or "!"

                # Resolve the sequence by combining the described object and its attributes
                combined_description = f"{owner}: {', '.join(descriptors)}"
                res.append(combined_description)

            def nested_sequence(self, tree):
                """
                Handles nested sequences, ensuring they remain part of their parent sequence.
                Example: ::hair_color::blonde!
                """
                sequence_elements = []
                for child in tree.children:
                    if isinstance(child, lark.Tree):
                        sequence_elements.append(resolve_tree(child))
                    elif isinstance(child, str):
                        sequence_elements.append(child.strip(" ~!"))  # Strip trailing "~" or "!"

                # Return formatted nested sequence
                return f"[{' | '.join(sequence_elements)}]"  # Format it like an embedded list

               
            def alternate1(self, tree):
                # Randomly resolve `alternate1`
                #options = [str(child) if not isinstance(child, lark.Tree) else resolve_tree(child) for child in tree.children]               
                options = [resolve_tree(child) for child in tree.children]
                res.append(random.choice(options))  # Random choice from options
                
            
            def grouped(self,tree):
                # Collect all descriptions within the group       
                ##group_descriptions = [
                    ##self._resolve_tree(child) if isinstance(child, lark.Tree) else str(child) 
                    ##for child in tree.children]
                
                #print(f"Group: {group_descriptions}") #debug
                
                # Handle the group as a cohesive unit (e.g., append to results)               
                ##res.append(", ".join(group_descriptions))
                group_descriptions = [
                    resolve_tree(child) if isinstance(child, lark.Tree) else str(child)
                    for child in tree.children
                ]
                res.append(", ".join(group_descriptions))
            def scheduled(self, tree):
                # Validate tree structure and children
                if not hasattr(tree, "children") or not tree.children:
                    return

                # Extract prompts and the scheduling number
                prompts = tree.children[:-2]  # All but the last two children are options
                number_node = tree.children[-2]  # Second-to-last child is the scheduling number
                additional_info = tree.children[-1] if len(tree.children) > 2 else None

                # Initialize parameters
                is_reverse = False
                step_intervals = []

                # Safeguard for missing or invalid children
                if not prompts or not number_node:
                    return

                # Convert number_node to a float (scheduling weight or total steps percentage)
                try:
                    v = float(number_node)
                except ValueError:
                    return

                # Handle additional parameters (reverse flag or step ranges)
                if additional_info:
                    if isinstance(additional_info, str) and additional_info.lower() in ("reverse", "r"):
                        is_reverse = True
                    elif isinstance(additional_info, list):
                        # Process step ranges
                        for r in additional_info:
                            start, end = r.split("-")
                            if "%" in start or "%" in end:  # Handle percentage-based ranges                                
                                start = round(float(start.strip("%")) / 100 * steps)
                                end = round(float(end.strip("%")) / 100 * steps)                                
                            else:  # Handle absolute step ranges
                                start, end = int(start), int(end)
                             # Clamp ranges to valid boundaries
                            if start > steps:
                                start = steps  # Adjust start to max steps
                            if end > steps:
                                end = steps  # Adjust end to max steps
                            # Ignore invalid ranges where start > end
                            if start > end:
                                print(f"Warning: Ignored invalid range {start}-{end}.")
                                continue
                            step_intervals.append((start, end))

                # If no step ranges are specified, generate default intervals based on weight
                if not step_intervals:
                    num_prompts = len(prompts)
                    step_intervals = [
                        (int(i * (v * steps) / num_prompts), int((i + 1) * (v * steps) / num_prompts))
                        for i in range(num_prompts)
                    ]

                # Handle reverse scheduling
                if is_reverse:
                    prompts = prompts[::-1]  # Reverse prompts
                    step_intervals = step_intervals[::-1]  # Reverse intervals

                # Replace number_node with numeric step intervals in the tree
                tree.children[-2] = step_intervals

                # Extend the results with calculated intervals
                res.extend(step_intervals)
    
        # Visit the tree and collect step intervals
        CollectSteps().visit(tree)
        #return sorted(set(res))  # Remove duplicates and sort
        return res #does not remove duplicates or sort them


    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def and_rule(self, args):
                # All elements in args must be resolved
                resolved_items = [self._resolve_tree(arg) if isinstance(arg, lark.Tree) else str(arg) for arg in args]
                return " and ".join(resolved_items)  # Join items with "and"
            
            def compound(self, args):
                # Return the compound phrase as a single string                
                return "_".join(str(arg) for arg in args)                
            
           
            def top_level_sequence(self, args):
                """
                Handles top-level sequences where an explicit owner is required.
                Example: woman:::hair_style::ponytail!, eye_color::blue!
                """
                owner = args[0]  # Extracts the first prompt (e.g., "woman")
                sequences = []

                for child in args[1:]:
                    if isinstance(child, list):
                        sequences.append(self.sequence(child, owner))
                    elif isinstance(child, str):
                        if child.strip() == "!!":  # Stop processing at `!!`
                            break
                        sequences.append(child.strip(" ~!"))

                # Return structured output with owner and its sequences
                return f"{owner} -> {', '.join(sequences)}"

            def sequence(self, args, parent=None):
                """
                Handles sequences by ensuring an owner (prompt) is defined before '::'.
                Example: hair_style::ponytail~hair_color::blonde!
                """
                if parent is None:
                    # If this is a root sequence, the first argument is the owner.
                    owner = args[0]
                    children = args[1:]
                else:
                    # If it's a child sequence, it inherits the parent's owner.
                    owner = parent
                    children = args

                descriptors = []
                for child in children:
                    if isinstance(child, str):
                        descriptors.append(child.strip(" ~!"))  # Remove sequence markers
                    elif isinstance(child, list):  
                        descriptors.append(self.nested_sequence(child))  # Handle nested sequences

                # Format the sequence
                return f"{owner}: {', '.join(descriptors)}"

            def nested_sequence(self, args):
                """
                Handles nested sequences within sequences.
                Example: ::hair_color::blonde!
                """
                sequence_elements = []
                for desc in args:
                    if isinstance(desc, str):
                        sequence_elements.append(desc.strip(" ~!"))  # Strip trailing markers
                    elif isinstance(desc, list):
                        sequence_elements.append(self.nested_sequence(desc))  # Handle deep nesting

                # Format the nested sequence
                return f"[{' | '.join(sequence_elements)}]"

            
            
            def alternate1(self, args):
                # Randomly select one of the alternates
                return random.choice(args)
            
            def alternate2(self, args):
                # Resolve all alternates into a list
                resolved_options = []
                for arg in args:
                    if isinstance(arg, str):  # If it's already plain text
                        resolved_options.append(arg)
                    elif isinstance(arg, lark.Tree):  # If it's a compound or nested structure
                        resolved_options.append(self._resolve_tree(arg))

                # Process each option to combine with `_` if necessary
                combined_options = []
                for option in resolved_options:
                    if "_" in option:  # Handle compounds like "green_eyes"
                        combined_options.append(option)  # Already combined
                    else:  # Combine plain text alternates
                        suffix = option.split("_")[-1] if "_" in resolved_options[0] else ""
                        combined_options.append(f"{option}_{suffix}" if suffix else option)

                # Return combined alternates separated by `|`
                return " | ".join(combined_options)
                
            def scheduled(self, args):  
                # Ensure args is valid
                if not args or len(args) < 2:
                    return

                # Extract components
                *prompts, when, _, is_reverse, weight = args
                step_intervals = args[-2]  # Step intervals
                is_reverse = args[-1].lower() in ("reverse", "r") if len(args) > 2 else False

                # Validate `when`
                if not isinstance(when, list):
                    return

                # Handle reverse scheduling
                if is_reverse:
                    prompts = prompts[::-1]  # Reverse prompts
                    when = when[::-1]        # Reverse boundaries

                # Iterate over the step intervals
                for i, (start, end) in enumerate(step_intervals):
                    # Skip invalid or clamped ranges
                    if start > end:
                        continue
                    if start <= step <= end:
                        #yield f"({prompts[i]}:focus)"  # Apply focus during the range
                        yield f"({prompts[i]}:{weight})"  # Uniform weight
                        return

                # Select the appropriate prompt based on the step
                for i, boundary in enumerate(when):
                    if step <= boundary:
                        yield f"({prompts[i]}:{weight})"  # Apply weight (de-emphasis)
                        return
                
                # Default to the last prompt with the weight if step exceeds boundaries
                yield f"({prompts[-1]}:{weight})"
                
            def alternate(self, args):
                # Handle alternates with a cycle
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]
            def start(self, args):
                #flatten nested structures into a single string
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)
                return ''.join(flatten(args))
            def plain(self, args):
                #handle plain text nodes
                yield args[0].value
            def grouped(self, args):
                ### Return the group as a cohesive string
                ##return ", ".join(args)
                # Combine all grouped elements into a single string
                return f"{{{', '.join(args)}}}"
            def __default__(self, data, children, meta):
                #handle all other nodes
                for child in children:
                    yield child
            def _resolve_tree(self, tree):
                """Recursively resolve a tree node to its final string representation."""
                if isinstance(tree, lark.Tree):
                    return "".join(self._resolve_tree(child) for child in tree.children)
                return str(tree)
                
        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
            #print(tree.pretty())  # Debugging: visualize the tree structure
       
        except lark.exceptions.LarkError as e:
            #print(f"Parsing error for prompt: {prompt}")
            #if 0:
            #    import traceback
            #    traceback.print_exc()
            return [[steps, prompt]]            
            
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", ["end_at_step", "cond"])


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)



def get_learned_conditioning(model, prompts: SdConditioning | list[str], steps, hires_steps=None, use_old_scheduling=False):
    """converts a list of prompts into a list of prompt schedules - each schedule is a list of ScheduledPromptConditioning, specifying the comdition (cond),
    and the sampling step at which this condition is to be replaced by the next one.

    Input:
    (model, ['a red crown', 'a [blue:green:5] jeweled crown'], 20)

    Output:
    [
        [
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0523,  ..., -0.4901, -0.3066,  0.0674], ..., [ 0.3317, -0.5102, -0.4066,  ...,  0.4119, -0.7647, -1.0160]], device='cuda:0'))
        ],
        [
            ScheduledPromptConditioning(end_at_step=5, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.0192,  0.3867, -0.4644,  ...,  0.1135, -0.3696, -0.4625]], device='cuda:0')),
            ScheduledPromptConditioning(end_at_step=20, cond=tensor([[-0.3886,  0.0229, -0.0522,  ..., -0.4901, -0.3067,  0.0673], ..., [-0.7352, -0.4356, -0.7888,  ...,  0.6994, -0.4312, -1.2593]], device='cuda:0'))
        ]
    ]
    """
    res = []

    prompt_schedules = get_learned_conditioning_prompt_schedules(prompts, steps, hires_steps, use_old_scheduling)
    cache = {}

    for prompt, prompt_schedule in zip(prompts, prompt_schedules):

        cached = cache.get(prompt, None)
        if cached is not None:
            res.append(cached)
            continue

        texts = SdConditioning([x[1] for x in prompt_schedule], copy_from=prompts)
        conds = model.get_learned_conditioning(texts)

        cond_schedule = []
        for i, (end_at_step, _) in enumerate(prompt_schedule):
            if isinstance(conds, dict):
                cond = {k: v[i] for k, v in conds.items()}
            else:
                cond = conds[i]

            cond_schedule.append(ScheduledPromptConditioning(end_at_step, cond))

        cache[prompt] = cond_schedule
        res.append(cond_schedule)

    return res


re_AND = re.compile(r"\bAND\b")
re_weight = re.compile(r"^((?:\s|.)*?)(?:\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+)))?\s*$")


def get_multicond_prompt_list(prompts: SdConditioning | list[str]):
    res_indexes = []

    prompt_indexes = {}
    prompt_flat_list = SdConditioning(prompts)
    prompt_flat_list.clear()

    for prompt in prompts:
        subprompts = re_AND.split(prompt)

        indexes = []
        for subprompt in subprompts:
            match = re_weight.search(subprompt)

            text, weight = match.groups() if match is not None else (subprompt, 1.0)

            weight = float(weight) if weight is not None else 1.0

            index = prompt_indexes.get(text, None)
            if index is None:
                index = len(prompt_flat_list)
                prompt_flat_list.append(text)
                prompt_indexes[text] = index

            indexes.append((index, weight))

        res_indexes.append(indexes)

    return res_indexes, prompt_flat_list, prompt_indexes


class ComposableScheduledPromptConditioning:
    def __init__(self, schedules, weight=1.0):
        self.schedules: list[ScheduledPromptConditioning] = schedules
        self.weight: float = weight


class MulticondLearnedConditioning:
    def __init__(self, shape, batch):
        self.shape: tuple = shape  # the shape field is needed to send this object to DDIM/PLMS
        self.batch: list[list[ComposableScheduledPromptConditioning]] = batch


def get_multicond_learned_conditioning(model, prompts, steps, hires_steps=None, use_old_scheduling=False) -> MulticondLearnedConditioning:
    """same as get_learned_conditioning, but returns a list of ScheduledPromptConditioning along with the weight objects for each prompt.
    For each prompt, the list is obtained by splitting the prompt using the AND separator.

    https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/
    """

    res_indexes, prompt_flat_list, prompt_indexes = get_multicond_prompt_list(prompts)

    learned_conditioning = get_learned_conditioning(model, prompt_flat_list, steps, hires_steps, use_old_scheduling)

    res = []
    for indexes in res_indexes:
        res.append([ComposableScheduledPromptConditioning(learned_conditioning[i], weight) for i, weight in indexes])

    return MulticondLearnedConditioning(shape=(len(prompts),), batch=res)


class DictWithShape(dict):
    def __init__(self, x, shape=None):
        super().__init__()
        self.update(x)

    @property
    def shape(self):
        return self["crossattn"].shape


def reconstruct_cond_batch(c: list[list[ScheduledPromptConditioning]], current_step):
    param = c[0][0].cond
    is_dict = isinstance(param, dict)

    if is_dict:
        dict_cond = param
        res = {k: torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype) for k, param in dict_cond.items()}
        res = DictWithShape(res, (len(c),) + dict_cond['crossattn'].shape)
    else:
        res = torch.zeros((len(c),) + param.shape, device=param.device, dtype=param.dtype)

    for i, cond_schedule in enumerate(c):
        target_index = 0
        for current, entry in enumerate(cond_schedule):
            if current_step <= entry.end_at_step:
                target_index = current
                break

        if is_dict:
            for k, param in cond_schedule[target_index].cond.items():
                res[k][i] = param
        else:
            res[i] = cond_schedule[target_index].cond

    return res


def stack_conds(tensors):
    # if prompts have wildly different lengths above the limit we'll get tensors of different shapes
    # and won't be able to torch.stack them. So this fixes that.
    token_count = max([x.shape[0] for x in tensors])
    for i in range(len(tensors)):
        if tensors[i].shape[0] != token_count:
            last_vector = tensors[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - tensors[i].shape[0], 1])
            tensors[i] = torch.vstack([tensors[i], last_vector_repeated])

    return torch.stack(tensors)



def reconstruct_multicond_batch(c: MulticondLearnedConditioning, current_step):
    param = c.batch[0][0].schedules[0].cond

    tensors = []
    conds_list = []

    for composable_prompts in c.batch:
        conds_for_batch = []
        #print(f"Conds_for_batch: {conds_for_batch}")

        for composable_prompt in composable_prompts:
            target_index = 0
            
            for current, entry in enumerate(composable_prompt.schedules):
                try:
                    end_at_step = int(entry.end_at_step)                    
                except ValueError:
                    raise TypeError(f"Invalid end_at_step value: {entry.end_at_step}, expected an integer.")
                    end_at_step = current_step
                
                if current_step <= end_at_step:
                    target_index = current
                    break
                
                

            conds_for_batch.append((len(tensors), composable_prompt.weight))
            tensors.append(composable_prompt.schedules[target_index].cond)

        conds_list.append(conds_for_batch)

    if isinstance(tensors[0], dict):
        keys = list(tensors[0].keys())
        stacked = {k: stack_conds([x[k] for x in tensors]) for k in keys}
        stacked = DictWithShape(stacked, stacked['crossattn'].shape)
    else:
        stacked = stack_conds(tensors).to(device=param.device, dtype=param.dtype)

    return conds_list, stacked


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:\s*([+-]?[.\d]+)\s*\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
else:
    import torch  # doctest faster
