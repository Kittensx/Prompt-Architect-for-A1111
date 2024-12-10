
# **01 Grouping Functionality**

## **Overview**
This update introduces the **Grouping** feature to the `prompt_parser`, allowing users to group multiple elements together using curly braces `{}`. Grouped prompts are treated as cohesive units, enabling more flexible and structured prompt design.

### **Features Added**
1. **Grouping with `{}`**:
   - Combine multiple elements into a single group.
   - Example: `{mountain, lake, forest}` groups these elements together.

2. **Balanced Outputs**:
   - Ensures that grouped elements are treated evenly during generation.
   - Group descriptions are resolved cohesively and handled as a unit.

---

## **Syntax Examples**

### **1. Basic Grouping**
- **Input**: `{mountain, lake, forest}`
- **Behavior**: Generates a balanced composition featuring all grouped elements (`mountain`, `lake`, and `forest`).

### **2. Nested Grouping**
- **Input**: `{mountain, {lake, forest}}`
- **Behavior**: Combines `mountain` with either `lake` or `forest` from the nested group.

---

## **Example Prompts and Results**

### **Example 1: Basic Grouping**
**Prompt**: `{snowy mountain, clear lake, dense forest}`
- **Expected Result**: An image combining a snowy mountain, clear lake, and dense forest.

### **Example 2: Nested Grouping**
**Prompt**: `{mountain, {lake, forest}}`
**Expected Result**: An image featuring a mountain as the backdrop, with a lake in the foreground and a forest harmoniously framing the scene.

---

## **Example Images**
Images generated for the examples above are stored in the `example_images/01_grouping` folder.

| **Prompt**                                | **Image**                                      |
|-------------------------------------------|-----------------------------------------------|
| `{snowy mountain, clear lake, dense forest}` 	| ![Example1](example_images/01_grouping/example 1_Grouped.png) |
| `snowy mountain, clear lake, dense forest` 	| ![Example1](example_images/01_grouping/example 1_UnGrouped.png) |
| `{mountain, {lake, forest}}` 					| ![Example1](example_images/01_grouping/example 2_Grouped.png) |



---

## **Known Limitations**
1. **Nested Depth**:
   - Deeply nested groups may result in complex and unpredictable outputs.
2. **Parsing Errors**:
   - Ensure that all groups are properly closed with matching curly braces `{}`.

---

## **How It Works**
The `01_prompt_parser.py` file introduces logic to:
1. **Parse Grouped Prompts**:
   - Group descriptions are resolved into cohesive strings.
   - Example: `{mountain, lake, forest}` resolves to `mountain, lake, forest`.

2. **Integration**:
   - Grouped prompts integrate seamlessly with other functionalities (e.g., OR logic).

---

## **Folder Structure**
All files for this update are stored in the `01_grouping` folder:
- `prompt_parser.py`: Updated parser with grouping logic.
- `example_images/`: Contains images for each example prompt.
