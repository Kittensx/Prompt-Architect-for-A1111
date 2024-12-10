# **02 OR Functionality**

## **Overview**
This update introduces the **OR Functionality** to the `prompt_parser`, enabling users to generate diverse outputs by using the `|` operator. The OR logic ensures that only one of the elements within the group is selected for each generation, adding controlled randomness to prompt design.

### **Features Added**
1. **OR Operator `|`**:
   - Randomly selects one element from the provided options.
   - Example: `mountain|lake|forest` will generate an image featuring either a mountain, a lake, or a forest.

2. **Nested OR Logic**:
   - Supports OR functionality within nested groups.
   - Example: `{lake, {mountain|forest}}` allows structured combinations of fixed and random elements.

---

## **Syntax Examples**

### **1. Basic OR**
- **Input**: `mountain|lake|forest`
- **Behavior**: Randomly selects one of the elements to feature in the output.

### **2. Nested OR**
- **Input**: `{lake, {mountain|forest}}`
- **Behavior**: Combines "lake" with a random selection of either "mountain" or "forest" for more dynamic compositions.

---

## **Example Prompts and Results**

### **Example 1: Basic OR**
**Prompt**: `mountain|lake|forest`
- **Expected Result**: An image featuring either a mountain, a lake, or a forest as the primary focus.

### **Example 2: Nested OR**
**Prompt**: `{lake, {mountain|forest}}`
- **Expected Result**: An image featuring a lake with a secondary focus on either a mountain or a forest.

---

## **Example Images**
Images generated for the examples above are stored in the `example_images/02_or_functionality` folder.

---

## **Known Limitations**
1. **Predictability**:
   - The randomness introduced by the OR logic may result in repeated selections during multiple generations.
2. **Nested Complexity**:
   - Nested OR logic can produce unexpected combinations if not carefully designed.

---

## **How It Works**
The `prompt_parser.py` file has been updated to:
1. **Parse OR Prompts**:
   - Resolves prompts containing the `|` operator by randomly selecting one of the options.
   - Example: `mountain|lake|forest` resolves to one element based on random selection.

2. **Integration**:
   - OR logic integrates seamlessly with other functionalities (e.g., grouping and nesting).

---

## **Folder Structure**
All files for this update are stored in the `02_or_functionality` folder:
- `prompt_parser.py`: Updated parser with OR logic.
- `example_images/`: Contains images for each example prompt.
