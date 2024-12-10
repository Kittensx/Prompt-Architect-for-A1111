# **Example 2: Grouped Prompts with Nesting**

## **Overview**
This example demonstrates the use of **nested grouping** in prompts. By nesting groups using `{}` within other groups, users can create more complex and dynamic compositions. The nesting ensures that some elements are prioritized within a subgroup while maintaining balance with the outer group.

---

## **Prompt Details**

### **Prompt**: `{mountain, {lake, forest}}`
- **Behavior**: 
  - The outer group combines the elements `mountain` with a nested group `{lake, forest}`.
  - The nested group `{lake, forest}` ensures that the lake and forest are treated as a cohesive unit within the composition.
  - The result is a scene where the mountain serves as the backdrop while the lake and forest harmoniously occupy the foreground and middle ground.

---

## **Generation Settings**
- **Steps**: 30
- **Sampler**: DPM++ 2M
- **Schedule Type**: Karras Exponential
- **CFG Scale**: 7
- **Seed**: 26067899193
- **Size**: 768x512
- **Model**: FusionX-Realistic_v3_float16
- **Upscaler**: R-ESRGAN 4x+ Anime6B
- **Version**: v1.10.1

---

## **Image Analysis**

### **Key Features of the Generated Image**:
1. **Mountain**:
   - Prominently displayed in the background, anchoring the scene.
   - Serves as the focal backdrop without overwhelming the foreground elements.

2. **Lake and Forest**:
   - Harmoniously integrated as part of the nested group.
   - The lake occupies the center foreground, reflecting the surroundings, while the forest lines the edges, adding depth and balance.

3. **Overall Composition**:
   - The grouping ensures a balance between the mountain and the combined lake-forest unit.
   - The nested structure prioritizes the lake and forest's relationship, resulting in a cohesive natural landscape.

---

## **How Nesting Affects the Composition**
1. **Without Nesting**:
   - Elements like `lake` and `forest` might compete independently for prominence, leading to a less cohesive scene.
2. **With Nesting**:
   - The nested group `{lake, forest}` ensures these elements are treated as a unit, leading to a more harmonious foreground and middle ground.
   - The outer group `{mountain, {lake, forest}}` balances the focus between the mountain and the nested unit, creating a visually pleasing hierarchy.

---

## **Example Image**
![Example2_Grouped](example_images/01_Grouping/example2_grouped.png)

---

## **Conclusion**
Nesting groups within prompts adds an additional layer of control to the composition. By combining elements into subgroups, users can emphasize relationships between specific components while maintaining balance with the overall scene. This approach is especially useful for creating natural, cohesive landscapes or other multi-layered compositions.
