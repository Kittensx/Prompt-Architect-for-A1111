# Changelog for `prompt_parser.py`
# Date: 1/30/2025

### 🔹 Key Changes:

### ✅ Updated Sequence Closures:
- Changed sequence closures from `;` and `_ ;` to:
  - `~` for separating elements within a sequence.
  - `!` for closing sequences.
  - `!!` for closing **top-level sequences**.

### ✅ New Sequence Definition Rules:
- Introduced **top-level sequences** (`top_level_sequence`) and **nested sequences** (`nested_sequence`).
- This ensures better structuring and scoping of sequences.

### ✅ Parent Requirement for Sequences:
- Sequences (`::`) must now have an explicit **preceding prompt**.
- Prevents orphan sequences without a defined parent.

### ✅ Refined `plain` Character Handling:
- Removed certain characters from the **plain text** category that caused parsing issues.
- Adjusted `plain` to explicitly include necessary characters.

---

### 🔹 Other Fixes & Improvements:
- ✅ Optimized sequence resolution to enforce structure without affecting compatibility.
- ✅ Refactored sequence handling logic to better distinguish **owners** from **attributes**.
- ✅ Minor optimizations for **parsing efficiency** and **error handling**.

---

📌 *Please update your prompts accordingly to ensure compatibility with the new parser changes.*
