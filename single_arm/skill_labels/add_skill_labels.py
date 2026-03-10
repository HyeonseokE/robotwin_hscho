"""
add_skill_labels.py — Use LLM to insert register_skill() calls before every self.move()
in each task's play_once() method.

Usage:
    # Single file
    python -m skill_labels.add_skill_labels --file ./envs/place_can_basket.py

    # Batch all tasks
    python -m skill_labels.add_skill_labels --batch --root ./envs

Pattern reused from script/add_annotation.py.
"""

import ast
import tokenize
import io
import re
import os
import time
import traceback
from threading import Thread
from pathlib import Path
from argparse import ArgumentParser


# ============================================================
# AST helpers (reused from add_annotation.py)
# ============================================================

def remove_comments_and_docstrings(source):
    """Remove comments and docstrings from Python source."""
    src = io.StringIO(source)
    out = []
    prev_tok_type = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(src.readline):
        token_type = tok.type
        token_string = tok.string
        start_line, start_col = tok.start
        end_line, end_col = tok.end

        if start_line > last_lineno:
            out.append("\n" * (start_line - last_lineno - 1))
            last_col = 0
        elif start_col > last_col:
            out.append(" " * (start_col - last_col))

        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_tok_type not in (tokenize.INDENT, tokenize.NEWLINE):
                if re.match(r'^\s*"""(?:[^"]|"{1,2})*"""$', token_string) or re.match(
                        r"^\s*'''(?:[^']|'{1,2})*'''$", token_string):
                    continue
                else:
                    out.append(token_string)
            else:
                continue
        else:
            out.append(token_string)

        prev_tok_type = token_type
        last_col = end_col
        last_lineno = end_line

    return "".join([i for i in out if i.strip() != ""]).strip()


def get_method_source(filename, method_name):
    """Extract the source of a method from a class in the given file."""
    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    lines = source.splitlines(keepends=True)
                    start_line = item.lineno - 1
                    end_line = _get_function_end_line(item, lines)
                    method_source = "".join(lines[start_line:end_line])
                    return method_source

    raise ValueError(f"Method '{method_name}' not found in {filename}.")


def _get_function_end_line(node, lines):
    last_child = None
    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            if last_child is None or child.lineno > last_child.lineno:
                last_child = child
    if last_child:
        return last_child.lineno
    return node.lineno


def normalize_code(code):
    return remove_comments_and_docstrings(code)


def count_move_calls(code):
    """Count self.move(...) calls in the code."""
    return len(re.findall(r'self\.move\s*\(', code))


def compare_move_calls(original, modified):
    """Verify that all original self.move() calls are preserved in modified code."""
    orig_count = count_move_calls(original)
    mod_count = count_move_calls(modified)
    return orig_count == mod_count


def replace_method_in_file(filename, method_name, new_method_source):
    """Replace a method in the file with new source code."""
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    start_line = item.lineno - 1
                    end_line = _get_function_end_line(item, lines)

                    new_lines = new_method_source.splitlines(keepends=True)
                    if not new_lines[-1].endswith('\n'):
                        new_lines[-1] += '\n'

                    lines[start_line:end_line] = new_lines

                    with open(filename, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                    return

    raise ValueError(f"Method '{method_name}' not found in {filename}.")


# ============================================================
# LLM prompt
# ============================================================

SYSTEM_PROMPT = r"""You are a robotics code annotator. Your task is to insert `self._skill_label_tracker.register_skill(...)` calls before every `self.move(...)` call in the given `play_once()` method.

## Rules

1. Before EACH `self.move(...)` call, insert a `self._skill_label_tracker.register_skill(...)` call.
2. The `register_skill` call uses **template-based slot filling**. You must provide these keyword arguments:
   - `skill_type`: one of "grasp", "place", "move_to_pose", "move_by_displacement", "close_gripper", "open_gripper", "back_to_origin"
     Determine this from the primitive inside self.move():
       - self.grasp_actor(...)    -> "grasp"
       - self.place_actor(...)    -> "place"
       - self.move_to_pose(...)   -> "move_to_pose"
       - self.move_by_displacement(...) -> "move_by_displacement"
       - self.close_gripper(...)  -> "close_gripper"
       - self.open_gripper(...)   -> "open_gripper"
       - self.back_to_origin(...) -> "back_to_origin"
   - `arm_tag`: use `str(self.arm_tag)` or the literal arm tag string (e.g., `"right"`, `"left"`, `str(arm_tag)`)
   - `target_object_name`: short name of the object being manipulated (e.g., `"can"`, `"basket"`, `"bottle"`). Use `""` if no specific object.
   - `target`: (for "place" and "move_to_pose" only) the destination or target location name (e.g., `"basket"`, `"table"`, `"display stand"`, `"adjusted pose"`). Omit if not applicable.
   - `direction`: (for "move_by_displacement" only) short direction description (e.g., `"upward"`, `"downward"`, `"forward"`, `"inward"`). Omit if not applicable.

3. The system will automatically generate language labels from these slots using templates like:
   - grasp: "grasp {object}"
   - place: "place {object} on {target}"
   - move_to_pose: "move to {target}"
   - move_by_displacement: "move {object} {direction}"
   - open_gripper: "release {object}"
   - close_gripper: "grip {object}"
   - back_to_origin: "return arm to home position"

4. Keep the EXACT same code — do NOT modify any existing lines. Only ADD `register_skill` calls.
5. Preserve indentation. The register_skill call should have the same indentation as the self.move() call it precedes.
6. If self.move() is inside an if/else/for block, place register_skill at the same indentation level, right before self.move().
7. Return the COMPLETE modified method wrapped in ```python ... ```.

## Example

Before:
```python
    def play_once(self):
        self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.06))
        self.move(self.place_actor(self.can, arm_tag=self.arm_tag, target_pose=basket_pose, constrain='free'))
        self.move(self.open_gripper(arm_tag=self.arm_tag))
        self.move(self.back_to_origin(arm_tag=self.arm_tag))
```

After:
```python
    def play_once(self):
        self._skill_label_tracker.register_skill(
            skill_type="grasp", arm_tag=str(self.arm_tag),
            target_object_name="can")
        self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05))
        self._skill_label_tracker.register_skill(
            skill_type="move_by_displacement", arm_tag=str(self.arm_tag),
            target_object_name="can", direction="upward")
        self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.06))
        self._skill_label_tracker.register_skill(
            skill_type="place", arm_tag=str(self.arm_tag),
            target_object_name="can", target="basket")
        self.move(self.place_actor(self.can, arm_tag=self.arm_tag, target_pose=basket_pose, constrain='free'))
        self._skill_label_tracker.register_skill(
            skill_type="open_gripper", arm_tag=str(self.arm_tag),
            target_object_name="can")
        self.move(self.open_gripper(arm_tag=self.arm_tag))
        self._skill_label_tracker.register_skill(
            skill_type="back_to_origin", arm_tag=str(self.arm_tag))
        self.move(self.back_to_origin(arm_tag=self.arm_tag))
```

Now process the following code. Output ONLY the complete modified method in a ```python``` block.
"""


## Vertex AI Gemini configuration
GEMINI_PROJECT_ID = "prism-485101"
GEMINI_LOCATION = "global"
GEMINI_MODEL = "gemini-3-flash-preview"


def call_llm(source, max_try=5, verbose=True):
    """Call Vertex AI Gemini to insert register_skill() calls."""
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    from google.api_core.exceptions import ResourceExhausted

    vertexai.init(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)
    model = GenerativeModel(GEMINI_MODEL, system_instruction=[SYSTEM_PROMPT])
    gen_config = GenerationConfig(temperature=0.0)

    start = time.time()
    try_times = 0
    while try_times < max_try:
        try_times += 1
        try:
            if verbose:
                print(f"  Calling {GEMINI_MODEL} (attempt {try_times})...", end="", flush=True)
            response = model.generate_content(source, generation_config=gen_config)
            answering = response.text
        except ResourceExhausted:
            delay = 30 * (2 ** (try_times - 1))
            if verbose:
                print(f"\n  [Rate limit] Waiting {delay}s...")
            time.sleep(delay)
            continue
        except Exception:
            print(traceback.format_exc())
            continue

        result = re.search(r"```python\n([\s\S]*?)\n```", answering, re.S)
        if result is not None:
            if verbose:
                print(f" done ({time.time()-start:.2f}s, try {try_times})", flush=True)
            return result.group(1)
        else:
            if verbose:
                print(f" no code block found, retrying...", flush=True)

    return None


# ============================================================
# Main processing
# ============================================================

def strip_register_skill_calls(source):
    """Remove existing register_skill() calls from source code."""
    lines = source.split('\n')
    result = []
    skip_continuation = False
    for line in lines:
        stripped = line.strip()
        if 'register_skill(' in stripped or skip_continuation:
            # Check if this line ends the call (has closing paren)
            open_count = line.count('(') - line.count(')')
            if skip_continuation:
                open_count += 1  # account for the opening paren we're tracking
            if open_count > 0:
                skip_continuation = True
            else:
                skip_continuation = False
            continue
        result.append(line)
    return '\n'.join(result)


def process_file(file_path, max_try=5, verbose=True, force=False):
    """Process a single task file: extract play_once, add register_skill calls, replace."""
    file_path = str(file_path)
    try_count = 0

    while try_count < max_try:
        try_count += 1

        # Step 1: Extract play_once
        try:
            method_source = get_method_source(file_path, "play_once")
        except ValueError as e:
            if verbose:
                print(f"  Skipping {file_path}: {e}")
            return False

        # Check if already has register_skill calls
        if "register_skill" in method_source:
            if force:
                if verbose:
                    print(f"  {file_path}: --force: stripping existing register_skill calls")
                clean_source = strip_register_skill_calls(method_source)
                replace_method_in_file(file_path, "play_once", clean_source)
                method_source = get_method_source(file_path, "play_once")
            else:
                if verbose:
                    print(f"  {file_path}: already has register_skill calls, skipping. (use --force to re-run)")
                return True

        # Step 2: Call LLM
        modified_source = call_llm(method_source, max_try=3, verbose=verbose)
        if modified_source is None:
            if verbose:
                print(f"  {file_path}: LLM returned None on attempt {try_count}")
            continue

        # Step 3: Validate — original self.move() calls must be preserved
        if not compare_move_calls(method_source, modified_source):
            orig_cnt = count_move_calls(method_source)
            mod_cnt = count_move_calls(modified_source)
            if verbose:
                print(f"  {file_path}: move() count mismatch (orig={orig_cnt}, mod={mod_cnt}), retrying...")
            continue

        # Step 4: Validate — register_skill count should match move count
        register_count = len(re.findall(r'register_skill\s*\(', modified_source))
        move_count = count_move_calls(modified_source)
        if register_count < move_count:
            if verbose:
                print(f"  {file_path}: register_skill count ({register_count}) < move count ({move_count}), retrying...")
            continue

        # Step 5: Replace in file
        try:
            replace_method_in_file(file_path, "play_once", modified_source)
            if verbose:
                print(f"  {file_path}: SUCCESS ({register_count} register_skill calls inserted)")
            return True
        except Exception as e:
            if verbose:
                print(f"  {file_path}: replace failed: {e}")
            continue

    # Exceeded max retries
    with open("skill_label_errors.log", "a", encoding="utf-8") as f:
        f.write(f"Error processing {file_path}: Exceeded maximum retries.\n")
    return False


def batch(batch_size=5, root="./envs", force=False):
    """Process all task files in batch using thread pool."""
    name_list = [
        "adjust_bottle",
        "beat_block_hammer",
        "blocks_ranking_rgb",
        "blocks_ranking_size",
        "click_alarmclock",
        "click_bell",
        "move_can_pot",
        "move_pillbottle_pad",
        "move_playingcard_away",
        "move_stapler_pad",
        "open_laptop",
        "open_microwave",
        "place_a2b_left",
        "place_a2b_right",
        "place_bread_basket",
        "place_can_basket",
        "place_container_plate",
        "place_empty_cup",
        "place_fan",
        "place_mouse_pad",
        "place_object_basket",
        "place_object_scale",
        "place_object_stand",
        "place_phone_stand",
        "place_shoe",
        "press_stapler",
        "rotate_qrcode",
        "shake_bottle",
        "shake_bottle_horizontally",
        "stack_blocks_three",
        "stack_blocks_two",
        "stack_bowls_three",
        "stack_bowls_two",
        "stamp_seal",
        "turn_switch",
    ]

    process_list = []
    for name in name_list:
        file = Path(root) / f"{name}.py"
        if file.exists():
            process_list.append(file)
        else:
            print(f"WARNING: {file.name} not found!")

    threads = []
    finish_count, total_count = 0, len(process_list)
    print(f"Processing {total_count} task files (batch_size={batch_size})...")

    for file in process_list:
        thread = Thread(target=process_file, args=(file, 5, False, force))
        thread.start()
        threads.append([file, thread])
        while len(threads) >= batch_size:
            for t in threads:
                if not t[1].is_alive():
                    threads.remove(t)
                    finish_count += 1
                    print(f"[{finish_count:>3d}/{total_count:03d}] done: {t[0].name}", flush=True)
            time.sleep(0.1)

    while len(threads) > 0:
        for t in threads:
            if not t[1].is_alive():
                threads.remove(t)
                finish_count += 1
                print(f"[{finish_count:>3d}/{total_count:03d}] done: {t[0].name}", flush=True)
        time.sleep(0.1)

    print(f"\nAll done: {finish_count}/{total_count} files processed.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Insert register_skill() calls into task play_once() methods")
    parser.add_argument("--file", type=str, default=None, help="Single file to process")
    parser.add_argument("--batch", action="store_true", help="Process all task files")
    parser.add_argument("--root", type=str, default="./envs", help="Root directory for task files")
    parser.add_argument("--batch-size", type=int, default=5, help="Concurrent threads for batch")
    parser.add_argument("--force", action="store_true", help="Re-run even if register_skill calls exist")
    args = parser.parse_args()

    if args.file:
        process_file(args.file, verbose=True, force=args.force)
    elif args.batch:
        batch(batch_size=args.batch_size, root=args.root, force=args.force)
    else:
        parser.print_help()
