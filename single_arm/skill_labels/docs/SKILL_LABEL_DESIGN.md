# Skill-Level Semantic Label Design

## Overview

Each `self.move()` call in a task's `play_once()` method corresponds to one **skill segment**.
Every frame within a segment is annotated with 8 labels, including two natural language labels:
- `language_subtask`: describes what the skill does
- `verification_question`: yes/no question to verify skill success

Both are generated from **predefined templates + slot values**, ensuring consistent formatting across all tasks.

---

## Template-Based Label Generation

### Principle

```
skill_type (from code)  →  selects template
slot values (from LLM)  →  fills template blanks
```

The system does NOT use free-form LLM-generated sentences.
Instead, templates are fixed in code, and the LLM only extracts short slot values from the task code context.

---

## language_subtask Templates

| skill_type             | Template                          | Required Slots     |
|------------------------|-----------------------------------|--------------------|
| `grasp`                | `grasp {object}`                  | object             |
| `place`                | `place {object} on {target}`      | object, target     |
| `move_to_pose`         | `move to {target}`                | target             |
| `move_by_displacement` | `move {object} {direction}`       | object, direction  |
| `open_gripper`         | `release {object}`                | object             |
| `close_gripper`        | `grip {object}`                   | object             |
| `back_to_origin`       | `return arm to home position`     | (none)             |

## verification_question Templates

| skill_type             | Template                              |
|------------------------|---------------------------------------|
| `grasp`                | `is {object} grasped?`                |
| `place`                | `is {object} placed on {target}?`     |
| `move_to_pose`         | `has arm reached {target}?`           |
| `move_by_displacement` | `has {object} moved {direction}?`     |
| `open_gripper`         | `is {object} released?`              |
| `close_gripper`        | `is {object} gripped?`               |
| `back_to_origin`       | `is arm at home position?`           |

---

## Slot Definitions

| Slot          | Description                              | Examples                                    |
|---------------|------------------------------------------|---------------------------------------------|
| `{object}`    | Name of the object being manipulated     | `can`, `basket`, `bottle`, `block`          |
| `{target}`    | Destination or target location/object    | `basket`, `table`, `display stand`, `adjusted pose` |
| `{direction}` | Movement direction description           | `upward`, `downward`, `forward`, `inward`, `upward and inward` |

---

## Slot Value Sources

### 1. Explicit (LLM-inserted via `add_skill_labels.py`)

`add_skill_labels.py` uses Vertex AI Gemini to read each task's `play_once()` code
and insert `register_skill()` calls with appropriate slot values:

```python
# LLM reads code context and extracts slot values:
self._skill_label_tracker.register_skill(
    skill_type="place", arm_tag=str(self.arm_tag),
    target_object_name="can", target="basket")
self.move(self.place_actor(self.can, arm_tag=self.arm_tag, target_pose=basket_pose))
```

Result:
- `language_subtask` = `"place can on basket"`
- `verification_question` = `"is can placed on basket?"`

### 2. Fallback (primitive auto-registration in `_base_task.py`)

When `add_skill_labels.py` has not been run, primitives automatically register
with default slot values:

```python
# Inside grasp_actor():
self._skill_label_tracker.register_skill(
    skill_type="grasp", arm_tag=str(arm_tag),
    target_object_name=actor.get_name())
```

Result:
- `language_subtask` = `"grasp {actor_name}"`
- `verification_question` = `"is {actor_name} grasped?"`

### 3. Default values (when slot is empty)

| Slot          | Default Value      |
|---------------|--------------------|
| `{object}`    | `object`           |
| `{target}`    | `target position`  |
| `{direction}` | `to target`        |

---

## Generation Flow

```
register_skill(skill_type="place", target_object_name="can", target="basket")
       │                                    │                      │
       ▼                                    ▼                      ▼
  Template lookup               Slot filling: {object}="can", {target}="basket"
       │                                    │
       ▼                                    ▼
  "place {object} on {target}"    →    "place can on basket"
  "is {object} placed on {target}?"  →  "is can placed on basket?"
```

---

## Full Example: place_can_basket

| # | skill_type             | Slots                              | language_subtask                | verification_question                  |
|---|------------------------|------------------------------------|---------------------------------|----------------------------------------|
| 0 | grasp                  | object=can                         | grasp can                       | is can grasped?                        |
| 1 | place                  | object=can, target=basket          | place can on basket             | is can placed on basket?               |
| 2 | open_gripper           | object=can                         | release can                     | is can released?                       |
| 3 | move_by_displacement   | object=can, direction=upward       | move can upward                 | has can moved upward?                  |
| 4 | back_to_origin         | (none)                             | return arm to home position     | is arm at home position?               |
| 5 | grasp                  | object=basket                      | grasp basket                    | is basket grasped?                     |
| 6 | close_gripper          | object=basket                      | grip basket                     | is basket gripped?                     |
| 7 | move_by_displacement   | object=basket, direction=upward and inward | move basket upward and inward | has basket moved upward and inward? |

---

## HDF5 Output

Per-frame labels are stored in the `skill_labels/` group:

```
episode{N}.hdf5
└── skill_labels/
    ├── skill_type             (T,)   vlen string   e.g., "grasp"
    ├── language_subtask       (T,)   vlen string   e.g., "grasp can"
    ├── verification_question  (T,)   vlen string   e.g., "is can grasped?"
    ├── skill_index            (T,)   int32
    ├── progress               (T,)   float32
    ├── goal_ee_pose           (T,7)  float32
    ├── goal_joint_positions   (T,6)  float32
    └── goal_gripper_position  (T,)   float32
```

---

## Implementation Files

| File | Role |
|------|------|
| `tracker.py` | Template definitions (`SUBTASK_TEMPLATES`, `VERIFICATION_TEMPLATES`) and `_render_template()` |
| `add_skill_labels.py` | LLM (Gemini) extracts slot values from task code and inserts `register_skill()` calls |
| `visualize.py` | Reads HDF5 labels and generates 4 types of PNG visualizations |
| `_base_task.py` | Fallback slot filling in 7 primitives + `move()` instrumentation + per-frame capture |
