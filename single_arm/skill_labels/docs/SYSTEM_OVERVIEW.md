# Skill Label System — Overview

## What It Does

데이터 취득 시 각 `self.move()` 호출 단위로 **프레임마다 8개의 skill label**을 자동 기록하여 HDF5에 저장한다.

---

## System Flow

```
[1단계] 사전 준비 (1회성, 선택)
─────────────────────────────
python -m skill_labels.add_skill_labels --batch

  태스크 .py 소스 코드 읽기
       ↓
  Gemini가 슬롯 값 추출 (object, target, direction)
       ↓
  각 self.move() 앞에 register_skill() 호출 삽입
       ↓
  소스 파일 저장 (로봇 제어 코드 자체는 변경 없음)


[2단계] 데이터 취득 (자동)
──────────────────────────
python script/collect_data.py <task_name> <config>

  _init_task_env_()
       ↓
  SkillLabelTracker 초기화, recording_enabled = True
       ↓
  play_once() 실행
       ↓
  ┌─ register_skill()  ← 슬롯 값으로 pending skill 등록
  │       ↓
  │  primitive 호출 (grasp_actor 등)
  │   └─ fallback register_skill() ← pending 있으면 스킵
  │       ↓
  └─ self.move()
      ├─ start_skill_segment()  ← pending → active 전환
      ├─ take_dense_action()
      │   └─ _take_picture()
      │       └─ capture_frame_labels()  ← 프레임 라벨 dict를 pkl에 저장
      ├─ update_goal()  ← 관절/EE 목표 갱신
      └─ finish_skill_segment()
       ↓
  merge_pkl_to_hdf5_video()  ← pkl → HDF5 변환 (skill_labels/ 그룹 포함)
       ↓
  generate_skill_visualizations()  ← 4종 PNG 생성
```

---

## Per-Frame Labels (8개)

| Label | Type | Source |
|-------|------|--------|
| `skill_type` | string | `register_skill(skill_type=...)` |
| `language_subtask` | string | 템플릿 + 슬롯 자동 생성 |
| `verification_question` | string | 템플릿 + 슬롯 자동 생성 |
| `skill_index` | int | `start_skill_segment()` 카운터 |
| `progress` | float | `capture_frame_labels()` 내부 계산 |
| `goal_ee_pose` | float[7] | `update_goal()` |
| `goal_joint_positions` | float[6] | `update_goal()` |
| `goal_gripper_position` | float | `register_skill(goal_gripper_position=...)` |

---

## First-Write-Wins

```
register_skill()  ← 태스크 코드 (명시적)
       ↓
primitive 내부 register_skill()  ← has_pending_skill() == True → 스킵
       ↓
move() → start_skill_segment()  ← 명시적 슬롯 값 사용
```

명시적 호출이 없으면 primitive fallback이 동작한다.

---

## File Structure

```
skill_labels/
├── __init__.py              # SkillLabelTracker export
├── tracker.py               # 트래커 + 템플릿 정의
├── add_skill_labels.py      # LLM 기반 슬롯 값 삽입 스크립트
├── visualize.py             # HDF5 → 4종 PNG
└── docs/
    ├── SKILL_LABEL_DESIGN.md  # 자연어 라벨 생성 규칙
    └── SYSTEM_OVERVIEW.md     # 이 문서
```

## Modified Files (in `envs/`)

| File | What Changed |
|------|-------------|
| `_base_task.py` | tracker 초기화, 7 primitive fallback, `move()` 계측, `_take_picture()` 라벨 캡처 |
| `utils/pkl2hdf5.py` | vlen string 데이터셋 지원 추가 |
| `script/collect_data.py` | HDF5 저장 후 시각화 호출 추가 |

---

## Usage

```bash
cd single_arm

# [선택] LLM으로 고품질 슬롯 값 삽입
python -m skill_labels.add_skill_labels --batch --root ./envs

# 데이터 취득 (기존과 동일)
python script/collect_data.py place_can_basket demo_clean

# 결과 확인
#   data/episode0.hdf5  → skill_labels/ 그룹
#   skill_viz/           → 4종 PNG
```
