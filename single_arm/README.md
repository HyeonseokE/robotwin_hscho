# RoboTwin Single-Arm

RoboTwin의 Dual-Arm 로봇 태스크를 Single-Arm 로봇용으로 변환한 파이프라인입니다.

원본 50개 태스크 중 35개를 Single-Arm으로 변환/유지하였으며, 양 팔 협동이 필수적인 15개 태스크는 제외되었습니다.

---

## 태스크 목록

### 1. 원래 Single-Arm 태스크 (26개)

원본에서 이미 한 팔만 사용하는 태스크입니다. `play_once()`에서 `self.move()` 호출 시 항상 단일 인자만 사용합니다.

| # | Task Name | 설명 |
|---|-----------|------|
| 1 | `adjust_bottle` | 기울어진 병을 잡아 세워 놓기 |
| 2 | `beat_block_hammer` | 망치로 블록 치기 |
| 3 | `click_alarmclock` | 알람시계 버튼 누르기 |
| 4 | `click_bell` | 벨 누르기 |
| 5 | `move_can_pot` | 캔을 냄비 옆으로 이동 |
| 6 | `move_pillbottle_pad` | 약병을 패드 위로 이동 |
| 7 | `move_playingcard_away` | 카드를 테이블 밖으로 밀기 |
| 8 | `move_stapler_pad` | 스테이플러를 패드 위로 이동 |
| 9 | `open_laptop` | 노트북 뚜껑 열기 |
| 10 | `open_microwave` | 전자레인지 문 열기 |
| 11 | `place_a2b_left` | 물체를 다른 물체 왼쪽에 놓기 |
| 12 | `place_a2b_right` | 물체를 다른 물체 오른쪽에 놓기 |
| 13 | `place_container_plate` | 그릇을 접시 위에 놓기 |
| 14 | `place_empty_cup` | 빈 컵을 코스터에 놓기 |
| 15 | `place_fan` | 선풍기를 패드에 놓기 |
| 16 | `place_mouse_pad` | 마우스를 패드에 놓기 |
| 17 | `place_object_scale` | 물체를 저울에 놓기 |
| 18 | `place_object_stand` | 물체를 진열대에 놓기 |
| 19 | `place_phone_stand` | 핸드폰을 거치대에 놓기 |
| 20 | `place_shoe` | 신발 한 짝 놓기 |
| 21 | `press_stapler` | 스테이플러 누르기 |
| 22 | `rotate_qrcode` | QR코드 표지판 회전 |
| 23 | `shake_bottle` | 병 잡고 세로로 흔들기 |
| 24 | `shake_bottle_horizontally` | 병 잡고 가로로 흔들기 |
| 25 | `stamp_seal` | 도장 찍기 |
| 26 | `turn_switch` | 스위치 전환 |

### 2. Dual-Arm Simultaneous → Single-Arm 변환 (6개)

원본에서 `self.move(arm1_action, arm2_action)` 형태로 양 팔을 동시에 사용하던 태스크를 순차 실행으로 변환한 태스크입니다.

| # | Task Name | 설명 | 변환 방식 |
|---|-----------|------|-----------|
| 1 | `blocks_ranking_rgb` | 블록 3개 RGB 색상순 정렬 | `back_to_origin` 동시 호출 제거, 순차 실행 |
| 2 | `blocks_ranking_size` | 블록 3개 크기순 정렬 | 동일 |
| 3 | `stack_blocks_two` | 블록 2개 쌓기 | 팔 고정, 순차 실행 |
| 4 | `stack_blocks_three` | 블록 3개 쌓기 | 동일 |
| 5 | `stack_bowls_two` | 그릇 2개 쌓기 | 동일 |
| 6 | `stack_bowls_three` | 그릇 3개 쌓기 | 동일 |

### 3. Dual-Arm Sequential/Handover → Single-Arm 변환 (3개)

원본에서 한 팔이 물체를 잡고 다른 팔로 넘기거나, 역할을 분담하던 태스크를 한 팔 순차 실행으로 변환한 태스크입니다.

| # | Task Name | 설명 | 변환 방식 |
|---|-----------|------|-----------|
| 1 | `place_bread_basket` | 빵을 바구니에 넣기 | 한 팔이 순차적으로 빵을 하나씩 넣기 |
| 2 | `place_can_basket` | 캔을 바구니에 넣고 들기 | 같은 팔이 캔 넣기 → 바구니 들기 순차 수행 |
| 3 | `place_object_basket` | 물체를 바구니에 넣고 들기 | 같은 팔이 물체 넣기 → 바구니 들기 순차 수행 |

### 4. 제외된 태스크 (15개)

양 팔 협동이 물리적으로 필수적이거나, 핸드오버 자체가 태스크 목적인 경우 제외되었습니다.

| # | Task Name | 설명 | 제외 사유 |
|---|-----------|------|-----------|
| 1 | `grab_roller` | 양손으로 롤러 잡기 | 양손 동시 잡기 필수 |
| 2 | `lift_pot` | 양손으로 냄비 들기 | 양손 동시 들기 필수 |
| 3 | `scan_object` | 스캐너+물체 양손 협동 | 양손 협동 필수 |
| 4 | `handover_block` | 블록을 반대 팔로 넘기기 | 핸드오버가 태스크 자체 |
| 5 | `handover_mic` | 마이크를 반대 팔로 넘기기 | 핸드오버가 태스크 자체 |
| 6 | `put_object_cabinet` | 서랍 열고 물체 넣기 | 서랍+물건 동시 조작 필요 |
| 7 | `pick_dual_bottles` | 양손으로 병 2개 동시 집기 | 미구현 (변환 가능) |
| 8 | `pick_diverse_bottles` | 다른 종류 병 2개 동시 집기 | 미구현 (변환 가능) |
| 9 | `place_bread_skillet` | 프라이팬+빵 동시 조작 | 미구현 (변환 가능) |
| 10 | `place_burger_fries` | 햄버거+감자튀김 트레이에 놓기 | 미구현 (변환 가능) |
| 11 | `place_cans_plasticbox` | 캔 2개를 상자에 넣기 | 미구현 (변환 가능) |
| 12 | `place_dual_shoes` | 신발 한 쌍 신발장에 넣기 | 미구현 (변환 가능) |
| 13 | `hanging_mug` | 머그잔 넘겨받아 걸기 | 미구현 (변환 가능) |
| 14 | `dump_bin_bigbin` | 쓰레기통 넘겨받아 비우기 | 미구현 (변환 가능) |
| 15 | `put_bottles_dustbin` | 병 3개 분류해 쓰레기통에 | 미구현 (변환 가능) |

> 7~15번은 `sort_task.doc`에서 변환 가능(O)으로 분류되었으나 아직 구현되지 않은 태스크입니다.

---

## 핵심 변환 패턴

### 작업 공간 제한

물체 배치 범위를 양쪽에서 한쪽으로 제한하여, 한 팔의 도달 범위 내에 모든 물체가 위치하도록 합니다.

```python
# 원본 (Dual-Arm) - 양쪽에 물체 분포
xlim = [-0.28, 0.28]

# Single-Arm - 한쪽으로 제한
xlim = [0, 0.28]
```

### 팔 선택 고정

원본에서는 물체 위치에 따라 좌/우 팔을 동적으로 선택했지만, Single-Arm에서는 한쪽 팔(주로 `"right"`)로 고정합니다.

```python
# 원본 (Dual-Arm)
arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

# Single-Arm
arm_tag = ArmTag("right")
```

### 병렬 동작 → 순차 실행

양 팔 동시 실행을 단일 팔 순차 실행으로 변환합니다.

```python
# 원본 (Dual-Arm) - 두 물체를 동시에 잡기
self.move(
    self.grasp_actor(obj_1, arm_tag="left"),
    self.grasp_actor(obj_2, arm_tag="right"),
)

# Single-Arm - 하나씩 순차 실행
self.move(self.grasp_actor(obj_1, arm_tag="right"))
self.move(self.grasp_actor(obj_2, arm_tag="right"))
```

### 핸드오버 제거

반대쪽 팔의 역할(바구니 잡기 등)을 같은 팔이 순차적으로 수행하도록 변경합니다.

```python
# 원본 (Dual-Arm) - 한 팔로 캔 놓고, 반대 팔로 바구니 잡기
self.move(
    self.back_to_origin(arm_tag=self.arm_tag),
    self.grasp_actor(self.basket, arm_tag=self.arm_tag.opposite),
)

# Single-Arm - 같은 팔로 순차 수행
self.move(self.back_to_origin(arm_tag=self.arm_tag))
self.move(self.grasp_actor(self.basket, arm_tag=self.arm_tag))
```

### Pose 방향 고정

양 팔 조건 분기 대신 고정된 팔의 방향값을 직접 사용합니다.

```python
# 원본 (Dual-Arm)
orientation = (-1, 0, 0, 0) if arm_tag == "left" else (0.05, 0, 0, 0.99)

# Single-Arm
orientation = (0.05, 0, 0, 0.99)
```

### 카메라/관측 축소

| 항목 | 원본 (Dual-Arm) | Single-Arm |
|-----|----------------|------------|
| Wrist 카메라 | 좌 + 우 2개 | 좌 1개 |
| 그리퍼 상태 | 좌/우 2개 기록 | 1개만 기록 |
| Joint State 벡터 | 좌+우 연결 (~15D) | 한쪽만 (~8D) |
| 포인트 클라우드 | 좌+우 카메라 합산 | 단일 카메라 |

---

## Skill-Level Semantic Labeling

### 개요

RoboTwin의 데이터 취득은 `play_once()` 내에서 **스킬 단위로 순차 실행**됩니다. 각 `self.move()` 호출이 하나의 스킬 실행 단위이므로, 에피소드 trajectory에 **스킬 간 명확한 시간적 boundary**가 존재합니다.

이 boundary를 활용하여, 전체 에피소드 레벨의 라벨(`"캔을 바구니에 넣고 들기"`)뿐 아니라 **각 스킬 구간마다 추가적인 semantic annotation**을 기록할 수 있습니다. 이를 **Skill-Level Semantic Label**이라 합니다.

### 라벨링 항목

각 스킬 구간(t_start ~ t_end)에 대해 다음 정보를 기록합니다:

| 항목 | 설명 | 예시 |
|-----|------|------|
| `skill_type` | 스킬 primitive 종류 | `grasp`, `place`, `lift`, `move`, `release`, `return` |
| `language_subtask` | 자연어 서브태스크 설명 | `"pick up the can"`, `"place the can into the basket"` |
| `goal_position` | 해당 스킬의 목표 위치 (x, y, z) | `[0.12, 0.05, 0.83]` |
| `goal_orientation` | 해당 스킬의 목표 자세 (quaternion) | `[0.05, 0, 0, 0.99]` |
| `target_object` | 조작 대상 물체 | `"can"`, `"basket"` |
| `t_start` | 스킬 시작 timestep | `0` |
| `t_end` | 스킬 종료 timestep | `45` |

### 예시: `place_can_basket`

`play_once()` 코드에서 각 `self.move()` 호출이 하나의 스킬 구간에 대응됩니다:

```python
# Skill 0: grasp — "pick up the can"
self.move(self.grasp_actor(self.can, arm_tag=self.arm_tag, pre_grasp_dis=0.05))

# Skill 1: place — "place the can into the basket"
self.move(self.place_actor(self.can, arm_tag=self.arm_tag, target_pose=place_pose, ...))

# Skill 2: return — "move arm back to home position"
self.move(self.back_to_origin(arm_tag=self.arm_tag))

# Skill 3: grasp — "grasp the basket handle"
self.move(self.grasp_actor(self.basket, arm_tag=self.arm_tag, pre_grasp_dis=0.08))

# Skill 4: lift — "lift the basket upward"
self.move(self.move_by_displacement(arm_tag=self.arm_tag, x=-0.02, z=0.05))
```

이에 대응하는 skill-level semantic label:

| 구간 | skill_type | language_subtask | target_object | goal_position |
|------|-----------|-----------------|---------------|---------------|
| t₀~t₁ | `grasp` | "pick up the can" | `can` | 캔 위치 |
| t₁~t₂ | `place` | "place the can into the basket" | `can` | 바구니 내부 위치 |
| t₂~t₃ | `return` | "move arm back to home position" | — | 홈 위치 |
| t₃~t₄ | `grasp` | "grasp the basket handle" | `basket` | 바구니 핸들 위치 |
| t₄~t₅ | `lift` | "lift the basket upward" | `basket` | 현재 위치 + Δz |

### 스킬 Primitive 목록

코드베이스에서 사용되는 스킬 primitive는 다음과 같습니다:

| Skill Primitive | 메서드 | 설명 |
|----------------|--------|------|
| `grasp` | `grasp_actor()` | 물체에 접근 → 접촉 → 그리퍼 닫기 |
| `place` | `place_actor()` | 목표 위치로 접근 → 배치 → 그리퍼 열기 |
| `lift` | `move_by_displacement(z=+)` | 물체를 잡은 상태에서 위로 들기 |
| `move` | `move_to_pose()` / `move_by_displacement()` | 팔을 특정 위치로 이동 |
| `release` | `open_gripper()` | 그리퍼 열기 |
| `close` | `close_gripper()` | 그리퍼 닫기 |
| `return` | `back_to_origin()` | 홈 위치로 복귀 |

### 현재 상태 및 활용

현재 trajectory 데이터는 **에피소드 단위**로 HDF5에 저장되며, 스킬 boundary는 명시적으로 기록되지 않습니다. 그러나 `play_once()` 코드의 `self.move()` 호출 순서가 스킬 구간을 정의하므로, 데이터 수집 파이프라인(`collect_data.py`)에서 각 `self.move()` 실행 시점을 기록하면 skill-level semantic label을 자동으로 생성할 수 있습니다.

이를 통해 다음과 같은 활용이 가능합니다:
- **Hierarchical policy learning**: 전체 태스크를 스킬 단위로 분해하여 계층적 정책 학습
- **Language-conditioned manipulation**: 스킬별 자연어 지시에 따른 조건부 제어
- **Skill-level success detection**: 전체 에피소드가 아닌 개별 스킬 단위의 성공/실패 판정
- **Goal-conditioned learning**: 스킬별 목표 위치를 조건으로 한 학습

---

## 요약

| 분류 | 개수 | 비고 |
|-----|------|------|
| 원래 Single-Arm | 26개 | 즉시 사용 가능 |
| Dual → Single 변환 완료 | 9개 | `play_once()` 재작성 |
| 변환 가능하나 미구현 | 9개 | `sort_task.doc` 참고 |
| 변환 불가 (양손 필수) | 6개 | 제외 |
| **합계 (현재 사용 가능)** | **35개** | |
