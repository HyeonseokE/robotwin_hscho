# Closed-Loop CaP 실패 감지·복구 전략

> RoboTwin 환경에서 closed-loop CaP 파이프라인의 3개 실행 레이어별 **실패 신호(signals)**와 **복구 전략(recovery strategy)**을 정의한다. 이 문서는 `closed_loop_cap/` 모듈 설계의 기준 문서이며, Phase 3(Planner) / Phase 4(Executor) / Phase 6(Orchestrator) 구현 시 참조된다.

---

## 1. 전체 구조

Closed-loop CaP는 다음 3개 레이어로 분해된다.

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1 — Task Layer (Planner)                              │
│   초기 RGB + task instruction  →  subtask instruction list  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2 — Subtask Layer (Code Generator)                    │
│   현재 RGB + subtask + skill API  →  Python 스니펫          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3 — Skill Execution Layer (Motion Planner / Sim)      │
│   skill call  →  joint trajectory  →  physics step          │
└─────────────────────────────────────────────────────────────┘
```

각 레이어는 **고유한 실패 모드**를 가지며, 상위 레이어는 하위 레이어의 결과를 **신호로 받아 복구 판단**에 사용한다. Layer 3은 자체 재시도가 없으며, 모든 Layer 3 실패는 Layer 2로 bubble up 되어 코드 재생성으로 복구된다.

---

## 2. Layer 1 — Task Layer (Planner)

### 2.1 역할
- 입력: 초기 RGB + task instruction + actor_list
- 출력: `list[SubtaskSpec]`
- 호출 횟수: 에피소드당 1회 + 검증 실패 시 최대 2회 재시도

### 2.2 실패 신호

| ID | 신호 | 종류 | 감지 방법 |
|---|---|---|---|
| L1-S1 | JSON 파싱 실패 | 문법적 | `json.loads()` 예외 |
| L1-S2 | 스키마 검증 실패 | 문법적 | Pydantic 검증 — 필수 필드(`id`, `instruction`, `skill_type`, `target_actor`, `arm_tag`, `success_hint`) 누락 |
| L1-S3 | `target_actor` ∉ actor_list | 의미적 (환각) | 출력 actor 이름을 actor_list와 집합 비교 |
| L1-S4 | `skill_type` ∉ 허용 목록 | 의미적 | 6개 skill 화이트리스트 비교 |
| L1-S5 | `arm_tag` ∉ {"left", "right"} | 의미적 | 값 검증 |
| L1-S6 | `len(subtasks) == 0` | 의미적 | 길이 검사 |
| L1-S7 | 2개 이상 연속 subtask 실패 (cascade) | 사후적 | 런타임 `consecutive_fails` 카운터 ✅ |

### 2.3 복구 전략

```python
for attempt in range(max_plan_retries=2):
    raw = vlm_call(planner_prompt + prev_error_hint)
    subtasks, errors = validate(raw, actor_list)
    if not errors:
        break
    prev_error_hint = format_errors(errors)   # 다음 호출 프롬프트에 주입
else:
    abort_episode(reason="L1_planner_failed")
```

| 신호 | 전략 |
|---|---|
| L1-S1 ~ L1-S6 | 검증 에러를 프롬프트에 삽입하여 planner 재호출 (최대 2회) |
| L1-S7 (cascade) | 구현: 1회 subtask 실패 시 다음 subtask로 진행 (best-effort), **2회 연속 실패 시 episode abort** (`max_consecutive_subtask_fails` 설정) |

---

## 3. Layer 2 — Subtask Layer (Code Generator)

### 3.1 역할
- 입력: 현재 RGB + subtask + skill API 목록 + actor_list + 이전 실패 힌트
- 출력: Python 스니펫 (fenced code block)
- 호출 횟수: subtask당 최대 `max_subtask_retries=3`

### 3.2 실패 신호

| ID | 신호 | 종류 | 감지 방법 |
|---|---|---|---|
| L2-S1 | `compile()` SyntaxError | 정적 | `compile(code, "<subtask>", "exec")` 예외 |
| L2-S2 | AST 화이트리스트 위반 | 정적 | 금지 노드: `Import`(허용 목록 외), `Call(name="open"/"exec"/"eval"/"__import__")`, `Attribute("__*__")` |
| L2-S3 | `self.<name>` ∉ actor_list ∪ Base_Task attrs | 정적 | AST로 `self.X` 참조 수집 후 대조 |
| L2-S4 | `ArmTag` 이외 표기 | 정적 | `"left"` 문자열 등 잘못된 인수 타입 |
| L2-S5 | `exec()` 런타임 예외 | 동적 | try/except에서 traceback 캡처 |
| L2-S6 | exec 정상 종료 + no-op | 동적 | (ΔEE < 1cm) AND (Δgripper_joint < 0.01) AND (`self.move()` 호출 0회) |
| L2-S7 | exec timeout 초과 | 동적 | `signal.SIGALRM` 또는 스레드 timeout (기본 30s) |

### 3.3 복구 전략

```python
for attempt in range(max_subtask_retries=3):
    rgb_before = capture_rgb(handle)
    ee_before, gripper_before = snapshot_robot_state(handle)

    code = vlm_call(codegen_prompt, image=rgb_before, subtask=st,
                    last_failure=prev_fail_hint)

    static_err = ast_validate(code, actor_list)
    if static_err:
        prev_fail_hint = f"[static] {static_err}"
        continue                   # VLM 재호출만, exec 안 함

    result = sandbox_exec(code, handle, timeout_s=30)
    judge = judge_execution(result, st, handle, ee_before, gripper_before)
    if judge.ok:
        break
    prev_fail_hint = judge.hint_for_codegen
else:
    abort_episode(reason=f"L2_subtask_{st.id}_max_retries")
```

| 신호 | 힌트 예시 (다음 재생성 프롬프트 주입) |
|---|---|
| L2-S1/S2/S3/S4 | `"[static] {error_type}: {detail}. Fix the syntax and re-generate."` |
| L2-S5 | `"[runtime] Previous code raised {ExceptionType}: {short_trace}. Check actor names and skill signatures."` |
| L2-S6 (no-op) | `"[no-op] Previous code did not move the robot. You MUST call self.move(...) with a valid action sequence."` |
| L2-S7 (timeout) | `"[timeout] Execution exceeded {T}s. Simplify the action or reduce steps."` |

---

## 4. Layer 3 — Skill Execution Layer

### 4.1 역할
- 입력: skill 호출 (예: `grasp_actor(self.hammer, arm_tag=ArmTag("left"))`)
- 출력: joint trajectory + physics step 결과
- **자체 재시도 없음**: deterministic이므로 같은 입력에 같은 결과. 모든 실패는 Layer 2로 bubble up.

### 4.2 실패 신호

#### 4.2.1 Planner 레벨

| ID | 신호 | 감지 방법 |
|---|---|---|
| L3-P1 | `self.plan_success == False` | Base_Task 플래그 |
| L3-P2 | `self.left_plan_success == False` | 왼팔 IK/RRT 실패 |
| L3-P3 | `self.right_plan_success == False` | 오른팔 IK/RRT 실패 |

#### 4.2.2 스킬 사후조건 (skill_type별)

| Skill | 판정 | 감지 방법 | 구현 상태 |
|---|---|---|---|
| `grasp` | 그리퍼 finger link가 target과 접촉 | `scene.get_contacts()` + finger name heuristic | ✅ `L3-SKL-GRASP` |
| `place` | 그리퍼 finger link가 target과 **접촉 없음** | 동 (기대 False) | ✅ `L3-SKL-PLACE` |
| `move_to_pose` | EE 포즈 ≈ target_pose | SubtaskSpec에 target_pose 필드 없음 → MVP 생략 | ⏸ (L3-P*로 커버) |
| `move_by_displacement` | ΔEE ≈ 요청 변위 | 동 | ⏸ |
| `open_gripper` | gripper joint ≈ 1.0 | qpos 비교 | ✅ `L3-SKL-OPEN` |
| `close_gripper` | gripper joint ≈ 0.0 | 동 | ✅ `L3-SKL-CLOSE` |
| `move_to_pose` | EE 포즈와 target_pose 거리 < ε | `‖EE_pose - target_pose‖ < 0.05m` |
| `move_by_displacement` | ΔEE가 요청 변위와 일치 | `‖ΔEE - requested‖ < 0.02m` |
| `open_gripper` | gripper joint ≈ 목표값 | `|gripper_joint - 1.0| < 0.05` |
| `close_gripper` | gripper joint ≈ 목표값 | `|gripper_joint - 0.0| < 0.05` |

#### 4.2.3 물리 안정성

| ID | 신호 | 감지 방법 | 처리 |
|---|---|---|---|
| L3-U1 | actor 선속도 > 2 m/s | `actor.get_linear_velocity()` 노름 | **즉시 episode abort** (복구 불가) |
| L3-U2 | actor pose NaN | `np.isnan(pose).any()` | **즉시 episode abort** |
| L3-U3 | actor가 테이블 밖 | z < table_height - 0.1 or xy ∈ workspace 밖 | **즉시 episode abort** |

> 물리 파탄(L3-U1~U3)은 SAPIEN 씬 복구가 불가능하므로 **재시도 없이 episode abort**한다 (MVP 결정).

### 4.3 복구 전략: Layer 2로 bubble up

Layer 3은 자체 복구가 없다. 신호를 `JudgeResult.hint_for_codegen`에 담아 Layer 2의 재생성 루프에 주입한다.

| Layer 3 신호 | Layer 2에 주입할 힌트 |
|---|---|
| L3-P1 (planner 전체 실패) | `"Motion planning failed. Try a different pre_grasp_dis, different arm_tag, or simpler target pose."` |
| L3-P2 / L3-P3 (한쪽 팔 실패) | `"Arm {tag} cannot reach the target. Try the other arm or adjust the approach pose."` |
| grasp 접촉 없음 | `"Grasp failed — gripper is not in contact with {target}. Try a different contact_point_id or reduce grasp_dis."` |
| place 접촉 유지 | `"Place failed — gripper is still holding the object. Open the gripper with wider pos."` |
| move 미달 | `"Target pose not reached (distance={d:.3f}m). Adjust target or use two-step move."` |
| gripper 상태 불일치 | `"Gripper joint did not reach target value. Increase pos margin or check gripper state."` |
| 물리 instability (L3-U1~U3) | 힌트 없음 — episode abort |

---

## 5. 통합 JudgeResult

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class JudgeResult:
    ok: bool
    layer: Literal[1, 2, 3]          # 어느 레이어에서 실패했나
    signal_id: str                    # 예: "L3-P1", "L2-S6"
    detail: str                       # 원본 에러/상태 문자열
    hint_for_codegen: str | None      # Layer 2 재생성 프롬프트 주입용
    episode_abort: bool = False       # 물리 파탄 등 즉시 중단 플래그
```

---

## 6. 재시도 예산과 종료 조건

| 레이어 | 예산 | 소진 시 |
|---|---|---|
| Layer 1 (planner) | `max_plan_retries = 2` | episode abort |
| Layer 2 (codegen) | `max_subtask_retries = 3` | episode abort |
| Layer 3 | 자체 재시도 없음 | Layer 2로 bubble up |
| 물리 instability | 0 (재시도 없음) | **즉시 episode abort** |
| Episode 전체 | `max_total_steps = 30` | episode abort |

---

## 7. 데이터 흐름 다이어그램

```
Layer 1 (최대 3회: 1 + 재시도 2)
   │  검증 실패 시 hint 주입
   ▼  subtask list
   ┌─────────────────────────────────────────────────────────┐
   │ for st in subtasks:                                     │
   │    for attempt in range(3):                             │
   │       rgb_before = capture_rgb()                        │
   │       code = codegen(rgb_before, st, last_hint) ◀──────┐│
   │       │                                                ││
   │       ▼ AST 정적 검증 (L2-S1~S4)                       ││
   │       ├─ fail ── hint="[static] ..." ──────────────────┘│
   │       │                                                 │
   │       ▼ sandbox exec (timeout)                          │
   │       ├─ L2-S5 exception ── hint="[runtime] ..." ───────┤
   │       ├─ L2-S6 no-op     ── hint="[no-op] ..." ─────────┤
   │       ├─ L2-S7 timeout   ── hint="[timeout] ..." ───────┤
   │       │                                                 │
   │       ▼ Layer 3 judgers                                 │
   │       ├─ L3-P* planner   ── hint="[motion] ..." ────────┤
   │       ├─ 스킬 사후조건    ── hint="[skill] ..." ────────┤
   │       ├─ L3-U* 물리파탄   ── episode_abort=True ────┐   │
   │       └─ ok ── break                                │   │
   │    else: abort(L2_max_retries)                      │   │
   │                                                     ▼   │
   │                                           abort(physics)│
   └─────────────────────────────────────────────────────────┘
   │
   ▼
check_success() → 성공 시 save_traj_data()
```

---

## 8. 확정된 설계 결정 (사용자 확인 완료)

1. **아키텍처**: 외부 드라이버 (옵션 B), 기존 `code_gen/`·`envs/` import만.
2. **대상 태스크**: 50개 전부.
3. **VLM**: Google AI Studio Gemini API.
4. **API 키**: `closed_loop_cap/gemini_api_key.json` (gitignored), 사용자가 문자열로 직접 입력.
5. **Verifier**: MVP 제외.
6. **데이터 저장**: 성공 에피소드는 기존 `save_traj_data()` (HDF5).
7. **샌드박스**: AST 화이트리스트 + timeout (subprocess 격리 안 함).
8. **평가**: MVP는 `check_num=10`.
9. **재시도 소진 시**: episode abort (옵션 a).
10. **no-op 감지 기준**: ΔEE + Δgripper_joint 모두 고려.
11. **물리 instability**: 즉시 episode abort (옵션 b).
12. **MVP skill_type**: `grasp`, `place`, `move_to_pose`, `move_by_displacement`, `open_gripper`, `close_gripper` (6종).

---

## 9. 향후 고려 사항 (MVP 이후)

현재 MVP에 포함되지 않았지만, 운영/확장 시 추가로 고려할 실패 신호 및 복구 전략:

### 9.1 API/인프라 레이어 (Layer 0으로 추가 가능)

| 신호 | 복구 |
|---|---|
| Gemini API 429 rate limit | exponential backoff, 최대 3회 재시도 |
| Gemini API 5xx | 동일하게 backoff 재시도 |
| 응답 토큰 truncation (`finish_reason=="MAX_TOKENS"`) | `max_output_tokens` 증액 + 재호출 |
| Safety filter block (`finish_reason=="SAFETY"`) | 프롬프트에서 민감 표현 제거, planner-level fail로 처리 |
| 네트워크 timeout | 재시도 + 최종 실패 시 episode abort |
| API key 무효 | 즉시 fatal, 실행 중단 |

### 9.2 이미지/지각 레이어

| 신호 | 복구 |
|---|---|
| 카메라 프레임 전부 검정 / NaN | 1 step 시뮬 진행 후 재캡처 |
| 타겟 actor가 프레임 밖 / 가림 | head + wrist 카메라 **멀티뷰** VLM 입력으로 전환 |
| 저해상도·저조도로 환각 유발 | 리사이즈 상한 완화, 조명 랜덤화 비활성화 |

### 9.3 의미적 일관성 검사 (cross-layer drift)

| 신호 | 복구 |
|---|---|
| planner의 `target_actor`와 codegen의 `self.<name>` 불일치 | AST로 추출해 비교 → 불일치 시 L2 재생성 |
| planner의 `arm_tag`와 code의 `ArmTag(...)` 불일치 | 동일, L2 재생성 |
| subtask instruction의 의도(예: "lift")와 코드의 실제 skill(예: `place`) 불일치 | VLM-based 의도 검증 또는 skill_type 화이트리스트 강제 |

### 9.4 진행성 / 루프 이탈

| 신호 | 복구 |
|---|---|
| 동일 subtask에서 3회 재시도 모두 **유사 코드** 생성 (시멘틱 해시 동일) | VLM temperature 상향 (0 → 0.3) 또는 few-shot 예시 삽입 |
| 여러 subtask에서 같은 에러 패턴 반복 | planner-level 재계획 트리거 (cascade recovery) |
| Wall-clock 시간 예산 초과 | episode abort (별도 예산) |

### 9.5 Trajectory 품질

| 신호 | 복구 |
|---|---|
| Joint jerk/acceleration 한계 초과 | TOPP 파라미터 완화 또는 target 세분화 |
| TOPP 실패 (feasible하나 최적화 실패) | 기본 속도 프로파일로 fallback |
| 예상치 못한 접촉 (테이블·벽 긁음) | SAPIEN contact reports로 감지, L2 hint "avoid collision with table/wall" |

### 9.6 Checkpoint & Rollback (고급)

- **SAPIEN 씬 상태 직렬화**: 각 subtask 시작 전 `pack()` / 실패 시 `unpack()` 으로 복원.
  - 장점: Layer 2 재시도 시 씬이 **깨끗한 상태**에서 다시 시도 가능.
  - 단점: SAPIEN의 완전한 state serialization은 actor/rigid/articulation 각각 다루어야 하며, 렌더 상태까지 포함하면 복잡.
  - MVP 이후 Phase 8 후보.

### 9.7 Planner-level Cascade Recovery

- 2개 이상의 subtask가 연속 실패하면 **계획 자체가 잘못됐을 가능성** 높음.
- 복구: 현재 씬 RGB + 실패 로그를 planner에 다시 넘겨 **재계획** (subtask list 재생성).
- MVP는 abort로 처리하지만, 성공률 향상에 가장 임팩트가 큰 확장 중 하나.

### 9.8 Few-shot 학습 신호

- 성공한 에피소드의 (subtask, code) 페어를 저장해둔 뒤, 실패가 잦은 태스크의 codegen 프롬프트에 **few-shot 예시**로 삽입.
- 레포 구조: `closed_loop_cap/output/_shots/<task>/<subtask_hash>.py`.
- Offline 업데이트로 점진적 성능 향상.

### 9.9 Multi-view / Depth 입력

- 현재는 `head_camera` RGB만 사용. Wrist camera(팔 시점)는 grasp·place 시 훨씬 유용.
- Depth map 또는 pointcloud를 텍스트 요약(예: "object at (0.3, 0.1, 0.05)")으로 변환해 프롬프트에 첨가하는 방식도 고려.

### 9.10 Verifier 재도입 (Phase 5)

MVP에서 제외했으나, Tier 2 사후조건 판정기가 애매한 subtask(예: 복잡한 조립)에서는 VLM-based verifier가 유일한 판단 수단. 선택적 경로(`config.verifier.enabled=true`)로 나중에 재활성화.

---

## 10. 참고 모듈

- `envs/_base_task.py` — `check_actors_contact`, `plan_success`, `left/right_plan_success`, 카메라 메서드
- `trajectory_refinement/vlm_analysis/capture_keyframes.py` — RGB 캡처 로직 (재사용)
- `code_gen/prompt.py` — `AVAILABLE_ENV_FUNCTION`, `FUNCTION_EXAMPLE` (skill catalog)
- `code_gen/observation_agent.py` — VLM(vision) 호출 패턴 (재사용, API 키만 `gemini_api_key.json`으로 대체)
