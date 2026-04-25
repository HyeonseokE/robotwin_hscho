# Closed-loop CaP Replay-based Dataset Logging (LeRobot v3.0)

> 성공/실패 에피소드의 `left/right_joint_path`를 별도 replay pass에서 재실행하여
> ADC 스타일 스트리밍 callback으로 LeRobot v3.0 포맷 데이터셋을 축적하는 2-phase
> 파이프라인 설계 문서.
>
> 관련 문서: [failure_detection_and_recovery.md](failure_detection_and_recovery.md)

---

## 1. 전체 구조

```
Phase 1: Closed-loop rollout                     Phase 2: Replay & record
┌─────────────────────────────────┐              ┌─────────────────────────────┐
│ make_env(task, seed)            │              │ make_env(task, seed)        │
│ planner → codegen/exec/judge    │              │   + RecordingContext.setup  │
│ → left/right_joint_path 누적     │  traj.pkl    │   + LeRobot recorder        │
│ → is_task_success()             │ ───────────▶ │ load traj.pkl               │
│ → save_traj_data() (성공+실패)  │              │ execute_path() 재실행       │
│                                 │              │  → _take_picture tap        │
│ 산출물:                         │              │    → add_frame per step     │
│   logs/seed_N/trial_K/traj.pkl  │              │ recorder.end_episode()      │
│   logs/seed_N/trial_K/report.json│             │                             │
│   logs/seed_N/trial_K/initial.png│             │ 산출물:                     │
│   logs/seed_N/trial_K/subtask_*/ │             │   recorded_data/ (LeRobot) │
│   logs/seed_N/trial_K/episode.mp4│             │   logs/.../replay.mp4 (확인)│
└─────────────────────────────────┘              └─────────────────────────────┘
```

**왜 2-phase인가**:
1. VLM 호출은 비싸고 느림. 성공/실패 판정만 1회 하고, 이미지·라벨 로깅은 deterministic replay로 따로.
2. Sim은 같은 seed + 같은 joint_path = 완전 비트 동일한 궤적 → replay 시 원하는 카메라·FPS·지터 주입 자유.
3. 라벨링(`skill.*`, `subtask.*`)은 replay 중 subtask 타임라인과 동기화해 정확히 찍힘.

---

## 2. 디렉터리 레이아웃

```
closed_loop_cap/output/
├── datasets/
│   └── <task>/
│       ├── recorded_data/                  # LeRobot v3.0 root (task당 1개)
│       │   ├── data/chunk-*/*.parquet      # state/action/labels/…
│       │   ├── videos/observation.images.<cam>/chunk-*/episode_*.mp4
│       │   └── meta/{info,stats,tasks,episodes}.json
│       └── logs/
│           └── seed_<N>/
│               └── trial_<KKK>/            # 항상 생성 (기본 trial_001)
│                   ├── report.json         # 1차 rollout 결과
│                   ├── initial.png         # 첫 RGB
│                   ├── subtask_XX/
│                   │   ├── code.py         # codegen 결과
│                   │   ├── judge.json      # JudgeResult
│                   │   ├── before.png
│                   │   └── after.png
│                   ├── episode.mp4         # 1차 rollout 확인용 mp4
│                   ├── traj.pkl            # save_traj_data 결과 (replay input)
│                   └── replay.mp4          # 2차 replay 확인용 (top cam 링크)
├── _gt_seeds/<task>.json                    # 기존 유지
├── _benchmark/<timestamp>/                  # 기존 유지
└── _sim_cache/ → DEPRECATE                  # traj.pkl은 logs/로 이동
```

**변경점**:
- `trial_KKK`는 **항상** 생성 (기본 단일 trial도 `trial_001`). 1-indexed, 3-digit zero-padded.
- LeRobot dataset은 task별 1개 root에서 seed/trial 횡단 누적.
- 1차 rollout 산출물(code, judge, mp4, traj.pkl)은 `logs/seed_N/trial_K/`에, 데이터는 `recorded_data/`에 분리.
- `_sim_cache/`는 deprecate. `save_traj_data`의 출력 경로를 `logs/seed_N/trial_K/traj.pkl`로 redirect.

---

## 3. LeRobot v3.0 Feature Schema

**정규화 규약 — ADC 호환**. 제어는 RoboTwin native (rad / 0~1) 그대로, **dataset 저장 시에만**
ADC 스키마(arm −100~+100, gripper 0~100)로 변환한다.

aloha-agilex dual-arm 기준 14-DoF:

### 3.1 Core features

| Key | dtype | shape | 단위/범위 | 설명 |
|---|---|---|---|---|
| `observation.state` | float32 | (14,) | arm −100~+100, gripper 0~100 | 6 arm_L + 1 grip_L + 6 arm_R + 1 grip_R (ADC 정규화) |
| `action` | float32 | (14,) | 동 | target qpos, 동일 정규화 |
| `observation.ee_pose.left` | float32 | (7,) | xyz(m) + quat(xyzw) | world frame |
| `observation.ee_pose.right` | float32 | (7,) | 동 | |
| `observation.images.top` | video | (H,W,3) | — | head_camera RGB |
| `observation.images.left_wrist` | video | (H,W,3) | — | wrist_camera L RGB |
| `observation.images.right_wrist` | video | (H,W,3) | — | wrist_camera R RGB |
| `task` | string | — | — | task description (전체 instruction) |

> **Note**: EE pose는 변환 없음. 학습 후 policy가 native로 돌려야 할 때를 위해
> qlimits는 메타(info.json custom field)에 함께 저장.

### 3.2 정규화 (변환 레이어)

제어 코드(`task_env.take_action`, `self.move` 등)는 **native(rad / 0~1) 그대로** 사용.
`DatasetRecorder` 내부에서만 다음 변환 적용:

**Arm joint (rad → −100~+100)**:
```python
# URDF limits: [lower, upper] per joint (from SAPIEN get_qlimits())
mid = (upper + lower) / 2
half = (upper - lower) / 2
norm = np.clip((qpos - mid) / half * 100.0, -100.0, 100.0)
```

**Gripper (0~1 → 0~100)**: `norm = gripper_val * 100.0`

**역변환** (meta에 저장되는 qlimits로 언제든 복원 가능):
```python
qpos = mid + norm / 100.0 * half
gripper_val = norm / 100.0
```

구현: `closed_loop_cap/dataset/calibration.py` (신규)
- `LimitsCache(task_env)`: 첫 호출 시 `robot.left_entity.get_qlimits()` 캐시.
- `to_normalized(state_native) → state_dataset`
- `to_native(state_dataset) → state_native` (replay/debug 용도)
- qlimits는 `meta/info.json`의 custom `calibration_limits` 필드에 dump 하여 dataset 자기기술.

### 3.3 Skill/subtask labels (ADC 미러링)

| Key | dtype | shape | 설명 |
|---|---|---|---|
| `skill.natural_language` | string | — | 현재 skill의 NL 라벨 |
| `skill.type` | string | — | `grasp`/`place`/`move_to_pose`/`move_by_displacement`/`open_gripper`/`close_gripper` |
| `skill.verification_question` | string | — | judger post-condition 문장 재활용 |
| `skill.progress` | float32 | (1,) | 0~1, replay 중 `elapsed_steps/total_steps` |
| `skill.goal_position.joint` | float32 | (14,) | target qpos |
| `skill.goal_position.left_ee` | float32 | (7,) | target EE pose (L) |
| `skill.goal_position.right_ee` | float32 | (7,) | target EE pose (R) |
| `skill.goal_position.gripper` | float32 | (2,) | target gripper [L, R] |
| `subtask.natural_language` | string | — | `SubtaskSpec.instruction` |
| `subtask.object_name` | string | — | `SubtaskSpec.target_actor` (bare) |
| `subtask.target_position` | float32 | (3,) | actor GT xyz (sim이라 오라클) |

### 3.4 Privileged GT features (yaml toggle, sim-only)

| Key | dtype | shape | 설명 |
|---|---|---|---|
| `observation.oracle.object_pose.<actor>` | float32 | (7,) | xyz+quat, per actor |
| `observation.oracle.contact.<actor>.<finger>` | float32 | (1,) | bool → float |
| `observation.oracle.ee_pose.left` | float32 | (7,) | FK 기반 (noise-free, state derived 동일 값) |
| `observation.oracle.ee_pose.right` | float32 | (7,) | 동 |
| `observation.oracle.table_height` | float32 | (1,) | 렌덤화된 테이블 높이 |

기본 off. `config.logging.privileged_features: true`일 때만 활성화. actor 목록은 `task_registry.load_task_meta(task).actor_names`에서.

### 3.5 Episode-level metadata (per-episode, meta/episodes.json custom fields)

```json
{
  "episode_index": 0,
  "length": 135,
  "tasks": ["Beat the block with the hammer"],
  "seed": 7,
  "trial": 1,
  "success": true,
  "abort_reason": null,
  "num_subtasks": 3,
  "completed_subtasks": 3,
  "task_name": "beat_block_hammer",
  "robotwin_embodiment": "aloha-agilex",
  "camera_latency_steps": 0,
  "physics_dt_s": 0.004
}
```

실패 에피소드:
```json
{
  "success": false,
  "abort_reason": "L2_subtask_2_max_retries",
  "completed_subtasks": 1,
  "failed_at_subtask_id": 2,
  "length": 84
}
```

LeRobot v3.0이 `episodes.json`에 custom 필드를 허용하므로 별도 side-car 파일 없이 여기에 담는다.

---

## 4. Phase 1 — Closed-loop rollout (1차)

### 4.1 변경 범위
- **`run_closed_loop.py`**: 대부분 기존대로. 경로만 새 구조로 redirect.
- `trial_KKK`는 항상 붙음 (기본 `trial_001`).
- `save_traj_data(0)`을 **성공/실패 모두**에서 호출 (실패 시에도 중단 시점까지의 joint_path 저장).
- 출력 pkl은 `logs/seed_N/trial_K/traj.pkl` (`save_path`를 override).

### 4.2 traj.pkl 스키마 (RoboTwin 원본)
```python
{
  "left_joint_path": np.ndarray,      # (T_left, 7)
  "right_joint_path": np.ndarray,     # (T_right, 7)
}
```
+ 추가로 `logs/seed_N/trial_K/subtask_timeline.json` 작성:
```json
[
  {"subtask_id": 1, "step_start": 0, "step_end": 42,
   "natural_language": "grasp the hammer",
   "skill_type": "grasp", "target_actor": "hammer", "arm_tag": "right",
   "verification_question": "Is gripper in contact with hammer?",
   "goal_joint": [...], "goal_ee_right": [...]},
  {"subtask_id": 2, "step_start": 42, "step_end": 84,
   "natural_language": "beat the block", ...},
  ...
]
```
→ replay 시 step index → skill/subtask 매핑에 사용.

### 4.3 1차 rollout 비디오는 유지
현재 `video/recorder.py`의 ffmpeg tee는 그대로 두고, 출력만 `logs/seed_N/trial_K/episode.mp4`로. 데이터셋용이 아니라 "1차 rollout 시 실제 VLM이 뭘 봤나" 디버깅 용도.

---

## 5. Phase 2 — Replay & streaming record (2차)

### 5.1 Entry point: `run_replay.py` (신규)
```
python -m closed_loop_cap.run_replay \
    --task beat_block_hammer \
    --seeds 1 2 3 --trials 5 \
    --include-failures               # 실패 에피소드도 replay 대상
    --privileged                     # GT features 활성
    --camera-latency-steps 0         # sim2real 지터 옵션
```
동작:
1. `output/<task>/logs/seed_<N>/trial_<K>/traj.pkl` 로드.
2. `make_env(task, seed, config, replay_mode=True)`로 새 환경 생성 (동일 seed → 동일 초기 상태).
3. `RecordingContext.setup(recorder, camera_manager, fps=30)`.
4. `subtask_timeline.json` 로드 → step index 기반 skill/subtask 라벨 주입 스케줄.
5. `task_env.replay_trajectory(left_path, right_path)` 호출 (신규 메서드, 5.3).
6. 매 sim step마다 `_take_picture` tap → `recording_context.on_step(state, action)` → `recorder.add_frame(frame)`.
7. `recorder.end_episode(discard=False)` + episode metadata에 `success/seed/trial/...` 기록.
8. 확인용 `replay.mp4`는 top 카메라 video를 `logs/.../replay.mp4`로 하드링크.

### 5.2 RecordingContext (sim-ported, ADC 미러링)
`closed_loop_cap/dataset/context.py` (신규):
- Singleton, `setup(recorder, camera_manager, task_env, fps, camera_latency_steps, ...)`.
- `set_skill(skill_type, nl, verification_q, goal_joint, goal_ee_left, goal_ee_right, goal_gripper, duration_steps)` — 새 skill 시작 시 호출, 내부 step counter reset.
- `set_subtask(nl, object_name, target_position)` — subtask 경계에서 호출.
- `on_step(state_qpos, action_qpos, ee_pose_left, ee_pose_right, gripper_targets)` — 매 step 호출, skill.progress 자동 계산.
- `_compute_privileged()` — `task_env.actor_name_dic`에서 actor pose 수집.
- `_get_images()` — `task_env.cameras.get_rgb()` 호출, `camera_latency_steps`만큼 지연 버퍼링.

ADC는 `AsyncCameraCapture` 백그라운드 스레드가 필요했지만 sim은 동기. **지연은 deque 버퍼로 시뮬레이션**:
```python
self._image_buffer = deque(maxlen=camera_latency_steps + 1)
def _get_images(self):
    current = self.task_env.cameras.get_rgb()
    self._image_buffer.append(current)
    return self._image_buffer[0]   # 가장 오래된 = latency_steps 전 프레임
```

### 5.3 `task_env.replay_trajectory()`
`Base_Task.execute_path()` 근처에 신규 메서드:
```python
def replay_trajectory(self, left_path, right_path, step_callback=None):
    # RoboTwin의 기존 path execution 로직 재사용
    # 매 step: apply joint targets → scene.step → step_callback()
```
기존 `self.move()` 내부 로직을 참고해서 path만 먹이면 되므로 작업량 적음.

### 5.4 Skill/subtask 타임라인 주입
`subtask_timeline.json`의 `step_start`/`step_end`는 `left_joint_path`의 인덱스 기준. replay 시 `RecordingContext`가 현재 step이 어느 subtask에 속하는지 판단해 `set_skill`/`set_subtask`를 자동 호출.

```python
# run_replay.py 내부
timeline = load_timeline(logs_dir)
def on_step(step_idx):
    active = find_subtask(timeline, step_idx)
    if active.id != ctx.current_subtask_id:
        ctx.set_subtask(active.natural_language, active.target_actor, active.target_position)
        ctx.set_skill(active.skill_type, active.natural_language, active.verification_question,
                      active.goal_joint, ..., duration_steps=active.step_end - active.step_start)
    ctx.on_step(...)
task_env.replay_trajectory(left_path, right_path, step_callback=on_step)
```

---

## 6. FPS & 제어 루프

- **Sim replay는 30Hz로 통일** (ADC 실제 동작과 일치: `skills_lerobot.py: RECORDING_FPS = 30`, 1 step = 1 record).
- RoboTwin 물리 step은 기본 `dt=0.004s` (250Hz). replay 메서드는 **sub-step**을 돌리되 **record tap은 30Hz 간격에서만** 발사.
  - `substeps_per_record = round(250/30) ≈ 8`
  - 8 physics step마다 1번 `step_callback` 호출.
- frame_skip_ratio = 1 (ADC의 50/30 대신 30/30). ADC의 `RecordingCallback.should_record()`처럼 복잡한 로직 불필요.

---

## 7. Sim2real 지터 옵션

### 7.1 실세계 지연 근거
- RealSense D435 30fps → frame period 33ms.
- ADC `AsyncCameraCapture` 60Hz → 제어 루프가 이미지 꺼낼 때 0~16ms stale.
- 30Hz 제어 루프에서 state read는 즉시, image는 ~0.5 step 과거 → **평균 ~0.5 step, 최악 1 step 지연**.

### 7.2 sim에 주입할 옵션
```yaml
logging:
  fps: 30
  camera_latency_steps: 0       # 0 = deterministic (기본), 1 = real-world worst-case 미러
  randomize_latency: false      # true면 매 step [0, camera_latency_steps] uniform
```
- `camera_latency_steps: 0`: 디버깅/통계 추출용 deterministic 데이터셋.
- `camera_latency_steps: 1, randomize_latency: true`: sim2real 학습 세트. 평균 0.5 step 지연 + ±0.5 step jitter → real 분포 근사.

### 7.3 State read jitter는 skip
sim의 `robot.get_left_arm_jointState()`는 즉시 반환 (μs 단위). 실세계에서도 serial read가 1ms 미만이라 무시 가능. 구현 복잡도 대비 이점 적어서 **생략**.

---

## 8. 실패 에피소드 처리

### 8.1 저장 경로
성공/실패 구분 없이 **동일한 `recorded_data/`에 episode로 append**. `meta/episodes.json`의 `success` 필드로 구분.

### 8.2 1차 rollout 변경
- 현재: 성공 시에만 `save_traj_data(0)`.
- 변경: **성공/실패 무관, 항상 호출**. `EpisodeReport`에 이미 `abort_reason`이 있으므로 meta에 그대로 전달.
- 검증: `Base_Task.__init__` (line 113)과 `setup_demo` (line 572)에서만 `left_joint_path=[]`로 초기화되고,
  `self.move()` / `execute_path()`는 매 step append만 함 (line 750, 783, 821). episode 중간에 reset 없음 →
  실패 중단 시에도 중단 시점까지의 path는 보존됨. 추가 검증 불필요.

### 8.3 2차 replay
- `--include-failures` 플래그가 있으면 `success=false` traj.pkl도 replay 대상.
- subtask_timeline.json의 마지막 subtask는 중단된 상태 → `step_end`는 실제 실행된 step 수.
- 확인용 `replay.mp4`는 성공/실패 공통.

### 8.4 학습 사용 패턴
```python
# DataLoader 예시
dataset = LeRobotDataset("output/datasets/beat_block_hammer/recorded_data")
success_only = [ep for ep in dataset.episode_data if ep["success"]]
failures = [ep for ep in dataset.episode_data if not ep["success"]]
```

---

## 9. Camera 구성

### 9.1 Sim camera 설정 (`configs/default.yaml` 추가)
```yaml
env:
  camera:
    head_camera_type: D435           # 기존
    wrist_camera_type: D435
    collect_head_camera: true
    collect_wrist_camera: true       # ← false → true
logging:
  cameras:
    - {name: top,         source: head_camera,          group: shared}
    - {name: left_wrist,  source: left_wrist_camera,    group: left_arm}
    - {name: right_wrist, source: right_wrist_camera,   group: right_arm}
```
feature 키 자동 생성: `observation.images.{name}` (ADC `CameraConfigRecord.feature_name` 규약 미러).

### 9.2 해상도/FPS
- 기본 D435 해상도(320×240) 유지, LeRobot에서 30fps video encode.
- Yaml에서 카메라별 오버라이드 가능.

### 9.3 비디오 이중 용도 (확인용 vs 데이터용)
- **데이터용**: LeRobot이 `videos/observation.images.top/.../episode_*.mp4` 자동 생성. 기본.
- **확인용 `replay.mp4`**: 같은 파일을 `logs/seed_N/trial_K/replay.mp4`로 **하드링크** (disk 중복 없음).
- 1차 rollout의 `episode.mp4`는 별도 (VLM이 본 시점 그대로 ffmpeg tee 출력).

---

## 10. 설정 (`configs/default.yaml` 추가 섹션)

```yaml
logging:
  enabled: true                       # 2차 replay pass 활성화
  fps: 30
  camera_latency_steps: 0             # 0=deterministic, 1=real-like
  randomize_latency: false
  privileged_features: false          # oracle.* 기본 off
  include_failures: true              # 실패 에피소드도 dataset에 포함
  cameras:                            # 9.1 참조
    - {name: top, source: head_camera, group: shared}
    - {name: left_wrist, source: left_wrist_camera, group: left_arm}
    - {name: right_wrist, source: right_wrist_camera, group: right_arm}

dataset:
  repo_id_template: "robotwin/{task}"  # push_to_hub 시 기본값
  root_template: "{output_dir}/datasets/{task}/recorded_data"
  use_videos: true
  image_writer_threads: 4
```

---

## 11. 신규/변경 파일 목록

| 경로 | 상태 | 목적 |
|---|---|---|
| `closed_loop_cap/dataset/__init__.py` | 신규 | |
| `closed_loop_cap/dataset/features.py` | 신규 | LeRobot features dict 동적 생성 (RoboTwin 스키마) |
| `closed_loop_cap/dataset/calibration.py` | 신규 | qlimits 캐시 + native↔ADC 정규화 변환 |
| `closed_loop_cap/dataset/recorder.py` | 신규 | `DatasetRecorder` wrapper — LeRobot v3.0 API 래핑 + 정규화 적용 |
| `closed_loop_cap/dataset/context.py` | 신규 | `RecordingContext` sim-ported |
| `closed_loop_cap/dataset/labels.py` | 신규 | subtask timeline ↔ skill label 매핑 |
| `closed_loop_cap/dataset/privileged.py` | 신규 | GT feature 수집기 |
| `closed_loop_cap/run_replay.py` | 신규 | Phase 2 엔트리 |
| `closed_loop_cap/run_closed_loop.py` | 변경 | 경로 redirect, `trial_KKK` 항상, 실패 시 save_traj_data |
| `closed_loop_cap/run_benchmark.py` | 변경 | replay pass 트리거 옵션 (`--with-replay`) |
| `closed_loop_cap/env/task_env.py` | 변경 | `replay_trajectory()`, wrist camera 활성 |
| `closed_loop_cap/configs/default.yaml` | 변경 | `logging`, `dataset` 섹션 |
| `closed_loop_cap/docs/replay_dataset_logging.md` | 신규 | 본 문서 |

---

## 12. 구현 순서 (phase)

1. **P1 — 디렉터리·경로 리팩토링** (기존 기능 유지)
   - `trial_KKK` 항상 생성
   - `save_traj_data` 출력 → `logs/seed_N/trial_K/traj.pkl`
   - `output/<task>/seed_N/` → `output/<task>/logs/seed_N/` 이동 (주: 상위 경로도 `datasets/<task>/logs/`로 이동)
   - 기존 `output/<task>/` 호환 필요하면 마이그레이션 주의
   - 기존 테스트 재실행 grace check
2. **P2 — 실패 시에도 `save_traj_data` 호출**
3. **P3 — Wrist camera 활성** 및 `task_env.replay_trajectory()` 구현
4. **P4 — LeRobot adapter** (`dataset/features.py`, `dataset/recorder.py`)
5. **P5 — `RecordingContext` + label pipeline** (`dataset/context.py`, `dataset/labels.py`)
6. **P6 — `run_replay.py`** + 확인용 replay.mp4 하드링크
7. **P7 — Privileged features** + `--privileged` 플래그
8. **P8 — `run_benchmark.py` 통합** (`--with-replay` 옵션)
9. **P9 — sim2real 지터** 옵션 (camera_latency_steps)
10. **P10 — push_to_hub 래퍼 + CI smoke**

각 phase마다 최소 smoke test 통과 후 다음으로.

---

## 13. 기존 코드와 호환성

### 13.1 Breaking
- `output/<task>/seed_<N>/` 경로 변경 → 기존 벤치마크 결과 재접근 스크립트 깨짐.
- `trial` 서브폴더 항상 생성 → 이전에 `seed_N/` 바로 아래 결과를 참조하던 테스트/툴 수정 필요.
- `_sim_cache/` deprecate.

### 13.1a 마이그레이션 정책
**기존 rollout 결과는 마이그레이션하지 않고 삭제 후 재생성한다**. 구체적으로:
- `output/beat_block_hammer/`, `output/shake_bottle/` 등 task별 결과 폴더 삭제.
- `output/_benchmark/*`, `output/_sim_cache/*` 삭제.
- `output/_gt_seeds/*.json`은 **보존** (GT seed 수집은 VLM과 무관, 재수집 비용 큼).
- `output/_phase1_smoke/`는 테스트 산출물이므로 삭제.

### 13.2 Non-breaking
- `run_closed_loop.py` CLI 인자 (`--trials`, `--start-trial`)는 기존 기본값으로 호환.
- `run_benchmark.py`의 기존 호출은 그대로 동작 (replay는 opt-in).
- LeRobot dataset은 **신규 생성**이라 기존 데이터와 충돌 없음.

---

## 14. 미결정 / 후속 고려

- **Multi-trial episode aggregation**: 같은 seed × 다른 trial이 거의 동일한 궤적을 낼 가능성. variance는 VLM stochasticity에서 나오므로 trial별 VLM output을 별도 저장해야 의미 있음.
- **Domain randomization augmentation**: replay 시 조명/배경만 랜덤화해서 같은 action으로 N variant 생성. 기본 구현 범위 밖, Phase 11 이후.
- **Depth / segmentation**: Schema에 자리는 있으나 (3.3) 기본 off. 수요 생기면 enable.
- **Dataset resume semantics**: 동일 task에 계속 append할 때 episode_index 충돌 방지. LeRobotDataset의 `resume=True` 옵션 검증 필요.

---

## 15. 참고

- ADC: `AutoDataCollector/record_dataset/{recorder,context,callback,async_camera,config}.py`
- ADC 문서: `AutoDataCollector/record_dataset/record_dataset.md`
- RoboTwin: `envs/_base_task.py` (`get_obs`, `take_action`, `save_traj_data`, `_take_picture`)
- RoboTwin robot: `envs/robot/robot.py` (`get_*_arm_jointState`, `get_*_gripper_val`)
- LeRobot: `AutoDataCollector/lerobot/src/lerobot/datasets/lerobot_dataset.py` (v3.0 API)
