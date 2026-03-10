# RoboTwin 데이터 생성 파이프라인

## 개요

RoboTwin v2.0은 **SAPIEN 3.0** 물리 시뮬레이션 엔진 기반의 양팔(bimanual) 로봇 조작 데이터 자동 생성 시스템이다. 태스크 이름과 설정 파일만 지정하면, 씬 구성부터 모션 플래닝, 관측 데이터 기록, 자연어 지시문 생성까지 사람 개입 없이 전 과정이 자동으로 수행된다.

```
bash collect_data.sh <task_name> <task_config> <gpu_id>
# 예시: bash collect_data.sh beat_block_hammer demo_randomized 0
```

---

## 전체 파이프라인 구조

```
collect_data.sh
 └─ script/collect_data.py
      │
      ├─ [Phase 1] Seed 탐색 (궤적 계획)
      │    └─ envs/<task>.setup_demo()  →  SAPIEN 씬 초기화
      │         ├─ load_actors()         →  물체 랜덤 배치
      │         ├─ load_robot()          →  URDF 로봇 로딩
      │         └─ load_camera()         →  RGBD 센서 설정
      │    └─ play_once()               →  모션 플래너로 관절 궤적 생성
      │    └─ check_success()           →  태스크 성공 판정
      │    └─ save_traj_data()          →  궤적 PKL 저장
      │
      ├─ [Phase 2] 관측 데이터 기록
      │    └─ setup_demo(seed=saved)    →  동일 시드로 씬 재구성
      │    └─ set_path_lst()            →  저장된 궤적 로드 (리플레이 모드)
      │    └─ play_once()               →  궤적 리플레이 + 카메라 녹화
      │         └─ _take_picture()      →  save_freq 간격으로 관측 캡처
      │    └─ merge_pkl_to_hdf5_video() →  PKL → HDF5 + MP4 변환
      │
      └─ [Phase 3] 자연어 지시문 생성
           └─ gen_episode_instructions.sh
                └─ generate_episode_instructions.py
                     ├─ scene_info.json              →  에피소드별 물체/팔 파라미터
                     ├─ task_instruction/<task>.json  →  태스크별 템플릿
                     └─ objects_description/*.json    →  물체 자연어 설명
```

---

## Phase 1: Seed 탐색 (궤적 계획)

성공적인 조작 궤적을 가진 시드를 찾는 단계이다.

### 동작 흐름

```python
# script/collect_data.py — run()
while suc_num < args["episode_num"]:
    TASK_ENV.setup_demo(now_ep_num=suc_num, seed=epid, **args)
    TASK_ENV.play_once()

    if TASK_ENV.plan_success and TASK_ENV.check_success():
        seed_list.append(epid)
        TASK_ENV.save_traj_data(suc_num)   # → _traj_data/episodeN.pkl
        suc_num += 1

    epid += 1
```

### 세부 과정

1. **씬 초기화** (`setup_demo` → `_init_task_env_`)
   - SAPIEN 엔진/씬 생성 (물리 시뮬레이션 250Hz)
   - 레이트레이싱 렌더러 설정 (32 spp, OIDN 디노이저)
   - 테이블/벽 생성 (도메인 랜덤화 시 텍스처 랜덤)
   - 로봇 URDF 로딩 및 홈 포지션 설정
   - 카메라(헤드 + 양쪽 손목) 설정

2. **물체 배치** (`load_actors`)
   - 태스크별 필요 물체를 3D 에셋에서 로딩
   - `rand_pose()`로 위치/회전 랜덤화
   - 방해 물체 10개 랜덤 배치 (cluttered table 옵션)

3. **안정성 검사** (`check_stable`)
   - 2000스텝 물리 시뮬 후 500스텝간 물체 자세 변화 측정
   - 3도 이상 회전 변화 시 `UnStableError` 발생, 해당 시드 폐기

4. **궤적 생성** (`play_once`)
   - 태스크별 스크립트 컨트롤러가 조작 프리미티브 순차 실행
   - 모션 플래너(mplib RRT 또는 CuRobo)가 충돌 회피 경로 계산
   - TOPP-RA로 시간 최적 궤적 보간

5. **성공 판정 및 저장**
   - `check_success()`로 물체 위치/접촉 조건 검증
   - 성공 시 좌/우 관절 경로를 `_traj_data/episodeN.pkl`에 저장
   - 시드 목록을 `seed.txt`에 기록 (중단 후 재개 지원)

---

## Phase 1과 Phase 2의 차이

두 Phase 모두 `scene.step()`으로 물리 시뮬레이션을 돌리는 것은 동일하지만, 목적이 다르다.

| | Phase 1 | Phase 2 |
|---|---|---|
| **목적** | 성공하는 시드/궤적 찾기 | 관측 데이터 기록 |
| `need_plan` | `True` → 모션 플래너 실행 | `False` → 저장된 궤적 리플레이 |
| `save_data` | `False` | `True` → 매 `save_freq` 스텝마다 카메라 캡처 |
| **출력** | `seed.txt` + `_traj_data/*.pkl` | `data/*.hdf5` + `video/*.mp4` |

Phase 1에서는 모션 플래너를 호출하여 경로를 **생성**한다. 실패하는 시드가 많으므로 카메라 캡처 같은 비용이 큰 작업은 수행하지 않고, 성공하는 시드만 빠르게 걸러낸다.

```python
# Phase 1: 플래너가 경로를 새로 계산
left_result = self.robot.left_plan_path(pose, ...)
self.left_joint_path.append(deepcopy(left_result))  # 경로 저장
```

Phase 2에서는 Phase 1에서 찾은 시드와 궤적을 그대로 **재생**하면서, RGB/깊이/관절값 등을 캡처한다.

```python
# Phase 2: 저장된 경로를 순차적으로 꺼내 씀
left_result = deepcopy(self.left_joint_path[self.left_cnt])
self.left_cnt += 1
```

2단계로 분리한 이유는 모션 플래닝 성공률이 100%가 아니기 때문이다. 비용이 큰 렌더링/캡처를 성공이 확정된 에피소드에서만 수행하기 위한 것이다.

같은 시드 + 같은 궤적이면 SAPIEN은 deterministic이므로 결과가 동일하다. Phase 2 끝에서 `assert TASK_ENV.check_success(), "Collect Error"`로 이를 재검증한다.

---

## Phase 2: 관측 데이터 기록

Phase 1에서 찾은 시드를 재사용하여 동일 궤적을 리플레이하면서 관측 데이터를 수집한다.

### 동작 흐름

```python
# script/collect_data.py — run()
args["need_plan"] = False      # 재계획 비활성화
args["save_data"] = True       # 데이터 저장 활성화

for episode_idx in range(st_idx, args["episode_num"]):
    TASK_ENV.setup_demo(seed=seed_list[episode_idx], ...)
    traj_data = TASK_ENV.load_tran_data(episode_idx)    # 저장된 궤적 로드
    TASK_ENV.set_path_lst(args)                          # 리플레이 모드 설정
    TASK_ENV.play_once()                                 # 궤적 리플레이 + 관측 기록
    TASK_ENV.merge_pkl_to_hdf5_video()                   # PKL → HDF5 + MP4
    TASK_ENV.remove_data_cache()                         # 캐시 정리
```

### 데이터 캡처 (`_take_picture`)

`save_freq` 스텝(기본 15) 간격으로 `get_obs()`를 호출하여 다음 데이터를 PKL 캐시에 저장:

| 데이터 타입 | 설명 | YAML 키 |
|---|---|---|
| RGB 이미지 | 헤드 카메라 + 좌/우 손목 카메라 | `rgb: true` |
| Depth 맵 | 헤드 카메라 깊이 영상 | `depth: true` |
| 포인트클라우드 | FPS 다운샘플 (기본 1024점) | `pointcloud: true` |
| 관절 상태 (qpos) | 좌/우 팔 관절각 + 그리퍼 | `qpos: true` |
| 엔드이펙터 포즈 | 7D (xyz + 쿼터니언) + 그리퍼 값 | `endpose: true` |
| 메시 세그멘테이션 | 메시 단위 시맨틱 라벨 | `mesh_segmentation: true` |
| 액터 세그멘테이션 | 물체 단위 시맨틱 라벨 | `actor_segmentation: true` |

### HDF5 변환 (`merge_pkl_to_hdf5_video`)

에피소드별 PKL 프레임 캐시를 하나의 HDF5 파일과 시각화 MP4로 병합:

```
data/<task>/<config>/data/episode0.hdf5
data/<task>/<config>/video/episode0.mp4
```

#### HDF5 내부 구조

```
episodeN.hdf5
├── observation/
│   ├── head_camera/
│   │   ├── rgb           # JPEG 인코딩된 프레임 시퀀스
│   │   ├── depth         # (옵션) 깊이 맵
│   │   ├── mesh_seg      # (옵션) 메시 세그멘테이션
│   │   └── actor_seg     # (옵션) 액터 세그멘테이션
│   ├── left_camera/
│   │   └── rgb
│   └── right_camera/
│       └── rgb
├── joint_action/
│   ├── left_arm          # 좌측 팔 관절각 (6 or 7 DoF)
│   ├── left_gripper      # 좌측 그리퍼
│   ├── right_arm         # 우측 팔 관절각
│   ├── right_gripper     # 우측 그리퍼
│   └── vector            # 좌+우 연결 벡터
├── endpose/
│   ├── left_endpose      # 좌측 엔드이펙터 포즈 (7D)
│   ├── left_gripper      # 좌측 그리퍼 값
│   ├── right_endpose     # 우측 엔드이펙터 포즈 (7D)
│   └── right_gripper     # 우측 그리퍼 값
└── pointcloud            # (옵션) FPS 다운샘플 포인트클라우드
```

---

## Phase 3: 자연어 지시문 생성

수집된 데이터에 대해 다양한 자연어 지시문을 자동 생성한다.

### 3계층 구조

```
1. 태스크 템플릿       description/task_instruction/<task>.json
   └─ "Pick {A} and strike the block."  (플레이스홀더 포함)

2. 물체 설명           description/objects_description/<model>/base<id>.json
   └─ {"seen": ["red-handled hammer"], "unseen": ["steel mallet"]}

3. 에피소드 지시문      data/<task>/<config>/instructions/episodeN.json
   └─ {"seen": ["Pick the red-handled hammer and ..."], "unseen": [...]}
```

### 생성 과정

1. `scene_info.json`에서 에피소드별 파라미터 로드 (예: `{A}: "020_hammer/base0"`, `{a}: "left"`)
2. 태스크 템플릿의 플레이스홀더와 에피소드 파라미터 매칭
3. 물체 설명 JSON에서 자연어 설명을 랜덤 선택하여 치환
4. 에피소드당 `language_num`(기본 100)개의 지시문 생성
5. `seen`(학습용) / `unseen`(일반화 테스트용) 분리 저장

---

## 설정 파일 (YAML)

### 기본 설정 항목

| 항목 | 설명 | 기본값 |
|---|---|---|
| `episode_num` | 수집할 에피소드 수 | 50 |
| `save_freq` | 관측 캡처 간격 (시뮬 스텝) | 15 |
| `embodiment` | 로봇 유형 | `[aloha-agilex]` |
| `language_num` | 에피소드당 지시문 수 | 100 |
| `use_seed` | 기존 시드 재사용 여부 | false |
| `collect_data` | Phase 2 실행 여부 | true |
| `save_path` | 출력 경로 | `./data` |
| `clear_cache_freq` | SAPIEN 캐시 정리 주기 | 5 |

### 도메인 랜덤화 설정

| 항목 | 설명 | randomized | clean |
|---|---|---|---|
| `cluttered_table` | 방해 물체 10개 랜덤 배치 | true | false |
| `random_background` | 테이블/벽 텍스처 랜덤화 | true | false |
| `clean_background_rate` | 깨끗한 배경 확률 | 0.02 | 1 |
| `random_light` | 조명 색상/방향 랜덤화 | true | false |
| `crazy_random_light_rate` | 프레임마다 조명 변동 확률 | 0.02 | 0 |
| `random_table_height` | 테이블 높이 변동 범위 (m) | 0.03 | 0 |
| `random_head_camera_dis` | 카메라 위치 지터 | 0 | 0 |

### 제공되는 설정 프리셋

| 파일 | 용도 |
|---|---|
| `task_config/demo_randomized.yml` | 전체 도메인 랜덤화 적용 (학습용) |
| `task_config/demo_clean.yml` | 랜덤화 없음 (디버깅/비교용) |
| `task_config/demo_test.yml` | 빠른 테스트 (소수 에피소드) |

---

## 태스크 정의

각 태스크는 `envs/<task_name>.py`에 `Base_Task`를 상속하는 클래스로 정의된다. 현재 **50개 태스크**가 구현되어 있다.

### 태스크 클래스 구조

```python
class beat_block_hammer(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)       # 씬/로봇/카메라 초기화

    def load_actors(self):                      # 물체 배치
        self.hammer = create_actor(scene=self, modelname="020_hammer", ...)
        self.block = create_box(scene=self, pose=rand_pose(...), ...)

    def play_once(self):                        # 스크립트 컨트롤러
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
        self.move(self.grasp_actor(self.hammer, arm_tag=arm_tag, ...))
        self.move(self.move_by_displacement(arm_tag, z=0.07, ...))
        self.move(self.place_actor(self.hammer, target_pose=..., ...))
        self.info["info"] = {"{A}": "020_hammer/base0", "{a}": str(arm_tag)}
        return self.info

    def check_success(self):                    # 성공 조건
        return np.all(abs(hammer_pos[:2] - block_pos[:2]) < eps) \
               and self.check_actors_contact(...)
```

### 주요 조작 프리미티브 (Base_Task)

| API | 설명 |
|---|---|
| `grasp_actor()` | 물체 파지 (접근 → 그리퍼 닫기) |
| `place_actor()` | 물체 배치 (이동 → 그리퍼 열기) |
| `move_by_displacement()` | 엔드이펙터 상대 변위 이동 |
| `left_move_to_pose()` / `right_move_to_pose()` | 목표 포즈로 이동 |
| `together_open_gripper()` / `together_close_gripper()` | 양쪽 그리퍼 동시 제어 |
| `move()` | 액션 시퀀스 실행 + 관측 기록 |

### 태스크 목록 (50개)

```
adjust_bottle          grab_roller             place_container_plate
beat_block_hammer      handover_block          place_dual_shoes
blocks_ranking_rgb     handover_mic            place_empty_cup
blocks_ranking_size    hanging_mug             place_fan
click_alarmclock       lift_pot                place_mouse_pad
click_bell             move_can_pot            place_object_basket
dump_bin_bigbin        move_pillbottle_pad     place_object_scale
open_laptop            move_playingcard_away   place_object_stand
open_microwave         move_stapler_pad        place_phone_stand
pick_diverse_bottles   place_a2b_left          place_shoe
pick_dual_bottles      place_a2b_right         press_stapler
put_bottles_dustbin    place_bread_basket      rotate_qrcode
put_object_cabinet     place_bread_skillet     scan_object
shake_bottle           place_burger_fries      shake_bottle_horizontally
stack_blocks_three     place_can_basket        stamp_seal
stack_blocks_two       place_cans_plasticbox   turn_switch
stack_bowls_three      stack_bowls_two
```

---

## 로봇 및 모션 플래닝

### 지원 로봇 (Embodiment)

| 이름 | 경로 |
|---|---|
| aloha-agilex (기본) | `assets/embodiments/aloha-agilex/` |
| piper | `assets/embodiments/piper/` |
| franka-panda | `assets/embodiments/franka-panda/` |
| ARX-X5 | `assets/embodiments/ARX-X5/` |
| ur5-wsg | `assets/embodiments/ur5-wsg/` |

### 모션 플래너

| 플래너 | 구현 | 특징 |
|---|---|---|
| **mplib RRT** | `envs/robot/planner.py` → `MplibPlanner` | 기본 플래너, `SapienPlanner` 통해 SAPIEN 연동 |
| **CuRobo** | `envs/robot/planner.py` → `CuroboPlanner` | NVIDIA GPU 가속 플래너 (옵션) |

두 플래너 모두 **TOPP-RA** (`toppra`)를 사용하여 시간 최적 궤적 보간을 수행한다.

### 카메라 설정

| 카메라 모델 | FOV | 해상도 |
|---|---|---|
| D435 (기본) | 37° | 320×240 |
| Large_D435 | 37° | 640×480 |
| L515 | 45° | 320×180 |
| Large_L515 | 45° | 640×360 |

카메라 3대 구성: **헤드 카메라** (고정) + **좌/우 손목 카메라** (로봇 팔에 부착)

---

## 출력 디렉토리 구조

```
data/<task_name>/<task_config>/
├── seed.txt                          # 성공 시드 목록
├── scene_info.json                   # 에피소드별 메타데이터
├── _traj_data/
│   ├── episode0.pkl                  # Phase 1 궤적 데이터
│   ├── episode1.pkl
│   └── ...
├── data/
│   ├── episode0.hdf5                 # Phase 2 관측 데이터
│   ├── episode1.hdf5
│   └── ...
├── video/
│   ├── episode0.mp4                  # 시각화 영상
│   ├── episode1.mp4
│   └── ...
└── instructions/
    ├── episode0.json                 # Phase 3 자연어 지시문
    ├── episode1.json
    └── ...
```

---

## 실행 예시

### 기본 데이터 수집 (도메인 랜덤화 적용)

```bash
bash collect_data.sh beat_block_hammer demo_randomized 0
```

### 클린 환경 데이터 수집

```bash
bash collect_data.sh beat_block_hammer demo_clean 0
```

### 커스텀 설정 생성 및 사용

```bash
bash task_config/create_task_config.sh my_config
# task_config/my_config.yml 편집 후:
bash collect_data.sh beat_block_hammer my_config 0
```

---

## 핵심 파일 참조

| 파일 | 역할 |
|---|---|
| `collect_data.sh` | 진입점 셸 스크립트 |
| `script/collect_data.py` | 2-Phase 데이터 수집 엔진 |
| `envs/_base_task.py` | 태스크 베이스 클래스 (씬/로봇/카메라/조작 API) |
| `envs/<task_name>.py` | 개별 태스크 정의 (50개) |
| `envs/robot/planner.py` | 모션 플래너 (mplib RRT + CuRobo) |
| `envs/camera/camera.py` | 카메라 센서 추상화 (RGBD + 포인트클라우드) |
| `envs/utils/pkl2hdf5.py` | PKL 캐시 → HDF5 + MP4 변환 |
| `task_config/*.yml` | 수집 설정 파일 |
| `description/utils/generate_episode_instructions.py` | 자연어 지시문 생성기 |
