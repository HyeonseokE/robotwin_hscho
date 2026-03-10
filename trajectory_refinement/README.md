# VLM 반복 궤적 최적화 파이프라인

기존 RoboTwin 파이프라인으로 생성한 궤적을 초기값으로 하여,
**VLM 관찰 → LLM 웨이포인트 수정 → IK only 재실행**의 반복 루프를 통해
태스크에 더 최적화된 궤적을 생성합니다.

## 파이프라인 구조

```
[Phase 0] 기존 파이프라인으로 초기 궤적 생성
    play_once() → IK+RRT planner → take_dense_action
    → 실행 중 EE 포즈를 Cartesian 웨이포인트로 추출
    → initial_waypoints.json 저장

[Phase 1+] VLM 반복 개선 루프
    ┌─→ 웨이포인트 → IK 연쇄 → TOPP 보간 → 시뮬 실행
    │   → 키프레임 캡처 + check_success
    │
    │   [Kimi VLM] 키프레임 이미지 분석 → 실행 문제점 피드백
    │   [DeepSeek LLM] VLM 피드백 + 현재 웨이포인트 → 수정된 JSON
    └─────────────────────────────────────────────────────┘

    → 최적 웨이포인트 JSON 저장
```

## 사용법

```bash
# 기본 실행
python trajectory_refinement/run_refinement.py shake_bottle

# 옵션 지정
python trajectory_refinement/run_refinement.py shake_bottle \
    --max-iterations 3 \
    --check-num 3 \
    --success-threshold 0.6

# 커스텀 설정 파일 사용
python trajectory_refinement/run_refinement.py shake_bottle \
    --config trajectory_refinement/configs/refinement_config.yml
```

## 폴더 구조

```
trajectory_refinement/
├── run_refinement.py                    # 메인 진입점
├── initial_trajectory/
│   ├── run_existing_pipeline.py         # 기존 play_once() 실행
│   └── extract_ee_poses_from_traj.py    # EE 포즈 추출
├── ee_waypoint_to_joint_trajectory/
│   ├── solve_ik_chain.py                # IK 연쇄 풀이
│   ├── smooth_with_topp.py              # TOPP 시간 최적 보간
│   └── replay_trajectory_in_sim.py      # 궤적 재실행
├── vlm_analysis/
│   ├── capture_keyframes.py             # 키프레임 캡처
│   └── analyze_execution_with_vlm.py    # VLM 분석
├── llm_waypoint_editing/
│   └── refine_waypoints_with_llm.py     # LLM 웨이포인트 수정
├── task_env_wrapper/
│   └── base_task_with_ik_execution.py   # Base_Task 확장
├── prompts/
│   ├── vlm_analysis_prompt.py           # VLM 프롬프트
│   └── llm_waypoint_edit_prompt.py      # LLM 프롬프트
├── configs/
│   └── refinement_config.yml            # 설정 파일
└── output/                              # 실행 결과
```

## 출력 파일

실행 후 `trajectory_refinement/output/<task_name>/` 디렉토리에:
- `initial_waypoints.json` - 초기 궤적 웨이포인트
- `refined_waypoints_iter{N}.json` - 각 반복의 수정된 웨이포인트
- `best_waypoints.json` - 최고 성공률의 웨이포인트

## 기존 코드 의존성 (import만, 수정 없음)

| 모듈 | 사용 방식 |
|---|---|
| `envs._base_task.Base_Task` | 상속 |
| `envs.<task_name>` | 다중 상속 |
| `envs.robot.Robot` | IK/TOPP 접근 |
| `code_gen.gpt_agent.generate` | LLM 호출 |
| `code_gen.observation_agent.observe_task_execution` | VLM 분석 |

## 설정

`configs/refinement_config.yml`에서 파라미터 조정:
- `max_iterations`: 최대 반복 횟수 (기본 5)
- `check_num`: 평가 에피소드 수 (기본 5)
- `success_threshold`: 목표 성공률 (기본 0.5)
- `record_ee_freq`: EE 기록 빈도 (기본 15)
- `topp_step`: TOPP 보간 스텝 (기본 1/250)
