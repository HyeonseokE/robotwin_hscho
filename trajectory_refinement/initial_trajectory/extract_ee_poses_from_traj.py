import numpy as np


def extract_ee_poses(task_env, arm_tag, sample_interval=1):
    """
    task_env.ee_trajectory_buffer에서 웨이포인트 추출.

    Args:
        task_env: BaseTaskWithIKExecution 인스턴스
        arm_tag: "left" or "right"
        sample_interval: 다운샘플 간격 (1이면 모든 포즈 사용)

    Returns:
        list of [x, y, z, qw, qx, qy, qz]
    """
    buffer = task_env.get_recorded_ee_trajectory(arm_tag)

    if len(buffer) == 0:
        return []

    # 다운샘플
    if sample_interval > 1:
        sampled = buffer[::sample_interval]
        # 마지막 포즈가 포함되지 않았으면 추가
        if len(buffer) - 1 not in range(0, len(buffer), sample_interval):
            sampled.append(buffer[-1])
    else:
        sampled = list(buffer)

    # 각 포즈를 [x,y,z,qw,qx,qy,qz] 리스트로 변환
    waypoints = []
    for pose in sampled:
        if isinstance(pose, (list, tuple)):
            waypoints.append(list(pose))
        elif isinstance(pose, np.ndarray):
            waypoints.append(pose.tolist())
        else:
            # sapien.Pose 등의 경우
            waypoints.append(list(pose))

    return waypoints


def filter_stationary_waypoints(waypoints, pos_threshold=0.001, rot_threshold=0.01):
    """
    연속적으로 거의 동일한 웨이포인트를 제거하여 의미 있는 변화가 있는 웨이포인트만 남김.

    Args:
        waypoints: list of [x,y,z,qw,qx,qy,qz]
        pos_threshold: 위치 변화 임계값 (m)
        rot_threshold: 회전 변화 임계값 (quaternion distance)

    Returns:
        list of [x,y,z,qw,qx,qy,qz]
    """
    if len(waypoints) <= 2:
        return waypoints

    filtered = [waypoints[0]]

    for i in range(1, len(waypoints)):
        prev = np.array(waypoints[i - 1])
        curr = np.array(waypoints[i])

        pos_diff = np.linalg.norm(curr[:3] - prev[:3])
        rot_diff = np.linalg.norm(curr[3:] - prev[3:])

        if pos_diff > pos_threshold or rot_diff > rot_threshold:
            filtered.append(waypoints[i])

    # 마지막 웨이포인트는 항상 포함
    if filtered[-1] != waypoints[-1]:
        filtered.append(waypoints[-1])

    return filtered
