import h5py, pickle
import numpy as np
import os
import cv2
from collections.abc import Mapping, Sequence
import shutil
from .images_to_video import images_to_video


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def parse_dict_structure(data):
    if isinstance(data, dict):
        parsed = {}
        for key, value in data.items():
            if isinstance(value, dict):
                parsed[key] = parse_dict_structure(value)
            elif isinstance(value, np.ndarray):
                parsed[key] = []
            else:
                parsed[key] = []
        return parsed
    else:
        return []


def append_data_to_structure(data_structure, data):
    for key in data_structure:
        if key in data:
            if isinstance(data_structure[key], list):
                # 如果是叶子节点，直接追加数据
                data_structure[key].append(data[key])
            elif isinstance(data_structure[key], dict):
                # 如果是嵌套字典，递归处理
                append_data_to_structure(data_structure[key], data[key])


def load_pkl_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_hdf5_from_dict(hdf5_group, data_dict):
    for key, value in data_dict.items():
        if isinstance(value, dict):
            subgroup = hdf5_group.create_group(key)
            create_hdf5_from_dict(subgroup, value)
        elif isinstance(value, list):
            value = np.array(value)
            if "rgb" in key:
                encode_data, max_len = images_encoding(value)
                hdf5_group.create_dataset(key, data=encode_data, dtype=f"S{max_len}")
            else:
                hdf5_group.create_dataset(key, data=value)
        else:
            return
            try:
                hdf5_group.create_dataset(key, data=str(value))
                print("Not np array")
            except Exception as e:
                print(f"Error storing value for key '{key}': {e}")


def concat_camera_views(head_imgs, wrist_imgs_list):
    """Concatenate head_camera and wrist cameras side by side.
    wrist_imgs_list is a list of image arrays (e.g. [left_wrist, right_wrist]).
    Resizes wrist images to match head image height if needed."""
    n = len(head_imgs)
    concat_frames = []
    for i in range(n):
        parts = [head_imgs[i]]
        h_h = head_imgs[i].shape[0]
        for wrist_imgs in wrist_imgs_list:
            w_img = wrist_imgs[i]
            w_h, w_w = w_img.shape[:2]
            if w_h != h_h:
                scale = h_h / w_h
                new_w = int(w_w * scale)
                w_img = cv2.resize(w_img, (new_w, h_h))
            parts.append(w_img)
        concat_frames.append(np.concatenate(parts, axis=1))
    return np.array(concat_frames)


def pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path):
    data_list = parse_dict_structure(load_pkl_file(pkl_files[0]))
    for pkl_file_path in pkl_files:
        pkl_file = load_pkl_file(pkl_file_path)
        append_data_to_structure(data_list, pkl_file)

    head_imgs = np.array(data_list["observation"]["head_camera"]["rgb"])
    images_to_video(head_imgs, out_path=video_path)

    # Save concatenated head + wrist camera(s) video
    obs = data_list.get("observation", {})
    wrist_imgs_list = []
    if "left_camera" in obs and obs["left_camera"].get("rgb"):
        wrist_imgs_list.append(np.array(obs["left_camera"]["rgb"]))
    if "right_camera" in obs and obs["right_camera"].get("rgb"):
        wrist_imgs_list.append(np.array(obs["right_camera"]["rgb"]))
    if wrist_imgs_list:
        concat_imgs = concat_camera_views(head_imgs, wrist_imgs_list)
        concat_video_path = video_path.replace(".mp4", "_concat.mp4")
        images_to_video(concat_imgs, out_path=concat_video_path)

    # Save observer camera (third_view) video
    if "third_view_rgb" in data_list and data_list["third_view_rgb"]:
        observer_imgs = np.array(data_list["third_view_rgb"])
        observer_video_path = video_path.replace(".mp4", "_observer.mp4")
        images_to_video(observer_imgs, out_path=observer_video_path)

    with h5py.File(hdf5_path, "w") as f:
        create_hdf5_from_dict(f, data_list)


def process_folder_to_hdf5_video(folder_path, hdf5_path, video_path):
    pkl_files = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl") and fname[:-4].isdigit():
            pkl_files.append((int(fname[:-4]), os.path.join(folder_path, fname)))

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found in {folder_path}")

    pkl_files.sort()
    pkl_files = [f[1] for f in pkl_files]

    expected = 0
    for f in pkl_files:
        num = int(os.path.basename(f)[:-4])
        if num != expected:
            raise ValueError(f"Missing file {expected}.pkl")
        expected += 1

    pkl_files_to_hdf5_and_video(pkl_files, hdf5_path, video_path)
