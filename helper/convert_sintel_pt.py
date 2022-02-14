# Standard
from pathlib import Path

# PIP
import cv2
import numpy as np
import torch
from tqdm import tqdm


def read_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float32)
    img /= 255.0
    return img


def read_flo(file_path):
    with open(file_path, "rb") as flo_file:
        magic = np.fromfile(flo_file, np.float32, count=1)
        if magic != 202021.25:
            raise ValueError(f"Incorrect magic number in {file_path}")
        else:
            width = np.fromfile(flo_file, np.int32, count=1)[0]
            height = np.fromfile(flo_file, np.int32, count=1)[0]
            flow = np.fromfile(flo_file, np.float32, count=2 * width * height)
            flow = flow.reshape((height, width, 2))
            flow = flow.transpose(2, 0, 1)
            return flow


def convert(converted_dir, image_path_list):
    count = 0
    for index in tqdm(range(len((image_path_list[:-1])))):
        flow_parts = list(image_path_list[index].parts)
        flow_parts[-3] = "flow"
        flow_parts[-1] = flow_parts[-1].replace("png", "flo")
        flow_path = Path(*(flow_parts))
        if not flow_path.is_file():
            continue

        front_image = read_image(image_path_list[index])
        back_image = read_image(image_path_list[index + 1])
        flow = read_flo(flow_path)
        array = np.concatenate((front_image, back_image, flow), 0)
        tensor = torch.from_numpy(array)
        torch.save(tensor, converted_dir / f"{count}.pt")
        count += 1


def convert_to_pt():
    project_dir = Path(__file__).absolute().parent.parent
    data_dir = project_dir / "data"
    original_dir = data_dir / "sintel" / "training"
    converted_dir = data_dir / "sintel_pt"

    state = "clean"  # clean, final

    image_dir = original_dir / state

    path_list = image_dir.glob("**/*")
    image_path_list = [path for path in path_list if path.is_file()]
    image_path_list.sort()

    convert(converted_dir / "train", image_path_list[:-50])  # length: 1014
    convert(converted_dir / "valid", image_path_list[-50:])  # length: 50


if __name__ == "__main__":
    convert_to_pt()
