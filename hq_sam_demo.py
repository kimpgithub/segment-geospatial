import os
import cv2
import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
from segment_anything_hq import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
    SamPredictor,
)

def download_checkpoint(model_type="vit_h", checkpoint_dir=None, hq=False):
    """Download the SAM model checkpoint.

    Args:
        model_type (str, optional): The model type. Can be one of ['vit_h', 'vit_l', 'vit_b'].
            Defaults to 'vit_h'. See https://bit.ly/3VrpxUh for more details.
        checkpoint_dir (str, optional): The checkpoint_dir directory. Defaults to None, "~/.cache/torch/hub/checkpoints".
        hq (bool, optional): Whether to use HQ-SAM model (https://github.com/SysCV/sam-hq). Defaults to False.
    """

    if not hq:
        model_types = {
            "vit_h": {
                "name": "sam_vit_h_4b8939.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            },
            "vit_l": {
                "name": "sam_vit_l_0b3195.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            },
            "vit_b": {
                "name": "sam_vit_b_01ec64.pth",
                "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            },
        }
    else:
        model_types = {
            "vit_h": {
                "name": "sam_hq_vit_h.pth",
                "url": [
                    "https://github.com/opengeos/datasets/releases/download/models/sam_hq_vit_h.zip",
                    "https://github.com/opengeos/datasets/releases/download/models/sam_hq_vit_h.z01",
                ],
            },
            "vit_l": {
                "name": "sam_hq_vit_l.pth",
                "url": "https://github.com/opengeos/datasets/releases/download/models/sam_hq_vit_l.pth",
            },
            "vit_b": {
                "name": "sam_hq_vit_b.pth",
                "url": "https://github.com/opengeos/datasets/releases/download/models/sam_hq_vit_b.pth",
            },
            "vit_tiny": {
                "name": "sam_hq_vit_tiny.pth",
                "url": "https://github.com/opengeos/datasets/releases/download/models/sam_hq_vit_tiny.pth",
            },
        }

    if model_type not in model_types:
        raise ValueError(
            f"Invalid model_type: {model_type}. It must be one of {', '.join(model_types)}"
        )

    if checkpoint_dir is None:
        checkpoint_dir = os.environ.get(
            "TORCH_HOME", os.path.expanduser("~/.cache/torch/hub/checkpoints")
        )

    checkpoint = os.path.join(checkpoint_dir, model_types[model_type]["name"])
    if not os.path.exists(checkpoint):
        print(f"Model checkpoint for {model_type} not found.")
        url = model_types[model_type]["url"]
        if isinstance(url, str):
            download_file(url, checkpoint)
        elif isinstance(url, list):
            download_files(url, checkpoint_dir, multi_part=True)
    return checkpoint


class SamGeo:
    def __init__(
        self,
        model_type="vit_h",
        automatic=True,
        device=None,
        checkpoint_dir=None,
        hq=False,
        sam_kwargs=None,
        **kwargs,
    ):
        hq = True  # HQ-SAM 사용
        checkpoint = download_checkpoint(model_type, checkpoint_dir, hq)

        # cuda 사용 여부 자동 설정
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                torch.cuda.empty_cache()

        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = device
        self.sam_kwargs = sam_kwargs if sam_kwargs is not None else {}

        # 모델 불러오기
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.sam.to(device=self.device)

        if automatic:
            self.mask_generator = SamAutomaticMaskGenerator(self.sam, **self.sam_kwargs)
        else:
            self.predictor = SamPredictor(self.sam, **self.sam_kwargs)

    def generate(
        self,
        source,
        output=None,
        foreground=True,
        erosion_kernel=(3, 3),
        mask_multiplier=255,
        unique=True,
        **kwargs,
    ):
        # 이미지 로드
        if isinstance(source, str):
            image = cv2.imread(source)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(source, np.ndarray):
            image = source
        else:
            raise ValueError("Input source must be a path or a numpy array.")
        
        self.image = image
        masks = self.mask_generator.generate(image)
        self.masks = masks

        if output is not None:
            self.save_masks(output, foreground, unique, erosion_kernel, mask_multiplier, **kwargs)

    def save_masks(
        self,
        output=None,
        foreground=True,
        unique=True,
        erosion_kernel=None,
        mask_multiplier=255,
        **kwargs,
    ):
        h, w, _ = self.image.shape
        masks = self.masks

        dtype = np.uint8 if len(masks) < 255 else np.uint16
        if unique:
            sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=False)
            objects = np.zeros((sorted_masks[0]["segmentation"].shape[0], sorted_masks[0]["segmentation"].shape[1]))
            for index, ann in enumerate(sorted_masks):
                m = ann["segmentation"]
                objects[m] = index + 1
        else:
            resulting_mask = np.zeros((h, w), dtype=dtype) if foreground else np.ones((h, w), dtype=dtype)
            resulting_borders = np.zeros((h, w), dtype=dtype)

            for m in masks:
                mask = (m["segmentation"] > 0).astype(dtype)
                resulting_mask += mask

                if erosion_kernel is not None:
                    mask_erode = cv2.erode(mask, erosion_kernel, iterations=1)
                    edge_mask = mask - mask_erode
                    resulting_borders += edge_mask

            resulting_mask = (resulting_mask > 0).astype(dtype)
            objects = resulting_mask - resulting_borders
            objects = objects * mask_multiplier

        self.objects = objects.astype(dtype)
        
        # 파일 저장 및 오류 처리
        success = cv2.imwrite(output, self.objects)
        if not success:
            raise IOError(f"Error: Unable to save the mask image to {output}")

    def show_masks(self, figsize=(12, 10), cmap="binary_r", axis="off"):
        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap)
        plt.axis(axis)
        plt.show()


# 사용 예시
if __name__ == "__main__":
    # 모델 설정
    sam_geo = SamGeo(
        model_type="vit_h",
        automatic=True,
        hq=True,
        checkpoint_dir="./checkpoints"
    )

    # 로컬 tiff 파일 경로
    tiff_path = "/path/to/your/image.tiff"

    # 출력 경로
    output_mask_path = "/path/to/output_mask.png"

    # 마스크 생성
    sam_geo.generate(
        source=tiff_path,
        output=output_mask_path,
        foreground=True,
        unique=True,
        mask_multiplier=255,
        erosion_kernel=(3, 3)
    )
