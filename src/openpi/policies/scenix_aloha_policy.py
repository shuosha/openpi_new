import dataclasses

import einops
import numpy as np

from openpi import transforms


def make_scenix_aloha_example() -> dict:
    """Creates a random input example for the Scenix Aloha policy."""
    return {
        "state": np.random.rand(14).astype(np.float32),
        "images": {
            "cam_top": np.random.randint(256, size=(160, 360, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(160, 360, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(160, 360, 3), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ScenixAlohaInputs(transforms.DataTransformFn):
    """Inputs for a dual-arm Aloha policy with 3 cameras and 14D state/actions."""

    EXPECTED_CAMERAS: tuple[str, ...] = ("cam_top", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = {k: _parse_image(v) for k, v in data["images"].items()}

        base_image = in_images["cam_top"]

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        wrist_mapping = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in wrist_mapping.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "state": data["state"],
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ScenixAlohaOutputs(transforms.DataTransformFn):
    """Outputs for the Scenix Aloha policy. Returns first 14 action dims."""

    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :self.action_dim])}
