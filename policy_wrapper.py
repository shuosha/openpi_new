"""Unified policy wrapper for pi05 and gr00t inference.

Provides a simple ``act(obs) -> (1, action_dim)`` interface with internal action
chunk queuing.  Only runs inference when the queue is empty; otherwise pops the
next action from the last chunk.

RTC terminology (matches rtc_readme.md §2):
    H = action_horizon — total chunk length the model emits per inference.
    s = ``execution_horizon`` — # of actions consumed per cycle before the
        wrapper kicks off the next inference (i.e. queue size per chunk).
    d = ``rtc_delay`` — training-time inference-delay parameter; the first
        ``d`` slots of each new chunk are clamped to the previous chunk's
        slots ``[s, s+d)`` to enforce inter-chunk continuity.
    Constraint: ``d + s <= H`` (the chunk must contain a full prefix plus
    the s actions to be executed). Validated on first inference.

The first inference of an episode (after construction or ``reset()``) has no
previous chunk to clamp against, so the wrapper falls through to vanilla
sampling (``inference_delay=0``). RTC kicks in from the second inference
onward, when ``_last_chunk_raw`` has been populated.

Usage (pi05, non-RTC)::

    wrapper = PolicyWrapper(
        checkpoint="checkpoints/pi05_aloha_cube_handover/...",
        policy_type="pi05",
        config_name="pi05_aloha_cube_handover",
    )
    action = wrapper.act(obs)  # (1, action_dim)
    wrapper.reset()            # clear queue + cached prev chunk on episode boundary

Usage (pi05, RTC-trained checkpoint)::

    wrapper = PolicyWrapper(
        checkpoint="...",
        policy_type="pi05",
        config_name="pi05_aloha_cube_handover_rtc",
        rtc_delay=8,            # d (set to the value the checkpoint trained against)
        execution_horizon=10,   # s; must satisfy s + d <= H
    )

Usage (gr00t)::

    from gr00t.data.embodiment_tags import EmbodimentTag

    wrapper = PolicyWrapper(
        checkpoint="/path/to/gr00t/checkpoint",
        policy_type="gr00t",
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )
    action = wrapper.act(obs)  # (1, action_dim)

Single-chunk test against a LeRobot dataset (pi05, non-RTC)::
    python policy_wrapper.py \
      --checkpoint shashuo0104/260413_pi05_aloha_cubehandover_v2:20000 \
      --policy_type pi05 \
      --config_name pi05_aloha_cube_handover \
      --dataset shashuo0104/260413_aloha_cube_handover_v2 \
      --prompt cube_handover --seed 42

Stitched-RTC test (pi05, RTC checkpoint) — runs ``num_cycles`` consecutive
inferences advancing obs by ``s`` each cycle and reports stitched MSE/MAE
plus the prefix→postfix boundary continuity (slot ``d-1`` → slot ``d``)
inside each RTC-clamped chunk vs the GT step at the same controller time::
    python policy_wrapper.py \
      --checkpoint <rtc-checkpoint> \
      --policy_type pi05 \
      --config_name pi05_aloha_cube_handover_rtc \
      --dataset shashuo0104/260413_aloha_cube_handover_v2 \
      --prompt cube_handover \
      --rtc_delay 8 --execution_horizon 10 --num_cycles 5 \
      --plot --save_plot /tmp/rtc.png --seed 42

Single-chunk test against a LeRobot dataset (gr00t)::
    python policy_wrapper.py \
      --checkpoint shashuo0104/260413_gr00t_aloha_cubehandover_v1 \
      --policy_type gr00t \
      --dataset shashuo0104/260413_aloha_cube_handover_v2 \
      --seed 42
"""

import argparse
import collections
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# LeRobot observation format — contract for act() / infer_chunk() input
# ---------------------------------------------------------------------------
# observation.images.{top, left_wrist, right_wrist}
#     uint8, (160, 320, 3), HWC, RGB, not normalized
# observation.state
#     float32, (14,)
#     [left_qpos(6), left_gripper(1), right_qpos(6), right_gripper(1)]
# action (ground-truth, same layout as state)
#     float32, (14,)
# prompt
#     str
#
# NOTE(shuo): LeRobotDataset returns images as CHW float32 [0, 1] internally.
# build_obs_from_lerobot_sample() converts back to HWC uint8.
# aloha_env._transform_obs() produces HWC uint8 RGB directly (BGR -> RGB).
#
# ---------------------------------------------------------------------------
# Per-policy input expectations (converted internally by PolicyWrapper)
# ---------------------------------------------------------------------------
# pi05:
#     images: dict[str, ndarray] — CHW float32 [0, 1]
#         keys: cam_top, cam_left_wrist, cam_right_wrist
#     state:  float32 (14,)
#     prompt: str
#
# gr00t:
#     video:    dict[key, ndarray] — (B=1, T=1, H, W, C) uint8 HWC RGB
#     state:    dict[limb_key, ndarray] — (B=1, T=1, D) float32, split per limb
#     language: dict[key, list[list[str]]] — [[prompt]]
# ---------------------------------------------------------------------------

# LeRobot image keys shared across the codebase.
_LEROBOT_IMAGE_KEYS = [
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]

# Mapping from LeRobot flat image keys to the camera names pi05 expects.
_PI05_IMAGE_KEY_MAP: dict[str, str] = {
    "observation.images.top": "cam_top",
    "observation.images.left_wrist": "cam_left_wrist",
    "observation.images.right_wrist": "cam_right_wrist",
}

# Mapping from gr00t state modality keys to slices of the 14D state vector.
_GR00T_STATE_SLICES: dict[str, slice] = {
    "left_arm": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_arm": slice(7, 13),
    "right_gripper": slice(13, 14),
}


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint path.  Supports local paths and HuggingFace repo IDs.

    A HuggingFace repo ID looks like ``user/repo`` or ``user/repo:subfolder``.
    """
    if Path(checkpoint).exists():
        return checkpoint

    parts = checkpoint.split(":")
    repo_id = parts[0]
    subfolder = parts[1] if len(parts) > 1 else None

    if "/" in repo_id and "://" not in checkpoint:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id)
        if subfolder:
            local_dir = str(Path(local_dir) / subfolder)
        return local_dir

    return checkpoint


class PolicyWrapper:
    """Unified inference wrapper for pi05 and gr00t policies.

    Accepts observations in LeRobot format (see module-level comment) and
    converts internally to whatever each policy expects.
    """

    def __init__(
        self,
        checkpoint: str,
        policy_type: Literal["pi05", "gr00t"],
        *,
        rtc_delay: int = 0,
        execution_horizon: int | None = None,
        warmup: bool = True,
        **kwargs: Any,
    ) -> None:
        """Args:
            rtc_delay: ``d`` in rtc_readme.md. Number of slots clamped to the
                previous chunk at the start of each new inference. Set to the
                value the checkpoint was trained with (``Pi0Config.rtc_max_delay``
                range — typically the upper end). 0 disables RTC entirely.
            execution_horizon: ``s`` in rtc_readme.md. Number of actions consumed
                per inference cycle. None = full chunk (``H``). Must satisfy
                ``execution_horizon + rtc_delay <= H``.
        """
        self._policy_type = policy_type
        self._action_queue: collections.deque[np.ndarray] = collections.deque()
        self._rtc_delay = rtc_delay
        # ``None`` means "use full action_chunk_size" — resolved on first inference.
        self._execution_horizon: int | None = execution_horizon
        # Most-recent full chunk (length H, raw/un-normalized joint space) used to
        # build the next chunk's RTC prefix. None until the first inference completes.
        self._last_chunk_raw: np.ndarray | None = None
        checkpoint = _resolve_checkpoint(checkpoint)

        if policy_type == "pi05":
            self._init_pi05(checkpoint, **kwargs)
        elif policy_type == "gr00t":
            self._init_gr00t(checkpoint, rtc_delay=rtc_delay, **kwargs)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type!r}")

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run a dummy inference so the JAX jit traces now.

        openpi traces ``sample_actions`` per input PyTree structure — the
        presence/absence of ``prev_action_chunk`` changes that structure, and
        each distinct ``inference_delay`` value compiles a separate variant.
        For RTC, we warm both the no-prefix path (first inference of an
        episode) and the with-prefix path so steady-state inference is fast.
        """
        import time as _time
        t0 = _time.time()
        dummy_obs = self._build_dummy_obs()
        # First-inference path: no prev chunk.
        self.infer_chunk(dummy_obs)
        if self._rtc_delay > 0:
            # Subsequent-inference path: prev chunk is now cached, so the next
            # call automatically takes the prefix-clamped path.
            self.infer_chunk(dummy_obs)
        print(f"[PolicyWrapper] warmup done in {(_time.time() - t0) * 1000:.0f} ms")
        self.reset()

    def _action_prefix_shape(self) -> tuple[int, int]:
        """Return ``(action_horizon, action_dim)`` expected by the underlying model."""
        if self._policy_type == "pi05":
            return (self._policy._model.action_horizon, self._policy._model.action_dim)
        model_cfg = self._policy.model.config
        return (model_cfg.action_horizon, model_cfg.action_dim)

    @property
    def rtc_delay(self) -> int:
        return self._rtc_delay

    def _build_dummy_obs(self) -> dict:
        """Synthesize a zero-valued LeRobot-format obs for warmup."""
        dummy: dict[str, Any] = {
            "observation.state": np.zeros(14, dtype=np.float32),
            "prompt": "",
        }
        for key in _LEROBOT_IMAGE_KEYS:
            dummy[key] = np.zeros((160, 320, 3), dtype=np.uint8)
        return dummy

    def _init_pi05(self, checkpoint: str, *, config_name: str, **kwargs: Any) -> None:
        from openpi.policies import policy_config as _policy_config
        from openpi.shared import download
        from openpi.training import config as _config

        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(checkpoint)
        self._policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        self._pi05_image_key_map = kwargs.get("image_key_map", _PI05_IMAGE_KEY_MAP)
        # Cache a single-key Normalize transform for the action prefix. The new openpi
        # API takes ``prev_action_chunk`` as a direct kwarg to ``Policy.infer``/``sample_actions``
        # which bypasses the input-transform pipeline, so we apply Normalize manually here.
        self._action_prefix_normalize = self._build_action_prefix_normalize()

    def _build_action_prefix_normalize(self):
        """Locate the Normalize step in the policy's input transform and return a
        single-key Normalize that applies just the actions stats.

        Returns ``None`` if no actions stats are present (no normalization needed).
        """
        from openpi import transforms as _tr
        comp = self._policy._input_transform
        if not isinstance(comp, _tr.CompositeTransform):
            return None
        for t in comp.transforms:
            if isinstance(t, _tr.Normalize) and t.norm_stats is not None and "actions" in t.norm_stats:
                return _tr.Normalize(
                    {"actions": t.norm_stats["actions"]},
                    use_quantiles=t.use_quantiles,
                    strict=False,
                )
        return None

    def _normalize_action_prefix(self, prefix_raw: np.ndarray) -> np.ndarray:
        """Map a raw-joint-space prefix into the model's normalized action space.

        ``prefix_raw`` shape: ``(d, action_dim)``. Returns the same shape.
        """
        if self._action_prefix_normalize is None:
            return prefix_raw.astype(np.float32)
        out = self._action_prefix_normalize({"actions": prefix_raw.astype(np.float32)})
        return np.asarray(out["actions"], dtype=np.float32)

    def _init_gr00t(
        self,
        checkpoint: str,
        *,
        embodiment_tag: Any,
        device: str = "cuda",
        rtc_delay: int = 0,
        **kwargs: Any,
    ) -> None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        gr00t_kwargs: dict[str, Any] = {}
        if rtc_delay > 0:
            gr00t_kwargs["rtc_delay"] = rtc_delay
        self._policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=checkpoint,
            device=device,
            **gr00t_kwargs,
        )
        self._modality_configs = self._policy.get_modality_config()
        self._action_keys = self._modality_configs["action"].modality_keys

    def act(self, obs: dict) -> np.ndarray:
        """Return the next action as shape ``(1, action_dim)``.

        Runs inference (via ``infer_chunk``) only when the internal queue is
        empty; otherwise pops the next action from the previously predicted
        chunk. The queue holds ``s = execution_horizon`` actions per cycle:
        for an RTC-trained checkpoint the first ``d`` of those are the
        clamped prefix (continuous with the previous chunk), the rest are
        fresh postfix predictions.
        """
        if len(self._action_queue) == 0:
            action_chunk = self.infer_chunk(obs)
            for step_idx in range(action_chunk.shape[0]):
                self._action_queue.append(action_chunk[step_idx])
        return self._action_queue.popleft()[np.newaxis, :]

    def reset(self) -> None:
        """Clear the action queue and the cached previous chunk.

        Call at episode boundaries. After reset, the next ``infer_chunk``
        call runs without an RTC prefix (vanilla sampling), since there is
        no previous chunk to clamp against.
        """
        self._action_queue.clear()
        self._last_chunk_raw = None
        # gr00t manages its own prefix internally; reset it too.
        if self._policy_type == "gr00t" and hasattr(self._policy, "reset"):
            self._policy.reset()

    def infer_chunk(
        self, obs: dict, noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run a single inference and return ``execution_horizon`` actions.

        For RTC-enabled wrappers (``rtc_delay > 0``), the wrapper internally
        constructs the action prefix from the previously cached chunk:
        ``prev_chunk[s : s+d]`` becomes the new chunk's slots ``[0, d)``.
        On the very first call (or after ``reset()``), there is no previous
        chunk so vanilla (no-prefix) sampling is used.

        Args:
            obs: LeRobot-format observation dict.
            noise: Optional ``(H, action_dim)`` noise array. Passed through
                to the underlying policy.

        Returns shape ``(execution_horizon, action_dim)``. The full ``H``
        chunk is also cached internally for the next call's prefix.
        Does **not** affect the action queue.
        """
        if self._policy_type == "pi05":
            raw = self._infer_pi05(obs, noise=noise)
        else:
            raw = self._infer_gr00t(obs, noise=noise)
        H = raw.shape[0]
        if self._execution_horizon is None:
            # Default s = H - d for RTC (max allowed) so the prefix slice [s, s+d) is in range;
            # default s = H for non-RTC (consume the whole chunk).
            self._execution_horizon = H - self._rtc_delay if self._rtc_delay > 0 else H
        self._validate_horizons(H=H)
        return raw[: self._execution_horizon]

    def _validate_horizons(self, *, H: int) -> None:
        """Ensure d + s <= H once the chunk size is known (rtc_readme.md §2)."""
        s = self._execution_horizon
        d = self._rtc_delay
        if s is None:
            return
        if s + d > H:
            raise ValueError(
                f"execution_horizon (s={s}) + rtc_delay (d={d}) = {s + d} exceeds the "
                f"chunk size H={H}. The chunk must contain a length-d prefix and at least "
                f"s actions to execute. Reduce s or d."
            )
        if s <= 0:
            raise ValueError(f"execution_horizon must be positive, got {s}")

    def _build_pi05_example(self, obs: dict) -> dict:
        """Convert LeRobot obs to pi05 input format.

        Images: HWC uint8 RGB -> CHW float32 [0, 1].
        Note: ``action_prefix`` is no longer carried via the obs dict — the
        new openpi API takes ``prev_action_chunk`` as a direct kwarg to
        ``Policy.infer``, plumbed by ``_infer_pi05``.
        """
        images = {}
        for lerobot_key, cam_name in self._pi05_image_key_map.items():
            img = obs[lerobot_key]
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            images[cam_name] = img
        return {
            "images": images,
            "state": obs["observation.state"],
            "prompt": obs.get("prompt", ""),
        }

    def _infer_pi05(
        self, obs: dict, *, noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """pi05 inference. Returns the full action chunk (length H, raw joint space).

        For RTC mode, builds ``prev_action_chunk`` from the cached previous chunk
        (slots ``[s, s+d)`` → next chunk's prefix slots ``[0, d)``), normalizes it,
        right-pads to ``(H, action_dim)``, and forwards via the new openpi kwargs.
        """
        example = self._build_pi05_example(obs)
        infer_kwargs: dict = {}
        if noise is not None:
            infer_kwargs["noise"] = noise

        if self._rtc_delay > 0 and self._last_chunk_raw is not None:
            d = self._rtc_delay
            H, ad = self._action_prefix_shape()
            s = self._execution_horizon if self._execution_horizon is not None else H
            self._validate_horizons(H=H)
            # Slots [s, s+d) of the previous chunk become next chunk's prefix [0, d).
            prefix_raw = self._last_chunk_raw[s : s + d]
            assert prefix_raw.shape == (d, ad), (
                f"prefix shape mismatch: got {prefix_raw.shape}, expected ({d}, {ad}). "
                f"Cached chunk shape was {self._last_chunk_raw.shape}, s={s}, d={d}."
            )
            prefix_normed = self._normalize_action_prefix(prefix_raw)
            prev_full = np.zeros((H, ad), dtype=np.float32)
            prev_full[:d] = prefix_normed
            infer_kwargs["prev_action_chunk"] = prev_full
            infer_kwargs["inference_delay"] = d

        result = self._policy.infer(example, **infer_kwargs)
        chunk = np.asarray(result["actions"], dtype=np.float32)  # (H, ad), raw joint space
        # Cache for next inference's RTC prefix.
        self._last_chunk_raw = chunk
        return chunk

    def _infer_gr00t(
        self, obs: dict, *, noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert LeRobot obs to gr00t nested format and run inference.

        State: 14D -> per-limb (B=1, T=1, D).
        Images: HWC uint8 -> (B=1, T=1, H, W, C).

        .. note:: RTC noise passthrough is not yet implemented for gr00t.
           The ``noise`` parameter is accepted for API compatibility but
           currently ignored.
        """
        parsed_obs = self._convert_obs_for_gr00t(obs)
        raw_actions, _ = self._policy.get_action(parsed_obs)

        # Unbatch and concatenate per-step across action keys.
        action_horizon = np.atleast_1d(raw_actions[self._action_keys[0]][0]).shape[0]
        actions = []
        for step_idx in range(action_horizon):
            step_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(raw_actions[key][0])[step_idx])
                    for key in self._action_keys
                ],
                axis=0,
            )
            actions.append(step_action)
        return np.stack(actions, axis=0)  # (action_horizon, 14)

    def _convert_obs_for_gr00t(self, obs: dict) -> dict:
        """Convert LeRobot-format obs to gr00t's nested ``{video, state, language}`` dict.

        Adds batch and time dimensions expected by ``Gr00tPolicy.get_action()``.
        """
        state_14d = obs["observation.state"]

        new_obs: dict[str, Any] = {}

        # State: split 14D vector into per-limb arrays with (B=1, T=1, D) shape.
        new_obs["state"] = {}
        for key in self._modality_configs["state"].modality_keys:
            sl = _GR00T_STATE_SLICES[key]
            new_obs["state"][key] = np.asarray(state_14d[sl], dtype=np.float32)[None, None, :]

        # Video: (H, W, C) uint8 -> (B=1, T=1, H, W, C).
        new_obs["video"] = {}
        for key in self._modality_configs["video"].modality_keys:
            new_obs["video"][key] = obs[key][None, None, :]

        # Language: wrap prompt string.
        new_obs["language"] = {}
        for key in self._modality_configs["language"].modality_keys:
            new_obs["language"][key] = [[obs.get("prompt", "")]]

        return new_obs


def plot_action_comparison(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Per-dimension 1D plot comparing predicted vs ground-truth action chunks.

    ``pred_actions`` and ``gt_actions`` have shape ``(horizon, action_dim)``.
    """
    from matplotlib import pyplot as plt

    horizon = min(len(pred_actions), len(gt_actions))
    pred_actions = pred_actions[:horizon]
    gt_actions = gt_actions[:horizon]
    action_dim = pred_actions.shape[1]

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(8, 3 * action_dim))
    if action_dim == 1:
        axes = [axes]

    for dim_idx in range(action_dim):
        ax = axes[dim_idx]
        ax.plot(gt_actions[:, dim_idx], label="gt")
        ax.plot(pred_actions[:, dim_idx], label="pred")
        ax.set_title(f"Action dim {dim_idx}")
        ax.set_xlabel("timestep")
        ax.legend()

    fig.suptitle("Predicted vs Ground-Truth Action Chunk", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def build_obs_from_lerobot_sample(
    sample: dict,
    image_keys: list[str],
    prompt: str | None = None,
) -> dict:
    """Build a LeRobot-format obs dict from a ``LeRobotDataset`` sample.

    Converts images from LeRobot's internal CHW float32 [0, 1] to the
    canonical HWC uint8 RGB format that ``act()`` / ``infer_chunk()`` expect.
    """
    obs: dict[str, Any] = {}
    for key in image_keys:
        img = sample[key].numpy()
        # LeRobot returns CHW float32 [0, 1]; convert to HWC uint8.
        if np.issubdtype(img.dtype, np.floating):
            img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        obs[key] = img
    obs["observation.state"] = sample["observation.state"].numpy()
    obs["prompt"] = prompt or sample.get("task", "")
    _print_obs_dict("[build_obs]", obs)
    return obs


def _print_obs_dict(tag: str, obs: dict) -> None:
    """Print shapes and dtypes of all entries in an obs dict."""
    print(f"{tag} obs dict:")
    for key, val in obs.items():
        if hasattr(val, "shape"):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        elif isinstance(val, str):
            print(f"  {key}: str = {repr(val)[:80]}")
        else:
            print(f"  {key}: {type(val).__name__}")


def _report_and_plot(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Print comparison metrics and optionally plot predicted vs GT actions."""
    horizon = min(len(pred_actions), len(gt_actions))
    pred_actions = pred_actions[:horizon]
    gt_actions = gt_actions[:horizon]

    mse = np.mean((pred_actions - gt_actions) ** 2)
    mae = np.mean(np.abs(pred_actions - gt_actions))

    pred_deltas = np.abs(np.diff(pred_actions, axis=0))
    gt_deltas = np.abs(np.diff(gt_actions, axis=0))

    print(f"Pred actions shape: {pred_actions.shape}")
    print(f"GT   actions shape: {gt_actions.shape}")
    print(f"Pred actions (first 3 steps):\n{pred_actions[:3]}")
    print(f"GT   actions (first 3 steps):\n{gt_actions[:3]}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Smoothness (mean |delta| per step):")
    print(f"  Pred: {np.mean(pred_deltas):.6f}  (max delta: {np.max(pred_deltas):.6f})")
    print(f"  GT:   {np.mean(gt_deltas):.6f}  (max delta: {np.max(gt_deltas):.6f})")
    print(f"Per-joint smoothness (mean |delta|):")
    print(f"  Pred: {np.mean(pred_deltas, axis=0)}")
    print(f"  GT:   {np.mean(gt_deltas, axis=0)}")

    if args.plot or args.save_plot:
        plot_action_comparison(pred_actions, gt_actions, save_path=args.save_plot)


def _test_pi05(args: argparse.Namespace) -> None:
    """Load a LeRobot dataset, run pi05 inference via infer_chunk(), compare to GT.

    Two test modes:
      * Single chunk (default, ``num_cycles=1``): run one inference and compare
        the full ``H`` chunk against ``H`` consecutive GT actions.
      * Stitched RTC (``num_cycles>1``): run ``num_cycles`` inference cycles,
        advancing the obs by ``s = execution_horizon`` steps each cycle so the
        wrapper exercises its prefix-clamp path. Reports stitched MSE and the
        critical RTC metric — the prefix→postfix continuity *inside* each
        chunk (slot ``d-1`` → slot ``d``) compared against the GT step at the
        same controller time.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if args.config_name is None:
        raise SystemExit("--config_name is required for pi05")

    print("Loading policy...")
    wrapper = PolicyWrapper(
        args.checkpoint,
        "pi05",
        config_name=args.config_name,
        rtc_delay=args.rtc_delay,
        execution_horizon=args.execution_horizon,
    )
    print(f"Policy loaded.  rtc_delay={wrapper._rtc_delay}, "
          f"execution_horizon={wrapper._execution_horizon}")

    print("Loading dataset...")
    ds = LeRobotDataset(args.dataset, revision=args.revision)
    print(f"Dataset loaded. Total frames: {len(ds)}")

    # Probe inference to learn chunk_size + resolve execution_horizon defaults.
    sample_0 = ds[0]
    obs_0 = build_obs_from_lerobot_sample(sample_0, _LEROBOT_IMAGE_KEYS, args.prompt)
    probe_chunk = wrapper.infer_chunk(obs_0)
    H = wrapper._last_chunk_raw.shape[0]
    s = wrapper._execution_horizon
    d = wrapper._rtc_delay
    print(f"Action chunk shape: H={H}, action_dim={probe_chunk.shape[1]}, s={s}, d={d}")

    # Reset so the test starts from a clean RTC state (vanilla cold-start on
    # the first cycle, then RTC-clamped on subsequent cycles).
    wrapper.reset()

    if args.num_cycles <= 1:
        # Single-chunk comparison.
        max_start = max(0, len(ds) - H - 1)
        timestep = random.randint(0, max_start)
        print(f"Sampling observation at timestep {timestep}")

        sample = ds[timestep]
        obs = build_obs_from_lerobot_sample(sample, _LEROBOT_IMAGE_KEYS, args.prompt)

        pred_actions = wrapper.infer_chunk(obs)
        gt_actions = np.stack(
            [ds[timestep + offset]["action"].numpy() for offset in range(H)],
            axis=0,
        )
        _report_and_plot(pred_actions, gt_actions, args)
        return

    # Stitched-RTC test.
    needed = args.num_cycles * s + max(d, 1)  # +1 so we can index gt[idx-1] at the last boundary
    max_start = max(0, len(ds) - needed - 1)
    if max_start <= 0:
        raise SystemExit(
            f"Dataset of length {len(ds)} is too short for {args.num_cycles} cycles "
            f"of s={s} (needed >= {needed} contiguous frames)."
        )
    start_t = random.randint(0, max_start)
    print(f"Stitched RTC test: num_cycles={args.num_cycles}, "
          f"start_t={start_t}, span=[{start_t}, {start_t + args.num_cycles * s})")

    chunks, executed, gt = _run_rtc_episode(
        wrapper, ds, start_t=start_t, num_cycles=args.num_cycles,
        s=s, prompt=args.prompt, image_keys=_LEROBOT_IMAGE_KEYS,
    )
    _report_rtc_metrics(chunks=chunks, executed=executed, gt=gt, d=d, s=s)
    if args.plot or args.save_plot:
        plot_action_comparison(executed, gt, save_path=args.save_plot)


def _run_rtc_episode(
    wrapper: "PolicyWrapper",
    ds: Any,
    *,
    start_t: int,
    num_cycles: int,
    s: int,
    prompt: str | None,
    image_keys: list[str],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Run ``num_cycles`` infer_chunk calls, advancing the obs by ``s`` each cycle.

    Returns:
        chunks: list of full ``(H, action_dim)`` chunks emitted per cycle.
        executed: ``(num_cycles * s, action_dim)`` — what the queue would feed
            the robot, stitched in order.
        gt: ``(num_cycles * s, action_dim)`` — recorded GT actions for the
            corresponding controller steps.
    """
    chunks: list[np.ndarray] = []
    executed_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    t = start_t
    for _ in range(num_cycles):
        sample = ds[t]
        obs = build_obs_from_lerobot_sample(sample, image_keys, prompt)
        chunk_executed = wrapper.infer_chunk(obs)  # (s, ad)
        full_chunk = np.asarray(wrapper._last_chunk_raw)  # (H, ad)
        chunks.append(full_chunk)
        executed_chunks.append(np.asarray(chunk_executed))
        gt_chunks.append(np.stack(
            [ds[t + i]["action"].numpy() for i in range(s)], axis=0,
        ))
        t += s
    return chunks, np.concatenate(executed_chunks, axis=0), np.concatenate(gt_chunks, axis=0)


def _report_rtc_metrics(
    *,
    chunks: list[np.ndarray],
    executed: np.ndarray,
    gt: np.ndarray,
    d: int,
    s: int,
) -> None:
    """Print stitched MSE/MAE plus the prefix→postfix boundary continuity check.

    The RTC-specific concern: inside each chunk emitted from cycle ``K >= 1``,
    slot ``d-1`` is a clamped prefix value (= prev chunk's slot ``s+d-1``)
    while slot ``d`` is the first freshly generated postfix prediction. A
    well-trained model produces those two adjacent actions with a step size
    consistent with the natural action smoothness of the task. A poorly
    trained one emits a visible jump.
    """
    num_cycles = len(chunks)
    horizon = min(len(executed), len(gt))
    executed = executed[:horizon]
    gt = gt[:horizon]

    mse = np.mean((executed - gt) ** 2)
    mae = np.mean(np.abs(executed - gt))

    # Per-step magnitudes for the stitched trajectories (excluding the very first step
    # which has no predecessor).
    pred_step_mag = np.linalg.norm(np.diff(executed, axis=0), axis=-1)  # [horizon-1]
    gt_step_mag = np.linalg.norm(np.diff(gt, axis=0), axis=-1)

    print(f"\n=== Stitched RTC test ({num_cycles} cycles, s={s}, d={d}) ===")
    print(f"Executed shape: {executed.shape}, GT shape: {gt.shape}")
    print(f"Stitched MSE: {mse:.6f}")
    print(f"Stitched MAE: {mae:.6f}")
    print(f"Per-step jump magnitude (mean ‖Δa‖): "
          f"pred={pred_step_mag.mean():.6f}, gt={gt_step_mag.mean():.6f}")

    if d == 0:
        print("(d=0: no prefix→postfix boundary to evaluate.)")
        return
    if num_cycles < 2:
        print("(<2 cycles: no RTC-clamped chunks to evaluate.)")
        return

    # RTC-specific boundary metric.
    #
    # For each cycle K in [1, num_cycles): chunk_K[d-1] is clamped (last prefix
    # slot, = prev_chunk[s+d-1]) and chunk_K[d] is the first fresh postfix
    # slot. In the stitched trajectory these land at executed indices
    # [K*s + d - 1, K*s + d].
    #
    # We compare the *predicted* jump at that boundary against the GT jump at
    # the same controller step. Ratio close to 1 = continuity preserved.
    pred_boundary_jumps = []
    gt_boundary_jumps = []
    for k in range(1, num_cycles):
        chunk = chunks[k]
        pred_boundary_jumps.append(np.linalg.norm(chunk[d] - chunk[d - 1]))
        gt_idx = k * s + d
        if 0 < gt_idx < len(gt):
            gt_boundary_jumps.append(np.linalg.norm(gt[gt_idx] - gt[gt_idx - 1]))

    pred_boundary_jumps = np.asarray(pred_boundary_jumps)
    gt_boundary_jumps = np.asarray(gt_boundary_jumps)
    pred_typical = float(pred_step_mag.mean())  # mean step-magnitude across the whole stitched traj

    print(f"\n--- Prefix→postfix boundary (slot {d-1} → {d}) inside each RTC chunk ---")
    print(f"  pred boundary ‖Δa‖: mean={pred_boundary_jumps.mean():.6f}, "
          f"max={pred_boundary_jumps.max():.6f}  (n={len(pred_boundary_jumps)})")
    if len(gt_boundary_jumps):
        print(f"  gt   boundary ‖Δa‖: mean={gt_boundary_jumps.mean():.6f}, "
              f"max={gt_boundary_jumps.max():.6f}")
        ratio = pred_boundary_jumps.mean() / max(gt_boundary_jumps.mean(), 1e-9)
        print(f"  pred/gt ratio: {ratio:.3f}  (≈1.0 means model preserves natural smoothness)")
    if pred_typical > 0:
        rel = pred_boundary_jumps.mean() / pred_typical
        print(f"  pred boundary / mean pred step: {rel:.3f}  "
              f"(>>1 means an artificial RTC-induced jump)")

    # Also: cross-chunk handover continuity in the executed trajectory. Each cycle k>=1
    # contributes its slot 0 = clamped(prev[s]); the handover from cycle k-1 to cycle k
    # is at executed indices [k*s - 1, k*s].
    handover_jumps = []
    handover_gt = []
    for k in range(1, num_cycles):
        idx = k * s
        if 0 < idx < len(executed):
            handover_jumps.append(np.linalg.norm(executed[idx] - executed[idx - 1]))
            handover_gt.append(np.linalg.norm(gt[idx] - gt[idx - 1]))
    if handover_jumps:
        hj = np.asarray(handover_jumps)
        gj = np.asarray(handover_gt)
        print(f"\n--- Cross-chunk handover (executed[{s*1}-1] → executed[{s*1}], etc.) ---")
        print(f"  pred ‖Δa‖: mean={hj.mean():.6f}, max={hj.max():.6f}")
        print(f"  gt   ‖Δa‖: mean={gj.mean():.6f}, max={gj.max():.6f}")


def _test_gr00t(args: argparse.Namespace) -> None:
    """Load a LeRobot dataset, run gr00t inference via infer_chunk(), compare to GT."""
    from gr00t.data.embodiment_tags import EmbodimentTag
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    print("Loading policy...")
    wrapper = PolicyWrapper(args.checkpoint, "gr00t", embodiment_tag=embodiment_tag)
    print("Policy loaded.")

    print("Loading dataset...")
    ds = LeRobotDataset(args.dataset, revision=args.revision)
    print(f"Dataset loaded. Total frames: {len(ds)}")

    # Probe inference to learn chunk_size.
    sample_0 = ds[0]
    obs_0 = build_obs_from_lerobot_sample(sample_0, _LEROBOT_IMAGE_KEYS, args.prompt)
    probe_chunk = wrapper.infer_chunk(obs_0)
    chunk_size = probe_chunk.shape[0]
    print(f"Action chunk shape: {probe_chunk.shape}")

    timestep = random.randint(0, max(0, len(ds) - chunk_size - 1))
    print(f"Sampling observation at timestep {timestep}")

    sample = ds[timestep]
    obs = build_obs_from_lerobot_sample(sample, _LEROBOT_IMAGE_KEYS, args.prompt)

    pred_actions = wrapper.infer_chunk(obs)
    gt_actions = np.stack(
        [ds[timestep + offset]["action"].numpy() for offset in range(chunk_size)],
        axis=0,
    )

    _report_and_plot(pred_actions, gt_actions, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PolicyWrapper against a LeRobot dataset.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path or HF repo id.")
    parser.add_argument(
        "--policy_type", type=str, required=True, choices=["pi05", "gr00t"],
    )
    parser.add_argument("--config_name", type=str, default=None, help="[pi05] Config name for get_config().")
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset repo id or local path.")
    parser.add_argument("--revision", type=str, default="v2.1", help="Dataset git revision.")
    parser.add_argument(
        "--embodiment_tag", type=str, default="new_embodiment",
        help="[gr00t] Embodiment tag (e.g. 'new_embodiment').",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Language prompt (overrides dataset task).")
    parser.add_argument("--plot", action="store_true", help="Plot predicted vs GT action chunks.")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save the plot (implies --plot).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--rtc_delay", type=int, default=0,
        help="[pi05] Training-time RTC delay d. Set to the value the checkpoint was trained with. "
             "0 = non-RTC (default).",
    )
    parser.add_argument(
        "--execution_horizon", type=int, default=None,
        help="[pi05] Per-cycle queue size s. None = full chunk for non-RTC, H-d for RTC.",
    )
    parser.add_argument(
        "--num_cycles", type=int, default=1,
        help="[pi05] Number of inference cycles to stitch together for the test. "
             ">1 enables the stitched-RTC test that reports prefix→postfix continuity.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.policy_type == "pi05":
        _test_pi05(args)
    else:
        _test_gr00t(args)


if __name__ == "__main__":
    main()
