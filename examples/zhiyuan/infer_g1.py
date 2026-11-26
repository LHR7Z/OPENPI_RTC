"""Offline prediction with our G1 pi05 checkpoint (trained with delta actions) from an observation NPZ.

Same CLI as the course example `infer.py`, plus the steps that example lacks for this model:
delta joint/head/waist actions are converted back to absolute targets by adding the current state,
and images are letterboxed to 224x224 exactly like training (any input size is accepted).

NPZ keys: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb (HxWx3 uint8), state (20,), prompt (str).
`predict()` also supports Real-Time Chunking guidance toward the previous chunk (`prev_actions`).
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools

from openpi import transforms as T
from openpi.models import model
from openpi.models import pi0_config
from openpi.models import tokenizer
from openpi.policies.policy import Policy
from openpi.shared import normalize

CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
STATE_DIM = 20
ACTION_DIM = 22
ACTION_HORIZON = 50
# Must match LeRobotG1DataConfig: make_bool_mask(14, -2, 2, 2) -> joints and head/waist are deltas, grippers absolute.
DELTA_MASK = np.array([True] * 14 + [False] * 2 + [True] * 4)
# RTC guides only the active left arm (0-6) and left gripper (14). The other dims barely move in this data, so their
# quantile ranges are ~1e-4 and any target mismatch would be amplified thousands of times by normalization.
GUIDANCE_DIM_MASK = np.array([True] * 7 + [False] * 7 + [True] + [False] * 7)


def load_policy(checkpoint: Path, asset: str) -> tuple[Policy, dict]:
    stats = normalize.load(checkpoint / "assets" / asset)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=ACTION_HORIZON, discrete_state_input=True)
    weights = model.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    policy = Policy(
        config.load(weights),
        transforms=[
            # Only act when `actions` is present, i.e. when an RTC target is passed; plain inference is unaffected.
            T.DeltaActions(DELTA_MASK),
            T.Normalize({"state": stats["state"], "actions": stats["actions"]}, use_quantiles=True),
            T.ResizeImages(224, 224),
            T.TokenizePrompt(tokenizer.PaligemmaTokenizer(200), discrete_state_input=True),
            T.PadStatesAndActions(32),
        ],
    )
    return policy, stats


def predict(
    policy: Policy,
    stats: dict,
    images: dict,
    state: np.ndarray,
    prompt: str,
    *,
    noise: np.ndarray | None = None,
    prev_actions: np.ndarray | None = None,
    inference_delay: int = 0,
    max_guidance_weight: float = 5.0,
) -> np.ndarray:
    """Returns a (50, 22) chunk of absolute action targets.

    For RTC, pass the previous chunk's absolute actions aligned to this chunk's timesteps (e.g. `prev[s:]` if `s`
    steps of the previous chunk ran before this observation) and the number of steps that execute while inferring.
    """
    state = np.asarray(state, dtype=np.float32)
    obs = {
        "image": {name: np.asarray(images[name]) for name in CAMERAS},
        "image_mask": {name: np.bool_(True) for name in CAMERAS},
        "state": state,
        "prompt": prompt,
    }
    # RTC kwargs only exist in this repo's Policy; omit them otherwise so plain inference still runs on upstream openpi
    # (e.g. in the course docker image).
    rtc = {}
    if prev_actions is not None:
        rtc = {
            "prev_actions": prev_actions,
            "inference_delay": inference_delay,
            "guidance_dim_mask": GUIDANCE_DIM_MASK,
            "max_guidance_weight": max_guidance_weight,
        }
    raw = policy.infer(obs, noise=noise, **rtc)["actions"][:, :ACTION_DIM]
    actions = T.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)({"actions": raw})["actions"]
    actions = np.array(actions, dtype=np.float32)
    actions[:, :STATE_DIM] += np.where(DELTA_MASK, state, 0)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError("Unexpected G1 action output")
    return actions


def letterbox_224(image: np.ndarray) -> np.ndarray:
    return image_tools.resize_with_pad(np.asarray(image), 224, 224)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--asset", required=True, help="Training asset name under checkpoint/assets")
    args = parser.parse_args()

    policy, stats = load_policy(args.checkpoint, args.asset)
    with np.load(args.observation, allow_pickle=False) as frame:
        actions = predict(policy, stats, {n: frame[n] for n in CAMERAS}, frame["state"], str(frame["prompt"]))
    np.save(args.output, actions)
    print(f"Saved {actions.shape} absolute actions to {args.output}")


if __name__ == "__main__":
    main()
