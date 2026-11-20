"""One offline prediction with a course G1 checkpoint and an observation saved as NPZ."""
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from openpi import transforms as T
from openpi.models import model, pi0_config, tokenizer
from openpi.policies.policy import Policy
from openpi.shared import normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("observation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--asset", required=True, help="Training asset name under checkpoint/assets")
    args = parser.parse_args()

    stats = normalize.load(args.checkpoint / "assets" / args.asset)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=50, discrete_state_input=True)
    weights = model.restore_params(args.checkpoint / "params", dtype=jnp.bfloat16)
    policy = Policy(config.load(weights), transforms=[
        T.Normalize({"state": stats["state"]}, use_quantiles=True),
        T.TokenizePrompt(tokenizer.PaligemmaTokenizer(200), discrete_state_input=True),
        T.PadStatesAndActions(32),
    ])
    cameras = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    with np.load(args.observation, allow_pickle=False) as frame:
        obs = {
            "image": {name: frame[name] for name in cameras},
            "image_mask": {name: np.bool_(True) for name in cameras},
            "state": frame["state"].astype(np.float32),
            "prompt": str(frame["prompt"]),
        }
    raw = policy.infer(obs)["actions"][:, :22]
    actions = T.Unnormalize({"actions": stats["actions"]}, use_quantiles=True)({"actions": raw})["actions"]
    if actions.shape != (50, 22) or not np.isfinite(actions).all():
        raise ValueError("Unexpected G1 action output")
    np.save(args.output, actions)
    print(f"Saved {actions.shape} actions to {args.output}")


if __name__ == "__main__":
    main()
