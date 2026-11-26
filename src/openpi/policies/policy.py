from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def rtc_step_weights(horizon: int, inference_delay: int, overlap: int) -> np.ndarray:
    """Real-Time Chunking soft mask: 1 for the first `inference_delay` steps (executed from the previous chunk while
    inferring), exponentially decaying over the rest of the `overlap` with the previous chunk, 0 afterwards."""
    if not 0 <= inference_delay <= overlap <= horizon:
        raise ValueError(f"Need 0 <= inference_delay ({inference_delay}) <= overlap ({overlap}) <= horizon ({horizon})")
    i = np.arange(horizon)
    c = (overlap - i) / (overlap - inference_delay + 1)
    decay = c * np.expm1(c) / np.expm1(1.0)
    return np.where(i < inference_delay, 1.0, np.where(i < overlap, decay, 0.0)).astype(np.float32)


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        prev_actions: np.ndarray | None = None,
        inference_delay: int = 0,
        guidance_dim_mask: Sequence[bool] | None = None,
        max_guidance_weight: float = 5.0,
    ) -> dict:
        """Runs inference. Passing `prev_actions` enables Real-Time Chunking guidance.

        Args:
            prev_actions: The not-yet-executed part of the previous chunk, aligned so that `prev_actions[i]` is the
                action for the same timestep as the new chunk's step i. Same (unnormalized) space as the policy output.
            inference_delay: Number of steps that will execute from the previous chunk while this call runs.
            guidance_dim_mask: Action dims to guide (in robot action space); unlisted/padded dims are left free.
            max_guidance_weight: Clip on the guidance strength (beta in the RTC paper).
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if prev_actions is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("Real-Time Chunking guidance is only implemented for JAX models.")
            # Map the previous chunk into model space with the same transforms the training targets went through.
            inputs["actions"] = np.array(prev_actions, dtype=np.float32)
        inputs = self._input_transform(inputs)
        rtc_kwargs = {}
        if prev_actions is not None:
            target = np.asarray(inputs.pop("actions"), dtype=np.float32)
            horizon, model_dim = self._model.action_horizon, self._model.action_dim
            overlap = min(len(target), horizon)
            padded_target = np.zeros((horizon, model_dim), np.float32)
            padded_target[:overlap] = target[:overlap]
            robot_mask = np.ones(np.shape(prev_actions)[-1]) if guidance_dim_mask is None else guidance_dim_mask
            dim_mask = np.zeros(model_dim, np.float32)
            dim_mask[: len(robot_mask)] = np.asarray(robot_mask, np.float32)
            weights = rtc_step_weights(horizon, inference_delay, overlap)[:, None] * dim_mask[None, :]
            rtc_kwargs = {
                "prev_chunk": jnp.asarray(padded_target)[None],
                "guidance_weights": jnp.asarray(weights)[None],
                "max_guidance_weight": max_guidance_weight,
            }
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs, **rtc_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
