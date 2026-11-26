import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


def test_rtc_step_weights():
    w = _policy.rtc_step_weights(horizon=50, inference_delay=6, overlap=25)
    assert w.shape == (50,)
    np.testing.assert_array_equal(w[:6], 1.0)
    assert np.all(np.diff(w[6:25]) < 0)
    assert 0 < w[24] < w[6] < 1
    np.testing.assert_array_equal(w[25:], 0.0)
    with pytest.raises(ValueError, match="inference_delay"):
        _policy.rtc_step_weights(horizon=50, inference_delay=30, overlap=25)
