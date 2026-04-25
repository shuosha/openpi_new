from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert metrics == {}

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert metrics == {}

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)
    assert metrics == {}

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)
    assert metrics == {}

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_pi05_rtc_compute_loss_shape():
    """Training-time RTC compute_loss returns the same [B, H] shape and is finite."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True, rtc_max_delay=4)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))
    # RTC metrics are present and scalar-shaped.
    expected_keys = {
        "rtc_mean_delay",
        "rtc_postfix_mse",
        "rtc_boundary_mse",
        "rtc_pred_boundary_jump",
        "rtc_true_boundary_jump",
    }
    assert set(metrics.keys()) == expected_keys
    for k, v in metrics.items():
        assert v.shape == (), f"{k} should be scalar, got {v.shape}"
        assert jnp.isfinite(v), f"{k} is not finite: {v}"


def test_pi05_rtc_metrics_dict_empty_without_rtc():
    """Without rtc_max_delay, the metrics dict is empty (no logging of RTC stats)."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)
    _, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert metrics == {}


def test_pi05_rtc_compute_loss_full_prefix_zero():
    """When delay == action_horizon (entire chunk is prefix), the masked loss is exactly zero."""
    key = jax.random.key(0)
    H = 50
    # Force every sampled delay to equal H by setting rtc_max_delay = H and patching the rng
    # path; simpler: use rtc_max_delay = H and rely on the rescale collapsing to 0 only when
    # the sampled delay actually hits H. Instead, monkeypatch jax.random.randint inside the
    # call to return all-H delays.
    config = pi0_config.Pi0Config(pi05=True, rtc_max_delay=H, action_horizon=H)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    original_randint = jax.random.randint

    def all_h_randint(rng, shape, minval, maxval):  # noqa: ARG001
        return jnp.full(shape, H, dtype=jnp.int32)

    jax.random.randint = all_h_randint
    try:
        # Don't jit; we need the monkeypatched randint to be called.
        loss, _ = model.compute_loss(key, obs, act)
    finally:
        jax.random.randint = original_randint
    assert loss.shape == (batch_size, H)
    np.testing.assert_array_equal(np.asarray(loss), np.zeros((batch_size, H), dtype=np.asarray(loss).dtype))


def test_pi05_rtc_sample_actions_clamps_prefix():
    """sample_actions with inference_delay > 0 returns an output whose prefix slots equal the GT prefix."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True)
    model = config.create(key)

    batch_size = 2
    obs = config.fake_obs(batch_size)

    delay = 3
    prev_chunk = jnp.full((batch_size, model.action_horizon, model.action_dim), 7.0, dtype=jnp.float32)

    actions = nnx_utils.module_jit(model.sample_actions)(
        key, obs, num_steps=4, prev_action_chunk=prev_chunk, inference_delay=delay
    )
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    # prefix slots are clamped exactly to prev_chunk
    np.testing.assert_array_equal(np.asarray(actions[:, :delay]), np.asarray(prev_chunk[:, :delay]))
    # postfix slots actually came from the model (not equal to 7.0)
    assert not jnp.all(actions[:, delay:] == 7.0)


def test_pi05_legacy_path_unchanged_without_rtc():
    """Loss with rtc_max_delay=None and sampling with inference_delay=0 work as before."""
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(pi05=True)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, metrics = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert metrics == {}

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_config_rejects_rtc_without_pi05():
    """Pi0Config refuses rtc_max_delay when pi05=False."""
    with pytest.raises(ValueError, match="pi05"):
        pi0_config.Pi0Config(pi05=False, rtc_max_delay=4)


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss, _ = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
