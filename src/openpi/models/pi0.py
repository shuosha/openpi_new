import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "*b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "*b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions.

    Supports an arbitrary leading shape on `pos` (e.g. `[B]` or `[B, H]`); the
    `embedding_dim` is appended as the new last axis.
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    # Generalized from `einsum("i,j->ij", ...)` to support arbitrary leading shapes on `pos`
    # (e.g. `[B]` for the legacy per-batch timestep, `[B, H]` for training-time RTC's per-token
    # timestep). Preserves HIGHEST precision to match the original einsum.
    sinusoid_input = jnp.einsum(
        "...,j->...j",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.rtc_max_delay = config.rtc_max_delay
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.state_noise_std = config.state_noise_std

        # Image augmentation config (applied in preprocess_observation when train=True).
        self.image_aug_color_jitter = config.image_aug_color_jitter
        self.image_aug_color_jitter_p = config.image_aug_color_jitter_p
        self.image_aug_crop_fraction = config.image_aug_crop_fraction
        self.image_aug_rotate_deg = config.image_aug_rotate_deg
        self.image_aug_geometric_p = config.image_aug_geometric_p
        self.image_aug_apply_geometric_to_wrist = config.image_aug_apply_geometric_to_wrist

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "*b"] | at.Float[at.Array, "*b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        if self.rtc_max_delay is None:
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        else:
            preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            state_noise_std=self.state_noise_std,
            image_aug_color_jitter=self.image_aug_color_jitter,
            image_aug_color_jitter_p=self.image_aug_color_jitter_p,
            image_aug_crop_fraction=self.image_aug_crop_fraction,
            image_aug_rotate_deg=self.image_aug_rotate_deg,
            image_aug_geometric_p=self.image_aug_geometric_p,
            image_aug_apply_geometric_to_wrist=self.image_aug_apply_geometric_to_wrist,
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        if self.rtc_max_delay is None:
            # Standard flow-matching loss: per-batch scalar tau.
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            timestep = time
            prefix_action_mask = None
            delay = None
        else:
            # Training-time RTC: per-token tau with a clean prefix of length d.
            # Note openpi's inverted convention: time=0 is clean (paper's tau=1), time=1 is noise.
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.rtc_max_delay + 1)  # [*b]
            prefix_action_mask = jnp.arange(self.action_horizon) < delay[..., None]  # [*b, ah]
            time_per_tok = jnp.where(prefix_action_mask, 0.0, time[..., None])  # [*b, ah]
            time_expanded = time_per_tok[..., None]  # [*b, ah, 1]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            timestep = time_per_tok

        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_tok_se = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [*b, ah]
        if prefix_action_mask is None:
            return per_tok_se, {}
        # Mask the prefix and rescale per example so that downstream `jnp.mean` over [*b, ah] equals
        # the per-postfix-token mean per example, then mean over the batch.
        loss_mask = jnp.logical_not(prefix_action_mask).astype(per_tok_se.dtype)
        postfix_count = jnp.maximum(loss_mask.sum(axis=-1, keepdims=True), 1.0)
        chunked_loss = per_tok_se * loss_mask * (self.action_horizon / postfix_count)

        metrics = self._compute_rtc_continuity_metrics(
            x_t=x_t,
            v_t=v_t,
            actions=actions,
            timestep=timestep,
            delay=delay,
            prefix_action_mask=prefix_action_mask,
            loss_mask=loss_mask,
        )
        return chunked_loss, metrics

    def _compute_rtc_continuity_metrics(
        self,
        *,
        x_t: at.Float[at.Array, "*b ah ad"],
        v_t: at.Float[at.Array, "*b ah ad"],
        actions: at.Float[at.Array, "*b ah ad"],
        timestep: at.Float[at.Array, "*b ah"],
        delay: at.Int[at.Array, "*b"],
        prefix_action_mask: at.Bool[at.Array, "*b ah"],
        loss_mask: at.Float[at.Array, "*b ah"],
    ) -> dict[str, at.Array]:
        """Continuity metrics measured at the prefix-postfix boundary.

        The key concern in RTC is that the model's first postfix prediction (slot `d`) joins
        smoothly onto the clamped prefix (slots `0..d-1`). These metrics are log-only — they
        do not flow into the gradient.

        Predicted clean action (openpi convention `x_t = t*noise + (1-t)*action`):
            x_hat = x_t - t * v_pred

        Reported scalars (mean over the batch unless noted; gradient stopped):
            rtc_mean_delay:        average sampled delay (sanity check on the distribution).
            rtc_postfix_mse:       MSE of predicted clean action vs ground truth, averaged
                                   over postfix tokens — overall postfix accuracy.
            rtc_boundary_mse:      MSE of predicted clean action vs ground truth at slot `d`
                                   only, averaged over examples where a postfix exists. This
                                   is the discontinuity at the join the robot would feel.
            rtc_pred_boundary_jump: ||x_hat[:, d, :] - actions[:, d-1, :]||_2 averaged over
                                   examples where d >= 1 and d < H. The clamped prefix
                                   ensures actions[:, d-1, :] equals what the prefix tail
                                   contains, so this is the predicted "step" at the join.
            rtc_true_boundary_jump: ||actions[:, d, :] - actions[:, d-1, :]||_2 over the
                                   same examples — the ground-truth step magnitude.
                                   Compare with rtc_pred_boundary_jump: ratio near 1 means
                                   the model preserves natural trajectory smoothness.
        """
        # x_hat: predicted clean action at every slot. At prefix slots (timestep=0) this
        # collapses to x_t = actions exactly (clamped GT), so x_hat values there are trivial.
        x_hat = x_t - timestep[..., None] * v_t  # [*b, ah, ad]
        clean_se = jnp.mean(jnp.square(x_hat - actions), axis=-1)  # [*b, ah]

        # postfix (excludes prefix slots).
        postfix_count_total = jnp.maximum(loss_mask.sum(), 1.0)
        rtc_postfix_mse = (clean_se * loss_mask).sum() / postfix_count_total

        H = self.action_horizon

        # Boundary slot is `delay` itself; only valid when delay < H (there *is* a postfix).
        boundary_valid = delay < H  # [*b]
        # Clip the index so take_along_axis is in-bounds; mask after.
        boundary_idx = jnp.minimum(delay, H - 1)[..., None]  # [*b, 1]
        boundary_clean_se = jnp.take_along_axis(clean_se, boundary_idx, axis=-1).squeeze(-1)  # [*b]
        rtc_boundary_mse = (boundary_clean_se * boundary_valid.astype(clean_se.dtype)).sum() / jnp.maximum(
            boundary_valid.sum().astype(clean_se.dtype), 1.0
        )

        # Boundary jump magnitude: only meaningful when 1 <= delay < H.
        jump_valid = (delay >= 1) & boundary_valid  # [*b]
        prev_idx = jnp.minimum(jnp.maximum(delay - 1, 0), H - 1)[..., None, None]  # [*b, 1, 1]
        cur_idx = jnp.minimum(delay, H - 1)[..., None, None]  # [*b, 1, 1]
        ad_idx_shape = (1,) * (actions.ndim - 1) + (actions.shape[-1],)
        prev_idx_b = jnp.broadcast_to(prev_idx, prev_idx.shape[:-1] + (actions.shape[-1],))
        cur_idx_b = jnp.broadcast_to(cur_idx, cur_idx.shape[:-1] + (actions.shape[-1],))
        del ad_idx_shape  # unused; kept for clarity
        x_hat_at_d = jnp.take_along_axis(x_hat, cur_idx_b, axis=-2).squeeze(-2)  # [*b, ad]
        action_at_d_minus_1 = jnp.take_along_axis(actions, prev_idx_b, axis=-2).squeeze(-2)  # [*b, ad]
        action_at_d = jnp.take_along_axis(actions, cur_idx_b, axis=-2).squeeze(-2)  # [*b, ad]
        pred_jump = jnp.linalg.norm(x_hat_at_d - action_at_d_minus_1, axis=-1)  # [*b]
        true_jump = jnp.linalg.norm(action_at_d - action_at_d_minus_1, axis=-1)  # [*b]
        jump_w = jump_valid.astype(pred_jump.dtype)
        jump_count = jnp.maximum(jump_w.sum(), 1.0)
        rtc_pred_boundary_jump = (pred_jump * jump_w).sum() / jump_count
        rtc_true_boundary_jump = (true_jump * jump_w).sum() / jump_count

        metrics = {
            "rtc_mean_delay": delay.astype(jnp.float32).mean(),
            "rtc_postfix_mse": rtc_postfix_mse,
            "rtc_boundary_mse": rtc_boundary_mse,
            "rtc_pred_boundary_jump": rtc_pred_boundary_jump,
            "rtc_true_boundary_jump": rtc_true_boundary_jump,
        }
        # Belt-and-suspenders: never let metrics leak into gradients.
        return jax.tree.map(jax.lax.stop_gradient, metrics)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prev_action_chunk: at.Float[at.Array, "b ah ad"] | None = None,
        inference_delay: int | at.Int[at.Array, ""] = 0,
    ) -> _model.Actions:
        """Sample actions via Euler integration of the velocity field.

        Training-time RTC:
            When `prev_action_chunk` is provided, the first `inference_delay` slots of the
            output are clamped to `prev_action_chunk[:, :inference_delay]` (the "action prefix"
            — the part of the previous chunk that overlaps with the new chunk during the
            inference latency window). The model only generates the postfix. This matches the
            runtime contract `(action_prefix, d) -> action_postfix` from the RTC paper.
            Pass both `prev_action_chunk` and `inference_delay` together; passing
            `prev_action_chunk=None` (default) takes the legacy sampling path unchanged.
        """
        rtc_active = prev_action_chunk is not None
        if rtc_active and not self.pi05:
            raise ValueError("RTC inference (prev_action_chunk!=None) requires pi05=True (adaRMS conditioning).")

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if rtc_active:
            # action-slot prefix mask, shape [ah]. Works whether `inference_delay` is a static
            # Python int or a 0-d traced array.
            action_prefix_mask = jnp.arange(self.action_horizon) < inference_delay  # [ah]
            # Clamp the initial state so the first denoising step sees clean prefix slots.
            noise = jnp.where(action_prefix_mask[None, :, None], prev_action_chunk, noise)

        def step(carry):
            x_t, time = carry
            if rtc_active:
                # Clamp prefix slots to the action prefix; per-token time=0 (clean) on those slots.
                x_t = jnp.where(action_prefix_mask[None, :, None], prev_action_chunk, x_t)
                timestep = jnp.where(
                    action_prefix_mask[None, :], 0.0, jnp.broadcast_to(time, (batch_size, self.action_horizon))
                )
            else:
                timestep = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, timestep
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            if rtc_active:
                # Re-clamp to keep the prefix exact across the Euler update.
                x_next = jnp.where(action_prefix_mask[None, :, None], prev_action_chunk, x_next)
            return x_next, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
