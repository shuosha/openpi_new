# Training-Time Real-Time Chunking (RTC): Integration Guide

A drop-in extension to a flow-matching action-chunking policy that enables smooth, asynchronous, latency-hiding execution at inference time. Based on Black et al., *Training-Time Action Conditioning for Efficient Real-Time Chunking* (2025).

---

## 1. Background

A standard chunking policy `p(A_t | o_t)` predicts an `H`-step action chunk `A_t = [a_t, …, a_{t+H-1}]` from observation `o_t`, then executes some prefix of it before re-querying the model. If we run the policy **synchronously**, the robot must wait the full inference latency between chunks → visible pauses, jerky motion.

**RTC (real-time chunking)** runs the policy **asynchronously**: the next chunk is computed *while the current one is still being executed*. The challenge is **inter-chunk continuity** — the new chunk must agree with the part of the old chunk that's already been committed to during the time inference was running.

- **Inference-time RTC** (Black, Galliker, Levine 2025) handles this with pseudoinverse-guided inpainting at sampling time. Flexible but adds VJP overhead per denoising step.
- **Training-time RTC** (this guide) trains the model directly to condition on a hard prefix of ground-truth actions, by simulating the inference delay during training. **Zero inference-time overhead.**

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `H` | prediction horizon (chunk length) |
| `s` | execution horizon (steps committed per chunk before the next inference is kicked off) |
| `d` | inference delay, in controller timesteps |
| `A_t` | chunk produced from observation at step `t`, indexed `[0, H)` and aligned with controller steps `[t, t+H)` |
| **prefix** | first `d` slots of the new chunk; **fixed** to match the previous chunk's predictions for those steps |
| **postfix** | remaining `H − d` slots; what the model actually generates |

Constraint: `d ≤ H − s` (otherwise the previous chunk doesn't have enough actions to bridge the latency gap).

---

## 3. How RTC Works — Timing Example

Concrete example with `H = 8`, `s = 4`, `d = 2`:

```
controller step :    0   1   2   3   4   5   6   7   8   9  10  11
                     |───────|───────|───────|───────|
                       wait    chunk0  chunk1  chunk2

chunk_0 inference:  [▓▓▓▓▓▓▓▓]                                    started t=0, arrives t=2
chunk_0 spans      [a0  a1  a2  a3  a4  a5  a6  a7]               covers controller [0, 8)
                            └── executed ──┘                      slots [2:6] used
                                                              
chunk_1 inference:                  [▓▓▓▓▓▓▓▓]                    started t=4, arrives t=6
   prefix from chunk_0:              ⤴ a4, a5  (slots 4,5 of chunk_0)
chunk_1 spans                      [a4  a5  a6  a7  a8  a9 a10 a11]   covers [4, 12)
                                    ├prefix┤└── executed ──┘      slots [2:6] used
                                                              
chunk_2 inference:                                  [▓▓▓▓▓▓▓▓]    started t=8, arrives t=10
   prefix from chunk_1:                              ⤴ a8, a9
```

**Per-cycle invariants:**
- A new inference is kicked off every `s` controller steps.
- When chunk `k+1` is generated, its first `d` slots are clamped to the previous chunk's predictions for the same controller steps (the **action prefix**).
- The model only generates the remaining `H − d` slots (the **postfix**).
- The robot always has actions ready: it bridges the latency window with the previous chunk's tail, then switches to the new chunk's postfix for `s` steps.

---

## 4. Training-Side Changes

Three minimal modifications to a standard conditional flow-matching policy. The reference loss is

```
A^τ_t = τ · A_t + (1 − τ) · ε,    ε ~ N(0, I)
L = E ‖ v_θ(A^τ_t, o_t, τ) − (ε − A_t) ‖²
```

### 4.1 Per-token flow-matching timestep

The velocity head must accept a **different `τ` per action token**, not a single scalar per sample.

- **DiT-style with adaLN-zero conditioning on `τ`**: trivial — embed `τ` per token, so (scale, shift, gate) modulation differs across tokens. **No new parameters.**
- **Other architectures**: ensure whatever module injects `τ` broadcasts across tokens rather than collapsing to a per-sample scalar.

### 4.2 Build the input as (clean prefix ‖ noisy postfix)

Per training example:

1. Sample a delay `d ~ p(d)` (e.g. `Unif{0, …, d_max−1}`).
2. Build `prefix_mask[i] = (i < d)` over chunk positions `i ∈ [0, H)`.
3. Sample a single flow timestep `τ ~ Unif[0, 1)` and broadcast to shape `(H,)`.
4. Set per-token `τ_i = 1.0` wherever `prefix_mask[i]` is true; leave `τ_i = τ` elsewhere.
5. Form the network input
   - `x_t[i] = τ_i · A_t[i] + (1 − τ_i) · ε[i]`
   - At prefix slots, `τ_i = 1` ⇒ `x_t[i] = A_t[i]` (clean GT, no noise).
   - At postfix slots, the standard noisy interpolant.

### 4.3 Mask the loss to postfix tokens only

Standard target is `(ε − A_t)`. Compute the per-token MSE as usual, then average **only over postfix tokens** (`~prefix_mask`). Prefix tokens contribute zero gradient.

### 4.4 Reference implementation (from the paper)

```python
def compute_loss(rng, model, observation, action_chunk, max_delay):
    b, H, D = action_chunk.shape
    rng_n, rng_t, rng_d = jax.random.split(rng, 3)

    tau   = jax.random.uniform(rng_t, (b,))
    noise = jax.random.normal(rng_n, (b, H, D))
    delay = jax.random.randint(rng_d, (b,), 0, max_delay)              # NEW

    prefix_mask = jnp.arange(H)[None, :] < delay[:, None]              # NEW: (b, H)
    tau_per_tok = jnp.where(prefix_mask, 1.0, tau[:, None])            # NEW: (b, H)

    x_t   = (tau_per_tok[..., None] * action_chunk
             + (1 - tau_per_tok[..., None]) * noise)
    v_pred = model(observation, x_t, tau_per_tok)                      # τ now per-token
    loss   = (v_pred - (action_chunk - noise)) ** 2

    postfix_mask = (~prefix_mask)[..., None]
    return (loss * postfix_mask).sum() / (postfix_mask.sum() + 1e-8)
```

### 4.5 Choosing the delay distribution

- Sample `d` per example from a distribution that **covers your expected real-world latency range**.
- Paper's real-world setup: `d ~ Unif{0, …, 10}` for a 50 Hz robot (max 200 ms latency).
- Simulated setup: exponentially decreasing weights over `{0, …, d_max}` — higher delays need less supervision since the postfix is shorter.
- Including `d = 0` preserves full synchronous behavior as a fallback.
- Fine-tuning a non-RTC checkpoint for a few thousand steps is sufficient (the paper fine-tunes π₀.₆ for 8k steps at batch size 512).

---

## 5. Inference-Side Changes

The runtime interface is `(action_prefix, d) → action_postfix`, identical to inference-time RTC, so any existing async controller works unchanged.

### 5.1 Sampling

Standard Euler integration of the velocity field, with two additions:

1. At every denoising step, **overwrite the prefix slots of `x_t` with the ground-truth prefix actions** (just like inpainting), and keep their per-token `τ = 1.0`.
2. Postfix slots evolve normally from `τ = 0` to `τ = 1`.

```python
def sample_actions(rng, model, observation, action_prefix, delay, num_steps):
    # action_prefix is padded to (b, H, D); only the first `delay` entries are valid
    b, H, D = action_prefix.shape
    x_t  = jax.random.normal(rng, (b, H, D))
    tau, dt = 0.0, 1.0 / num_steps
    prefix_mask = jnp.arange(H)[None, :] < delay   # (b, H)

    for _ in range(num_steps):
        x_t         = jnp.where(prefix_mask[..., None], action_prefix, x_t)
        tau_per_tok = jnp.where(prefix_mask, 1.0, tau)
        v_t         = model(observation, x_t, tau_per_tok)
        x_t         = x_t + dt * v_t
        tau         = tau + dt
    return x_t
```

After the loop, the prefix slots equal `action_prefix` exactly; only the postfix slots are "real" model output.

### 5.2 Async controller loop

No change from any standard async RTC controller. Each cycle:

1. Maintain a buffer holding the most recent committed chunk and the timestep at which it was generated.
2. Every `s` controller steps, kick off a new inference. Pass:
   - the current observation,
   - the slice of the previous chunk corresponding to the next `d` controller steps as `action_prefix` (pad to length `H`),
   - the expected delay `d` (rounded measured latency in controller steps).
3. Continue executing the previous chunk while inference runs.
4. When the new chunk arrives, swap it in for execution starting at slot `d`.

`d` can be set per-call from the actual measured latency. Since training covered the full distribution, the model generalizes across delays.

---

## 6. Drop-In Summary

| Component | Change |
|---|---|
| Architecture | Broadcast `τ` per token (one-line change for adaLN-zero DiTs). No new params. |
| Training loop | ~5 added lines: sample `d`, build prefix mask, set `τ = 1` on prefix, mask loss to postfix. |
| Sampling loop | ~3 added lines: clamp prefix slots to GT each step, set per-token `τ = 1` on those slots. |
| Robot runtime | Unchanged from any existing async RTC controller. |

---

## 7. Limitations

- Unlike inference-time RTC's "soft masking" (pseudoinverse guidance with exponentially decaying weights over the full `H − s` overlap), training-time RTC supports only a **hard prefix** of length `d`. There's no equivalent of weighting the rest of the overlap.
- The training delay distribution must reasonably match deployment latency. If real-world `d` exceeds the training range, expect degradation.
- At small `d` (e.g. 0 or 1), few training tokens get supervision under that condition, so the model may be marginally weaker there than a non-RTC baseline trained on equivalent compute. In practice this gap is small and well worth the asynchronous-execution gains at higher `d`.

---

## Appendix A. Inference-Time RTC (for reference)

The original RTC method (Black, Galliker, Levine 2025, arXiv:2506.07339) operates entirely at inference time and works on top of any pre-trained flow or diffusion policy with **no retraining**. It is more flexible than training-time RTC, at the cost of a vector-Jacobian product through the model on every denoising step (~2.5× the vanilla per-step cost). Use it when you can't retrain or when you need soft cross-chunk continuity.

### A.1 Three-region chunk structure

Inference-time RTC partitions the new chunk into **three** regions, not two:

```
chunk position:    [0, d)        [d, H − s)           [H − s, H)
region:          hard prefix    intermediate         freshly generated
guidance W_i:        1         exp decay (1 → 0)            0
```

- **Hard prefix** (length `d`) — same role as in training-time RTC.
- **Intermediate** (length `H − d − s`) — actions that still overlap with the previous chunk's predictions but where inference will have completed before they're needed for execution. Used as *soft* guidance, with weights interpolating from ~1 down to 0.
- **Freshly generated** (length `s`) — beyond the previous chunk; no guidance.

Training-time RTC merges the right two regions into a single postfix of length `H − d`, generated with no reference to the previous chunk for the intermediate region. This is the precise sense in which training-time RTC is "less flexible".

### A.2 Soft-mask weight formula (Eq. 5)

For each position `i ∈ [0, H)`:

```
W_i = 1                                  if i < d
W_i = c_i · (e^{c_i} − 1) / (e − 1)      if d ≤ i < H − s
W_i = 0                                  if i ≥ H − s

where c_i = (H − s − i) / (H − s − d + 1)
```

`c_i` is a linear ramp from ~1 (at `i = d`) down to 0 (at `i = H − s`); wrapping in `(e^c − 1) / (e − 1)` shapes it exponentially. Ablations (Fig. 8 in the paper) show linear decay is nearly as good; hard masking underperforms at small `d`.

### A.3 ΠGDM correction (Eqs. 2–4)

At every denoising step, the velocity field is augmented with a guidance term that pulls the predicted final chunk toward the previous chunk's predictions, weighted by `W`:

```
v̂(A^τ, o, τ) = v(A^τ, o, τ)
              + min(β, (1 − τ) / (τ · r²_τ))
              · (Y − Â¹)ᵀ diag(W) · ∂Â¹/∂A^τ

where  Â¹    = A^τ + (1 − τ) · v(A^τ, o, τ)         (one-step denoising estimate)
       r²_τ = (1 − τ)² / (τ² + (1 − τ)²)
       Y    = previous chunk's predictions, right-padded to length H
       β    = guidance-weight clip (paper uses β = 5)
```

The `(Y − Â¹)ᵀ ∂Â¹/∂A^τ` term is a vector-Jacobian product computed via reverse-mode autodiff — this is the source of the inference-time overhead. The `β` clip is needed because `(1 − τ) / (τ · r²_τ) → ∞` as `τ → 0`, which destabilizes generation when only `n = 5` denoising steps are used.

### A.4 Hyperparameters (paper's real-world π₀.₅ setup)

| Symbol | Description | Value |
|---|---|---|
| `n` | Denoising steps | 5 |
| `H` | Prediction horizon | 50 |
| `s_min` | Minimum execution horizon | 25 |
| `β` | Guidance weight clip | 5 |
| `b` | Delay buffer size | 10 |

The execution horizon adapts per chunk as `s = max(d, s_min)`. The delay estimate for the next chunk is `max(buffer)` — a conservative choice that prevents underestimating latency.

### A.5 Choosing between the two

| Pick inference-time RTC if… | Pick training-time RTC if… |
|---|---|
| You can't retrain the base policy | You can spend a few thousand fine-tuning steps |
| You want soft cross-chunk continuity, especially at small `d` | You operate at higher delays (paper shows training-time wins for `d ≥ 2`) |
| Inference compute is not a bottleneck | You need zero inference-time overhead |
| You want a fully drop-in inference-only addition | You want a simpler implementation (no VJP, no β tuning) |