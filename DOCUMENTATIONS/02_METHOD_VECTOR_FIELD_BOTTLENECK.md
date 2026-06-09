# Method

At a selected denoising or flow step, the adapter exposes the frozen generator prediction:

```text
v_base = v_theta(z_t, t, prompt)
```

For each unsafe concept and safe sibling:

```text
b_unsafe = v_theta(z_t, t, prompt + unsafe_concept) - v_theta(z_t, t, prompt + neutral_concept)
b_safe = v_theta(z_t, t, prompt + safe_sibling_concept) - v_theta(z_t, t, prompt + neutral_concept)
activation = relu(cosine(v_base, b_unsafe) - cosine(v_base, b_safe) + margin)
v_steered = v_base + lambda_t * mask * (b_safe - b_unsafe)
```

The mask is computed locally over latent tokens by reducing cosine similarity along a feature/channel dimension. For image latents this is typically channel dimension `1`; video latents use the same default for channel-first layouts.

