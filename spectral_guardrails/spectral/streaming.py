"""
Compute attention features one layer at a time, without ever holding the
whole stack.

``output_attentions=True`` returns the attention probabilities for every
layer at once, and that tensor is quadratic in prompt length:

    layers x heads x T^2

For a 16-layer model with 32 heads at three thousand tokens this is over ten
gigabytes in bfloat16, which is the whole budget of a consumer GPU and is why
a conversation history in the prompt makes the pass spill into host memory.

The features we keep from a layer are tiny: five numbers per head, a short
diagonal profile, and two attention shares. So instead of collecting the
matrices and reducing afterwards, we attach a hook to each attention module
that reduces the layer as soon as it is produced and then drops the matrix by
returning it as None. Peak attention memory becomes one layer rather than all
of them.
"""
import torch


def _attention_modules(model):
    """Every submodule that looks like a self-attention block."""
    mods = []
    for name, mod in model.named_modules():
        base = name.rsplit(".", 1)[-1]
        if base in ("self_attn", "attention", "attn") and hasattr(mod, "forward"):
            mods.append((name, mod))
    return mods


class StreamingAttentionFeatures:
    """
    Context manager that runs `reducer(layer_index, attn_weights)` on each
    layer's attention as it is computed and then releases the matrix.

    Usage::

        with StreamingAttentionFeatures(model, reducer) as stream:
            model(input_ids=ids, output_attentions=True,
                  output_hidden_states=True)
        results = stream.results

    The model must still be called with ``output_attentions=True``, since that
    is what makes the attention weights exist at all; the hook prevents them
    from accumulating.
    """

    def __init__(self, model, reducer, drop=True):
        self.model = model
        self.reducer = reducer
        self.drop = drop
        self.results = []
        self._handles = []
        self._order = {}

    def __enter__(self):
        mods = _attention_modules(self.model)
        for idx, (name, mod) in enumerate(mods):
            self._order[name] = idx

            def make_hook(layer_idx):
                def hook(module, args, output):
                    weights = None
                    if isinstance(output, tuple):
                        for item in output[1:]:
                            if torch.is_tensor(item) and item.dim() == 4:
                                weights = item
                                break
                    if weights is None:
                        return output
                    with torch.no_grad():
                        self.results.append(
                            (layer_idx, self.reducer(layer_idx, weights)))
                    if not self.drop:
                        return output
                    # replace the weights with nothing so the stack does not
                    # accumulate them
                    new = tuple(None if (torch.is_tensor(o) and o.dim() == 4
                                         and o is weights) else o
                                for o in output)
                    return new
                return hook

            self._handles.append(mod.register_forward_hook(make_hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def ordered(self):
        """Reduced results in layer order."""
        return [r for _, r in sorted(self.results, key=lambda kv: kv[0])]
