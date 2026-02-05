import torch

def extract_activations(model, tokenizer, input_ids, attention_mask=None, capture="down_proj_in", positions=None):
    activations = {}
    handles = []

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("Could not find model.model.layers")

    def _slice(x):
        if positions is None:
            return x
        return x[:, positions, :]

    if capture == "gate_act":
        def make_gate_hook(layer_idx, act_fn):
            def hook(module, inp, out):
                act = act_fn(out)
                activations[layer_idx] = _slice(act).detach().cpu()
            return hook

        for i, layer in enumerate(layers):
            mlp = layer.mlp
            h = mlp.gate_proj.register_forward_hook(make_gate_hook(i, mlp.act_fn))
            handles.append(h)

    elif capture == "down_proj_in":
        def make_downproj_prehook(layer_idx):
            def pre_hook(module, inputs):
                x_in = inputs[0]  # [B, T, D_intermediate]
                activations[layer_idx] = _slice(x_in).detach().cpu()
                return None
            return pre_hook

        for i, layer in enumerate(layers):
            mlp = layer.mlp
            h = mlp.down_proj.register_forward_pre_hook(make_downproj_prehook(i))
            handles.append(h)

    else:
        raise ValueError("capture must be 'gate_act' or 'down_proj_in'")

    try:
        with torch.inference_mode():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
    finally:
        for h in handles:
            h.remove()

    return dict(sorted(activations.items(), key=lambda kv: kv[0]))


def extract_hidden_states(model, input_ids, attention_mask=None):
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False
        )
    return tuple(h.detach().cpu() for h in out.hidden_states)
