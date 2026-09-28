"""Compact 2-channel multi-label ink-segmentation U-Net, defined twice with identical layer names:

  * build_model()      -> Keras 3 functional model (the deliverable / export source)
  * make_torch_model() -> PyTorch twin used for FAST training on the M1 GPU (MPS).
                          (TF has no GPU on this Mac and the Keras-torch backend is ~10x slower
                          than raw torch here, so training runs in torch and the weights are
                          ported 1:1 into the Keras model by layer name: port_torch_to_keras().)

Channel 0 = "printed ink present", channel 1 = "handwriting ink present" (independent sigmoids,
so a pixel where handwriting crosses printed text can be 1 in both).

Interface (fixed, the app is written against it):
  input  float32 [1, 256, 256, 3] RGB, raw 0..255 (the /255 rescaling is inside the model)
  output float32 [1, 256, 256, 2] sigmoid probabilities (print, handwriting)

Design: depthwise-separable convs (cheap on phone CPU/GPU/ANE), BatchNorm + ReLU (BN folds into
the conv at conversion time), 4 downsamplings to 16x16 with a dilated context block, nearest
upsampling + 1x1 projections in the decoder, final 1x1 conv + sigmoid.
Stride-2 convs use TF "same" padding semantics (pad right/bottom); the torch twin pads the same
way explicitly so both frameworks compute identical results.
"""
from __future__ import annotations

TILE = 256
NUM_CHANNELS = 2
WIDTHS = (24, 40, 64, 112, 192)
BN_EPS = 1e-3


def arch(widths=WIDTHS):
    """List of ops describing the graph (shared by both implementations); tensors by name."""
    ops = []
    w = widths

    def cbr(name, src, cin, cout, k=3, stride=1):
        ops.append(("conv", name, src, cin, cout, k, stride, 1))
        ops.append(("bn", name + "_bn", name, cout))
        ops.append(("relu", name + "_relu", name + "_bn"))
        return name + "_relu"

    def sep(name, src, cin, cout, dilation=1, stride=1):
        ops.append(("dw", name + "_dw", src, cin, 3, stride, dilation))
        ops.append(("bn", name + "_dwbn", name + "_dw", cin))
        ops.append(("relu", name + "_dwrelu", name + "_dwbn"))
        ops.append(("conv", name + "_pw", name + "_dwrelu", cin, cout, 1, 1, 1))
        ops.append(("bn", name + "_pwbn", name + "_pw", cout))
        ops.append(("relu", name + "_out", name + "_pwbn"))
        return name + "_out"

    def res(name, src, cin, cout, dilations=(1, 1)):
        sc = src
        if cin != cout:
            ops.append(("conv", name + "_sc", src, cin, cout, 1, 1, 1))
            ops.append(("bn", name + "_scbn", name + "_sc", cout))
            sc = name + "_scbn"
        y, c = src, cin
        for i, d in enumerate(dilations):
            y = sep(f"{name}_s{i}", y, c, cout, dilation=d)
            c = cout
        ops.append(("add", name + "_add", y, sc))
        return name + "_add"

    def up(name, src, skip, cin, cskip, cout):
        ops.append(("up", name + "_up", src))
        ops.append(("conv", name + "_proj", name + "_up", cin, cout, 1, 1, 1))
        ops.append(("bn", name + "_projbn", name + "_proj", cout))
        ops.append(("relu", name + "_projrelu", name + "_projbn"))
        ops.append(("cat", name + "_cat", name + "_projrelu", skip))
        return res(name + "_res", name + "_cat", cout + cskip, cout)

    e0 = cbr("stem", "rescaled", 3, w[0])
    e0 = sep("e0", e0, w[0], w[0])
    e1 = sep("down1", e0, w[0], w[1], stride=2)
    e1 = res("e1", e1, w[1], w[1])
    e2 = sep("down2", e1, w[1], w[2], stride=2)
    e2 = res("e2", e2, w[2], w[2])
    e3 = sep("down3", e2, w[2], w[3], stride=2)
    e3 = res("e3", e3, w[3], w[3])
    b = sep("down4", e3, w[3], w[4], stride=2)
    b = res("b1", b, w[4], w[4], dilations=(1, 2))
    b = res("b2", b, w[4], w[4], dilations=(4, 1))
    d3 = up("d3", b, e3, w[4], w[3], w[3])
    d2 = up("d2", d3, e2, w[3], w[2], w[2])
    d1 = up("d1", d2, e1, w[2], w[1], w[1])
    d0 = up("d0", d1, e0, w[1], w[0], w[0])
    d0 = sep("head", d0, w[0], w[0])
    ops.append(("conv_bias", "logits", d0, w[0], NUM_CHANNELS, 1, 1, 1))
    ops.append(("sigmoid", "probs", "logits"))
    return ops


# ------------------------------------------------------------------------------------------------
# Keras
# ------------------------------------------------------------------------------------------------
def build_model(widths=WIDTHS, tile: int | None = TILE):
    import keras
    from keras import layers

    inp = keras.Input(shape=(tile, tile, 3), name="input")
    t = {"rescaled": layers.Rescaling(1.0 / 255.0, name="rescale")(inp)}
    for op in arch(widths):
        kind, name = op[0], op[1]
        if kind in ("conv", "conv_bias"):
            _, _, src, cin, cout, k, stride, dil = op
            t[name] = layers.Conv2D(cout, k, strides=stride, padding="same", use_bias=(kind == "conv_bias"), name=name)(t[src])
        elif kind == "dw":
            _, _, src, c, k, stride, dil = op
            t[name] = layers.DepthwiseConv2D(k, strides=stride, padding="same", dilation_rate=dil, use_bias=False, name=name)(t[src])
        elif kind == "bn":
            t[name] = layers.BatchNormalization(epsilon=BN_EPS, name=name)(t[op[2]])
        elif kind == "relu":
            t[name] = layers.ReLU(name=name)(t[op[2]])
        elif kind == "add":
            t[name] = layers.Add(name=name)([t[op[2]], t[op[3]]])
        elif kind == "up":
            t[name] = layers.UpSampling2D(2, interpolation="nearest", name=name)(t[op[2]])
        elif kind == "cat":
            t[name] = layers.Concatenate(name=name)([t[op[2]], t[op[3]]])
        elif kind == "sigmoid":
            t[name] = layers.Activation("sigmoid", name=name)(t[op[2]])
    return keras.Model(inp, t["probs"], name="ink_segmenter_v2")


# ------------------------------------------------------------------------------------------------
# PyTorch twin (training only)
# ------------------------------------------------------------------------------------------------
def make_torch_model(widths=WIDTHS):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def same_pad(x, k, stride, dil):
        if k == 1:
            return x
        eff = (k - 1) * dil + 1
        size = x.shape[-1]
        if stride == 1:
            total = eff - 1
        else:  # TF SAME: total = max(eff - stride, 0) for size % stride == 0; extra on right/bottom
            total = max(eff - stride, 0) if size % stride == 0 else max(eff - size % stride, 0)
        p0 = total // 2
        return F.pad(x, (p0, total - p0, p0, total - p0))

    class InkSegTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.ops = arch(widths)
            self.m = nn.ModuleDict()
            for op in self.ops:
                kind, name = op[0], op[1]
                if kind in ("conv", "conv_bias"):
                    _, _, src, cin, cout, k, stride, dil = op
                    self.m[name] = nn.Conv2d(cin, cout, k, stride=stride, bias=(kind == "conv_bias"))
                elif kind == "dw":
                    _, _, src, c, k, stride, dil = op
                    self.m[name] = nn.Conv2d(c, c, k, stride=stride, dilation=dil, groups=c, bias=False)
                elif kind == "bn":
                    self.m[name] = nn.BatchNorm2d(op[3], eps=BN_EPS, momentum=0.01)

        def forward(self, x):  # NCHW raw 0..255 -> logits NCHW (sigmoid applied outside)
            t = {"rescaled": x * (1.0 / 255.0)}
            for op in self.ops:
                kind, name = op[0], op[1]
                if kind in ("conv", "conv_bias", "dw"):
                    k, stride, dil = (op[5], op[6], op[7]) if kind != "dw" else (op[4], op[5], op[6])
                    t[name] = self.m[name](same_pad(t[op[2]], k, stride, dil))
                elif kind == "bn":
                    t[name] = self.m[name](t[op[2]])
                elif kind == "relu":
                    t[name] = F.relu(t[op[2]])
                elif kind == "add":
                    t[name] = t[op[2]] + t[op[3]]
                elif kind == "up":
                    t[name] = F.interpolate(t[op[2]], scale_factor=2, mode="nearest")
                elif kind == "cat":
                    t[name] = torch.cat([t[op[2]], t[op[3]]], 1)
                elif kind == "sigmoid":
                    return t[op[2]]
            raise RuntimeError("no output")

    return InkSegTorch()


def port_torch_to_keras(tmodel, kmodel):
    """Copy weights from the torch twin into the Keras model by layer name. Returns #layers set."""
    import numpy as np

    sd = {k: v.detach().cpu().float().numpy() for k, v in tmodel.state_dict().items()}
    n = 0
    for layer in kmodel.layers:
        if not layer.weights:
            continue
        name, cls = layer.name, layer.__class__.__name__
        if cls == "Conv2D":
            ws = [sd[f"m.{name}.weight"].transpose(2, 3, 1, 0)]  # (out,in,kh,kw)->(kh,kw,in,out)
            if layer.use_bias:
                ws.append(sd[f"m.{name}.bias"])
        elif cls == "DepthwiseConv2D":
            ws = [sd[f"m.{name}.weight"].transpose(2, 3, 0, 1)]  # (C,1,kh,kw)->(kh,kw,C,1)
        elif cls == "BatchNormalization":
            ws = [sd[f"m.{name}.weight"], sd[f"m.{name}.bias"], sd[f"m.{name}.running_mean"], sd[f"m.{name}.running_var"]]
        else:
            raise ValueError(f"unexpected weighted layer {name} ({cls})")
        layer.set_weights([np.ascontiguousarray(w, np.float32) for w in ws])
        n += 1
    return n


if __name__ == "__main__":
    import os
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    m = build_model()
    print("keras params:", m.count_params())
    tm = make_torch_model()
    print("torch params:", sum(p.numel() for p in tm.parameters()) + sum(b.numel() for n_, b in tm.named_buffers() if "running" in n_))
