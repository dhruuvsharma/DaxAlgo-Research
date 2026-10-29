"""Restore the frozen seed-17 FDT-N weights from the exported ONNX into the PyTorch model.

Handles torch.onnx.export constant folding:
- Linear weights referenced via MatMul(x, folded_W) -> folded_W is [in, out] by MatMul
  semantics; PyTorch weight = folded_W.T
- Gemm(x, W, b) with named W/b -> original orientation, direct copy
- Cross-attention MHA exports q/k/v as 3 separate MatMuls (k, v share the memory-norm
  input; q consumes the query-norm output); in_proj_weight = cat([q, k, v], 0)
- Self-attention MHA exports one packed MatMul (stored [d, 3d]) or 3 decomposed pieces
- auction_queries / time_embedding / horizon sums are fully folded constants,
  recovered by unique shape (horizon_embedding absorbed into horizon_queries)

Gate: parity-fixture window must reproduce within 1e-4, plus random-window parity vs
ONNX Runtime (same tolerance as export.py's frozen gate).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from daxalgo_footprint_transformer.config import EncodingConfig, ModelConfig
from daxalgo_footprint_transformer.model import FootprintDistributionTransformer

DOC = Path(__file__).resolve().parent.parent / "doc" / "model"
OUT = Path(__file__).resolve().parent

m_onnx = onnx.load(str(DOC / "fdt-n.onnx"))
init = {t.name: numpy_helper.to_array(t) for t in m_onnx.graph.initializer}

model = FootprintDistributionTransformer(EncodingConfig(), ModelConfig())
state_keys = set(model.state_dict().keys())
sd: dict[str, torch.Tensor] = {}


def np2t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a).copy()).float()


def scope_of(tensor_name: str) -> str:
    """'/decoder_blocks.1/self_attention/MatMul_output_0' -> 'decoder_blocks.1.self_attention'.

    Attribute-called Sequentials produce doubled scopes ('/row_projection/row_projection.0/...')
    because the call site and the inner child both contribute a name; collapse those.
    """
    s = tensor_name.lstrip("/")
    parts = s.split("/")
    stem = parts[-1].rsplit("_output_", 1)[0]
    module_parts: list[str] = []
    for p in parts[:-1]:
        if module_parts and p.startswith(module_parts[-1] + "."):
            module_parts.pop()
        module_parts.append(p)
    if not module_parts:
        return stem
    return ".".join(module_parts)


# ---- 1. named initializers (biases, norm weights, layer scales, Gemm weights) --------
for k in state_keys:
    if k in init:
        sd[k] = np2t(init[k])
named_count = len(sd)
print(f"[1] named initializers mapped directly: {named_count}")

# ---- 2. folded MatMul / Gemm weights, keyed by node scope --------------------------
matmul_by_scope: dict[str, list] = defaultdict(list)  # scope -> [(init_name, input0, order)]
gemm_by_scope: dict[str, list] = defaultdict(list)
add_const_by_scope: dict[str, list[str]] = defaultdict(list)  # scope -> folded bias consts

nodes = list(m_onnx.graph.node)
for n in nodes:
    sc = scope_of(n.output[0])
    if n.op_type == "MatMul" and len(n.input) == 2 and n.input[1] in init \
            and n.input[1].startswith("onnx::"):
        matmul_by_scope[sc].append((n.input[1], n.input[0]))
    elif n.op_type == "Gemm" and len(n.input) >= 2 and n.input[1] in init \
            and n.input[1].startswith("onnx::"):
        trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
        gemm_by_scope[sc].append((n.input[1], trans_b))
    elif n.op_type == "Add" and len(n.input) == 2:
        for a, b in ((n.input[0], n.input[1]), (n.input[1], n.input[0])):
            if a in init and a.startswith("onnx::") and b not in init:
                add_const_by_scope[sc].append(a)
                break

for sc, pairs in sorted(matmul_by_scope.items()):
    if sc.endswith(".attention") or sc.endswith(".self_attention"):
        if len(pairs) == 3:
            # decomposed q/k/v. k and v share input[0] (memory path); q is the lone input.
            by_input: dict[str, list] = defaultdict(list)
            for w, x in pairs:
                by_input[x].append(w)
            if len(by_input) == 2:
                (shared_xs := [xs for xs in by_input.values() if len(xs) == 2])
                lone = [xs for xs in by_input.values() if len(xs) == 1]
                if len(shared_xs) == 1 and len(lone) == 1:
                    order = [w for w, _ in pairs]
                    # q = the lone one; k, v = shared ones in node order
                    q_w = lone[0][0]
                    shared_seq = [w for w, _ in pairs if w in shared_xs[0]]
                    wq, wk, wv = q_w, shared_seq[0], shared_seq[1]
                else:
                    order = [w for w, _ in pairs]
                    wq, wk, wv = order
            else:
                order = [w for w, _ in pairs]
                wq, wk, wv = order
            sd[f"{sc}.in_proj_weight"] = torch.cat(
                [np2t(init[w]).T for w in (wq, wk, wv)], dim=0
            )
            bias_key = f"{sc}.in_proj_bias"
            if bias_key not in sd:
                consts = add_const_by_scope.get(sc, [])
                if len(consts) == 3:
                    sd[bias_key] = torch.cat([np2t(init[c]) for c in consts], dim=0)
        elif len(pairs) == 1:
            # packed: stored [d, 3d] -> in_proj_weight [3d, d]
            sd[f"{sc}.in_proj_weight"] = np2t(init[pairs[0][0]]).T
        else:
            raise RuntimeError(f"attention scope {sc}: unexpected {len(pairs)} matmuls")
    else:
        if len(pairs) != 1:
            raise RuntimeError(f"scope {sc}: unexpected {len(pairs)} matmuls")
        key = f"{sc}.weight"
        if key in sd:
            raise RuntimeError(f"double assignment for {key}")
        sd[key] = np2t(init[pairs[0][0]]).T

for sc, items in sorted(gemm_by_scope.items()):
    for w, trans_b in items:
        key = f"{sc}.weight"
        arr = init[w]
        sd[key] = np2t(arr) if trans_b == 1 else np2t(arr).T

print(f"[2] folded weights mapped: {len(matmul_by_scope)} matmul scopes, "
      f"{len(gemm_by_scope)} gemm scopes")

# ---- 3. fully folded parameters by unique shape --------------------------------------
folded = {n: a for n, a in init.items() if n.startswith("onnx::") and a.dtype != bool}
by_shape: dict[tuple[int, ...], list[str]] = defaultdict(list)
for n, a in folded.items():
    by_shape[tuple(a.shape)].append(n)
used: set[str] = set()


def pick(shapes: list[tuple[int, ...]]) -> np.ndarray:
    options = [n for s in shapes for n in by_shape.get(s, []) if n not in used]
    if len(options) != 1:
        raise RuntimeError(f"expected unique folded init for {shapes}, got {options}")
    used.add(options[0])
    return init[options[0]]


# Each fully-folded parameter keeps its unsqueeze(0) leading dim -> unique shapes:
# auction_queries [4,64] -> (1,4,64); time_embedding [64,64] -> (1,64,64) (the only
# non-(64,64)-MatMul const of that rank); horizon sum [8,64] -> (1,8,64).
sd["auction_queries"] = np2t(pick([(1, 4, 64)]).reshape(4, 64))
sd["time_embedding"] = np2t(pick([(1, 64, 64)]).reshape(64, 64))
# horizon = horizon_queries + horizon_embedding(arange): fully folded sum [8,64].
# Functionally identical reconstruction: put the sum in queries, zeros in the embedding.
sd["horizon_queries"] = np2t(pick([(1, 8, 64)]).reshape(8, 64))
sd["horizon_embedding.weight"] = torch.zeros(8, 64)
print(f"[3] folded parameters recovered by shape: auction_queries, time_embedding, horizon sum")

# ---- 4. coverage check + load ---------------------------------------------------------
missing = state_keys - set(sd)
extra = set(sd) - state_keys
assert not missing, f"unmapped keys: {sorted(missing)}"
assert not extra, f"unexpected keys: {sorted(extra)}"
model.load_state_dict(sd, strict=True)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"loaded strict; parameter count = {n_params}")
assert n_params == 371861, "parameter count mismatch vs frozen metadata"

# ---- 5. parity gate -------------------------------------------------------------------
fixture = json.loads((DOC / "parity-fixture.json").read_text())
row_f = torch.tensor(fixture["row_features"]).unsqueeze(0)
row_m = torch.tensor(fixture["row_mask"]).unsqueeze(0)
bar_f = torch.tensor(fixture["bar_features"]).unsqueeze(0)
bar_m = torch.tensor(fixture["bar_mask"]).unsqueeze(0)
expected = torch.tensor(fixture["expected_quantiles"])

model.eval()
with torch.inference_mode():
    out = model(row_f, row_m, bar_f, bar_m)
err = (out - expected).abs().max().item()
print("fixture parity max abs err:", err)
assert err <= 1e-4, f"parity FAILED ({err})"

import onnxruntime as ort

sess = ort.InferenceSession(str(DOC / "fdt-n.onnx"), providers=["CPUExecutionProvider"])
rng = np.random.default_rng(11)
ok = True
for trial in range(3):
    R = int(rng.integers(5, 200))
    rf = rng.normal(size=(2, 64, R, 11)).astype(np.float32)
    rm = np.ones((2, 64, R), dtype=bool)
    bf = rng.normal(size=(2, 64, 23)).astype(np.float32)
    bm = np.ones((2, 64), dtype=bool)
    ref = sess.run(None, {"row_features": rf, "row_mask": rm, "bar_features": bf,
                          "bar_mask": bm})[0]
    with torch.inference_mode():
        got = model(torch.from_numpy(rf), torch.from_numpy(rm), torch.from_numpy(bf),
                    torch.from_numpy(bm)).numpy()
    d = np.abs(ref - got).max()
    print(f"random window R={R}: max err {d:.3e}")
    ok = ok and d <= 1e-4
assert ok, "random-window parity FAILED"

torch.save(model.state_dict(), OUT / "restored-seed17.pt")
print("\nOK: restored weights saved ->", OUT / "restored-seed17.pt")
