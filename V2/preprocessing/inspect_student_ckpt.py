from pathlib import Path

import torch
import numpy as np


def main() -> None:
    v2_dir = Path(__file__).resolve().parents[1]
    model_path = v2_dir / "student_model.pt"

    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("student_state") or ckpt.get("state") or ckpt
    else:
        state = ckpt

    print("model_path:", model_path)
    print("ckpt_type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("top_level_keys:", list(ckpt.keys()))

    if isinstance(ckpt, dict) and "kmeans" in ckpt:
        km = ckpt["kmeans"]
        print("kmeans_type:", type(km))
        if hasattr(km, "cluster_centers_"):
            print("kmeans_centers_dtype:", km.cluster_centers_.dtype)
            print("kmeans_centers_shape:", km.cluster_centers_.shape)
            x64 = np.zeros((1, km.cluster_centers_.shape[1]), dtype=np.float64)
            x32 = np.zeros((1, km.cluster_centers_.shape[1]), dtype=np.float32)
            try:
                print("kmeans_predict_float64_ok:", km.predict(x64)[:1])
            except Exception as e:
                print("kmeans_predict_float64_err:", repr(e))
            try:
                print("kmeans_predict_float32_ok:", km.predict(x32)[:1])
            except Exception as e:
                print("kmeans_predict_float32_err:", repr(e))

    if not isinstance(state, dict):
        print("state_type:", type(state))
        return

    print("state_keys:", len(state))

    wanted = [
        "proj.0.weight",
        "proj.0.bias",
        "proj.2.weight",
        "proj.2.bias",
        "encoder.weight_ih_l0",
        "encoder.weight_hh_l0",
        "encoder.bias_ih_l0",
        "encoder.bias_hh_l0",
        "encoder.weight_ih_l0_reverse",
        "encoder.weight_hh_l0_reverse",
        "encoder.bias_ih_l0_reverse",
        "encoder.bias_hh_l0_reverse",
    ]
    for k in wanted:
        v = state.get(k)
        if v is not None:
            print(k, tuple(v.shape))

    # Print inferred dims
    w0 = state.get("proj.0.weight")
    w2 = state.get("proj.2.weight")
    enc_ih = state.get("encoder.weight_ih_l0")

    if w0 is not None:
        print("proj_in_features:", w0.shape[1])
        print("proj0_out_features:", w0.shape[0])

    if w2 is not None:
        print("proj2_out_features:", w2.shape[0])

    if enc_ih is not None:
        print("encoder_gate_rows:", enc_ih.shape[0])
        print("encoder_input_size:", enc_ih.shape[1])

    if isinstance(ckpt, dict) and "features" in ckpt:
        feats = ckpt["features"]
        print("features_in_ckpt:", type(feats), "len=", len(feats) if hasattr(feats, "__len__") else None)
        if isinstance(feats, (list, tuple)):
            print("features:", feats)


if __name__ == "__main__":
    main()
