#!/usr/bin/env python3
"""gguf -> safetensors 转换工具: 供 vLLM 等原生加载器使用。

用法:
    python3 tests/gguf2safetensors.py <model.gguf> [outdir] [f16|bf16] [outname]

参数:
    model.gguf   源模型(必需)
    outdir       输出目录(默认 safetensors-out)
    dtype        输出精度: f16(默认) / bf16(vLLM CPU 推荐, 范围大不易 NaN)
    outname      输出文件名(默认 <gguf名去后缀>.safetensors)
    --config-only 只从 gguf 元数据生成 config.json(跳过 safetensors 转换)

依赖: pip install gguf safetensors torch numpy
支持张量类型: F32 / F16 / Q4_K / Q6_K(qwen3-8B Q4_K_M 实测)。
Q4_K/Q6_K 反量化按块向量化(numpy), 按行分块控制峰值内存。
转换正确性已与 yllm 引擎逐值对比验证(embed/gate/q_proj 等)。

示例:
    # 转换到 qwen3-st/ 目录, 输出 Qwen3-8B-Q4_K_M.safetensors(f16)
    python3 tests/gguf2safetensors.py Qwen3-8B-Q4_K_M.gguf qwen3-st
    # 输出到 models/, bf16, 自定义文件名 qwen3-8b.safetensors
    python3 tests/gguf2safetensors.py Qwen3-8B-Q4_K_M.gguf models bf16 qwen3-8b.safetensors
"""
import json
import numpy as np
import os
import sys
import time


_ARCH_CLASS = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen3": "Qwen3ForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
}


def _gguf_field(reader, name):
    """读取 gguf 元数据字段值(字符串/数字), 不存在返回 None。"""
    f = reader.fields.get(name)
    if f is None:
        return None
    try:
        data = f.parts[f.data[0]]
        if data.dtype.kind in ("S", "O", "U") or data.dtype == np.uint8:
            s = data.tobytes().decode("utf-8", "ignore").split("\x00")[0]
            if s and not s.replace(".", "").replace("-", "").isdigit():
                return s
        v = data[0]
        if isinstance(v, (bytes, bytearray)):
            s = bytes(v).decode("utf-8", "ignore").split("\x00")[0]
            return s
        return float(v) if np.issubdtype(data.dtype, np.floating) else int(v)
    except Exception:
        return None


def _gguf_str(reader, name):
    """读取标量字符串字段。"""
    f = reader.fields.get(name)
    if f is None:
        return None
    try:
        return f.parts[4].tobytes().decode("utf-8", "ignore").split("\x00")[0]
    except Exception:
        return None


def _gguf_arr_str(reader, name):
    """读取字符串数组字段(gguf-python parts 布局: [0]=总长 [1]=名 [2]=类型 [3]=元素类型 [4]=数量 [5..]=每元素 2 part)。"""
    f = reader.fields.get(name)
    if f is None:
        return None
    try:
        parts = f.parts
        count = int.from_bytes(parts[4].tobytes(), "little")
        vals = []
        i = 5
        for _ in range(count):
            n = int.from_bytes(parts[i].tobytes(), "little")
            vals.append(parts[i + 1].tobytes().decode("utf-8", "ignore") if n else "")
            i += 2
        return vals
    except Exception:
        return None


def _gguf_arr_i32(reader, name):
    """读取 int32 数组字段。"""
    f = reader.fields.get(name)
    if f is None:
        return None
    try:
        parts = f.parts
        count = int.from_bytes(parts[4].tobytes(), "little")
        return [int.from_bytes(parts[5 + i].tobytes(), "little") for i in range(count)]
    except Exception:
        return None


def _gguf_i64(reader, name):
    f = reader.fields.get(name)
    if f is None:
        return None
    try:
        return int.from_bytes(f.parts[4].tobytes(), "little")
    except Exception:
        return None


def _token_spec(content, lstrip=False, normalized=False, single_word=False):
    return {"content": content, "lstrip": lstrip, "normalized": normalized,
            "rstrip": False, "single_word": single_word}


def write_tokenizer(outdir, reader):
    """从 gguf 提取 tokenizer 数据, 生成 HF 格式 tokenizer.json + tokenizer_config.json + vocab.json + merges.txt。"""
    tokens = _gguf_arr_str(reader, "tokenizer.ggml.tokens")
    merges = _gguf_arr_str(reader, "tokenizer.ggml.merges")
    if not tokens or not merges:
        print("skip tokenizer: no tokens/merges in gguf")
        return
    types = _gguf_arr_i32(reader, "tokenizer.ggml.token_type") or [1] * len(tokens)
    bos_id = _gguf_i64(reader, "tokenizer.ggml.bos_token_id")
    eos_id = _gguf_i64(reader, "tokenizer.ggml.eos_token_id")
    pad_id = _gguf_i64(reader, "tokenizer.ggml.padding_token_id")
    unk_id = None
    for i, t in enumerate(types):
        if t == 2:
            unk_id = i
            break
    if unk_id is None:
        unk_id = bos_id
    add_bos = bool(_gguf_i64(reader, "tokenizer.ggml.add_bos_token"))
    model = _gguf_str(reader, "tokenizer.ggml.model") or "gpt2"
    pre = _gguf_str(reader, "tokenizer.ggml.pre") or "bytelevel"
    chat_template = _gguf_str(reader, "tokenizer.chat_template")

    arch = _gguf_str(reader, "general.architecture") or "llama"
    cls = "Qwen2Tokenizer" if arch.startswith("qwen") else "LlamaTokenizer"

    # --- tokenizer.json(经 tokenizers 库构造, 保证格式正确) ---
    import tokenizers
    from tokenizers import Tokenizer as TK
    from tokenizers import models as tk_models, pre_tokenizers as tk_pre, decoders as tk_dec
    vocab = {p: i for i, p in enumerate(tokens)}
    merges_tuples = [tuple(m.split(" ")) for m in merges if m]
    bpe = tk_models.BPE(vocab=vocab, merges=merges_tuples,
                        unk_token=tokens[unk_id] if unk_id is not None else None)
    tk = TK(bpe)
    if (pre or "").lower() in ("bytelevel", "gpt2", "qwen2", "qwen3"):
        tk.pre_tokenizer = tk_pre.ByteLevel(add_prefix_space=False, use_regex=True)
        tk.decoder = tk_dec.ByteLevel()
    else:
        tk.pre_tokenizer = tk_pre.Metaspace(replacement="\u2581")
        tk.decoder = tk_dec.Metaspace(replacement="\u2581")
    # 特殊 token 元数据(type != 1)
    added = []
    for i, t in enumerate(types):
        if t != 1:
            tk.add_special_tokens([tokens[i]])
            added.append({"id": i, "content": tokens[i], "lstrip": False,
                          "normalized": False, "rstrip": False, "single_word": False,
                          "special": t == 3})
    tk.save(os.path.join(outdir, "tokenizer.json"))
    if added:
        import json as _json
        with open(os.path.join(outdir, "tokenizer.json"), "r") as f:
            tj = _json.load(f)
        tj["added_tokens"] = added
        with open(os.path.join(outdir, "tokenizer.json"), "w") as f:
            _json.dump(tj, f, ensure_ascii=False, indent=2)

    # --- vocab.json + merges.txt(慢 tokenizer 兼容) ---
    with open(os.path.join(outdir, "vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(outdir, "merges.txt"), "w") as f:
        f.write("\n".join(merges) + "\n")

    # --- tokenizer_config.json ---
    cfg = {
        "add_bos_token": add_bos,
        "add_eos_token": False,
        "bos_token": _token_spec(tokens[bos_id]) if bos_id is not None else None,
        "clean_up_tokenization_spaces": False,
        "eos_token": _token_spec(tokens[eos_id]) if eos_id is not None else None,
        "pad_token": _token_spec(tokens[pad_id]) if pad_id is not None else None,
        "unk_token": _token_spec(tokens[unk_id]) if unk_id is not None else None,
        "model_max_length": 1000000000000000019884624838656,
        "tokenizer_class": cls,
        "tokenizer_config": {
            "add_prefix_space": True,
            "clean_up_tokenization_spaces": False,
            "model_input_names": ["input_ids", "attention_mask"],
            "tokenizer_class": cls,
        },
        "model_input_names": ["input_ids", "attention_mask"],
    }
    if chat_template:
        cfg["chat_template"] = chat_template
    with open(os.path.join(outdir, "tokenizer_config.json"), "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"tokenizer -> {outdir}/tokenizer.json + vocab.json + merges.txt + tokenizer_config.json "
          f"({len(tokens)} tokens, {len(merges)} merges)")


def write_config_json(outdir, reader):
    """从 gguf 元数据生成 HF config.json(与 safetensors 同目录)。

    至少覆盖 convert_safetensors 需要的字段:
    num_attention_heads / num_key_value_heads / head_dim /
    rms_norm_eps / rope_theta / model_type。
    """
    arch = _gguf_field(reader, "general.architecture")
    if not arch:
        return
    p = arch + "."
    cfg = {
        "architectures": [_ARCH_CLASS.get(arch, arch.capitalize() + "ForCausalLM")],
        "model_type": arch,
    }
    mapping = [
        ("hidden_size", "embedding_length"),
        ("num_hidden_layers", "block_count"),
        ("num_attention_heads", "attention.head_count"),
        ("num_key_value_heads", "attention.head_count_kv"),
        ("head_dim", "attention.key_length"),
        ("rms_norm_eps", "attention.layer_norm_rms_epsilon"),
        ("rope_theta", "rope.freq_base"),
        ("vocab_size", "vocab_size"),
        ("intermediate_size", "feed_forward_length"),
        ("max_position_embeddings", "context_length"),
    ]
    for hf_key, g_key in mapping:
        v = _gguf_field(reader, p + g_key)
        if v is None and g_key == "attention.key_length":
            v = _gguf_field(reader, p + "rope.dimension_count")
        if v is not None:
            if isinstance(v, float):
                v = round(v, 7)
            cfg[hf_key] = v
    # head_dim 兜底: hidden / heads(标准 GQA)
    if "head_dim" not in cfg and "hidden_size" in cfg and "num_attention_heads" in cfg:
        cfg["head_dim"] = int(cfg["hidden_size"] // cfg["num_attention_heads"])
    # 补充: 从张量形状兜底 vocab_size
    if "vocab_size" not in cfg:
        for t in reader.tensors:
            if t.name in ("token_embd.weight", "output.weight"):
                cfg["vocab_size"] = int(max(t.shape[0], t.shape[1]))
                break
    path = os.path.join(outdir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"config -> {path} ({len(cfg)} keys)")


def f16_bytes(b):
    return np.frombuffer(bytes(b), dtype=np.float16).astype(np.float32)


# 元素 -> 字节索引映射(常量)
_E = np.arange(256)
_G = _E >> 5                       # q4k 组
_Q4_BYTE = (_G >> 1) * 32 + (_E & 31)
_Q4_HIGH = (_G & 1) == 1
_Q6_HALF = _E >> 7
_Q6_QUAD = (_E & 0x7F) >> 5
_Q6_LL = _E & 31
_Q6_QL = _Q6_HALF * 64 + (_Q6_QUAD & 1) * 32 + _Q6_LL
_Q6_QH = _Q6_HALF * 32 + _Q6_LL      # 相对 qh 区(128)的偏移, 0..63
_Q6_SC = _Q6_HALF * 8 + _Q6_QUAD * 2 + (_Q6_LL >> 4)


def gsm_k4(scm):
    """(..., 12) -> (..., 8) scale 与 min"""
    sc = np.empty(scm.shape[:-1] + (8,), np.uint8)
    m = np.empty(scm.shape[:-1] + (8,), np.uint8)
    sc[..., 0:4] = scm[..., 0:4] & 63
    m[..., 0:4] = scm[..., 4:8] & 63
    sc[..., 4:8] = (scm[..., 8:12] & 0xF) | ((scm[..., 0:4] >> 6) << 4)
    m[..., 4:8] = (scm[..., 8:12] >> 4) | ((scm[..., 4:8] >> 6) << 4)
    return sc, m


def dequant_q4k(blk):
    """(nblk, 144) -> (nblk, 256) f16"""
    d = f16_bytes(blk[:, :, 0:2].tobytes()).reshape(blk.shape[0], blk.shape[1])
    dmin = f16_bytes(blk[:, :, 2:4].tobytes()).reshape(blk.shape[0], blk.shape[1])
    sc, m = gsm_k4(blk[:, :, 4:16])
    q = np.take(blk[:, :, 16:144], _Q4_BYTE, axis=2)
    nib = np.where(_Q4_HIGH, (q >> 4) & 0xF, q & 0xF)
    # sc/nib 需转 float: uint8 乘法会溢出(53*15=795 > 255)
    s = sc[:, :, _G].astype(np.float32)
    mi = m[:, :, _G].astype(np.float32)
    out = d[:, :, None] * (s * nib.astype(np.float32)) - dmin[:, :, None] * mi
    return out.astype(np.float16)


def dequant_q6k(blk):
    """(nblk, 210) -> (nblk, 256) f16
    布局: ql[0:128] qh[128:192] sc[192:208] d@208 (f16, 2字节)"""
    d = f16_bytes(blk[:, :, 208:210].tobytes()).reshape(blk.shape[0], blk.shape[1])
    ql = np.take(blk[:, :, 0:128], _Q6_QL, axis=2)
    qh = np.take(blk[:, :, 128:192], _Q6_QH, axis=2)
    bits = np.where((_Q6_QUAD & 2) != 0, (ql >> 4) & 0xF, ql & 0xF) | (((qh >> (_Q6_QUAD * 2)) & 3) << 4)
    q = bits.astype(np.int16) - 32
    sc = np.frombuffer(blk[:, :, 192:208].tobytes(), dtype=np.int8).reshape(blk.shape[0], blk.shape[1], 16)
    s = np.take(sc, _Q6_SC, axis=2).astype(np.int16)
    return (d[:, :, None] * s * q).astype(np.float16)


def dequant_tensor(data, dt, rows, cols, blk_bytes):
    """整张量分块反量化 -> (rows, cols) f32。data 为 (rows, blk_bytes*cols/256) uint8 视图。"""
    nb = data.shape[1] // blk_bytes
    blk = data.reshape(rows, nb, blk_bytes)
    out = np.empty((rows, cols), np.float32)
    CH = 128
    for r0 in range(0, rows, CH):
        r1 = min(r0 + CH, rows)
        b = blk[r0:r1]
        if dt == "Q4_K":
            out[r0:r1] = dequant_q4k(b).reshape(r1 - r0, cols)
        else:
            out[r0:r1] = dequant_q6k(b).reshape(r1 - r0, cols)
    return out


NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
BLK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
BLK_BYTES = {12: 144, 14: 210}  # Q4_K / Q6_K


def hf_name(name):
    if name in NAME_MAP:
        return NAME_MAP[name]
    p = name.split(".")
    if p[0] == "blk" and len(p) >= 3:
        suf = ".".join(p[2:])
        if suf in BLK_MAP:
            return f"model.layers.{int(p[1])}.{BLK_MAP[suf]}"
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "safetensors-out"
    os.makedirs(outdir, exist_ok=True)
    out_name = sys.argv[4] if len(sys.argv) > 4 else os.path.basename(src)[:-5] + ".safetensors"
    out_path = os.path.join(outdir, out_name)

    import torch
    import safetensors.torch
    from gguf import GGUFReader

    out_dtype = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] in ("f16", "bf16") else "f16"
    torch_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    reader = GGUFReader(src)
    write_config_json(outdir, reader)
    write_tokenizer(outdir, reader)
    if len(sys.argv) > 5 and sys.argv[5] == "--config-only":
        print(f"config only -> {os.path.join(outdir, 'config.json')} (safetensors 已存在, 跳过转换)")
        return 0
    print(f"output: {out_path} (dtype {out_dtype})")
    state = {}
    n = 0
    for t in reader.tensors:
        hf = hf_name(t.name)
        if hf is None:
            continue
        t0 = time.time()
        rows = t.data.shape[0]
        if len(t.shape) == 1:
            arr = np.frombuffer(t.data.tobytes(), dtype=np.float32).reshape(rows)
            state[hf] = torch.from_numpy(arr).to(torch_dtype)
        else:
            cols = int(t.shape[0])
            dt = int(t.tensor_type)
            if dt in BLK_BYTES:
                arr = dequant_tensor(t.data, "Q4_K" if dt == 12 else "Q6_K", rows, cols, BLK_BYTES[dt])
            elif dt == 0:  # F32
                arr = t.data.view(np.float32).reshape(rows, cols)
            elif dt == 1:  # F16
                arr = t.data.view(np.float16).astype(np.float32).reshape(rows, cols)
            else:
                print(f"skip {t.name}: unsupported type {dt}")
                continue
            state[hf] = torch.from_numpy(arr).to(torch_dtype)
        n += 1
        print(f"[{n}] {hf}: {tuple(t.shape)} {out_dtype} ({state[hf].numel() * 2 / 1e6:.0f}MB, {time.time() - t0:.0f}s)", flush=True)

    safetensors.torch.save_file(state, out_path)
    print(f"DONE -> {out_path} ({n} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
