"""Re-encode Mix-256G's dense Q5_K / Q6_K tensors as Q8_0 into a raw blob + manifest.

Maya's dense path accepts F32/F16/BF16/Q8_0 only, and its routed-expert path already
handles IQ2_S/IQ3_XXS/IQ3_S/MXFP4.  Only the 203 Q6_K + 43 Q5_K dense tensors are
converted here, so the rest of the artifact is untouched and nothing is rewritten in place.

Dequant is a literal transcription of ggml's dequantize_row_q6_K / dequantize_row_q5_K
(including get_scale_min_k4) and the Q8_0 encode of quantize_row_q8_0_ref
(roundf = round half away from zero).
"""
import json, pathlib, struct, sys
import numpy as np

QK = 256
BLK = {13: 176, 14: 210}
TYPES = {13: 'Q5_K', 14: 'Q6_K'}

_KV_FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def _kv_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def _kv_skip(f, t):
    """Skip one typed KV value (same type map / array handling as pack_gguf._val)."""
    if t in _KV_FMT:
        f.read(struct.calcsize('<' + _KV_FMT[t]))
    elif t == 8:
        _kv_str(f)
    elif t == 9:
        at, n = struct.unpack('<IQ', f.read(12))
        if at in _KV_FMT:
            f.read(n * struct.calcsize('<' + _KV_FMT[at]))
        else:
            for _ in range(n):
                _kv_skip(f, at)
    else:
        raise ValueError('unknown GGUF KV type %d' % t)


def read_gguf_alignment(path):
    """general.alignment declared in the GGUF header, or the spec default 32."""
    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError('%s: not GGUF' % path)
        f.read(4)
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        align = 32
        for _ in range(n_kv):
            k = _kv_str(f)
            t = struct.unpack('<I', f.read(4))[0]
            if k == 'general.alignment' and t == 4:
                align = struct.unpack('<I', f.read(4))[0]
            else:
                _kv_skip(f, t)
        del n_tensors
    return align


def require_align32(path):
    """Data-start math in this script assumes 32; pack_gguf honours other values,
    so a non-32 shard would make dense reads silently land at wrong offsets."""
    a = read_gguf_alignment(path)
    if a != 32:
        sys.exit('%s: general.alignment=%d but this requant assumes 32'
                 % (path.name, a))
    return a


def dequant_q6k(b):
    """b: uint8 [n, 210] -> float32 [n, 256] (ggml dequantize_row_q6_K)."""
    n = b.shape[0]
    ql = b[:, 0:128].astype(np.int32)
    qh = b[:, 128:192].astype(np.int32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.int32)
    d = np.ascontiguousarray(b[:, 208:210]).view(np.float16).astype(np.float32).reshape(n)
    y = np.empty((n, 256), dtype=np.float32)
    for h in range(2):
        q = ql[:, h * 64:(h + 1) * 64]
        u = qh[:, h * 32:(h + 1) * 32]
        s = sc[:, h * 8:(h + 1) * 8]
        base = h * 128
        for l in range(32):
            i2 = l // 16
            q1 = (q[:, l] & 0xF) | (((u[:, l] >> 0) & 3) << 4)
            q2 = (q[:, l + 32] & 0xF) | (((u[:, l] >> 2) & 3) << 4)
            q3 = (q[:, l] >> 4) | (((u[:, l] >> 4) & 3) << 4)
            q4 = (q[:, l + 32] >> 4) | (((u[:, l] >> 6) & 3) << 4)
            y[:, base + l] = d * s[:, i2 + 0] * (q1 - 32)
            y[:, base + l + 32] = d * s[:, i2 + 2] * (q2 - 32)
            y[:, base + l + 64] = d * s[:, i2 + 4] * (q3 - 32)
            y[:, base + l + 96] = d * s[:, i2 + 6] * (q4 - 32)
    return y


def scale_min_k4(j, q):
    if j < 4:
        return q[:, j] & 63, q[:, j + 4] & 63
    return ((q[:, j + 4] & 0xF) | ((q[:, j - 4] >> 6) << 4),
            ((q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)))


def dequant_q5k(b):
    """b: uint8 [n, 176] -> float32 [n, 256] (ggml dequantize_row_q5_K; qh is not advanced)."""
    n = b.shape[0]
    d = np.ascontiguousarray(b[:, 0:2]).view(np.float16).astype(np.float32).reshape(n)
    dmin = np.ascontiguousarray(b[:, 2:4]).view(np.float16).astype(np.float32).reshape(n)
    scales = b[:, 4:16].astype(np.int32)
    qh = b[:, 16:48].astype(np.int32)
    ql = b[:, 48:176].astype(np.int32)
    y = np.empty((n, 256), dtype=np.float32)
    u1, u2 = 1, 2
    for g in range(4):
        sc1, m1v = scale_min_k4(2 * g + 0, scales)
        sc2, m2v = scale_min_k4(2 * g + 1, scales)
        d1, m1 = d * sc1, dmin * m1v
        d2, m2 = d * sc2, dmin * m2v
        lo = ql[:, g * 32:(g + 1) * 32]
        for l in range(32):
            h1 = 16 * ((qh[:, l] & u1) != 0)
            h2 = 16 * ((qh[:, l] & u2) != 0)
            y[:, g * 64 + l] = d1 * ((lo[:, l] & 0xF) + h1) - m1
            y[:, g * 64 + 32 + l] = d2 * ((lo[:, l] >> 4) + h2) - m2
        u1 <<= 2
        u2 <<= 2
    return y


def quant_q8(x):
    """x: float32 [n, 256] -> uint8 [n, 272]: 8 block_q8_0 (f16 d + 32 int8)."""
    n = x.shape[0]
    out = np.empty((n, 8 * 34), dtype=np.uint8)
    for blk in range(8):
        v = x[:, blk * 32:(blk + 1) * 32]
        amax = np.abs(v).max(axis=1)
        d = (amax / 127.0).astype(np.float32)
        idv = np.where(d != 0, 1.0 / np.where(d == 0, np.float32(1), d), np.float32(0)).astype(np.float32)
        q = np.floor(np.abs(v) * idv[:, None] + 0.5).astype(np.int32)
        q = np.sign(v).astype(np.int32) * np.minimum(q, 127)
        base = blk * 34
        out[:, base:base + 2] = np.ascontiguousarray(d.astype(np.float16)).view(np.uint8).reshape(n, 2)
        out[:, base + 2:base + 34] = np.ascontiguousarray(q.astype(np.int8)).view(np.uint8)
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=pathlib.Path, required=True,
                        help='Directory containing the input GGUF shards')
    parser.add_argument('--index', type=pathlib.Path, required=True,
                        help='Shard metadata JSON (file, metadata_bytes, tensors)')
    parser.add_argument('--out', type=pathlib.Path, required=True,
                        help='NEW directory for mixq8.bin and mixq8.json')
    args = parser.parse_args()
    SRC, OUT = args.model.resolve(), args.out.resolve()
    if OUT.exists():
        parser.error('output already exists; refusing to overwrite a previous conversion')
    sj = json.loads(args.index.read_text())
    for shard in sj:
        if pathlib.Path(shard['file']).name != shard['file']:
            parser.error('index shard filenames must be plain basenames')
    starts = {}
    for s in sj:
        align = 32
        starts[s['file']] = (s['metadata_bytes'] + align - 1) // align * align
    want = [(s['file'], t) for s in sj for t in s['tensors'] if t['type'] in TYPES]
    if not want:
        sys.exit('no Q5_K/Q6_K tensors')
    for fname in dict.fromkeys(f for f, _ in want):
        require_align32(SRC / fname)   # loud gate: real header, not an assumption
    OUT.mkdir(parents=True, exist_ok=False)
    handles, man, off, done = {}, {}, 0, 0
    with open(OUT / 'mixq8.bin', 'wb') as blob:
        for fname, t in want:
            fh = handles.setdefault(fname, open(SRC / fname, 'rb'))
            blk = BLK[t['type']]
            nblk = t['elements'] // QK
            fh.seek(starts[fname] + t['offset'])
            raw = fh.read(nblk * blk)
            if len(raw) != nblk * blk:
                sys.exit('short read %s %s' % (fname, t['name']))
            b = np.frombuffer(raw, dtype=np.uint8).reshape(nblk, blk)
            x = dequant_q6k(b) if t['type'] == 14 else dequant_q5k(b)
            enc = quant_q8(x).tobytes()
            pad = (-off) % 256
            if pad:
                blob.write(b'\0' * pad)
                off += pad
            blob.write(enc)
            man[t['name']] = {'off': off, 'bytes': len(enc),
                              'shape': list(reversed(t['dims'])), 'src_type': TYPES[t['type']]}
            off += len(enc)
            done += 1
            if done % 20 == 0:
                print('converted %d/%d' % (done, len(want)), flush=True)
    (OUT / 'mixq8.json').write_text(json.dumps(man, indent=1))
    print('DONE %d tensors, %.2f GiB blob of %d requested' % (done, off / 2**30, len(want)))


if __name__ == '__main__':
    main()
