import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
import struct

LLF = r'E:\workspace\yllm\test\tinyllama-1.1b-chat-v1.0.Q4_K_M.llf'

# attn_k row 0: Q4_K, 8 blocks of 144 bytes. Extract first 2 blocks + golden decode
f = open(LLF, 'rb')
f.seek(36880384 + 2367488)
krow = f.read(1152)
f.close()
krow = np.frombuffer(krow, dtype=np.uint8)
k_ref = gguf.dequantize(krow.copy().view(np.uint8), QT.Q4_K)

# attn_v row 0: Q6_K, 8 blocks of 210. block5 has negative d (sign bug test)
f = open(LLF, 'rb')
f.seek(36880384 + 2662400)
vrow = f.read(430080 // 256)  # 1680 bytes
f.close()
vrow = np.frombuffer(vrow, dtype=np.uint8)
v_ref = gguf.dequantize(vrow.copy().view(np.uint8), QT.Q6_K)

def c_arr(name, data, per_line=12):
    lines = []
    for i in range(0, len(data), per_line):
        lines.append('    ' + ', '.join('0x%02x' % b for b in data[i:i+per_line]))
    return 'static const uint8_t %s[] = {\n%s\n};\n' % (name, ',\n'.join(lines))

def f_arr(name, data, per_line=6):
    vals = []
    for v in data:
        if v == 0:
            vals.append('0.0f')
        else:
            vals.append('%.9gf' % float(v))
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append('    ' + ', '.join(vals[i:i+per_line]))
    return 'static const float %s[] = {\n%s\n};\n' % (name, ',\n'.join(lines))

print('/* Q4_K attn_k row0: first 2 blocks (288 bytes) + golden decode (512 floats) */')
print(c_arr('K4_RAW', krow[:288].tolist()))
print(f_arr('K4_REF', k_ref[:512]))

print('/* Q6_K attn_v row0: block5 only (210 bytes, negative d) + golden decode (256 floats) */')
print(c_arr('V6_RAW', vrow[5*210:6*210].tolist()))
print(f_arr('V6_REF', v_ref[5*256:6*256]))

# embedding row 15043 golden (Q4_K)
f = open(LLF, 'rb')
f.seek(16384)  # layer0 emb offset
embrow = f.read(1152)
f.close()
emb = gguf.dequantize(np.frombuffer(embrow, dtype=np.uint8).copy().view(np.uint8), QT.Q4_K)
print('/* embedding token 15043: first 8 floats */')
print(f_arr('EMB_REF', emb[:8]))
