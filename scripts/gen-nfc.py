"""Writes llm/src/main/resources/mlx/llm/nfc.txt from Python's unicodedata: canonical decompositions (full),
canonical combining classes, and the primary composites. `python3 scripts/gen-nfc.py`"""
import sys, unicodedata as u

def canon(c):
    d = u.decomposition(chr(c))
    if not d or d.startswith('<'): return None
    return [int(x, 16) for x in d.split()]

def full(c):
    d = canon(c)
    if d is None: return [c]
    out = []
    for x in d: out += full(x)
    return out

decomp, ccc, comp = [], [], []
for c in range(0x110000):
    if 0xAC00 <= c <= 0xD7A3: continue               # Hangul: by formula
    k = u.combining(chr(c))
    if k: ccc += [c, k]
    d = canon(c)
    if d is not None:
        decomp.append((c, full(c)))
        if len(d) == 2 and u.normalize('NFC', chr(d[0]) + chr(d[1])) == chr(c):
            comp += [d[0], d[1], c]

def b36(n):
    d = '0123456789abcdefghijklmnopqrstuvwxyz'; s = ''
    while True:
        s = d[n % 36] + s; n //= 36
        if n == 0: return s

flat = []
for c, d in decomp: flat += [c, len(d)] + d
out = 'llm/src/main/resources/mlx/llm/nfc.txt'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    f.write(f"# Unicode {u.unidata_version} canonical data, by scripts/gen-nfc.py from Python's unicodedata; numbers in base 36\n")
    f.write("# decompositions: code point, length, code points\n" + ' '.join(map(b36, flat)) + "\n")
    f.write("# combining classes: code point, class\n" + ' '.join(map(b36, ccc)) + "\n")
    f.write("# primary composites: first, second, composed\n" + ' '.join(map(b36, comp)) + "\n")
print(len(decomp), len(ccc) // 2, len(comp) // 3, u.unidata_version)
