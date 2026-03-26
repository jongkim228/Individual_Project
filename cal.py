def EUCLIDEX(m, n):
    if m == 0:
        return (n, 0, 1)
    else:
        r = n - (n // m) * m
        (d, x, y) = EUCLIDEX(r, m)
        s = y - (n // m) * x
        return (d, s, x)

d, x, y = EUCLIDEX(121, 16800)
print(f"d = {d}")
print(f"x = {x}  (이게 바로 역원 d)")
print(f"검증: 121 × {x} mod 16800 = {(121 * x) % 16800}")
