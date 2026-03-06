MOD = 10**9 + 7

k = int(input())

if k == 1:
    print(1, 1)
else:
    i = k - 2
    n = i // 2 + 2

    n_mod = n % MOD
    n3_mod = pow(n, 3, MOD)

    if i % 2 == 0:
        print(n_mod, n3_mod)
    else:
        print(n3_mod, n_mod)
