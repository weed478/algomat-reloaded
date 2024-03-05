import numpy as np
import subprocess


if __name__ == '__main__':
    subprocess.run(['cabal', 'build', '--ghc-options=-O2'], check=True, capture_output=True)

    for k in [2, 3, 4, 5, 6, 7, 8]:
        n = 2**k
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        c_expected = a @ b

        print(f'Testing n = {n}...')

        res = subprocess.run(['cabal', 'run', '--ghc-options=-O2'], input=(f'{n} ' + ' '.join(a.flatten().astype(str).tolist()) + ' ' + ' '.join(b.flatten().astype(str).tolist())).encode(), capture_output=True, check=True)
        c = np.array(list(map(float, res.stdout.decode().strip().split()))).reshape(n, n)

        assert np.allclose(c, c_expected), f'Failed for n = {n}'
