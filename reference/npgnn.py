"""NumPy re-implementation of the students' SpatioTemporalGNN (src/detection/graph_model.py).

Same layers, same PyTorch default initialisation, same Adam + MSE training loop
(batch 64, lr 1e-3, 50 epochs, dropout 0.2). Used because PyTorch cannot be
installed in this session. Weights will not be bit-identical to a PyTorch run.
"""
from __future__ import annotations
import numpy as np


class NpSTGNN:
    def __init__(self, F, W=12, Hd=64, H=12, dropout=0.2, seed=42):
        self.F, self.W, self.Hd, self.H, self.p = F, W, Hd, H, dropout
        rng = np.random.default_rng(seed)
        u = lambda fan_in, shape: rng.uniform(-1 / np.sqrt(fan_in), 1 / np.sqrt(fan_in), shape).astype(np.float64)
        # Conv1d(F -> Hd, k=3): torch weight (Hd, F, 3) -> stored as (3F, Hd), row index k*F + f
        wc = u(3 * F, (Hd, F, 3))
        self.P = {
            "Wc": np.transpose(wc, (2, 1, 0)).reshape(3 * F, Hd), "bc": u(3 * F, (Hd,)),
            "Ws": u(Hd, (Hd, Hd)).T.copy(), "bs": u(Hd, (Hd,)),
            "W1": u(W * Hd, (Hd, W * Hd)).T.copy(), "b1": u(W * Hd, (Hd,)),
            "W2": u(Hd, (F * H, Hd)).T.copy(), "b2": u(Hd, (F * H,)),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0
        self.rng = rng

    def _cols(self, xs):
        B, W, F = xs.shape
        pad = np.zeros((B, W + 2, F), dtype=xs.dtype)
        pad[:, 1:-1] = xs
        return np.concatenate([pad[:, k:k + W] for k in range(3)], axis=2)  # (B, W, 3F)

    def forward(self, x, A, train=False):
        P = self.P
        B = x.shape[0]
        xs = x @ A if A is not None else x
        cols = self._cols(xs)
        a1 = cols @ P["Wc"] + P["bc"]; h1 = np.maximum(a1, 0)
        a2 = h1 @ P["Ws"] + P["bs"]; h2 = np.maximum(a2, 0)
        flat = h2.reshape(B, -1)
        a3 = flat @ P["W1"] + P["b1"]; z = np.maximum(a3, 0)
        if train and self.p > 0:
            mask = (self.rng.random(z.shape) >= self.p) / (1 - self.p)
        else:
            mask = np.ones_like(z)
        zd = z * mask
        out = zd @ P["W2"] + P["b2"]
        cache = (x, A, cols, a1, h1, a2, h2, flat, a3, mask, zd)
        return out.reshape(B, self.H, self.F), cache

    def backward(self, dout, cache):
        x, A, cols, a1, h1, a2, h2, flat, a3, mask, zd = cache
        P = self.P
        B = x.shape[0]
        g = {}
        dout = dout.reshape(B, -1)
        g["W2"] = zd.T @ dout; g["b2"] = dout.sum(0)
        dz = (dout @ P["W2"].T) * mask * (a3 > 0)
        g["W1"] = flat.T @ dz; g["b1"] = dz.sum(0)
        dh2 = (dz @ P["W1"].T).reshape(h2.shape) * (a2 > 0)
        g["Ws"] = h1.reshape(-1, self.Hd).T @ dh2.reshape(-1, self.Hd); g["bs"] = dh2.sum((0, 1))
        dh1 = (dh2 @ P["Ws"].T) * (a1 > 0)
        g["Wc"] = cols.reshape(-1, cols.shape[2]).T @ dh1.reshape(-1, self.Hd); g["bc"] = dh1.sum((0, 1))
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.P:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            self.P[k] -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, Y, A, epochs=50, batch=64, lr=1e-3, seed=42, log=False):
        rng = np.random.default_rng(seed + 1)
        n = len(X); hist = []
        for ep in range(epochs):
            idx = rng.permutation(n); tot = 0.0
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                out, cache = self.forward(X[b], A, train=True)
                diff = out - Y[b]
                tot += float(np.mean(diff ** 2)) * len(b)
                self.adam(self.backward(2 * diff / diff.size, cache), lr=lr)
            hist.append(tot / n)
            if log and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  epoch {ep+1}: train MSE {hist[-1]:.4f}", flush=True)
        return hist

    def predict(self, X, A, batch=2048):
        return np.concatenate([self.forward(X[i:i + batch], A)[0] for i in range(0, len(X), batch)])

    def n_params(self):
        return sum(v.size for v in self.P.values())


def gradcheck():
    rng = np.random.default_rng(0)
    m = NpSTGNN(F=5, W=4, Hd=6, H=3, dropout=0.0, seed=1)
    x = rng.normal(size=(3, 4, 5)); y = rng.normal(size=(3, 3, 5)); A = rng.random((5, 5))
    out, c = m.forward(x, A, train=True)
    g = m.backward(2 * (out - y) / out.size, c)
    worst = 0
    for k, v in m.P.items():
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in v.shape)
            old = v[i]
            v[i] = old + 1e-6; lp = np.mean((m.forward(x, A)[0] - y) ** 2)
            v[i] = old - 1e-6; lm = np.mean((m.forward(x, A)[0] - y) ** 2)
            v[i] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[k][i]) / max(1e-8, abs(num) + abs(g[k][i])))
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradcheck())
    print("params F=43:", NpSTGNN(43).n_params())
