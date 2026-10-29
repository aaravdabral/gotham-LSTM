

import os
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np



# Config

@dataclass
class Cfg:
    
    # architecture
    emsize: int = 64
    nhid: int = 128
    nlayers: int = 3
    max_vocab: int = 10000      
   
    dropouti: float = 0.65
    dropouth: float = 0.3
    dropout: float = 0.4
    dropoute: float = 0.1
    wdrop: float = 0.5
    
    alpha: float = 2.0
    beta: float = 1.0
    wdecay: float = 1.2e-6      

    # optimisation
    lr: float = 30.0
    clip: float = 0.25
    bptt: int = 35
    variable_bptt: bool = True
    nt_asgd: bool = True        # False -> plain SGD, lr/2 on stagnation 
    nonmono: int = 5
    batch_size: int = 20
    eval_batch_size: int = 10
    epochs: int = 30
    seed: int = 1111
    dtype: str = "float32"      # gradcheck uses float64

    @staticmethod
    def paper():
        return Cfg(emsize=400, nhid=1150, max_vocab=None, bptt=70, batch_size=80, epochs=750)


def count_params(model):
    return sum(p.size for p in model.params.values())



# Data




class Corpus:
    def __init__(self, splits, max_vocab=None):
        toks = {k: [w for line in v for w in (line.split() + ["<eos>"])] for k, v in splits.items()}
        counts = {}
        for w in toks["train"]:
            counts[w] = counts.get(w, 0) + 1
        ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
        specials = ["<unk>", "<eos>"]
        ranked = [w for w in ranked if w not in specials]
        if max_vocab:
            ranked = ranked[: max_vocab - len(specials)]
        self.i2w = specials + ranked
        self.w2i = {w: i for i, w in enumerate(self.i2w)}
        unk = self.w2i["<unk>"]
        enc = lambda ws: np.array([self.w2i.get(w, unk) for w in ws], dtype=np.int64)
        self.train, self.valid, self.test = enc(toks["train"]), enc(toks["valid"]), enc(toks["test"])


def save_corpus(c, path="data_cache/wt2.npz"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, train=c.train, valid=c.valid, test=c.test, i2w=np.array(c.i2w))


def load_corpus(path="data_cache/wt2.npz"):
    d = np.load(path)
    return SimpleNamespace(train=d["train"], valid=d["valid"], test=d["test"], i2w=list(d["i2w"]))


def batchify(data, bsz):
    """1-D token stream -> (T, bsz); each column is a contiguous stretch of text."""
    nb = len(data) // bsz
    return np.ascontiguousarray(data[: nb * bsz].reshape(bsz, nb).T)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    return source[i:i + seq_len], source[i + 1:i + 1 + seq_len].reshape(-1)



# LSTM layer: forward and backward pass

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def lstm_forward(x, h0, c0, Wx, Wh, b):
    
    T, B, nin = x.shape
    H = h0.shape[1]
    xw = (x.reshape(T * B, nin) @ Wx + b).reshape(T, B, 4 * H)   # input part for all steps at once
    hs = np.empty((T, B, H), x.dtype)
    cs = np.empty((T, B, H), x.dtype)
    tcs = np.empty((T, B, H), x.dtype)
    gates = np.empty((T, B, 4 * H), x.dtype)                      # stores i, f, g, o (post-activation)
    h, c = h0, c0
    for t in range(T):
        z = xw[t] + h @ Wh
        i = sigmoid(z[:, :H])
        f = sigmoid(z[:, H:2 * H])
        g = np.tanh(z[:, 2 * H:3 * H])
        o = sigmoid(z[:, 3 * H:])
        c = f * c + i * g                                         # c_t = i*g + f*c_{t-1}
        tc = np.tanh(c)
        h = o * tc
        gates[t, :, :H], gates[t, :, H:2 * H], gates[t, :, 2 * H:3 * H], gates[t, :, 3 * H:] = i, f, g, o
        cs[t], tcs[t], hs[t] = c, tc, h
    return hs, (x, h0, c0, Wx, Wh, gates, cs, tcs, hs)


def lstm_backward(dh_out, cache):
    
    x, h0, c0, Wx, Wh, gates, cs, tcs, hs = cache
    T, B, H = hs.shape
    dz_all = np.empty((T, B, 4 * H), x.dtype)
    dWh = np.zeros_like(Wh)
    dh_next = np.zeros((B, H), x.dtype)
    dc_next = np.zeros((B, H), x.dtype)
    for t in range(T - 1, -1, -1):
        i, f, g, o = gates[t, :, :H], gates[t, :, H:2 * H], gates[t, :, 2 * H:3 * H], gates[t, :, 3 * H:]
        dh = dh_out[t] + dh_next
        do = dh * tcs[t]
        dc = dc_next + dh * o * (1.0 - tcs[t] ** 2)
        c_prev = cs[t - 1] if t > 0 else c0
        h_prev = hs[t - 1] if t > 0 else h0
        di, dg, df = dc * g, dc * i, dc * c_prev
        dc_next = dc * f
        dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
        dz_all[t] = dz
        dWh += h_prev.T @ dz
        dh_next = dz @ Wh.T
    dz2 = dz_all.reshape(T * B, 4 * H)
    dWx = x.reshape(T * B, -1).T @ dz2
    db = dz2.sum(0)
    dx = (dz2 @ Wx.T).reshape(x.shape)
    return dx, dWx, dWh, db



# Model

class AWDLSTM:
    def __init__(self, ntoken, cfg, rng=None):
        self.cfg, self.V = cfg, ntoken
        self.dt = np.dtype(cfg.dtype)
        rng = rng or np.random.default_rng(cfg.seed)
        L, H, Em = cfg.nlayers, cfg.nhid, cfg.emsize
        P = {"E": rng.uniform(-0.1, 0.1, (ntoken, Em)), "bdec": np.zeros(ntoken)}
        for l in range(L):
            nin = Em if l == 0 else H                 
            nout = Em if l == L - 1 else H            
            k = 1.0 / np.sqrt(nout)
            P[f"Wx{l}"] = rng.uniform(-k, k, (nin, 4 * nout))
            P[f"Wh{l}"] = rng.uniform(-k, k, (nout, 4 * nout))
            P[f"b{l}"] = rng.uniform(-k, k, (4 * nout,))
        self.params = {k: v.astype(self.dt) for k, v in P.items()}

    def init_hidden(self, bsz):
        L, H, Em = self.cfg.nlayers, self.cfg.nhid, self.cfg.emsize
        sizes = [Em if l == L - 1 else H for l in range(L)]
        return [(np.zeros((bsz, s), self.dt), np.zeros((bsz, s), self.dt)) for s in sizes]

    def sample_masks(self, T, B, rng):
        """One set of dropout masks per forward/backward pass (inverted dropout: scaled by 1/(1-p))."""
        c, dt = self.cfg, self.dt

        def bern(shape, p):
            return ((rng.random(shape) >= p) / (1.0 - p)).astype(dt) if p > 0 else np.ones(shape, dt)

        m = {"emb": bern((self.V, 1), c.dropoute),          
             "in": bern((1, B, c.emsize), c.dropouti),      
             "out": bern((1, B, c.emsize), c.dropout)}
        for l in range(c.nlayers - 1):
            m[f"h{l}"] = bern((1, B, c.nhid), c.dropouth)
        for l in range(c.nlayers):                           
            m[f"wh{l}"] = bern(self.params[f"Wh{l}"].shape, c.wdrop)
        return m

    #  forward pass

    def forward(self, tokens, hidden, masks=None):
        """masks=None -> eval mode (no dropout)."""
        P, L = self.params, self.cfg.nlayers
        E = P["E"]
        E_eff = E * masks["emb"] if masks else E
        x = E_eff[tokens]
        if masks:
            x = x * masks["in"]
        layers, new_hidden = [], []
        for l in range(L):
            Wh = P[f"Wh{l}"] * masks[f"wh{l}"] if masks else P[f"Wh{l}"]
            hs, lc = lstm_forward(x, hidden[l][0], hidden[l][1], P[f"Wx{l}"], Wh, P[f"b{l}"])
            layers.append(lc)
            new_hidden.append((hs[-1].copy(), lc[6][-1].copy()))
            if l < L - 1:
                x = hs * masks[f"h{l}"] if masks else hs
        raw_last = hs
        xd = raw_last * masks["out"] if masks else raw_last
        T, B, Em = xd.shape
        logits = xd.reshape(T * B, Em) @ E.T + P["bdec"]     # weight tying: decoder uses the UNmasked E
        return logits, new_hidden, raw_last, xd, layers

    #  loss + gradients

    def loss_and_grads(self, tokens, targets, hidden, masks):
        """Returns (total_loss, cross_entropy, grads, new_hidden). total = CE + AR + TAR."""
        c, P, L = self.cfg, self.params, self.cfg.nlayers
        logits, new_hidden, raw, xd, layers = self.forward(tokens, hidden, masks)
        T, B, Em = xd.shape
        N = T * B
        shifted = logits - logits.max(1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(1))
        ce = float((lse - shifted[np.arange(N), targets]).mean())
        dlog = np.exp(shifted - lse[:, None])                # softmax probabilities
        dlog[np.arange(N), targets] -= 1.0
        dlog /= N

        grads = {"bdec": dlog.sum(0), "E": dlog.T @ xd.reshape(N, Em)}   # decoder side of the tied matrix
        dxd = (dlog @ P["E"]).reshape(T, B, Em)
        total = ce
        if c.alpha:                                          # AR: alpha * mean((m*h)^2)
            total += c.alpha * float(np.mean(xd ** 2))
            dxd = dxd + c.alpha * 2.0 * xd / xd.size
        dh = dxd * masks["out"]
        if c.beta and T > 1:                                 # TAR: beta * mean((h_t - h_{t+1})^2), raw output
            diff = raw[1:] - raw[:-1]
            total += c.beta * float(np.mean(diff ** 2))
            d = c.beta * 2.0 * diff / diff.size
            dh[1:] += d
            dh[:-1] -= d

        for l in range(L - 1, -1, -1):
            dx, dWx, dWh_eff, db = lstm_backward(dh, layers[l])
            grads[f"Wx{l}"], grads[f"b{l}"] = dWx, db
            grads[f"Wh{l}"] = dWh_eff * masks[f"wh{l}"]       # only kept weights get gradient
            dh = dx * masks[f"h{l - 1}"] if l > 0 else dx
        demb = dh * masks["in"]
        dE_eff = np.zeros_like(P["E"])
        np.add.at(dE_eff, tokens, demb)                       # scatter-add into used rows
        grads["E"] = grads["E"] + dE_eff * masks["emb"]
        return total, ce, grads, new_hidden



# Gradient checker 

def gradcheck(model, tokens, targets, rng, n_probe=6, eps=1e-4, scale=3.0):
   
    assert model.dt == np.float64, "build the model with Cfg(dtype='float64') for gradcheck"
    orig = {k: p.copy() for k, p in model.params.items()}
    for p in model.params.values():
        p *= scale
    try:
        B = tokens.shape[1]
        hidden = [(0.1 * rng.standard_normal(h.shape), 0.1 * rng.standard_normal(c.shape))
                  for h, c in model.init_hidden(B)]
        masks = model.sample_masks(tokens.shape[0], B, rng)
        _, _, grads, _ = model.loss_and_grads(tokens, targets, hidden, masks)
        worst = {}
        for k, p in model.params.items():
            flat = p.reshape(-1)
            idx = list(rng.choice(flat.size, min(n_probe, flat.size), replace=False))
            if k == "E":                                      # also probe rows that were actually looked up
                idx += [int(r) * p.shape[1] + int(rng.integers(p.shape[1])) for r in tokens.reshape(-1)[:3]]
            errs = []
            for j in idx:
                old = flat[j]
                flat[j] = old + eps
                lp = model.loss_and_grads(tokens, targets, hidden, masks)[0]
                flat[j] = old - eps
                lm = model.loss_and_grads(tokens, targets, hidden, masks)[0]
                flat[j] = old
                num, ana = (lp - lm) / (2 * eps), grads[k].reshape(-1)[j]
                errs.append(abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
            worst[k] = max(errs)
        return worst
    finally:
        for k, p in model.params.items():
            p[...] = orig[k]



# Optimisers 

def clip_global_norm(grads, max_norm):
    norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if norm > max_norm:
        for g in grads.values():
            g *= max_norm / (norm + 1e-6)
    return norm


class SGD:
    def __init__(self, params, lr, wd=0.0):
        self.params, self.lr, self.wd = params, lr, wd

    def step(self, grads, lr_scale=1.0):
        for k, p in self.params.items():
            p -= self.lr * lr_scale * (grads[k] + self.wd * p)


class ASGD(SGD):
    """SGD step, then a running average of the iterates (t0=0, lambd=0, i.e. average from the start)."""

    def __init__(self, params, lr, wd=0.0):
        super().__init__(params, lr, wd)
        self.ax = {k: p.copy() for k, p in params.items()}
        self.n = 0

    def step(self, grads, lr_scale=1.0):
        super().step(grads, lr_scale)
        self.n += 1
        mu = 1.0 / self.n                                     
        for k, p in self.params.items():
            self.ax[k] += (p - self.ax[k]) * mu


@contextmanager
def averaged(model, opt):
    
    saved = {k: p.copy() for k, p in model.params.items()}
    for k in model.params:
        model.params[k][...] = opt.ax[k]
    try:
        yield
    finally:
        for k in model.params:
            model.params[k][...] = saved[k]



# Training 

def evaluate(model, data, bsz, bptt=70):
    """Mean cross-entropy in nats/token. Perplexity = exp(loss)."""
    hidden = model.init_hidden(bsz)
    total = 0.0
    for i in range(0, data.shape[0] - 1, bptt):
        x, y = get_batch(data, i, bptt)
        logits, hidden, _, _, _ = model.forward(x, hidden, masks=None)
        shifted = logits - logits.max(1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(1))
        total += len(x) * float((lse - shifted[np.arange(len(y)), y]).mean())
    return total / (data.shape[0] - 1)


def ppl(loss):
    return float(np.exp(loss))


def train_epoch(model, data, opt, cfg, rng, max_steps=None):
    hidden = model.init_hidden(data.shape[1])
    total, n, i = 0.0, 0, 0
    while i < data.shape[0] - 2:
        if cfg.variable_bptt:                                         
            bptt = cfg.bptt if rng.random() < 0.95 else cfg.bptt / 2.0
            seq_len = max(5, int(rng.normal(bptt, 5)))                
        else:
            seq_len = cfg.bptt
        x, y = get_batch(data, i, seq_len)
        masks = model.sample_masks(x.shape[0], x.shape[1], rng)
        _, ce, grads, hidden = model.loss_and_grads(x, y, hidden, masks)
        clip_global_norm(grads, cfg.clip)
        opt.step(grads, lr_scale=seq_len / cfg.bptt)                  # LR rescale for the sampled length
        total += ce
        n += 1
        i += seq_len
        if max_steps and n >= max_steps:
            break
    return total / n


def save_params(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **model.params)


def load_params(model, path):
    d = np.load(path)
    for k in model.params:
        model.params[k][...] = d[k]


def fit(model, train_data, val_data, cfg, ckpt="ckpt/best.npz", finetune=False, log=print, max_steps=None):
    
    rng = np.random.default_rng(cfg.seed + (1 if finetune else 0))
    asgd = finetune
    opt = (ASGD if finetune else SGD)(model.params, cfg.lr, cfg.wdecay)
    logs, best = [], float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_data, opt, cfg, rng, max_steps)
        with (averaged(model, opt) if asgd and opt.n > 0 else nullcontext()):
            val = evaluate(model, val_data, cfg.eval_batch_size)
            if val < best:
                best = val
                save_params(model, ckpt)
        log(f"ep {epoch:3d} | {'ASGD' if asgd else 'SGD '} | train ppl {ppl(tr):8.2f} | "
            f"val ppl {ppl(val):8.2f} | best {ppl(best):8.2f} | {time.time() - t0:5.0f}s")
        if len(logs) > cfg.nonmono and val > min(logs[:-cfg.nonmono]):
            if finetune:
                log("fine-tune: criterion hit, stopping")
                break
            if cfg.nt_asgd and not asgd:
                log(">>> NT-ASGD triggered: switching to averaging (T = now)")
                opt, asgd = ASGD(model.params, cfg.lr, cfg.wdecay), True
            elif not cfg.nt_asgd:
                opt.lr /= 2                                   # ablation: plain SGD, halve LR
        logs.append(val)
    return best