"""Integer ring (post hoc play): the trained PHS models run with no floating point in the dynamics, in the spirit of
s0fractal/genesis omega_v2/src/math.rs (integer phase circle, sine LUT, xorshift, wrap by bitmask).

  phase     theta in Z / 2^A (A state bits); the circle closes by the mask, so the winding number is an exact integer;
            a phase step is rounded half up (rnd = 1, default) or floored (rnd = 0: a -1/2 LSB drift per step, a seam);
            Python only, for the rounding controls (rounding.py): rnd = 2 stochastic rounding (a uniform sub-LSB
            offset from a second xorshift stream, unbiased in expectation), rnd = 3 error feedback (each rotor keeps
            its sub-LSB remainder and adds it to the next step, unbiased over time and deterministic)
  trig      sin/cos through a LUT of 2^B entries indexed by the top B bits of theta, values in Q(F)
  softmax   exp through a LUT on [-EXP_RANGE, 0] in steps of 2^-EXP_STEP, then an integer division
  numbers   momenta, forces, the Cayley rotation G, 1/m, sqrt(m), drives in Q(F); products are floored shifts
            (prnd = 1, Python only: products, the force sums, the rotation and the noise scale round to nearest;
            the softmax division, the symmetrisation and the divisions by d still floor)
  noise     xorshift32 per sequence; Irwin-Hall (sum of 4 uniforms of 16 bits, centred, scaled by sqrt(3)): variance
            1 up to the discrete centring, the rounded factor and the floors; a shared contract of the three runtimes,
            not exactly Gaussian and not the torch OU noise
  readout   logits = W z with z in Q(F) (float only after the dynamics, for argmax and the bars)

Two knobs: B, the resolution the trigonometry sees, and A, the resolution the state keeps. With A = B the state has no
sub-step memory below one LUT step, as in genesis' 8-bit phase.
Every operation is an integer operation on values below 2^53, so the JS port in web/intring.js reproduces it bit for
bit (check with web/check_intring.js). The bound is measured, not assumed: with ring.track = True every product and
accumulation records an upper bound of its magnitude in ring.peak (max|a| max|b|, times the length of a sum), and
write_case refuses a case whose peak reaches 2^53."""
from __future__ import annotations

import hashlib
import json
import math
import sys

import numpy as np

EXP_RANGE, EXP_STEP = 16, 10          # exp LUT: arguments -16..0 in steps of 2^-10


class IntRing:
    def __init__(self, export: dict, A: int = 24, B: int = 16, F: int = 16, rnd: int = 1, prnd: int = 0):
        assert B <= A <= 30 and B <= 16 and F <= 16 and rnd in (0, 1, 2, 3) and prnd in (0, 1)
        self.A, self.B, self.F = A, B, F
        self.rnd = rnd                    # phase increment: 1 rounds half up (default), 0 floors (biased by -1/2 LSB)
        self.prnd = prnd                  # 1: products and sums round to nearest instead of floor
        self.stats = None                 # a dict to collect phase-increment statistics in run()
        self.N, self.d, self.S = export["N"], export["d"], export["S"]
        self.mode = export["mode"]
        one = 1 << F
        q = lambda v: np.round(np.asarray(v, dtype=np.float64) * one).astype(np.int64)  # noqa: E731
        k = np.arange(1 << B)
        self.SIN = q(np.sin(2 * np.pi * k / (1 << B)))
        self.EXP = q(np.exp(-np.arange((EXP_RANGE << EXP_STEP) + 1) / (1 << EXP_STEP)))
        h = export["h"]
        self.h2 = q(0.5 * h)                                            # h/2 in Q(F)
        self.c1 = q(math.exp(-export["gamma"] * h))
        self.c2u = q(math.sqrt(1 - math.exp(-2 * export["gamma"] * h)))  # c2 = c2u * sigma
        mass = np.asarray(export["mass"])
        self.invm = q(1 / mass)[:, None]
        self.sqm = q(np.sqrt(mass))[:, None]
        self.G = q(np.asarray(export["G"]).reshape(self.N, self.N))
        self.kappa = q(export["kappa"])
        self.g_lat, self.g_pin, self.m_pin = q(export["g_lat"]), q(export["g_pin"]), export["m_pin"]
        self.lattice = export["g_lat"] > 0 or export["g_pin"] > 0
        self.ema = q(export.get("ema", 0.3))
        two_pi_over = (1 << A) / (2 * math.pi)                          # radians -> phase units
        self.phi = np.round((np.asarray(export["phi"]).reshape(-1, self.N, self.d) % (2 * math.pi)) * two_pi_over
                            ).astype(np.int64) & ((1 << A) - 1)
        self.u = q(np.asarray(export["u"]).reshape(-1, self.N, self.d))
        self.psi = (np.round((np.asarray(export["psi"]) % (2 * math.pi)) * two_pi_over).astype(np.int64)
                    & ((1 << A) - 1))[:, None]
        self.W, self.b = q(np.asarray(export["W"]).reshape(len(export["b"]), -1)), q(export["b"])
        # phase increment: dtheta[phase units] = (h/2 * invm * P) [Q(F) radians] * 2^A / (2 pi) / 2^F, via a Q20 factor
        self.ph_mul = int(round((1 << A) / (2 * math.pi) * 2 ** -F * (1 << 20)))   # Q20 multiplier
        self.track, self.peak = False, 0

    def _bound(self, a, b, k=1, add=0):
        """Record an upper bound of |a * b| summed over k terms (+ |add|), as exact Python integers."""
        if self.track:
            m = lambda v: int(np.max(np.abs(np.asarray(v, dtype=np.int64)))) if np.size(v) else 0  # noqa: E731
            self.peak = max(self.peak, k * m(a) * m(b) + m(add))

    # --- integer helpers (floor semantics everywhere unless prnd) ---
    def sh(self, v, k):
        return (v + (1 << (k - 1))) >> k if self.prnd else v >> k

    def mul(self, a, b):
        self._bound(a, b)
        return self.sh(a * b, self.F)

    def sin(self, th):
        return self.SIN[(th >> (self.A - self.B)) & ((1 << self.B) - 1)]

    def cos(self, th):
        return self.SIN[((th >> (self.A - self.B)) + (1 << (self.B - 2))) & ((1 << self.B) - 1)]

    def rad_to_phase(self, x):
        """x radians in Q(F) -> phase units, floored (rnd = 0) or rounded half up (rnd = 1). Flooring a step smaller
        than one LSB gives 0 or -1, a drift of -1/2 LSB per step that turns every phase together."""
        self._bound(x, self.ph_mul, add=1 << 20)
        y = x * self.ph_mul
        if self.rnd == 2:                                               # stochastic: + uniform 20-bit offset
            self._sr ^= self._sr << np.uint32(13)
            self._sr ^= self._sr >> np.uint32(17)
            self._sr ^= self._sr << np.uint32(5)
            inc = (y + (self._sr >> np.uint32(12)).astype(np.int64)) >> 20
        elif self.rnd == 3:                                             # error feedback: carry the remainder
            t = y + self._res
            inc = t >> 20
            self._res = t & ((1 << 20) - 1)
        else:
            inc = (y + (self.rnd << 19)) >> 20
        if self.stats is not None:
            err = inc - y / (1 << 20)
            st = self.stats
            st["n"] = st.get("n", 0) + err.size
            st["err"] = st.get("err", 0.0) + float(err.sum())
            st["err2"] = st.get("err2", 0.0) + float((err * err).sum())
            st["frozen"] = st.get("frozen", 0) + int(((inc == 0) & (y != 0)).sum())
            st["sub"] = st.get("sub", 0) + int((np.abs(y) < (1 << 20)).sum())
        return inc

    def force(self, th, gpin):
        F, N, d = self.F, self.N, self.d
        c, s = self.cos(th), self.sin(th)                               # (n, N, d) Q(F)
        self._bound(c, c, 2 * d)
        C = self.sh(np.einsum("nik,njk->nij", c, c) + np.einsum("nik,njk->nij", s, s), F)
        C = C // d                                                      # Q(F), d = 2: exact shift
        z = self.mul(self.kappa, C)                                     # kappa C in Q(F)
        z = z - z.max(axis=2, keepdims=True)                            # <= 0
        idx = np.minimum((-z) >> (F - EXP_STEP), EXP_RANGE << EXP_STEP)
        e = self.EXP[idx]
        self._bound(e, 1 << F, N)
        A = (e << F) // e.sum(axis=2, keepdims=True)                    # softmax rows, Q(F)
        As = (A + A.transpose(0, 2, 1)) >> 1
        self._bound(As, c, N)
        ac = self.sh(np.einsum("nij,njk->nik", As, c), F)
        as_ = self.sh(np.einsum("nij,njk->nik", As, s), F)
        ds = self.mul(s, ac) - self.mul(c, as_)
        self._bound(self.kappa, ds)
        f = -self.sh(self.kappa * ds, F) // d
        if self.lattice:
            fwd, bwd = np.roll(th, -1, axis=1) - th, th - np.roll(th, 1, axis=1)
            f = f + self.mul(self.g_lat, self.sin(fwd) - self.sin(bwd))
            f = f - self.m_pin * self.mul(gpin, self.sin(self.m_pin * (th - self.psi)))
        return f

    def rotate(self, P):
        self._bound(self.G, P, self.N)
        return self.sh(np.einsum("ij,njk->nik", self.G, P), self.F)

    def run(self, x, sigma_q, seed=1, record=False):
        """x: (n, T) tokens; sigma_q: (n,) noise in Q(F). Returns logits (float), theta, P, per-token winding."""
        n, T = x.shape
        F, N, d, mask = self.F, self.N, self.d, (1 << self.A) - 1
        th = np.zeros((n, N, d), np.int64)
        P = np.zeros((n, N, d), np.int64)
        self._bound(self.c2u, sigma_q)
        c2 = self.sh(self.c2u * sigma_q, F)                                # Q(F), per sequence
        c2 = c2[:, None, None]
        # xorshift32 state per (sequence, node, channel): (seed * 2654435761 + i * 40503 + e * 2246822519 + 1) mod 2^32, odd
        i = np.arange(n, dtype=np.uint64)[:, None, None]
        e = np.arange(N * d, dtype=np.uint64).reshape(1, N, d)
        rng = ((np.uint64(seed) * np.uint64(2654435761) + i * np.uint64(40503) + e * np.uint64(2246822519) + np.uint64(1))
               & np.uint64(0xFFFFFFFF)).astype(np.uint32) | np.uint32(1)
        # second stream for stochastic rounding (rnd = 2), apart from the physical noise; remainders for rnd = 3
        self._sr = ((np.uint64(seed) * np.uint64(2246822519) + i * np.uint64(2654435761) + e * np.uint64(40503)
                     + np.uint64(7)) & np.uint64(0xFFFFFFFF)).astype(np.uint32) | np.uint32(1)
        self._res = np.full((n, N, d), 1 << 19, np.int64)
        Dhat = np.full(n, 1 << F, np.int64)
        winding = []
        for t in range(T):
            gpin = np.full((n, 1, 1), self.g_pin, np.int64)
            if self.mode == "homeo":
                D = self.cos(self.m_pin * th).sum(axis=(1, 2)) // (N * d)
                Dhat = D if t == 0 else Dhat + self.mul(self.ema, D - Dhat)
                self._bound(self.g_pin, 1 << F)
                gpin = ((self.g_pin << F) // np.maximum(Dhat, int(0.3 * (1 << F))))[:, None, None]
            tok = x[:, t]
            phi, u = self.phi[tok], self.u[tok]
            f = self.force(th, gpin) - self.mul(u, self.sin(th - phi))
            for _ in range(self.S):
                P = P + self.mul(self.h2, f)
                P = self.rotate(P)
                th = (th + self.rad_to_phase(self.mul(self.h2, self.mul(self.invm, P)))) & mask
                # O step: Irwin-Hall Gaussian from xorshift32, variance 1 in Q(F)
                acc = np.zeros((n, N, d), np.int64)
                for _k in range(4):
                    rng ^= rng << np.uint32(13)
                    rng ^= rng >> np.uint32(17)
                    rng ^= rng << np.uint32(5)
                    acc += (rng >> np.uint32(16)).astype(np.int64)
                self._bound(acc - (1 << 17), 113512)
                xi = self.sh((acc - (1 << 17)) * 113512, 16)             # sqrt(3) * 2^16 = 113512; Q16 Gaussian
                xi = xi >> (16 - F) if F < 16 else xi
                P = self.mul(self.c1, P) + self.mul(c2, self.mul(self.sqm, xi))
                th = (th + self.rad_to_phase(self.mul(self.h2, self.mul(self.invm, P)))) & mask
                f = self.force(th, gpin) - self.mul(u, self.sin(th - phi))
                P = self.rotate(P)
                P = P + self.mul(self.h2, f)
            if record:
                half = 1 << (self.A - 1)
                bonds = ((np.roll(th, -1, axis=1) - th + half) & mask) - half
                winding.append(bonds.sum(axis=1) >> self.A)              # exact integer, per channel
        z = np.concatenate([self.cos(th), self.sin(th), P], axis=2).reshape(n, -1)
        self._bound(z, self.W, z.shape[1], add=self.b * (1 << F))
        logits = z @ self.W.T + self.b * (1 << F)                        # exact, Q(2F)
        return logits, th, P, (np.stack(winding, 1) if record else None)

    def export(self) -> dict:
        """Every integer the dynamics use, so another runtime needs no rounding of its own."""
        assert self.rnd in (0, 1) and not self.prnd, "the JS and C engines implement floor and nearest only"
        L = lambda a: np.asarray(a).astype(np.int64).ravel().tolist()  # noqa: E731
        return {"A": self.A, "B": self.B, "F": self.F, "N": self.N, "d": self.d, "S": self.S, "mode": self.mode,
                "EXP_RANGE": EXP_RANGE, "EXP_STEP": EXP_STEP, "SIN": L(self.SIN), "EXP": L(self.EXP), "h2": int(self.h2),
                "c1": int(self.c1), "c2u": int(self.c2u), "invm": L(self.invm), "sqm": L(self.sqm), "G": L(self.G),
                "kappa": int(self.kappa), "g_lat": int(self.g_lat), "g_pin": int(self.g_pin), "m_pin": self.m_pin,
                "lattice": bool(self.lattice), "ema": int(self.ema), "phi": L(self.phi), "u": L(self.u), "psi": L(self.psi),
                "W": L(self.W), "b": L(self.b), "ph_mul": self.ph_mul, "V": int(self.W.shape[0]), "rnd": self.rnd}


def state_hash(th, P):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(th, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(P, dtype="<i8").tobytes())
    return h.hexdigest()[:16]


def write_case(ring, path, name, n=24, seed=3):
    """Sequences at several sigma with the Python integer states and logits, for web/check_intring.js."""
    rs = np.random.default_rng(11)
    x = np.concatenate([np.repeat(rs.integers(1, 8, size=(n, 1)), 2, 1), np.where(rs.random((n, 29)) < 0.25,
                        rs.integers(1, 8, size=(n, 29)), 0), np.full((n, 1), 8)], 1).astype(np.int64)
    sq = np.round(np.linspace(0, 1.2, n) * (1 << ring.F)).astype(np.int64)
    ring.track, ring.peak = True, 0
    lg, th, P, _ = ring.run(x, sq, seed=seed)
    ring.track = False
    assert ring.peak < 1 << 53, f"{name}: an intermediate may reach 2^{math.log2(ring.peak):.1f}, beyond exact doubles"
    seqs = [{"x": x[k].tolist(), "sigmaQ": int(sq[k]), "th": th[k].ravel().tolist(), "P": P[k].ravel().tolist(),
             "logits": lg[k].tolist()} for k in range(n)]
    json.dump({"name": name, "seed": seed, "peak_bits": round(math.log2(max(ring.peak, 1)), 2), "export": ring.export(),
               "seqs": seqs}, open(path, "w"))


if __name__ == "__main__" and sys.argv[1] == "--case":
    exp_path, A, B, out = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rnd = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    write_case(IntRing(json.load(open(exp_path)), A=A, B=B, rnd=rnd), out, exp_path.split("/")[-1])
    sys.exit(0)

if __name__ == "__main__":
    LAB, exp_path = sys.argv[1], sys.argv[2]
    A, B = int(sys.argv[3]), int(sys.argv[4])
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 512
    rnd = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    sys.path.insert(0, f"{LAB}/phs-soliton")
    from phs import tasks
    ring = IntRing(json.load(open(exp_path)), A=A, B=B, rnd=rnd)
    SIG = [round(0.1 * i, 1) for i in range(13)]
    bs = [tasks.test_set("A", 32, s, n=n, namespace="comp") for s in SIG]
    X = np.concatenate([b["x"] for b in bs]); Y = np.concatenate([b["targets"][:, -1] for b in bs])
    S = np.round(np.concatenate([b["sigma"] for b in bs]) * (1 << ring.F)).astype(np.int64)
    lg, th, P, w = ring.run(X, S, seed=3, record=True)
    acc = (lg.argmax(1) == Y).reshape(len(SIG), n).mean(1)
    if len(sys.argv) > 6:
        json.dump({"acc": acc.tolist(), "sigma": SIG, "A": A, "B": B, "n": n, "rnd": rnd}, open(sys.argv[6], "w"))
    print(f"{exp_path.split('/')[-1]} A={A} B={B} {'round' if rnd else 'floor'}: " + " ".join(f"{s:.1f}:{a:.2f}" for s, a in zip(SIG, acc))
          + f"  | hash {state_hash(th[:n], P[:n])} | max|Q| {int(np.abs(w).max())}")
