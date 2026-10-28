/* Integer ring: the JS twin of fringes/intring.py. It reads the integer export (IntRing.export) and repeats every
   operation in the same order with floored divisions by powers of two, so states and logits agree bit for bit.
   All values stay below 2^53, where JS numbers are exact integers. One sequence per call. */
const IntRing = (() => {
  const P2 = (k) => 2 ** k;
  const fl = (v, k) => Math.floor(v / P2(k));

  function load(j) {
    const L = (a) => Float64Array.from(a);
    return { ...j, SIN: L(j.SIN), EXP: L(j.EXP), invm: L(j.invm), sqm: L(j.sqm), G: L(j.G), phi: L(j.phi), u: L(j.u),
      psi: L(j.psi), W: L(j.W), b: L(j.b), Nd: j.N * j.d, mask: P2(j.A) - 1, Lm: P2(j.B) - 1, sh: P2(j.A - j.B),
      expMax: j.EXP_RANGE * P2(j.EXP_STEP), floor03: Math.trunc(0.3 * P2(j.F)) };
  }

  function makeRun(R) {
    const F = R.F, N = R.N, d = R.d, Nd = R.Nd, mul = (a, b) => fl(a * b, F);
    const sinI = (v) => R.SIN[Math.floor(v / R.sh) & R.Lm];
    const cosI = (v) => R.SIN[(Math.floor(v / R.sh) + P2(R.B - 2)) & R.Lm];
    const c = new Float64Array(Nd), s = new Float64Array(Nd), A = new Float64Array(N * N), e = new Float64Array(N * N);

    function force(th, gpin, out) {
      for (let x = 0; x < Nd; x++) { c[x] = cosI(th[x]); s[x] = sinI(th[x]); }
      for (let i = 0; i < N; i++) {
        let mx = -Infinity;
        for (let j = 0; j < N; j++) {
          let v = 0;
          for (let q = 0; q < d; q++) v += c[i * d + q] * c[j * d + q];
          let w = 0;
          for (let q = 0; q < d; q++) w += s[i * d + q] * s[j * d + q];
          const C = Math.floor(fl(v + w, F) / d);
          const z = mul(R.kappa, C);
          A[i * N + j] = z; if (z > mx) mx = z;
        }
        let sum = 0;
        for (let j = 0; j < N; j++) {
          const idx = Math.min(fl(-(A[i * N + j] - mx), F - R.EXP_STEP), R.expMax);
          e[i * N + j] = R.EXP[idx]; sum += e[i * N + j];
        }
        for (let j = 0; j < N; j++) A[i * N + j] = Math.floor((e[i * N + j] * P2(F)) / sum);
      }
      for (let i = 0; i < N; i++) for (let q = 0; q < d; q++) {
        let ac = 0, as = 0;
        for (let j = 0; j < N; j++) {
          const a = fl(A[i * N + j] + A[j * N + i], 1);
          ac += a * c[j * d + q]; as += a * s[j * d + q];
        }
        const x = i * d + q;
        const acF = fl(ac, F), asF = fl(as, F);
        let f = Math.floor(-fl(R.kappa * (mul(s[x], acF) - mul(c[x], asF)), F) / d);
        if (R.lattice) {
          const ip = ((i + 1) % N) * d + q, im = ((i - 1 + N) % N) * d + q;
          f += mul(R.g_lat, sinI(th[ip] - th[x]) - sinI(th[x] - th[im]));
          f -= R.m_pin * mul(gpin, sinI(R.m_pin * (th[x] - R.psi[i])));
        }
        out[x] = f;
      }
    }

    const tmp = new Float64Array(Nd);
    function rotate(P) {
      for (let i = 0; i < N; i++) for (let q = 0; q < d; q++) {
        let v = 0; for (let j = 0; j < N; j++) v += R.G[i * N + j] * P[j * d + q]; tmp[i * d + q] = fl(v, F);
      }
      P.set(tmp);
    }

    /* tokens: int array; sigmaQ: round(sigma * 2^F); index: the sequence's index in its batch (seeds the RNG) */
    return function run(tokens, sigmaQ, seed, index, onToken) {
      const th = new Float64Array(Nd), P = new Float64Array(Nd), f = new Float64Array(Nd), rng = new Uint32Array(Nd);
      for (let x = 0; x < Nd; x++) {
        const v = ((Math.imul(seed, 2654435761) >>> 0) + (Math.imul(index, 40503) >>> 0) + (Math.imul(x, 2246822519) >>> 0) + 1) >>> 0;
        rng[x] = (v | 1) >>> 0;
      }
      const c2 = fl(R.c2u * sigmaQ, F);
      let Dhat = P2(F);
      const phiOf = (tok, x) => R.phi[tok * Nd + x], uOf = (tok, x) => R.u[tok * Nd + x];
      for (let t = 0; t < tokens.length; t++) {
        let gpin = R.g_pin;
        if (R.mode === "homeo") {
          let D = 0; for (let x = 0; x < Nd; x++) D += cosI(R.m_pin * th[x]);
          D = Math.floor(D / Nd);
          Dhat = t === 0 ? D : Dhat + mul(R.ema, D - Dhat);
          gpin = Math.floor((R.g_pin * P2(F)) / Math.max(Dhat, R.floor03));
        }
        const tok = tokens[t];
        const drive = () => { for (let x = 0; x < Nd; x++) f[x] -= mul(uOf(tok, x), sinI(th[x] - phiOf(tok, x))); };
        force(th, gpin, f); drive();
        for (let sub = 0; sub < R.S; sub++) {
          for (let x = 0; x < Nd; x++) P[x] += mul(R.h2, f[x]);
          rotate(P);
          for (let x = 0; x < Nd; x++) th[x] = (th[x] + fl(mul(R.h2, mul(R.invm[(x / d) | 0], P[x])) * R.ph_mul, 20)) & R.mask;
          for (let x = 0; x < Nd; x++) {
            let acc = 0;
            for (let k = 0; k < 4; k++) {
              let r = rng[x]; r ^= r << 13; r ^= r >>> 17; r ^= r << 5; rng[x] = r >>> 0; acc += rng[x] >>> 16;
            }
            let xi = fl((acc - P2(17)) * 113512, 16);
            if (F < 16) xi = fl(xi, 16 - F);
            P[x] = mul(R.c1, P[x]) + mul(c2, mul(R.sqm[(x / d) | 0], xi));
          }
          for (let x = 0; x < Nd; x++) th[x] = (th[x] + fl(mul(R.h2, mul(R.invm[(x / d) | 0], P[x])) * R.ph_mul, 20)) & R.mask;
          force(th, gpin, f); drive();
          rotate(P);
          for (let x = 0; x < Nd; x++) P[x] += mul(R.h2, f[x]);
        }
        if (onToken) onToken(t, th, P);
      }
      const Z = 3 * Nd, lg = new Float64Array(R.V);
      for (let v = 0; v < R.V; v++) {
        let acc = R.b[v] * P2(F);
        for (let i = 0; i < N; i++) for (let q = 0; q < d; q++) {
          const base = v * Z + i * 3 * d;
          acc += R.W[base + q] * cosI(th[i * d + q]) + R.W[base + d + q] * sinI(th[i * d + q]) + R.W[base + 2 * d + q] * P[i * d + q];
        }
        lg[v] = acc;
      }
      return { logits: lg, th, P };
    };
  }

  return { load, makeRun };
})();
if (typeof module !== "undefined") module.exports = IntRing;
