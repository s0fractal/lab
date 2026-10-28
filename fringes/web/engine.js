/* PHS engine: a float64 port of phs-soliton's PortHamiltonianCore.token (B -> G -> A -> O -> A -> G -> B per
   sub-step, Cayley gyroscope, attention + Frenkel-Kontorova + drive forces) and the PHS readout. One sequence. */
const PHS = (() => {
  const TAU = 2 * Math.PI;
  const wrap = (x) => { const y = (x + Math.PI) % TAU; return (y < 0 ? y + TAU : y) - Math.PI; };

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * rng());
  }

  function load(j) {
    const N = j.N, d = j.d;
    return { ...j, mass: Float64Array.from(j.mass), G: Float64Array.from(j.G), phi: Float64Array.from(j.phi),
      u: Float64Array.from(j.u), W: Float64Array.from(j.W), b: Float64Array.from(j.b), psi: Float64Array.from(j.psi),
      sqm: Float64Array.from(j.mass, Math.sqrt), invm: Float64Array.from(j.mass, (v) => 1 / v), Nd: N * d,
      _c: new Float64Array(N * d), _s: new Float64Array(N * d), _A: new Float64Array(N * N),
      _tmp: new Float64Array(N * d) };
  }

  function internalForce(M, th, gPin, out) {
    const N = M.N, d = M.d, c = M._c, s = M._s, A = M._A, k = M.kappa;
    for (let x = 0; x < M.Nd; x++) { c[x] = Math.cos(th[x]); s[x] = Math.sin(th[x]); }
    for (let i = 0; i < N; i++) {
      let mx = -Infinity;
      for (let j = 0; j < N; j++) {
        let v = 0;
        for (let q = 0; q < d; q++) v += c[i * d + q] * c[j * d + q] + s[i * d + q] * s[j * d + q];
        v = k * v / d; A[i * N + j] = v; if (v > mx) mx = v;
      }
      let z = 0;
      for (let j = 0; j < N; j++) { const e = Math.exp(A[i * N + j] - mx); A[i * N + j] = e; z += e; }
      for (let j = 0; j < N; j++) A[i * N + j] /= z;
    }
    for (let i = 0; i < N; i++) {
      for (let q = 0; q < d; q++) {
        let ac = 0, as = 0;
        for (let j = 0; j < N; j++) {
          const a = 0.5 * (A[i * N + j] + A[j * N + i]);
          ac += a * c[j * d + q]; as += a * s[j * d + q];
        }
        const x = i * d + q;
        let f = -(k / d) * (s[x] * ac - c[x] * as);
        if (M.g_lat > 0 || gPin > 0) {
          const ip = ((i + 1) % N) * d + q, im = ((i - 1 + N) % N) * d + q;
          f += M.g_lat * (Math.sin(th[ip] - th[x]) - Math.sin(th[x] - th[im]));
          f -= M.m_pin * gPin * Math.sin(M.m_pin * (th[x] - M.psi[i]));
        }
        out[x] = f;
      }
    }
  }

  function rotate(M, P) {
    const N = M.N, d = M.d, t = M._tmp;
    for (let i = 0; i < N; i++) for (let q = 0; q < d; q++) {
      let v = 0; for (let j = 0; j < N; j++) v += M.G[i * N + j] * P[j * d + q]; t[i * d + q] = v;
    }
    P.set(t);
  }

  /* Run one sequence. tokens: int array; sigma: noise; rng: () => [0,1); onToken(t, theta, P, D) for animation.
     homeo: pinning normalised by the EMA of <cos m theta> (the homeostatic architecture). */
  function run(M, tokens, sigma, rng, onToken) {
    const Nd = M.Nd, d = M.d, h = M.h;
    const th = new Float64Array(Nd), P = new Float64Array(Nd), f = new Float64Array(Nd), fi = new Float64Array(Nd);
    const c1 = Math.exp(-M.gamma * h), c2 = Math.sqrt(sigma * sigma * (1 - c1 * c1));
    let Dhat = 1;
    for (let t = 0; t < tokens.length; t++) {
      let gPin = M.g_pin;
      let D = 0; for (let x = 0; x < Nd; x++) D += Math.cos(M.m_pin * th[x]); D /= Nd;
      if (M.mode === "homeo") { Dhat = t === 0 ? D : (1 - M.ema) * Dhat + M.ema * D; gPin = M.g_pin / Math.max(Dhat, 0.3); }
      const base = tokens[t] * Nd;
      const drive = (out) => { for (let x = 0; x < Nd; x++) out[x] += -M.u[base + x] * Math.sin(th[x] - M.phi[base + x]); };
      internalForce(M, th, gPin, fi); f.set(fi); drive(f);
      for (let sub = 0; sub < M.S; sub++) {
        for (let x = 0; x < Nd; x++) P[x] += 0.5 * h * f[x];
        rotate(M, P);
        for (let x = 0; x < Nd; x++) th[x] = wrap(th[x] + 0.5 * h * M.invm[(x / d) | 0] * P[x]);
        if (c2 > 0) for (let x = 0; x < Nd; x++) P[x] = c1 * P[x] + c2 * M.sqm[(x / d) | 0] * gauss(rng);
        else if (c1 !== 1) for (let x = 0; x < Nd; x++) P[x] *= c1;
        for (let x = 0; x < Nd; x++) th[x] = wrap(th[x] + 0.5 * h * M.invm[(x / d) | 0] * P[x]);
        internalForce(M, th, gPin, fi); f.set(fi); drive(f);
        rotate(M, P);
        for (let x = 0; x < Nd; x++) P[x] += 0.5 * h * f[x];
      }
      if (onToken) onToken(t, th, P, D);
    }
    return readout(M, th, P);
  }

  function readout(M, th, P) {
    const N = M.N, d = M.d, C = M.b.length, Z = 3 * N * d, z = new Float64Array(Z);
    for (let i = 0; i < N; i++) for (let q = 0; q < d; q++) {
      z[i * 3 * d + q] = Math.cos(th[i * d + q]); z[i * 3 * d + d + q] = Math.sin(th[i * d + q]); z[i * 3 * d + 2 * d + q] = P[i * d + q];
    }
    const lg = new Float64Array(C);
    for (let c = 0; c < C; c++) { let v = M.b[c]; for (let x = 0; x < Z; x++) v += M.W[c * Z + x] * z[x]; lg[c] = v; }
    return lg;
  }

  /* task A: marker y (1..7) at t = 0, 1; blanks with distractors p = 0.25; query token 8 at t = T-1 */
  function taskA(y, T, rng) {
    const x = [y, y];
    for (let t = 2; t < T - 1; t++) x.push(rng() < 0.25 ? 1 + Math.floor(rng() * 7) : 0);
    x.push(8);
    return x;
  }

  return { load, run, taskA, mulberry32, wrap };
})();
if (typeof module !== "undefined") module.exports = PHS;
