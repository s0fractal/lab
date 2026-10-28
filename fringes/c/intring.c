/* Integer ring in C: the third twin of fringes/intring.py and web/intring.js. No floating point anywhere, only
   int64 arithmetic with floored shifts and floored division (written out, so the result does not depend on how the
   compiler shifts negative numbers). Scaling by 2^k is a multiplication, never a left shift of a value that can be
   negative (undefined in C99); fringes/ci.sh also builds this file with -fsanitize=undefined. Reads a case in the
   plain-text format that c/check_c.py writes from an `intring.py --case` file and prints, per sequence, the final
   phases, momenta and logits, one line each.

   Build: cc -O2 -std=c99 -o intring intring.c
   Run:   ./intring case.txt > out.txt */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int64_t i64;

static i64 fshift(i64 a, int k) { return a >= 0 ? (a >> k) : -((-a + ((i64)1 << k) - 1) >> k); }
static i64 fdiv(i64 a, i64 b) { i64 q = a / b; if ((a % b) != 0 && ((a < 0) != (b < 0))) q -= 1; return q; }

static int A, B, F, N, D, S, V, homeo, lattice, m_pin, EXP_RANGE, EXP_STEP, nphi, rnd;
static i64 h2, c1, c2u, kappa, g_lat, g_pin, ema, ph_mul;
static i64 *SIN, *EXP, *invm, *sqm, *G, *phi, *u, *psi, *W, *b;

static i64 rd(FILE *f) { long long v; if (fscanf(f, "%lld", &v) != 1) { fprintf(stderr, "bad input\n"); exit(2); } return (i64)v; }
static i64 *rdv(FILE *f, int n) { i64 *a = malloc(sizeof(i64) * (size_t)n); for (int i = 0; i < n; i++) a[i] = rd(f); return a; }

static i64 mul(i64 a, i64 x) { return fshift(a * x, F); }
static i64 sinI(i64 v) { return SIN[fshift(v, A - B) & (((i64)1 << B) - 1)]; }
static i64 cosI(i64 v) { return SIN[(fshift(v, A - B) + ((i64)1 << (B - 2))) & (((i64)1 << B) - 1)]; }

static void force(const i64 *th, i64 gpin, i64 *out, i64 *c, i64 *s, i64 *Am) {
  int Nd = N * D;
  for (int x = 0; x < Nd; x++) { c[x] = cosI(th[x]); s[x] = sinI(th[x]); }
  for (int i = 0; i < N; i++) {
    i64 mx = INT64_MIN;
    for (int j = 0; j < N; j++) {
      i64 v = 0, w = 0;
      for (int q = 0; q < D; q++) v += c[i * D + q] * c[j * D + q];
      for (int q = 0; q < D; q++) w += s[i * D + q] * s[j * D + q];
      i64 C = fdiv(fshift(v + w, F), D);
      i64 z = mul(kappa, C);
      Am[i * N + j] = z; if (z > mx) mx = z;
    }
    i64 sum = 0, e[256];
    for (int j = 0; j < N; j++) {
      i64 idx = fshift(-(Am[i * N + j] - mx), F - EXP_STEP);
      if (idx > (i64)EXP_RANGE << EXP_STEP) idx = (i64)EXP_RANGE << EXP_STEP;
      e[j] = EXP[idx]; sum += e[j];
    }
    for (int j = 0; j < N; j++) Am[i * N + j] = fdiv(e[j] * ((i64)1 << F), sum);
  }
  for (int i = 0; i < N; i++) for (int q = 0; q < D; q++) {
    i64 ac = 0, as = 0;
    for (int j = 0; j < N; j++) {
      i64 a = fshift(Am[i * N + j] + Am[j * N + i], 1);
      ac += a * c[j * D + q]; as += a * s[j * D + q];
    }
    int x = i * D + q;
    i64 f = fdiv(-fshift(kappa * (mul(s[x], fshift(ac, F)) - mul(c[x], fshift(as, F))), F), D);
    if (lattice) {
      int ip = ((i + 1) % N) * D + q, im = ((i - 1 + N) % N) * D + q;
      f += mul(g_lat, sinI(th[ip] - th[x]) - sinI(th[x] - th[im]));
      f -= (i64)m_pin * mul(gpin, sinI((i64)m_pin * (th[x] - psi[i])));
    }
    out[x] = f;
  }
}

static void rotate(i64 *P, i64 *tmp) {
  for (int i = 0; i < N; i++) for (int q = 0; q < D; q++) {
    i64 v = 0; for (int j = 0; j < N; j++) v += G[i * N + j] * P[j * D + q]; tmp[i * D + q] = fshift(v, F);
  }
  for (int x = 0; x < N * D; x++) P[x] = tmp[x];
}

int main(int argc, char **argv) {
  FILE *f = argc > 1 ? fopen(argv[1], "r") : stdin;
  if (!f) { perror("open"); return 2; }
  A = rd(f); B = rd(f); F = rd(f); N = rd(f); D = rd(f); S = rd(f); V = rd(f); homeo = rd(f); lattice = rd(f);
  m_pin = rd(f); EXP_RANGE = rd(f); EXP_STEP = rd(f); rnd = (int)rd(f);
  h2 = rd(f); c1 = rd(f); c2u = rd(f); kappa = rd(f); g_lat = rd(f); g_pin = rd(f); ema = rd(f); ph_mul = rd(f);
  SIN = rdv(f, 1 << B); EXP = rdv(f, (EXP_RANGE << EXP_STEP) + 1);
  invm = rdv(f, N); sqm = rdv(f, N); G = rdv(f, N * N);
  nphi = rd(f); phi = rdv(f, nphi); u = rdv(f, nphi); psi = rdv(f, N);
  W = rdv(f, V * 3 * N * D); b = rdv(f, V);
  int seed = (int)rd(f), nseq = (int)rd(f), T = (int)rd(f);
  int Nd = N * D;
  i64 *th = malloc(sizeof(i64) * Nd), *P = malloc(sizeof(i64) * Nd), *fo = malloc(sizeof(i64) * Nd);
  i64 *c = malloc(sizeof(i64) * Nd), *s = malloc(sizeof(i64) * Nd), *Am = malloc(sizeof(i64) * N * N), *tmp = malloc(sizeof(i64) * Nd);
  uint32_t *rng = malloc(sizeof(uint32_t) * Nd);
  i64 mask = ((i64)1 << A) - 1, floor03 = (i64)(3 * ((i64)1 << F) / 10);
  int *x = malloc(sizeof(int) * T);
  for (int k = 0; k < nseq; k++) {
    for (int t = 0; t < T; t++) x[t] = (int)rd(f);
    i64 sigq = rd(f);
    for (int e = 0; e < Nd; e++) {
      th[e] = 0; P[e] = 0;
      uint32_t v = (uint32_t)((uint64_t)seed * 2654435761u + (uint64_t)k * 40503u + (uint64_t)e * 2246822519u + 1u);
      rng[e] = v | 1u;
    }
    i64 c2 = fshift(c2u * sigq, F), Dhat = (i64)1 << F;
    for (int t = 0; t < T; t++) {
      i64 gpin = g_pin;
      if (homeo) {
        i64 Dm = 0; for (int e = 0; e < Nd; e++) Dm += cosI((i64)m_pin * th[e]);
        Dm = fdiv(Dm, Nd);
        Dhat = t == 0 ? Dm : Dhat + mul(ema, Dm - Dhat);
        gpin = fdiv(g_pin * ((i64)1 << F), Dhat > floor03 ? Dhat : floor03);
      }
      const i64 *ph = phi + (i64)x[t] * Nd, *uu = u + (i64)x[t] * Nd;
      force(th, gpin, fo, c, s, Am);
      for (int e = 0; e < Nd; e++) fo[e] -= mul(uu[e], sinI(th[e] - ph[e]));
      for (int sub = 0; sub < S; sub++) {
        for (int e = 0; e < Nd; e++) P[e] += mul(h2, fo[e]);
        rotate(P, tmp);
        for (int e = 0; e < Nd; e++) th[e] = (th[e] + fshift(mul(h2, mul(invm[e / D], P[e])) * ph_mul + ((i64)rnd << 19), 20)) & mask;
        for (int e = 0; e < Nd; e++) {
          i64 acc = 0;
          for (int r = 0; r < 4; r++) {
            uint32_t z = rng[e]; z ^= z << 13; z ^= z >> 17; z ^= z << 5; rng[e] = z; acc += (i64)(z >> 16);
          }
          i64 xi = fshift((acc - ((i64)1 << 17)) * 113512, 16);
          if (F < 16) xi = fshift(xi, 16 - F);
          P[e] = mul(c1, P[e]) + mul(c2, mul(sqm[e / D], xi));
        }
        for (int e = 0; e < Nd; e++) th[e] = (th[e] + fshift(mul(h2, mul(invm[e / D], P[e])) * ph_mul + ((i64)rnd << 19), 20)) & mask;
        force(th, gpin, fo, c, s, Am);
        for (int e = 0; e < Nd; e++) fo[e] -= mul(uu[e], sinI(th[e] - ph[e]));
        rotate(P, tmp);
        for (int e = 0; e < Nd; e++) P[e] += mul(h2, fo[e]);
      }
    }
    for (int e = 0; e < Nd; e++) printf("%lld ", (long long)th[e]);
    printf("| ");
    for (int e = 0; e < Nd; e++) printf("%lld ", (long long)P[e]);
    printf("| ");
    for (int v = 0; v < V; v++) {
      i64 acc = b[v] * ((i64)1 << F);   /* b may be negative: a left shift of it is undefined in C99 */
      for (int i = 0; i < N; i++) for (int q = 0; q < D; q++) {
        i64 base = (i64)v * 3 * Nd + (i64)i * 3 * D;
        acc += W[base + q] * cosI(th[i * D + q]) + W[base + D + q] * sinI(th[i * D + q]) + W[base + 2 * D + q] * P[i * D + q];
      }
      printf("%lld ", (long long)acc);
    }
    printf("\n");
  }
  return 0;
}
