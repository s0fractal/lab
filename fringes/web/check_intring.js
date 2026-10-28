// Bit-for-bit check of web/intring.js against fringes/intring.py. Reads the case file written by
// `intring.py --case` (integer export + sequences + the Python states and logits) and compares every integer.
// Usage: node check_intring.js case.json [--bite]
// --bite runs the engine with the other phase rounding (floor <-> round half up) and must see a difference: a check
// that cannot fail on a changed engine proves nothing (primitives PATTERNS.md: negatives must be shown to bite).
const IntRing = require("./intring.js");
const fs = require("fs");
const cs = JSON.parse(fs.readFileSync(process.argv[2]));
const bite = process.argv.includes("--bite");
const R = IntRing.load(bite ? { ...cs.export, rnd: 1 - (cs.export.rnd ?? 0) } : cs.export), run = IntRing.makeRun(R);
let bad = 0;
cs.seqs.forEach((s, k) => {
  const out = run(s.x, s.sigmaQ, cs.seed, k);
  const eq = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  if (!eq(Array.from(out.th), s.th) || !eq(Array.from(out.P), s.P) || !eq(Array.from(out.logits), s.logits)) bad++;
});
if (bite) {
  console.log(`${cs.name}: bite ${bad ? "ok" : "FAILED"}: ${bad}/${cs.seqs.length} sequences differ with the other rounding`);
  process.exit(bad ? 0 : 1);
}
console.log(`${cs.name}: ${cs.seqs.length - bad}/${cs.seqs.length} sequences bit-identical (A=${R.A}, B=${R.B}, F=${R.F})`);
process.exit(bad ? 1 : 0);
