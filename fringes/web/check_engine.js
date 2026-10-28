// Check the JS engine against PyTorch: logits of the reference sequence at sigma = 0 for each exported model.
// Usage: node check_engine.js runs/js/soliton.json [runs/js/pure.json ...]   (exports from export_js.py)
const PHS = require("./engine.js");
const fs = require("fs");
let worst = 0;
for (const f of process.argv.slice(2)) {
  const j = JSON.parse(fs.readFileSync(f));
  if (!j.ref) { console.log(f, "no reference logits (homeo export)"); continue; }
  const lg = PHS.run(PHS.load(j), j.ref.x, 0, PHS.mulberry32(1));
  const diff = Math.max(...Array.from(lg, (v, i) => Math.abs(v - j.ref.logits[i])));
  worst = Math.max(worst, diff);
  console.log(f, "max |logit diff|", diff.toExponential(2));
}
process.exit(worst < 1e-3 ? 0 : 1);
