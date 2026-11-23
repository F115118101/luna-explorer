// Export luna_v3.html's toy dataset + LUNA pipeline (k = 3, m = 5, N = 1) to toy-data.json, so the
// Manim scenes show exactly the numbers the page's widgets show. The code is not re-implemented:
// rng / gauss and the PIPELINE-START..PIPELINE-END block are cut out of the page and evaluated.
//   node web/assets/export_toy.mjs
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = fs.readFileSync(path.join(here, '..', 'luna_v3.html'), 'utf8');

const grab = re => { const m = html.match(re); if (!m) throw new Error('not found: ' + re); return m[0]; };
const rngSrc = grab(/function rng\(seed\)\{[^\n]*\n/);
const gaussSrc = grab(/function gauss\(r\)\{[^\n]*\n/);
const pipeSrc = grab(/\/\* PIPELINE-START \*\/[\s\S]*?\/\* PIPELINE-END \*\//);
const promptSrc = grab(/const GEN_PROMPT = [^\n]*\n/);
const genSrc = grab(/const GEN = \[[\s\S]*?\]\];\n/);

const hidSrc = grab(/const HID = \(\(\) => \{[\s\S]*?return \{E0, fin, J\}; \}\)\(\);/);
const rmsSrc = grab(/const rmsOf = [^\n]*\n/);

const vtok = s => s.replace(/^ /, '▁');
const out = new Function('vtok', `${rngSrc}${gaussSrc}${rmsSrc}${promptSrc}${genSrc}${pipeSrc}
  const RUN_TOK = GEN.map(g => vtok(g[0][0]));
  ${hidSrc}
  return {TOY, pipeline, RUN_TOK, GEN_PROMPT, HID, rmsOf};`)(vtok);

const {TOY, pipeline, RUN_TOK, GEN_PROMPT, HID, rmsOf} = out;
const p = pipeline(5, 1);
/* the page's hidden-state table at layer 32 (hidden() with l = 32): fin / rms(fin) * 0.9, columns 1-6, 4095, 4096 */
const HCOLS = [0, 1, 2, 3, 4, 5, 46, 47];
const hid32 = HID.fin.map(row => { const r = rmsOf(row); return HCOLS.map(d => Math.round(row[d] / r * .9 * 100) / 100); });
/* the page's DTMC widget: the 12 most visited states, and the example's transitions */
const visits = [...p.cnt.entries()].map(([s, c]) => [s, c.pos + c.neg]).sort((a, b) => b[1] - a[1]);
const top12 = visits.slice(0, 12).map(v => v[0]);
const ra = p.run.a;
const runTrans = ra.slice(0, -1).map((w, i) => { const a = w.s, b = ra[i + 1].s, n = p.trans.get(a + '>' + b) || 0, o = p.out.get(a) || 0;
  return {a, b, n, out: o, p: o ? n / o : 0}; });
const r6 = v => Math.round(v * 1e6) / 1e6;
const trace = tr => ({y: tr.y, run: !!tr.run, z: tr.z.map(z => z.slice(0, 3).map(r6)),
  a: tr.a.map(w => ({i: w.i, zm: w.zm.map(r6), c: w.c, s: w.s}))});
const zip = (sets, aset) => sets.map((tr, i) => ({...trace({...tr, a: aset[i].a}), score: aset[i].score === undefined ? null : r6(aset[i].score)}));

const data = {
  config: {model: 'Llama2-7b', d: 4096, layers: 32, k: p.k, m: p.m, N: p.N},
  prompt: GEN_PROMPT, tokens: RUN_TOK,
  pca: {D: TOY.D, N: TOY.N, vals: TOY.vals.map(r6), mu: TOY.mu.map(r6), W: TOY.vecs.slice(0, 3).map(v => v.map(r6))},
  run_h: TOY.run.h.map(h => h.map(r6)),
  grid: {lo: p.lo.map(r6), hi: p.hi.map(r6), unit: p.unit.map(r6)},
  train: zip(TOY.train, p.train),
  test: zip(TOY.test, p.test),
  counts: Object.fromEntries([...p.cnt].map(([s, c]) => [s, c])),
  trans: Object.fromEntries(p.trans),
  out: Object.fromEntries(p.out),
  auc: r6(p.auc),
  hidden32: {cols: [1, 2, 3, 4, 5, 6, 4095, 4096], rows: hid32},
  top12, runTrans,
};
fs.writeFileSync(path.join(here, 'toy-data.json'), JSON.stringify(data));
const run = data.test.find(t => t.run);
console.log('tokens', RUN_TOK.join(' '));
console.log('train', data.train.length, 'test', data.test.length, 'states seen', Object.keys(data.counts).length, 'auc', data.auc);
console.log('run states', run.a.map(w => w.s).join(','), 'score', run.score);
console.log('lo', data.grid.lo, 'unit', data.grid.unit);
