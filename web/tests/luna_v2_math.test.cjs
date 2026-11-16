// Run from any directory: node web/tests/luna_v2_math.test.cjs
// Exercise the actual embedded calculations, without duplicating them in tests.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname, '..', 'luna_v2.html'), 'utf8');
const code = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]).join('\n');
new vm.Script(code);
function extract(from, to) {
  const start = code.indexOf(from), end = code.indexOf(to, start);
  assert(start >= 0 && end > start, `Missing source boundaries: ${from}`);
  return code.slice(start, end);
}
const close = (a,b,tolerance=1e-7) => assert(Math.abs(a-b) < tolerance, `${a} != ${b}`);
const random = extract('function rng(seed)', '/* PIPELINE-START */');

// Every E, W and b derivative agrees with central finite differences.
const embedding = vm.createContext({
  $: () => ({value:'1'}), embRender: () => {}
});
vm.runInContext(random + extract('const EV =', 'function embRender('), embedding);
vm.runInContext(`
  embReset(); EM.r = () => 0;
  globalThis.parameters = EM;
  globalThis.selectedPair = EFLAT[0];
  globalThis.sampleLoss = () => -Math.log(embProbs(EM.E[selectedPair[0]]).p[selectedPair[1]]);
`, embedding);
const params = embedding.parameters, numerical = [], delta = 1e-5;
for (const name of ['E','W','b']) {
  for (let row=0; row<params[name].length; row++) {
    const target = name === 'b' ? params[name] : params[name][row];
    const columns = name === 'b' ? [row] : [0,1];
    for (const col of columns) {
      const old = target[col];
      target[col] = old + delta; const plus = embedding.sampleLoss();
      target[col] = old - delta; const minus = embedding.sampleLoss();
      target[col] = old;
      numerical.push({name,row,col,old,gradient:(plus-minus)/(2*delta)});
    }
  }
}
const oldLoss = embedding.sampleLoss();
embedding.embStep();
let maxGradientError = 0;
for (const item of numerical) {
  const updated = item.name === 'b' ? params.b[item.col] : params[item.name][item.row][item.col];
  const analytic = (item.old-updated)/0.01;
  maxGradientError = Math.max(maxGradientError, Math.abs(analytic-item.gradient));
  close(analytic,item.gradient);
  if (item.name === 'W') close(params.last.W[item.row][item.col],item.old);
}
assert(embedding.sampleLoss() < oldLoss);
console.log(`PASS: ${numerical.length} parameter derivatives; max error ${maxGradientError.toExponential(2)}; SGD lowers sample loss.`);
let embeddingFormulas;
Object.assign(embedding,{
  T:en=>en,esc:String,H:(key,value)=>value,FX:(name,tex,role)=>({name,tex,role}),
  f2:x=>Number(x).toFixed(2),f3:x=>Number(x).toFixed(3),
  setFx:(anchor,items)=>{embeddingFormulas=items;}
});
vm.runInContext(extract('function embFx(){','function embLoop()'),embedding);
embedding.embFx();
assert.equal(embeddingFormulas.length,4);
assert(embeddingFormulas[3].tex.includes('W\\leftarrow W-'));
assert(embeddingFormulas[3].tex.includes('\\mathbf b\\leftarrow'));
assert(embeddingFormulas[0].tex.includes('\\begin{pmatrix}'));
assert(embeddingFormulas[2].tex.includes('\\eta=0.01'));
vm.runInContext('EM.last=null;',embedding);embedding.embFx();
assert.equal(embeddingFormulas.length,4);
assert(embeddingFormulas[0].tex.includes('E^\\top\\mathrm{onehot}'));
console.log('PASS: initial and post-update formula tabs include E/W/b updates, column-vector lookup, and recorded learning rate.');

// PCA eigenvectors, reconstruction-variance identity, and every Grid setting.
const luna = vm.createContext({});
vm.runInContext(random + extract('/* PIPELINE-START */', '/* PIPELINE-END */') + '\nglobalThis.data=TOY;', luna);
const data = luna.data;
data.vecs.forEach((w,i) => {
  close(w.reduce((a,x)=>a+x*x,0),1);
  data.vecs.forEach((v,j) => {if(i!==j)close(w.reduce((a,x,k)=>a+x*v[k],0),0);});
  data.C.forEach((row,j)=>close(row.reduce((a,x,k)=>a+x*w[k],0),data.vals[i]*w[j],1e-6));
});
close(data.vals.reduce((a,x)=>a+x,0),data.C.reduce((a,row,i)=>a+row[i],0),1e-6);
const k=3;
let residual=0;
for (const trace of data.train) for (const h of trace.h) {
  const centered=h.map((x,i)=>x-data.mu[i]);
  const z=data.vecs.slice(0,k).map(w=>w.reduce((a,x,i)=>a+x*centered[i],0));
  centered.forEach((x,i)=>{const reconstruction=z.reduce((a,value,j)=>a+value*data.vecs[j][i],0);residual+=(x-reconstruction)**2;});
}
close(residual/(data.N-1),data.vals.slice(k).reduce((a,x)=>a+x,0),1e-6);
for (const m of [5,10,15]) for (const N of [1,2,3]) {
  const p=luna.pipeline(m,N);
  assert.equal(p.run.a.length,data.run.h.length-N); // explicit author-code loop contract
  assert.equal(p.cell(p.lo.map((x,i)=>x-0.2*p.unit[i]))[0],-0);
  assert.equal(p.theta(Number.MAX_SAFE_INTEGER),0);
  for (const [state,total] of p.out) {
    let sum=0;for(const [pair,count] of p.trans) if(pair.startsWith(state+'>'))sum+=count/total;
    close(sum,1);
  }
  for(const trace of p.test){
    assert(trace.score>=0 && trace.score<=1);
    close(trace.score,trace.a.reduce((a,w)=>a+p.theta(w.s),0)/trace.a.length);
  }
  // Independent ROC area agrees with pairwise ranking, including tied scores.
  const thresholds=[-Infinity,...new Set([...p.pos,...p.neg].sort((a,b)=>a-b)),Infinity];
  const points=thresholds.map(t=>[p.pos.filter(x=>x<t).length/p.pos.length,p.neg.filter(x=>x<t).length/p.neg.length]);
  let area=0;for(let i=1;i<points.length;i++)area+=(points[i][0]-points[i-1][0])*(points[i][1]+points[i-1][1])/2;
  close(area,p.auc);
}
console.log('PASS: PCA eigenpairs/orthogonality/reconstruction; 9 Grid/window settings; transition rows; state scores; ROC/AUC with ties.');

// Run real attention rendering with a minimal DOM facade to inspect its formulas.
const nodes = new Map();
const node = id => {
  if(!nodes.has(id))nodes.set(id,{checked:true,innerHTML:'',querySelectorAll:()=>[]});
  return nodes.get(id);
};
let captured;
const attention = vm.createContext({
  $:node, $$:()=>[], RUN_TOK:Array.from({length:9},(_,i)=>String(i)),
  tx:()=>'',lk:()=>'',bindTips:()=>{},T:en=>en,Tx:en=>en,esc:String,
  f1:x=>x.toFixed(1),f2:x=>x.toFixed(2),f3:x=>x.toFixed(3),
  H:(key,x)=>x,FX:(name,tex,role)=>({name,tex,role}),setFx:(anchor,items)=>{captured=items;}
});
vm.runInContext(random+extract('const ATS =', "$('#attn-mask').addEventListener"),attention);
for(const mask of [true,false])for(const query of [0,4]){
  node('#attn-mask').checked=mask;
  vm.runInContext(`attnQ=${query};attn();globalThis.weights=ATTN.row;`,attention);
  close(attention.weights.reduce((a,x)=>a+x,0),1);
  if(mask)attention.weights.slice(query+1).forEach(x=>assert.equal(x,0));
  else attention.weights.forEach(x=>assert(x>0));
  assert(captured[1].tex.includes('+M_'));
  assert(captured[3].tex.includes('W_O'));
  if(!mask)assert(captured[2].tex.includes('M_{ij}=0'));
}
console.log('PASS: masked/unmasked attention formulas and weights, including first-token behavior and multi-head output projection.');

// Every formula diagram has three localized steps and links only to present terms.
let diagramLanguage='en';
const diagrams=vm.createContext({
  T:(en,zh)=>diagramLanguage==='en'?en:zh,
  esc:s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))
});
vm.runInContext(extract('const FORMULA_FLOWS =','const FXS =')+'\nglobalThis.flowDefinitions=FORMULA_FLOWS;',diagrams);
let diagramCount=0;
for(const [anchor,tabs] of Object.entries(diagrams.flowDefinitions))for(let i=0;i<tabs.length;i++){
  for(const language of ['en','zh']){
    diagramLanguage=language;
    const markup=diagrams.formulaFlowMarkup(anchor,i,[]);
    assert.equal((markup.match(/class="flow-step"/g)||[]).length,3);
    assert(!markup.includes('undefined'));
    assert(!markup.includes('data-link='));
  }
  diagramCount++;
}
diagramLanguage='en';
assert(diagrams.formulaFlowMarkup('attn-mat',2,[],String.raw`M_{ij}=0\quad`).includes('all 9 positions'));
assert(diagrams.formulaFlowMarkup('sem-svg',0,[],String.raw`\mathrm{pos}+\mathrm{neg}=0`).includes('Do not divide'));
assert(diagrams.formulaFlowMarkup('dtmc-svg',0,[],String.raw`\#=0`).includes('no estimate'));
assert(diagrams.formulaFlowMarkup('blk-svg',1,[],String.raw`\mathrm{FFN}`).includes('residual off'));
assert(!diagrams.formulaFlowMarkup('sem-svg',2,['sc']).includes('data-link="tau"'));
assert(diagrams.formulaFlowMarkup('sem-svg',2,['sc']).includes('data-link="sc"'));
console.log(`PASS: ${diagramCount} formula diagrams in both languages; real link keys only; mask-off, residual-off and zero-count variants.`);
