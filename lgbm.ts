// LightGBM tree evaluation from the flattened dump exported by allelion/export/bundle.py.
// Each node is either [leafValue] or [feature, threshold, defaultLeft, missingType, left, right]
// with missingType 0 = None, 1 = Zero, 2 = NaN, following LightGBM's NumericalDecision.

export type Node = number[];
export interface Model { num_class: number; num_tree_per_iteration: number; objective: string; features: string[]; trees: Node[][] }

const ZERO = 1e-35;

function leaf(tree: Node[], x: Float64Array): number {
  let i = 0;
  for (;;) {
    const n = tree[i];
    if (n.length === 1) return n[0];
    const [f, thr, defLeft, mt, l, r] = n;
    let v = x[f];
    if (Number.isNaN(v) && mt !== 2) v = 0;
    if ((mt === 1 && Math.abs(v) <= ZERO) || (mt === 2 && Number.isNaN(v))) {
      i = defLeft ? l : r;
    } else {
      i = v <= thr ? l : r;
    }
  }
}

export function raw(model: Model, x: Float64Array): number[] {
  const k = model.num_tree_per_iteration || 1;
  const out = new Array(k).fill(0);
  model.trees.forEach((t, j) => { out[j % k] += leaf(t, x); });
  return out;
}

export function predictBinary(model: Model, x: Float64Array): number {
  return 1 / (1 + Math.exp(-raw(model, x)[0]));
}

export function predictMulti(model: Model, x: Float64Array): number[] {
  const r = raw(model, x);
  const m = Math.max(...r);
  const e = r.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export function vector(model: Model, feats: Record<string, number>): Float64Array {
  const x = new Float64Array(model.features.length);
  model.features.forEach((f, i) => { const v = feats[f]; x[i] = v === undefined || v === null ? NaN : v; });
  return x;
}
