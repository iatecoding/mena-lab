// The page's state in one place, a tiny event bus, and the job request built from it.
import { store } from './util.js';

export const state = {
  info: null,            // GET /api/info
  structure: null,       // {format, text, name, title}
  analysis: null,        // POST /api/structure analysis
  atoms: null,           // viewer atoms of the input {symbols, positions, cell}
  structure2: null, analysis2: null, atoms2: null,
  task: store('task') || 'singlepoint',
  params: {},            // per task: {taskId: {name: value}}
  runtime: {},           // overrides from "How to run" (only non-auto values)
  system: {},            // charge, spin
  plan: null, planError: null, planPending: false,
  running: null,         // id of the job being followed
};

const handlers = new Map();
export function on(evt, fn) { let l = handlers.get(evt); if (!l) handlers.set(evt, l = []); l.push(fn); }
export function emit(evt, data) { (handlers.get(evt) || []).forEach(fn => { try { fn(data); } catch (e) { console.error(e); } }); }

export function task(id = state.task) { return state.info ? state.info.catalog.tasks.find(t => t.id === id) : null; }
export function domain() { return state.analysis ? state.analysis.domain : null; }

export function value(p, taskId = state.task) {
  const v = (state.params[taskId] || {})[p.name];
  return v === undefined ? p.default : v;
}
export function setValue(p, v, taskId = state.task) { (state.params[taskId] ||= {})[p.name] = v; }

// A parameter is shown (and sent) when the structure's domain allows it and its show_if conditions hold.
export function visible(p, params, current, dom) {
  if (p.domains && dom && !p.domains.includes(dom)) return false;
  if (p.show_if) {
    for (const [name, allowed] of Object.entries(p.show_if)) {
      const q = params.find(x => x.name === name);
      if (!q || !allowed.includes(current(q))) return false;
    }
  }
  return true;
}

export function taskAllowed(t, dom = domain()) { return !dom || t.domains.includes(dom); }

// The job request; keys left out take the server's defaults.
export function buildJob() {
  const t = task();
  if (!t || !state.structure) return null;
  const params = {};
  t.params.forEach(p => { if (visible(p, t.params, q => value(q), domain())) params[p.name] = value(p); });
  const job = { task: t.id, structure: pick(state.structure), params, runtime: { ...state.runtime } };
  if (t.needs_second && state.structure2) job.structure2 = pick(state.structure2);
  if (domain() === 'molecule' || !domain()) job.system = { ...state.system };
  return job;
}
const pick = s => ({ format: s.format, text: s.text, name: s.name });
