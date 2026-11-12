// Entry point: loads /api/info, mounts the 2-column workbench, and wires runs, results and history together.
import { setLotLabels, formulaText } from './labels.js';
import { el, clear, api, store, toast } from './util.js';
import { state, on, emit, task } from './state.js';
import { Viewer } from './viewer.js';
import { mountHeader, setView } from './header.js';
import { mountStructure, loadStructure } from './structure.js';
import { mountTasks, shownTasks } from './tasks.js';
import { mountPlan, setRunning, refreshPlan } from './plan.js';
import { mountRunOpts } from './runopts.js';
import { mountSettings } from './settings.js';
import { RunPanel } from './run.js';
import { renderReport } from './report.js';
import { mountHistory, renderHistory } from './history.js';
import { mountPerf, showPerf } from './perf.js';

const $ = id => document.getElementById(id);
let viewer, panel = null, report = null;

function inputFrames(slot = 1) {
  const a = slot === 1 ? state.atoms : state.atoms2, an = slot === 1 ? state.analysis : state.analysis2;
  return a ? { frames: [{ symbols: a.symbols, positions: a.positions, label: (slot === 2 ? 'End structure: ' : '') + formulaText(an.formula) }], cell: a.cell } : null;
}

// Automatically sizes the 3D viewer & side-by-side chart stage based on whether live progress, charts, or results are shown.
function syncStageClasses() {
  const sc = $('stage-charts'), rp = $('run-panel'), rt = $('result-top');
  $('centre').classList.toggle('staged', !sc.hidden && sc.childElementCount > 0);
  $('centre').classList.toggle('active-output', !rp.hidden || !rt.hidden);
}

function watchStage() {
  const obs = new MutationObserver(syncStageClasses);
  ['stage-charts', 'run-panel', 'result-top'].forEach(id => obs.observe($(id), { childList: true, attributes: true, attributeFilter: ['hidden'] }));
  syncStageClasses();
}

function clearOutput(resetViewer = false) {
  if (panel) { panel.destroy(); panel = null; }
  if (report) { report.destroy(); report = null; }
  clear($('run-panel')); clear($('result-top')); clear($('result-panel')); clear($('stage-charts'));
  $('stage-charts').hidden = true; $('run-panel').hidden = true; $('result-top').hidden = true; $('result-panel').hidden = true;
  syncStageClasses();
  if (resetViewer && state.atoms) viewer.show(inputFrames(1), { keepRepeat: true });
}

function showResult(full) {
  if (report) { report.destroy(); report = null; }
  const res = full && full.result;
  if (!res) return;
  if (res.viewer && res.viewer.frames && res.viewer.frames.length) viewer.show(res.viewer, { frame: full.task === 'relax' ? 'last' : 0 });
  report = renderReport($('result-panel'), res, {
    job: full,
    viewer,
    topHost: $('result-top'),
    chartHost: $('stage-charts'),
    onClear: () => clearOutput(true),
  });
  syncStageClasses();
}

function follow(job, title, estimate) {
  state.running = job.id; emit('running');
  $('run-panel').hidden = false;
  syncStageClasses();
  panel = new RunPanel($('run-panel'), job, { title, estimate, viewer, chartHost: $('stage-charts'), onEnd: full => {
    state.running = null; setRunning(false); emit('running');
    if (full && full.result) {
      $('run-panel').hidden = true;
      showResult(full);
    }
    syncStageClasses();
    if (full && full.status === 'done') toast(`${title}: finished.`);
  } });
}

async function run() {
  const job = state.plan && state.plan.spec;
  if (!job || state.running) return;
  clearOutput();
  if (state.atoms) viewer.show(inputFrames(1), { keepRepeat: true });
  try {
    const r = await api('POST', '/api/jobs', job);
    follow({ id: r.id, status: 'queued' }, task().title, state.plan.decision.seconds);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (e) { state.planError = e.message; toast(e.message, 'error'); refreshPlan(); }
}

async function restoreInputs(full) {
  const sp = full.spec || {};
  if (!sp.task || !shownTasks().some(t => t.id === sp.task)) return;
  state.task = sp.task;
  state.params[sp.task] = { ...(sp.params || {}) };
  state.runtime = {};
  Object.entries(sp.runtime || {}).forEach(([k, v]) => { if (v !== 'auto' && v !== null) state.runtime[k] = v; });
  state.system = { ...(sp.system || {}) };
  if (!sp.structure2) {
    state.structure2 = null; state.analysis2 = null; state.atoms2 = null;
  }
  emit('restore-runtime');
  const usable = s => s && typeof s.text === 'string' && !/^<\d+ characters of text>$/.test(s.text);
  const norm = x => ({ ...x, title: x.format === 'builtin' ? undefined : (x.format === 'smiles' ? x.text : x.name) });
  if (usable(sp.structure2)) await loadStructure(norm(sp.structure2), 2, { keepViewer: true });
  if (usable(sp.structure)) await loadStructure(norm(sp.structure), 1, { keepViewer: true, keepSystem: true });
  emit('restore-task');
}

async function openJob(id) {
  if (state.running && state.running !== id) { toast('A job is running; open past results when it has finished.', 'error'); return; }
  let full;
  try { full = await api('GET', `/api/jobs/${id}`); } catch (e) { toast(e.message, 'error'); return; }
  if (full.task === 'benchmark') {
    setView('performance');
    showPerf(full);
    return;
  }
  setView('workspace');
  if (state.running === id) return;
  clearOutput();
  await restoreInputs(full);
  if (full.status === 'running' || full.status === 'queued') { follow(full, full.test_title || full.task, null); return; }
  if (full.error && !full.result) {
    $('result-top').hidden = false;
    $('result-top').append(
      el('div', { class: 'res-bar' }, el('h2', { class: 'section-h' }, 'Failed'),
        el('button', { class: 'btn ghost small', type: 'button', onclick: () => clearOutput(true) }, '× Clear')),
      el('div', { class: 'error-box' }, el('strong', {}, `${full.test_title || full.task} failed. `), full.error));
    syncStageClasses();
  }
  showResult(full);
}

async function boot() {
  let info;
  try { info = await api('GET', '/api/info'); setLotLabels(info.catalog && info.catalog.lot_labels); }
  catch (e) { $('boot').textContent = e.message; return; }
  state.info = info;
  $('boot').remove();
  $('view-workspace').hidden = false;
  viewer = new Viewer($('viewer-host'));
  mountHeader(info, () => viewer.schedule(), v => { setView(v); if (v === 'history') renderHistory(); if (v === 'performance') showPerf(); });
  mountPerf($('view-performance'));
  mountStructure($('structure-col'), $('struct-bar'));
  mountTasks($('task-strip'), $('task-form'), $('task-desc'));
  watchStage();
  mountPlan($('plan-card'), $('run-actions'));
  mountRunOpts($('runtime-opts'), $('repro'));
  mountSettings();
  mountHistory($('view-history'));
  on('structure', ({ slot, keepViewer }) => {
    if (slot !== 1 || keepViewer || state.running) return;
    clearOutput(); viewer.show(inputFrames(1));
  });
  on('show-atoms', slot => { if (!state.running) viewer.show(inputFrames(slot), { keepRepeat: true }); });
  on('run', run);
  on('cancel', () => panel && panel.cancel());
  on('open-job', openJob);
  emit('task');
  const ex = store('example') || 'caffeine';
  if (!(await loadStructure({ format: 'builtin', text: ex, name: ex }, 1)) && ex !== 'caffeine') await loadStructure({ format: 'builtin', text: 'caffeine', name: 'caffeine' }, 1);
  try {
    const live = (await api('GET', '/api/jobs')).jobs.find(j => j.status === 'running' || j.status === 'queued');
    if (live) openJob(live.id);
  } catch (e) { /* history is optional at start */ }
}

boot();
