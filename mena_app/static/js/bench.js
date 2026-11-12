// "Run it on this computer": top-of-page interactive 3D molecular & crystal simulation + live speed/memory benchmark.
// Supports both "Live visual MD (3D dynamics)" (real-time NVE droplet/hotspot dynamics + live atoms/s HUD) and
// "Scaling ladder (grow system)" (multi-process cluster/supercell growth with and without low-memory mode).
import { el, clear, api, toast } from './util.js';
import { state } from './state.js';
import { paramInput } from './tasks.js';
import { Viewer } from './viewer.js';
import { RunPanel } from './run.js';
import { renderReport } from './report.js';
import { deviceLabel, devicesLabel, formulaText } from './labels.js';

const SYSTEMS = [
  ['water', 'Water droplet — Molecules (Mena-LAM-s)', 'Molecules — Mena-LAM-s'],
  ['ethanol', 'Ethanol cluster — Molecules (Mena-LAM-s)', 'Molecules — Mena-LAM-s'],
  ['cssni3', 'CsSnI₃ perovskite — Crystal (Mena-LAM-preview)', 'Periodic Crystals — Mena-LAM-preview'],
  ['diamond', 'Dense diamond (158 nbr/at) — Crystal (Mena-LAM-preview)', 'Periodic Crystals — Mena-LAM-preview'],
  ['si', 'Silicon lattice — Crystal (Mena-LAM-preview)', 'Periodic Crystals — Mena-LAM-preview'],
];
const MOL_SYSTEMS = new Set(['water', 'ethanol']);
const SIZES_MD = [500, 1000, 2000, 5000];
const SIZES_LADDER = [2000, 5000, 10000, 20000];
const COLORS = { standard: 0, 'low-memory': 1, throughput: 0, temperature: 1, 'energy drift': 2 };
const CHARTS = {
  bench_speed: { title: 'Speed vs. system size', x_label: 'atoms', y_label: 'atoms per ms', logx: true, seriesColors: COLORS, markers: true },
  bench_memory: { title: 'Peak memory vs. system size', x_label: 'atoms', y_label: 'GB', logx: true, seriesColors: COLORS, markers: true },
  bench_ms: { title: 'Time per step', x_label: 'atoms', y_label: 'ms per step', logx: true, logy: true, seriesColors: COLORS, markers: true },
  bench_live_speed: { title: 'Live throughput (atoms/s)', x_label: 'MD step', y_label: 'atoms/s', seriesColors: COLORS, markers: true, legend: false },
  bench_live_temp: { title: 'Temperature equilibration', x_label: 'time (fs)', y_label: 'K', seriesColors: COLORS, legend: false },
  bench_live_energy: { title: 'NVE energy drift', x_label: 'time (fs)', y_label: 'meV/atom', seriesColors: COLORS, legend: false },
};

let panel = null, report = null, viewer = null, openJobFn = null;

// Build a synthetic gentle vibrational/phonon preview mode on the input system so the 3D viewer can animate immediately.
function previewModes(v) {
  if (!v || !v.frames || !v.frames.length) return v;
  const pos = v.frames[0].positions, n = pos.length;
  const disp = pos.map((p, i) => {
    const s = (i % 2 === 0 ? 1 : -1) * 0.22;
    return [s * Math.cos(i * 1.3), s * Math.sin(i * 1.7), s * Math.cos(i * 0.9)];
  });
  return { ...v, modes: [{ freq: 180, disp: disp.slice(0, n) }] };
}

export function openBenchJob(full) {
  if (openJobFn && full) openJobFn(full);
}

export function mountBench(host) {
  clear(host);
  if (viewer) { viewer.destroy(); viewer = null; }
  const t = state.info && state.info.catalog.tasks.find(x => x.id === 'benchmark');
  if (!t) { host.append(el('p', { class: 'muted' }, 'This version of the server has no benchmark task.')); return; }

  const head = el('div', { class: 'bench-head' },
    el('div', {},
      el('h2', { class: 'section-h' }, 'Run it on this computer — live 3D simulation & speed benchmark'),
      el('p', { class: 'muted bench-intro' }, 'Benchmark molecules on Mena-LAM-s or crystals on Mena-LAM-preview right on your hardware — either as real-time 3D NVE dynamics or a multi-process scaling ladder.')),
    el('span', { class: 'pill accent-pill' }, state.info.machine.chip || devicesLabel(state.info.machine.devices)));
  host.append(head);

  const values = { system: 'water', device: 'auto', mode: 'visual_md', max_atoms: 1000, compare_low_memory: true, steps: 22 };
  const form = el('div', { class: 'bench-form' });

  const systemSubLabel = sysId => MOL_SYSTEMS.has(sysId)
    ? 'active molecular cluster (Mena-LAM-s)'
    : 'active crystal supercell (Mena-LAM-preview)';

  // Mode segmented control
  const modeSeg = el('div', { class: 'seg', role: 'group', 'aria-label': 'Simulation mode' });
  const modes = [['visual_md', 'Live visual MD (3D dynamics)'], ['ladder', 'Scaling ladder (grow system)']];
  const syncModeUI = () => {
    [...modeSeg.children].forEach((b, i) => b.setAttribute('aria-selected', String(modes[i][0] === values.mode)));
    const sizes = values.mode === 'visual_md' ? SIZES_MD : SIZES_LADDER;
    if (!sizes.includes(values.max_atoms)) values.max_atoms = values.mode === 'visual_md' ? 1000 : 5000;
    values.steps = values.mode === 'visual_md' ? (MOL_SYSTEMS.has(values.system) ? 22 : 28) : 5;
    sizeLbl.textContent = values.mode === 'visual_md' ? 'System size' : 'Largest system';
    clear(sizeSel).append(...sizes.map(s => el('option', { value: s, selected: s === values.max_atoms }, `${s.toLocaleString('en-US')} atoms`)));
    lowField.hidden = values.mode !== 'ladder';
    runBtn.textContent = values.mode === 'visual_md' ? 'Run live 3D simulation' : 'Run scaling benchmark';
  };
  modes.forEach(([val, lbl]) => {
    modeSeg.append(el('button', { type: 'button', 'aria-selected': String(val === values.mode), onclick: () => { values.mode = val; syncModeUI(); } }, lbl));
  });

  const groups = new Map();
  SYSTEMS.forEach(([v, l, g]) => { if (!groups.has(g)) groups.set(g, []); groups.get(g).push([v, l]); });
  const sys = el('select', {
    id: 'b-system',
    onchange: () => {
      values.system = sys.value;
      hudVal.atoms_sub.textContent = systemSubLabel(values.system);
      syncModeUI();
      loadPreview(values.system);
    },
  }, ...[...groups].map(([g, items]) => el('optgroup', { label: g }, ...items.map(([v, l]) => el('option', { value: v, selected: v === values.system }, l)))));

  const sizeLbl = el('label', { for: 'b-size' }, 'System size');
  const sizeSel = el('select', { id: 'b-size', onchange: () => { values.max_atoms = Number(sizeSel.value); } });
  const dev = el('select', { id: 'b-device', onchange: () => { values.device = dev.value; } },
    el('option', { value: 'auto' }, 'Automatic (fastest)'),
    ...state.info.machine.devices.map(d => el('option', { value: d }, deviceLabel(d))));

  const lowParam = t.params.find(p => p.name === 'compare_low_memory') || { name: 'compare_low_memory', label: 'Also measure low-memory mode', type: 'bool', default: true };
  const lowField = paramInput(lowParam, values.compare_low_memory, v => { values.compare_low_memory = v; }, { prefix: 'b-' });

  const runBtn = el('button', { class: 'btn primary', type: 'button', onclick: () => start() }, 'Run live 3D simulation');
  const cancelBtn = el('button', { class: 'btn danger', type: 'button', hidden: true, onclick: () => { cancelBtn.disabled = true; panel && panel.cancel(); } }, 'Cancel');

  form.append(
    el('div', { class: 'field bench-mode-field' }, el('span', { class: 'lbl' }, 'Mode'), modeSeg),
    el('div', { class: 'field bench-sys-field' }, el('label', { for: 'b-system' }, 'System'), sys),
    el('div', { class: 'field' }, sizeLbl, sizeSel),
    el('div', { class: 'field' }, el('label', { for: 'b-device' }, 'Device'), dev),
    lowField,
    el('div', { class: 'bench-actions' }, runBtn, cancelBtn)
  );
  syncModeUI();

  // Live telemetry HUD strip right above the 3D viewer & live charts
  const hudVal = {
    atoms_s: el('div', { class: 'hud-v num' }, '—'),
    atoms_ms: el('div', { class: 'hud-s num' }, 'Click Run to measure on this machine'),
    ms_step: el('div', { class: 'hud-v num' }, '—'),
    ms_sub: el('div', { class: 'hud-s' }, 'ms per force step'),
    atoms: el('div', { class: 'hud-v num' }, '—'),
    atoms_sub: el('div', { class: 'hud-s' }, systemSubLabel(values.system)),
    peak: el('div', { class: 'hud-v num' }, '—'),
    peak_sub: el('div', { class: 'hud-s' }, 'process + GPU memory'),
  };
  const hud = el('div', { class: 'bench-hud', 'aria-label': 'Live telemetry' },
    el('div', { class: 'hud-card accent' }, el('div', { class: 'hud-k' }, 'Live throughput'), hudVal.atoms_s, hudVal.atoms_ms),
    el('div', { class: 'hud-card' }, el('div', { class: 'hud-k' }, 'Step time'), hudVal.ms_step, hudVal.ms_sub),
    el('div', { class: 'hud-card' }, el('div', { class: 'hud-k' }, 'System size'), hudVal.atoms, hudVal.atoms_sub),
    el('div', { class: 'hud-card' }, el('div', { class: 'hud-k' }, 'Peak memory'), hudVal.peak, hudVal.peak_sub),
  );

  const err = el('div', { class: 'msg error', role: 'alert', hidden: true });
  const progress = el('div', { class: 'bench-progress', hidden: true });
  const vHost = el('div', { class: 'bench-viewer' });
  const chartTabs = el('div', { class: 'seg bench-chart-tabs', role: 'tablist', 'aria-label': 'Select chart', hidden: true });
  const makePlaceholder = () => el('div', { class: 'bench-chart-placeholder muted' }, 'Live chart appears here beside the 3D viewer as soon as you start the simulation.');
  const chartBody = el('div', { class: 'bench-chart-body' }, makePlaceholder());
  const chartsCol = el('div', { class: 'stage-charts bench-charts', 'aria-label': 'Live benchmark chart' }, chartTabs, chartBody);
  const stage = el('div', { class: 'bench-stage' }, vHost, chartsCol);
  const result = el('div', { class: 'bench-result', hidden: true });

  host.append(form, hud, err, progress, stage, result);
  viewer = new Viewer(vHost);

  let activeChartIdx = 0;
  function syncChartTabs() {
    const figs = [...chartBody.querySelectorAll('figure.chart')];
    if (figs.length <= 1) {
      chartTabs.hidden = true;
      figs.forEach(f => { f.hidden = false; });
      return;
    }
    if (activeChartIdx >= figs.length) activeChartIdx = 0;
    chartTabs.hidden = false;
    clear(chartTabs);
    figs.forEach((f, idx) => {
      const titleEl = f.querySelector('.chart-title');
      const lbl = (titleEl && titleEl.textContent) || `Chart ${idx + 1}`;
      f.hidden = idx !== activeChartIdx;
      chartTabs.append(el('button', {
        type: 'button',
        role: 'tab',
        'aria-selected': String(idx === activeChartIdx),
        onclick: () => {
          activeChartIdx = idx;
          syncChartTabs();
          window.dispatchEvent(new Event('resize'));
        },
      }, lbl));
    });
  }
  const chartObserver = new MutationObserver(() => syncChartTabs());
  chartObserver.observe(chartBody, { childList: true, subtree: true });

  async function loadPreview(sysId) {
    try {
      const r = await api('POST', '/api/structure', { structure: { format: 'builtin', text: sysId, name: sysId } });
      if (r && r.viewer && r.viewer.frames && r.viewer.frames.length) {
        const deg = r.analysis.n_atoms ? Math.round(r.analysis.edges / r.analysis.n_atoms) : 0;
        r.viewer.frames[0].label = r.analysis.domain === 'molecule'
          ? `${formulaText(r.analysis.formula)} unit (${r.analysis.n_atoms} atoms · tiled into 3D cluster on run)`
          : `${formulaText(r.analysis.formula)} (${deg} neighbours/atom at 6 Å)`;
        viewer.show(previewModes(r.viewer));
      }
    } catch (e) { /* ignore preview error */ }
  }
  loadPreview(values.system);

  function resetHud() {
    hudVal.atoms_s.textContent = '—';
    hudVal.atoms_ms.textContent = 'Click Run to measure on this machine';
    hudVal.ms_step.textContent = '—';
    hudVal.atoms.textContent = '—';
    hudVal.atoms_sub.textContent = systemSubLabel(values.system);
    hudVal.peak.textContent = '—';
  }

  function clearBenchOutput() {
    if (panel) { panel.destroy(); panel = null; }
    if (report) { report.destroy(); report = null; }
    clear(result); result.hidden = true;
    err.hidden = true;
    progress.hidden = true;
    chartTabs.hidden = true;
    chartBody.hidden = false;
    clear(chartBody).append(makePlaceholder());
    resetHud();
    loadPreview(values.system);
  }

  function updateHudFromStats(stats) {
    if (!stats) return;
    if (stats['atoms/s']) hudVal.atoms_s.textContent = `${stats['atoms/s']} atoms/s`;
    if (stats['atoms/ms']) hudVal.atoms_ms.textContent = `${stats['atoms/ms']} atoms/ms`;
    if (stats['ms/step']) hudVal.ms_step.textContent = stats['ms/step'];
    if (stats.atoms) hudVal.atoms.textContent = `${stats.atoms} atoms`;
    if (stats.peak) hudVal.peak.textContent = stats.peak;
  }

  function showFinishedReport(full) {
    runBtn.hidden = false; cancelBtn.hidden = true;
    chartBody.hidden = false;
    if (!full) return;
    const sp = full.spec || {};
    if (sp.structure && sp.structure.format === 'builtin' && SYSTEMS.some(([v]) => v === sp.structure.text)) {
      values.system = sp.structure.text;
      sys.value = values.system;
      hudVal.atoms_sub.textContent = systemSubLabel(values.system);
    }
    if (!full.result) {
      if (chartBody.childElementCount === 0) chartBody.append(makePlaceholder());
      err.textContent = full.error || (full.status === 'cancelled' ? 'Benchmark was stopped before completing a step.' : 'Benchmark did not finish.');
      err.hidden = false;
      hudVal.atoms_s.textContent = '—';
      hudVal.atoms_ms.textContent = full.status === 'cancelled' ? 'Stopped early' : 'Benchmark failed';
      return;
    }
    err.hidden = true;
    const res = full.result;
    if (res.raw) {
      if (res.raw.atoms_per_s) {
        hudVal.atoms_s.textContent = `${Math.round(res.raw.atoms_per_s).toLocaleString('en-US')} atoms/s`;
        hudVal.atoms_ms.textContent = `${res.raw.atoms_per_ms.toFixed(1)} atoms/ms`;
        hudVal.ms_step.textContent = `${res.raw.median_ms.toFixed(1)} ms`;
        hudVal.atoms.textContent = `${res.raw.n.toLocaleString('en-US')} atoms`;
        hudVal.peak.textContent = `${res.raw.peak_gb.toFixed(2)} GB`;
      } else if (res.raw.rows && res.raw.rows.length) {
        const std = res.raw.rows.filter(q => q.mode === 'standard');
        const pool = std.length ? std : res.raw.rows;
        const fast = pool.reduce((a, b) => (b.atoms_per_ms > a.atoms_per_ms ? b : a), pool[0]);
        const top = pool[pool.length - 1];
        hudVal.atoms_s.textContent = `${Math.round(fast.atoms_per_ms * 1000).toLocaleString('en-US')} atoms/s`;
        hudVal.atoms_ms.textContent = `${fast.atoms_per_ms.toFixed(1)} atoms/ms best`;
        hudVal.ms_step.textContent = `${top.ms.toFixed(0)} ms`;
        hudVal.atoms.textContent = `${top.n.toLocaleString('en-US')} atoms`;
        hudVal.peak.textContent = `${top.peak_gb.toFixed(2)} GB`;
      }
    }
    if (res.viewer && res.viewer.frames && res.viewer.frames.length) {
      viewer.show(res.viewer);
      if (res.viewer.frames.length > 1 && !viewer.playing) {
        viewer.interval = 90;
        viewer.toggle();
      }
    }
    result.hidden = false;
    activeChartIdx = 0;
    report = renderReport(result, res, {
      job: full,
      viewer,
      chartHost: chartBody,
      compact: true,
      chartMeta: { seriesColors: COLORS, markers: true },
      onClear: clearBenchOutput,
    });
    if (chartBody.childElementCount === 0) chartBody.append(makePlaceholder());
    syncChartTabs();
  }

  openJobFn = full => {
    if (panel) { panel.destroy(); panel = null; }
    if (report) { report.destroy(); report = null; }
    clear(result); result.hidden = true;
    progress.hidden = true;
    chartBody.hidden = false;
    showFinishedReport(full);
  };

  async function start() {
    err.hidden = true;
    const params = {
      mode: values.mode,
      max_atoms: values.max_atoms,
      compare_low_memory: values.compare_low_memory,
      steps: values.steps,
    };
    const job = {
      task: 'benchmark',
      structure: { format: 'builtin', text: values.system, name: values.system },
      params,
      runtime: values.device === 'auto' ? {} : { device: values.device },
    };
    if (panel) { panel.destroy(); panel = null; }
    if (report) { report.destroy(); report = null; }
    clear(result); result.hidden = true;
    activeChartIdx = 0;
    chartTabs.hidden = true;
    clear(chartBody); chartBody.hidden = false;
    hudVal.atoms_s.textContent = 'Starting…';
    hudVal.atoms_ms.textContent = 'warming up model & device kernels';
    hudVal.atoms_sub.textContent = systemSubLabel(values.system);
    let r;
    try {
      r = await api('POST', '/api/jobs', job);
    } catch (e) {
      err.textContent = e.message; err.hidden = false;
      chartBody.append(makePlaceholder());
      resetHud();
      return;
    }
    runBtn.hidden = true; cancelBtn.hidden = false; cancelBtn.disabled = false; progress.hidden = false;
    panel = new RunPanel(progress, { id: r.id, status: 'queued' }, {
      title: values.mode === 'visual_md' ? 'Live 3D visual MD' : t.title,
      viewer,
      chartHost: chartBody,
      liveCharts: CHARTS,
      chartMeta: { compact: true },
      onEnd: full => {
        progress.hidden = true;
        showFinishedReport(full);
        if (full && full.status === 'done') toast(values.mode === 'visual_md' ? 'Live 3D simulation finished.' : 'Benchmark finished.');
      },
    });
    const origProgress = panel.progress.bind(panel);
    panel.progress = d => {
      origProgress(d);
      if (d && d.stats) updateHudFromStats(d.stats);
    };
  }
}
