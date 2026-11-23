// Performance view:
// 1. TOP: "Run it on this computer" — live 3D visual simulation & on-machine speed/memory benchmark for both Molecules
//    (Mena-LAM-s / lam5-v1) and Periodic Crystals (Mena-LAM-6 / lam6-v1) via bench.js.
// 2. MIDDLE: which model runs molecules and which runs crystals and surfaces.
// 3. BOTTOM: Published Benchmarks organized into scannable Domain sections (Molecules · Mena-LAM-s & Periodic Materials).
import { el, clear, api } from './util.js';
import { hbars, scatter } from './perfcharts.js';
import { mountBench, openBenchJob } from './bench.js';

let host, loaded = false;

function figure(title, subBadge, body, src = {}, extraClass = '') {
  const hasSrc = src && (src.note || src.source);
  return el('figure', { class: ('pfig ' + extraClass).trim() },
    el('figcaption', { class: 'pfig-head' },
      el('span', { class: 'pfig-title' }, title),
      subBadge ? el('span', { class: 'hw-badge' }, subBadge) : null),
    body,
    hasSrc ? el('div', { class: 'pfig-src' },
      src.note ? el('p', { class: 'caveat' }, src.note) : null,
      src.source ? el('p', {}, 'Source: ', src.source) : null) : null);
}

function factsFigure(sec, extraClass = '') {
  if (!sec || !sec.facts) return null;
  const dl = el('dl', { class: 'facts showcase-facts' },
    Object.entries(sec.facts).map(([k, v]) => el('div', { class: 'fact-pair' }, el('dt', {}, k), el('dd', { class: 'num' }, String(v)))));
  return figure(sec.title, sec.badge || 'Apple M3 Max 36 GB', dl, { note: sec.note, source: sec.source }, ('showcase ' + extraClass).trim());
}

function archStatusCallout(st) {
  if (!st || !st.molecules || !st.materials) return null;
  const card = (item, kind) => el('div', { class: `arch-card ${kind}` },
    el('div', { class: 'arch-badge-row' },
      el('span', { class: `arch-pill ${kind}` }, item.badge)),
    el('h3', { class: 'arch-card-title' }, item.title),
    el('p', { class: 'arch-card-body' }, item.body));
  return el('section', { class: 'arch-callout', 'aria-label': st.title || 'Architecture and checkpoint status' },
    el('div', { class: 'arch-head' },
      el('span', { class: 'arch-kicker' }, 'Models'),
      el('span', { class: 'muted arch-sub' }, 'Automatic picks the model by structure type; change it in Settings')),
    el('div', { class: 'arch-grid' },
      card(st.molecules, st.molecules.status || 'active'),
      card(st.materials, st.materials.status || 'active')));
}

function hardwareCallout(hb) {
  if (!hb) return null;
  return el('div', { class: 'hw-banner', role: 'region', 'aria-label': 'Hardware comparison context' },
    el('div', { class: 'hw-banner-title' },
      el('span', { class: 'hw-tag mena' }, hb.mena_tag || 'Mena-LAM-6 (lam6-v1): 1× RTX PRO 6000 / Apple M3 Max laptop (strict FP32)'),
      el('span', { class: 'hw-vs' }, 'vs.'),
      el('span', { class: 'hw-tag other' }, hb.other_tag || 'Competitors: NVIDIA H100 / H200 / RTX PRO 6000 / H20 96 GB')),
    el('div', { class: 'hw-banner-grid' },
      el('p', {}, el('strong', {}, 'Matbench Discovery DynaMat MD (RTX PRO 6000 vs. H100/H200/Blackwell): '), hb.dynamat),
      el('p', {}, el('strong', {}, 'Dense Diamond 158 nbr/atom (M3 Max laptop vs. H20 96 GB): '), hb.diamond)));
}

function hero(items) {
  if (!items || !items.length) return null;
  return el('div', { class: 'hero' }, items.map(h => el('div', { class: 'hero-item' },
    h.badge ? el('div', { class: 'hero-badge' }, h.badge) : null,
    el('div', { class: 'hero-v num' }, String(h.value), h.unit ? el('span', { class: 'hero-u' }, h.unit) : null),
    el('div', { class: 'hero-l' }, h.label),
    h.sub ? el('div', { class: 'hero-s' }, h.sub) : null)));
}

function molScalingFigure(ts) {
  if (!ts || !ts.rows) return null;
  const cols = ts.columns || ['System', 'Atoms', 'PyTorch (ms)', 'Fused (ms)', 'Fused atoms/s', 'Speedup', 'PyTorch RAM', 'Fused RAM'];
  const tbl = el('table', { class: 'tbl perf-tbl' },
    el('thead', {}, el('tr', {}, cols.map(c => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, ts.rows.map(r => {
      const oom = r.t1_ms == null;
      return el('tr', { class: oom ? 'perf-highlight-row' : '' },
        el('td', {}, r.system),
        el('td', { class: 'num' }, r.n.toLocaleString('en-US')),
        el('td', { class: 'num' + (oom ? ' oom-cell' : '') }, r.t1_ms != null ? r.t1_ms.toFixed(1) : 'OOM'),
        el('td', { class: 'num strong-cell' }, r.t2_ms.toFixed(1)),
        el('td', { class: 'num strong-cell' }, r.atoms_per_s.toLocaleString('en-US')),
        el('td', { class: 'num accent-cell' }, r.speedup),
        el('td', { class: 'num' + (oom ? ' oom-cell' : '') }, r.t1_gb),
        el('td', { class: 'num' }, r.t2_gb));
    })));
  return figure(ts.title, ts.badge, el('div', { class: 'perf-tbl-wrap' }, tbl), ts, 'wide-showcase');
}

function moleculesSection(b) {
  const m = b.molecules_lam5;
  if (!m) return null;
  const sec = el('section', { class: 'domain-section', 'aria-label': 'Molecules benchmarks (Mena-LAM-s)' },
    el('div', { class: 'domain-head' },
      el('div', {},
        el('h3', { class: 'domain-title' }, 'Molecules & Biomolecules — Mena-LAM-s (lam5-v1)'),
        el('p', { class: 'muted domain-desc' }, 'Accuracy against CCSD(T) and DFT references, IR and Raman spectra, and speed and memory on a laptop GPU and CPU.')),
      el('span', { class: 'domain-tag active' }, 'Default for molecules')),
    hero(m.headline));

  const grid = el('div', { class: 'pgrid' });
  if (m.plf547 && m.plf547.rows) {
    const figPlf = el('div');
    grid.append(figure(m.plf547.title, m.plf547.badge, figPlf, m.plf547));
    const rows = m.plf547.rows.map(r => ({
      label: r.name,
      value: r.mae,
      mena: r.mena,
      title: `${r.name}: ${r.detail || (r.mae + ' kcal/mol')}`,
    }));
    hbars(figPlf, { rows, unit: 'kcal/mol', x_label: 'MAE vs. CCSD(T) (kcal/mol, linear scale, lower is better)', linear: true, digits: 3 });
  }
  const scaleFig = molScalingFigure(m.tier2_scaling);
  if (scaleFig) grid.append(scaleFig);
  const cpuFig = factsFigure(m.cpu_and_workflows, 'compact-facts');
  if (cpuFig) grid.append(cpuFig);
  const accFig = factsFigure(m.accuracy_and_spectroscopy, 'compact-facts');
  if (accFig) grid.append(accFig);
  sec.append(grid);
  return sec;
}

function lam6SpeedFigure(sec) {
  if (!sec || !sec.rows) return null;
  const cols = sec.columns || ['Hardware', 'Workload (E+F+Stress)', 'Atoms', 'Latency', 'Throughput', 'Memory'];
  const tbl = el('table', { class: 'tbl perf-tbl' },
    el('thead', {}, el('tr', {}, cols.map(c => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, sec.rows.map(r => el('tr', { class: r.highlight ? 'perf-highlight-row' : '' },
      el('td', { class: 'strong-cell' }, r.hw),
      el('td', {}, r.workload),
      el('td', { class: 'num' }, r.n.toLocaleString('en-US')),
      el('td', { class: 'num' }, r.latency),
      el('td', { class: 'num accent-cell' }, `${r.atoms_per_s.toLocaleString('en-US')} at/s`),
      el('td', { class: 'num' }, r.mem)))));
  return figure(sec.title, sec.badge, el('div', { class: 'perf-tbl-wrap' }, tbl), sec, 'wide-showcase');
}

function materialsSection(b) {
  const sec = el('section', { class: 'domain-section', 'aria-label': 'Periodic materials benchmarks' },
    el('div', { class: 'domain-head' },
      el('div', {},
        el('h3', { class: 'domain-title' }, 'Crystals & Surfaces — Mena-LAM-6 (lam6-v1) Leaderboard & Speed'),
        el('p', { class: 'muted domain-desc' }, 'Default materials model (strict IEEE FP32): Matbench Discovery CPS 0.874, F1 0.904, κ_SRME 0.182, WBM-257k in 4m 38s, DynaMat MD in 43.6 min (0.46 GB VRAM).')),
      el('span', { class: 'domain-tag active' }, 'Default: Mena-LAM-6 (lam6-v1)')),
    hero(b.headline),
    hardwareCallout(b.hardware_banner));

  const box = el('div', { class: 'pgrid' });
  const si = b.silicon_speed;
  if (si && si.rows) {
    const siRows = si.rows.map(r => ({
      label: r.name,
      value: r.atoms_per_ms,
      mena: r.mena,
      title: `${r.name}: ${r.detail || (r.atoms_per_ms + ' atoms/ms')}`,
    }));
    const figSi = el('div');
    box.append(figure(si.title, si.badge, figSi, si));
    if (siRows.length) hbars(figSi, { rows: siRows, unit: 'at/ms', x_label: 'throughput (atoms/ms, log scale, higher is better)', digits: 1 });
  }
  const cc = b.cps_vs_compute;
  if (cc && cc.rows) {
    const ccPts = cc.rows.map(r => ({
      name: r.name,
      x: r.gpu_h,
      y: r.cps,
      mena: r.mena,
      title: `${r.name}: CPS ${r.cps}, ${r.gpu_h.toLocaleString('en-US')} GPU-h (${r.scope})`,
    }));
    const figCc = el('div');
    box.append(figure(cc.title, cc.badge, figCc, cc));
    if (ccPts.length) scatter(figCc, { points: ccPts, x_label: 'training compute (GPU-hours, log scale, lower is better)', y_label: 'Matbench Discovery CPS (higher is better)', x_unit: 'h' });
  }
  const dm = b.dynamat;
  if (dm && dm.rows) {
    const memRows = dm.rows.filter(r => r.mem_gb != null).sort((a, c) => a.mem_gb - c.mem_gb)
      .map(r => ({
        label: `${r.name} · ${r.gpu}`,
        value: r.mem_gb,
        mena: r.mena,
        title: `${r.name} (${r.gpu}): ${r.mem_gb} GB peak memory, ${r.hours} h, CMDS ${r.cmds}`,
      }));
    const fig1 = el('div');
    box.append(figure(dm.title || 'Peak GPU memory in DynaMat MD', 'Mena-LAM-6: RTX PRO 6000 · Others: H100/H200/Blackwell', fig1, dm));
    if (memRows.length) hbars(fig1, { rows: memRows, unit: 'GB', x_label: 'peak GPU memory (GB, log scale, lower is better)' });

    const pts = dm.rows.filter(r => r.hours != null && r.cmds != null)
      .map(r => ({
        name: `${r.name} (${r.gpu})`,
        x: r.hours,
        y: r.cmds,
        mena: r.mena,
        title: `${r.name} on ${r.gpu}: ${r.hours} h, CMDS ${r.cmds}${r.cps ? `, CPS ${r.cps}` : ''}`,
      }));
    const fig2 = el('div');
    box.append(figure('Speed vs. accuracy (CMDS) in DynaMat MD', '#1 active speed · 43.6 min (strict FP32)', fig2,
      { note: dm.note, source: dm.source + ' Up and to the left is better (faster on x, more accurate on y).' }));
    if (pts.length) scatter(fig2, { points: pts, x_label: 'MD wall time (hours, log scale, lower is better)', y_label: 'CMDS (higher is better)', x_unit: ' h' });
  }
  const di = b.diamond;
  if (di && di.rows) {
    const rows = di.rows.filter(r => r.mb_per_atom != null).sort((a, c) => a.mb_per_atom - c.mb_per_atom)
      .map(r => ({
        label: `${r.name} · ${r.gpu || (r.mena ? 'M3 Max' : 'H20')}`,
        value: r.mb_per_atom,
        mena: r.mena,
        title: `${r.name} (${r.hardware}): ${r.mb_per_atom} MB/atom${r.atoms_per_ms != null ? `, ${r.atoms_per_ms} atoms/ms (${Math.round(r.atoms_per_ms * 1000).toLocaleString('en-US')} atoms/s)` : ''}`,
      }));
    const fig3 = el('div');
    box.append(figure(di.title || 'Memory per atom, dense diamond', 'Mena-LAM-6: M3 Max · Others: H20 96GB', fig3, di));
    if (rows.length) hbars(fig3, { rows, unit: 'MB', x_label: 'memory per atom (MB, log scale, lower is better)' });

    const dPts = di.rows.filter(r => r.atoms_per_ms != null && r.cps != null)
      .map(r => ({
        name: `${r.name} (${r.gpu || (r.mena ? 'M3 Max' : 'H20')})`,
        x: r.atoms_per_ms,
        y: r.cps,
        mena: r.mena,
        title: `${r.name} on ${r.hardware}: ${r.atoms_per_ms} atoms/ms (${Math.round(r.atoms_per_ms * 1000).toLocaleString('en-US')} atoms/s), CPS ${r.cps}, ${r.mb_per_atom} MB/atom`,
      }));
    const fig4 = el('div');
    box.append(figure('Discovery accuracy (CPS) vs. speed on dense diamond (158 nbr/atom)', 'M3 Max laptop (300 GB/s) vs. H20 (4,000 GB/s)', fig4, {
      note: (di.rows.filter(r => r.mena && r.cps != null).map(r => `${r.name} (${r.atoms_per_ms} atoms/ms = ${r.atoms_per_s.toLocaleString('en-US')} atoms/s, CPS ${r.cps})`).join('; ') || 'Mena') + ' measured on an Apple M3 Max laptop; all other models ran on an NVIDIA H20 96 GB server GPU (DPA4 Fig. 1c / Table S-17).',
      source: 'Matbench Discovery CPS vs. saturated diamond E+F+S throughput (1 atom/ms = 1,000 atoms/s). Up and to the right is better.',
    }));
    if (dPts.length) scatter(fig4, { points: dPts, x_label: 'saturated throughput (atoms/ms, log scale, higher is better; 10 = 10,000 atoms/s)', y_label: 'Matbench Discovery CPS (higher is better)', x_unit: '' });
  }
  const lam6SpeedFig = lam6SpeedFigure(b.lam6_speed_scaling);
  if (lam6SpeedFig) box.append(lam6SpeedFig);
  const lam6BenchFig = factsFigure(b.lam6_materials_benchmarks, 'wide-showcase');
  if (lam6BenchFig) box.append(lam6BenchFig);

  const extra = [];
  const sc = b.showcase;
  if (sc && sc.facts) {
    extra.push(el('figure', { class: 'pfig showcase wide-showcase' },
      el('figcaption', { class: 'pfig-head' },
        el('span', { class: 'pfig-title' }, sc.title),
        el('span', { class: 'hw-badge' }, 'Apple M3 Max 36 GB')),
      el('dl', { class: 'facts showcase-facts' }, Object.entries(sc.facts).map(([k, v]) => el('div', { class: 'fact-pair' }, el('dt', {}, k), el('dd', { class: 'num' }, String(v))))),
      sc.note ? el('div', { class: 'pfig-src' }, el('p', {}, sc.note)) : null));
  }
  const t2p = factsFigure(b.materials_tier2_preview, 'wide-showcase');
  if (t2p) extra.push(t2p);
  if (extra.length) {
    box.append(el('details', { class: 'perf-more wide-showcase' },
      el('summary', {}, 'Additional laptop & earlier-model benchmarks (CsSnI₃ 10k showcase, Mena-LAM-s on crystals)'),
      el('div', { class: 'perf-more-body' }, ...extra)));
  }

  sec.append(box);
  return sec;
}

export async function showPerf(openFull) {
  if (!loaded) {
    loaded = true;
    clear(host);
    const live = el('section', { class: 'panel bench', 'aria-label': 'Run it on this computer' });
    const domainTabs = el('div', { class: 'seg perf-domain-tabs', role: 'tablist', 'aria-label': 'Benchmark domain filter' });
    const pubWrap = el('section', { class: 'pub-section', 'aria-label': 'Published leaderboard benchmarks' },
      el('div', { class: 'perf-head perf-head-row' },
        el('div', {},
          el('h2', { class: 'section-h' }, 'Published Benchmarks, Scaling & Hardware Comparisons'),
          el('p', { class: 'muted' }, 'Ground-truth molecular & periodic benchmarks with explicit hardware tags on every measurement.')),
        domainTabs));
    const statusBox = el('div');
    const pub = el('div', {}, el('p', { class: 'muted' }, 'Loading published numbers…'));
    pubWrap.prepend(statusBox);
    pubWrap.append(pub);
    host.append(live, pubWrap);
    mountBench(live);
    try {
      const b = await api('GET', '/api/benchmarks');
      clear(statusBox);
      const callout = archStatusCallout(b.architecture_status);
      if (callout) statusBox.append(callout);

      const matSec = materialsSection(b);
      const molSec = moleculesSection(b);
      clear(pub).append(...[matSec, molSec].filter(Boolean));

      let activeDomain = 'materials';
      const tabDefs = [
        ['materials', 'Materials (Mena-LAM-6)'],
        ['molecules', 'Molecules (Mena-LAM-s)'],
        ['all', 'Both domains'],
      ];
      const syncDomain = () => {
        if (molSec) molSec.hidden = activeDomain === 'materials';
        if (matSec) matSec.hidden = activeDomain === 'molecules';
        [...domainTabs.children].forEach((btn, i) => btn.setAttribute('aria-selected', String(tabDefs[i][0] === activeDomain)));
        window.dispatchEvent(new Event('resize'));
      };
      tabDefs.forEach(([id, lbl]) => {
        domainTabs.append(el('button', {
          type: 'button',
          role: 'tab',
          'aria-selected': String(id === activeDomain),
          onclick: () => { activeDomain = id; syncDomain(); },
        }, lbl));
      });
      syncDomain();
    } catch (e) {
      clear(pub).append(el('div', { class: 'msg error' }, `Published numbers are not available: ${e.message}`));
      loaded = false;
    }
  }
  if (openFull) openBenchJob(openFull);
}

export function mountPerf(h) { host = h; }
