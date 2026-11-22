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
    Object.entries(sec.facts).flatMap(([k, v]) => [el('dt', {}, k), el('dd', { class: 'num' }, String(v))]));
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
      el('span', { class: 'hw-tag mena' }, 'Mena-LAM-preview (lam-400k): 1× A100 / Apple M3 Max laptop'),
      el('span', { class: 'hw-vs' }, 'vs.'),
      el('span', { class: 'hw-tag other' }, 'Competitors: NVIDIA H100 / H200 / H20 96 GB')),
    el('div', { class: 'hw-banner-grid' },
      el('p', {}, el('strong', {}, 'Matbench Discovery DynaMat MD (A100 vs. H100/H200): '), hb.dynamat),
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
  return figure(ts.title, ts.badge, el('div', { class: 'perf-tbl-wrap' }, tbl), ts);
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

function materialsSection(b) {
  const sec = el('section', { class: 'domain-section', 'aria-label': 'Periodic materials benchmarks' },
    el('div', { class: 'domain-head' },
      el('div', {},
        el('h3', { class: 'domain-title' }, 'Crystals & Surfaces — Mena-LAM-6 (default) & Published Benchmarks'),
        el('p', { class: 'muted domain-desc' }, 'Crystals and surfaces run on Mena-LAM-6 (lam6-v1: Matbench Discovery CPS 0.874, F1 0.904, κ_SRME 0.182) by default. Below are the published DynaMat v1.0 and DPA4 dense-diamond comparisons for Mena-LAM-preview (lam-400k) and large-crystal scaling for Mena-LAM-s.')),
      el('span', { class: 'domain-tag active' }, 'Default: Mena-LAM-6 (lam6-v1)')),
    hardwareCallout(b.hardware_banner),
    hero(b.headline));

  const box = el('div', { class: 'pgrid' });
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
    box.append(figure(dm.title || 'Peak GPU memory in DynaMat MD', 'Mena: A100 · Others: H100/H200', fig1, dm));
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
    box.append(figure('Speed vs. accuracy (CMDS) in DynaMat MD', '1× A100 beats 41 H100/H200s', fig2,
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
    box.append(figure(di.title || 'Memory per atom, dense diamond', 'Mena: M3 Max · Others: H20 96GB', fig3, di));
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
      note: (di.rows.filter(r => r.mena && r.cps != null).map(r => `${r.name} (${r.atoms_per_ms} atoms/ms = ${r.atoms_per_s.toLocaleString('en-US')} atoms/s, CPS ${r.cps})`).join('; ') || 'Mena') + ' is measured on an Apple M3 Max laptop; all other models ran on an NVIDIA H20 96 GB server GPU (DPA4 Fig. 1c / Table S-17).',
      source: 'Matbench Discovery CPS vs. saturated diamond E+F+S throughput (1 atom/ms = 1,000 atoms/s). Up and to the right is better.',
    }));
    if (dPts.length) scatter(fig4, { points: dPts, x_label: 'saturated throughput (atoms/ms, log scale, higher is better; 10 = 10,000 atoms/s)', y_label: 'Matbench Discovery CPS (higher is better)', x_unit: '' });
  }
  const sc = b.showcase;
  if (sc && sc.facts) {
    box.append(el('figure', { class: 'pfig showcase wide-showcase' },
      el('figcaption', { class: 'pfig-head' },
        el('span', { class: 'pfig-title' }, sc.title),
        el('span', { class: 'hw-badge' }, 'Apple M3 Max 36 GB')),
      el('dl', { class: 'facts showcase-facts' }, Object.entries(sc.facts).flatMap(([k, v]) => [el('dt', {}, k), el('dd', { class: 'num' }, String(v))])),
      sc.note ? el('div', { class: 'pfig-src' }, el('p', {}, sc.note)) : null));
  }
  const t2p = factsFigure(b.materials_tier2_preview, 'wide-showcase');
  if (t2p) box.append(t2p);

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

      const molSec = moleculesSection(b);
      const matSec = materialsSection(b);
      clear(pub).append(...[molSec, matSec].filter(Boolean));

      let activeDomain = 'all';
      const tabDefs = [
        ['all', 'Both domains'],
        ['molecules', 'Molecules (Mena-LAM-s)'],
        ['materials', 'Periodic Materials'],
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
    } catch (e) {
      clear(pub).append(el('div', { class: 'msg error' }, `Published numbers are not available: ${e.message}`));
      loaded = false;
    }
  }
  if (openFull) openBenchJob(openFull);
}

export function mountPerf(h) { host = h; }
