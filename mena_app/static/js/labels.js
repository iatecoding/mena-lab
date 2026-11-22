// Display names the API does not carry. LOT names mirror mena_app/tasks/common.py LOT_LABELS.

export const LOT = {
  mp_pbe: 'PBE (Materials Project, PBE_52)', mp_pbeU: 'PBE+U (Materials Project, PBE_52)',
  omat_pbe: 'PBE (OMat24, PBE_54)', omat_pbeU: 'PBE+U (OMat24, PBE_54)',
  matpes_pbe: 'PBE (MatPES, PBE_64)', matpes_r2scan: 'r²SCAN (MatPES, PBE_64)',
  oc20_rpbe: 'RPBE (OC20 catalysis)', oc22_pbeU: 'PBE+U (OC22 oxide catalysis)',
  oc25_rpbe: 'RPBE (OC25 catalysis)',
  mad16_r2scan: 'r²SCAN (MAD-1.6 aims_tight)', mad16_pbe: 'PBE (MAD-1.6 aims_tight)',
  omc25_pbe: 'PBE+D3 (OMC25 molecular crystals)', omol_wb97mv: 'ωB97M-V/def2-TZVPD (OMol25)',
  opoly26_wb97mv: 'ωB97M-V/def2-TZVPD (OPoly26)',
  openrxn_wb97xd: 'ωB97X-D3/def2-TZVPPD (OpenRxn)', spice_wb97md3: 'ωB97M-D3(BJ)/def2-TZVPPD (SPICE)',
};

export const DEVICE = { cpu: 'CPU', mps: 'Apple GPU', cuda: 'NVIDIA GPU' };
export const DTYPE = { float32: 'Single (float32)', float64: 'Double (float64)' };
export const PRECISION = { auto: 'Automatic', fp32_ieee: 'Strict FP32 (IEEE)', exact: 'Double (FP64 exact)' };
export const DOMAIN = { molecule: 'Molecule', crystal: 'Crystal', slab: 'Surface slab' };
export const DOMAIN_PLURAL = { molecule: 'molecules', crystal: 'crystals', slab: 'surfaces' };

export const STATUS = {
  queued: 'Waiting', running: 'Running', done: 'Finished', cancelled: 'Stopped', error: 'Failed', timeout: 'Timed out', stopped: 'Stopped by guard',
};

// Titles for the live charts a running job streams (ids from mena_app/tasks/*.py).
export const LIVE_CHARTS = {
  opt_energy: { title: 'Energy', x_label: 'step', y_label: 'energy change (eV)' },
  opt_fmax: { title: 'Largest force', x_label: 'step', y_label: 'eV/Å', logy: true },
  md_temperature: { title: 'Temperature', x_label: 'time (ps)', y_label: 'K' },
  md_energy: { title: 'Total energy (per atom, from start)', x_label: 'time (ps)', y_label: 'meV/atom' },
  neb_barrier: { title: 'Barrier', x_label: 'step', y_label: 'eV' },
  neb_fmax: { title: 'Largest force', x_label: 'step', y_label: 'eV/Å', logy: true },
};

// Live chart metadata, including the NEB end-point relaxations (end_start_energy, end_end_fmax, ...).
export function liveChart(id) {
  if (LIVE_CHARTS[id]) return LIVE_CHARTS[id];
  const m = /^end_(\w+?)_(energy|fmax)$/.exec(id);
  if (m) {
    const base = LIVE_CHARTS['opt_' + m[2]], who = m[1] === 'start' ? 'Start structure' : m[1] === 'end' ? 'End structure' : m[1];
    return { ...base, title: `${who} relax: ${base.title.toLowerCase()}` };
  }
  return { title: id.replace(/_/g, ' '), x_label: 'step', y_label: '' };
}

export function lotLabel(id) { return LOT[id] || id; }
export function setLotLabels(labels) { Object.assign(LOT, labels || {}); }  // the API's names win
export function deviceLabel(id) { return DEVICE[id] || String(id).toUpperCase(); }
export function devicesLabel(list) { return (list || []).map(deviceLabel).join(' · '); }

// Plain-text subscripts, for places that only take text (the viewer label).
export function formulaText(f) { return String(f || '').replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[d]); }

// "C8H10N4O2" style formulas get subscripts in the page.
export function formulaNode(f) {
  const span = document.createElement('span');
  span.className = 'formula';
  String(f || '').split(/(\d+)/).forEach((part, i) => {
    if (!part) return;
    if (i % 2) { const s = document.createElement('sub'); s.textContent = part; span.append(s); }
    else span.append(part);
  });
  return span;
}
