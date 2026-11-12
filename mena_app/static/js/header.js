// Header: version, device and free-memory pills (memory refreshed every 10 s), light / dark toggle, view tabs.
import { api, fmt, store } from './util.js';
import { devicesLabel } from './labels.js';

function effectiveTheme() {
  const t = document.documentElement.getAttribute('data-theme');
  if (t) return t;
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function setThemeButton(btn) {
  const dark = effectiveTheme() === 'dark';
  btn.textContent = dark ? 'Light' : 'Dark';
  btn.setAttribute('aria-label', dark ? 'Switch to the light theme' : 'Switch to the dark theme');
}

export function mountHeader(info, onTheme, onView) {
  document.getElementById('version').textContent = 'v' + info.version;
  const dev = document.getElementById('dev-pill');
  dev.textContent = devicesLabel(info.machine.devices);
  dev.title = 'Compute devices on this computer' + (info.machine.torch ? ` (PyTorch ${info.machine.torch})` : '');
  const mem = document.getElementById('mem-pill');
  const showMem = m => {
    const hasAvail = m.available_gb != null && Number.isFinite(m.available_gb);
    mem.textContent = hasAvail ? `${fmt(m.available_gb, 1)} GB free` : (m.total_gb ? `${fmt(m.total_gb, 1)} GB RAM` : 'Memory n/a');
    mem.title = hasAvail ? `Free memory: ${fmt(m.available_gb, 1)} of ${fmt(m.total_gb, 1)} GB` : `Total memory: ${fmt(m.total_gb, 1)} GB`;
    mem.classList.toggle('warn', hasAvail && m.available_gb < 4);
  };
  showMem(info.machine);
  setInterval(async () => { if (document.hidden) return; try { showMem(await api('GET', '/api/machine')); } catch (e) { /* server gone: keep the last value */ } }, 10000);

  const btn = document.getElementById('theme-btn');
  setThemeButton(btn);
  btn.addEventListener('click', () => {
    const next = effectiveTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    store('theme', next); setThemeButton(btn); onTheme();
  });
  if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { setThemeButton(btn); onTheme(); });

  document.querySelectorAll('.tabs [role=tab]').forEach(t => t.addEventListener('click', () => onView(t.dataset.view)));
}

export function setView(view) {
  document.querySelectorAll('.tabs [role=tab]').forEach(t => t.setAttribute('aria-selected', String(t.dataset.view === view)));
  ['workspace', 'history', 'performance'].forEach(v => { document.getElementById('view-' + v).hidden = v !== view; });
}
