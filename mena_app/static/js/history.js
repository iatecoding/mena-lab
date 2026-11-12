// History: past jobs (task, structure, model, status, duration, headline). Click opens one in the workspace.
import { el, clear, api, fmtDuration, fmtDate, toast } from './util.js';
import { emit } from './state.js';
import { STATUS, formulaNode } from './labels.js';

let host;

export async function renderHistory() {
  clear(host);
  host.append(el('div', { class: 'hist-head' }, el('h2', { class: 'section-h' }, 'History'),
    el('button', { class: 'btn ghost small', type: 'button', onclick: renderHistory }, 'Refresh')));
  let jobs;
  try { jobs = (await api('GET', '/api/jobs')).jobs; } catch (e) { host.append(el('div', { class: 'msg error' }, e.message)); return; }
  if (!jobs.length) { host.append(el('p', { class: 'empty' }, 'No runs yet. Results you run in the workspace appear here and are kept between sessions.')); return; }
  const tbl = el('table', { class: 'tbl hist' });
  tbl.append(el('thead', {}, el('tr', {}, ['When', 'Task', 'Structure', 'Model', 'Status', 'Time', 'Result', ''].map(h => el('th', { scope: 'col' }, h)))));
  const body = el('tbody');
  [...jobs].reverse().forEach(j => {
    const a = j.analysis || {}, live = j.status === 'running' || j.status === 'queued';
    const open = () => emit('open-job', j.id);
    const del = el('button', { class: 'btn ghost small', type: 'button', disabled: live || null, 'aria-label': 'Delete this run',
      onclick: async e => {
        e.stopPropagation();
        try { await api('DELETE', `/api/jobs/${j.id}`); tr.remove(); toast('Run deleted.'); } catch (err) { toast(err.message, 'error'); }
      } }, 'Delete');
    const tr = el('tr', { class: 'clickable', tabindex: 0, onclick: open, onkeydown: e => { if (e.key === 'Enter') open(); } },
      el('td', { class: 'num nowrap' }, fmtDate(j.created)),
      el('td', {}, j.test_title || j.task),
      el('td', {}, a.formula ? formulaNode(a.formula) : '–', a.n_atoms ? el('span', { class: 'muted' }, ` · ${a.n_atoms} atom${a.n_atoms === 1 ? '' : 's'}`) : null),
      el('td', {}, (j.model && j.model.title) || '–'),
      el('td', {}, el('span', { class: 'status ' + j.status }, STATUS[j.status] || j.status)),
      el('td', { class: 'num nowrap' }, fmtDuration(j.duration_s)),
      el('td', { class: 'hist-headline' }, j.headline || j.error || ''),
      el('td', {}, del));
    body.append(tr);
  });
  tbl.append(body);
  host.append(el('div', { class: 'table-wrap tall' }, tbl));
}

export function mountHistory(h) { host = h; }
