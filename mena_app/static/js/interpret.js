// "Interpretation" panel of a single-point result: each physics-head quantity with its value, a trust badge,
// what it means and its caveat. Optional viewer hook: a button switches the 3D colouring to that quantity.
import { el, fmt } from './util.js';

const VIEW_MODE = { unpaired: 'unpaired', pauli: 'pauli', charges: 'charges', spin: 'spins' };

function row(it, viewer) {
  const mode = VIEW_MODE[it.key];
  const canColor = viewer && mode && viewer.colorModes && viewer.colorModes.some(m => m.id === mode);
  return el('div', { class: 'interp-item' },
    el('div', { class: 'interp-head' },
      el('span', { class: 'interp-label' }, it.label),
      el('span', { class: 'interp-val num' }, fmt(it.value, it.digits), it.unit ? el('span', { class: 'tile-u' }, it.unit) : null),
      el('span', { class: 'trust ' + it.trust, title: 'How far this quantity can be trusted, from tests against known chemistry' }, it.trust_label),
      canColor ? el('button', { class: 'btn small ghost', type: 'button', title: 'Colour the 3D structure by this quantity',
        onclick: () => viewer.setColorMode(mode) }, 'Show on structure') : null),
    it.detail ? el('div', { class: 'interp-detail num' }, it.detail) : null,
    el('p', { class: 'interp-meaning' }, it.meaning),
    el('p', { class: 'interp-caveat' }, el('b', {}, 'Caveat: '), it.caveat));
}

export function renderInterpretation(panel, viewer) {
  const items = panel.items || [];
  return el('details', { class: 'more table-box interp', open: '' },
    el('summary', {}, panel.title || 'Interpretation', el('span', { class: 'count' }, panel.model || '')),
    el('div', { class: 'interp-body' },
      panel.intro ? el('p', { class: 'note', style: 'margin-top:0' }, panel.intro) : null,
      items.map(it => row(it, viewer)),
      (panel.notes || []).length ? el('ul', { class: 'notes' }, panel.notes.map(n => el('li', {}, n))) : null));
}
