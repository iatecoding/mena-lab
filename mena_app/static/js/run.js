// Live view of one running job over Server-Sent Events: status, progress bar, elapsed / remaining time, live charts,
// live frames in the viewer, the log, and the error if it fails.
import { el, clear, api, fmt, fmtDuration, toast, approx } from './util.js';
import { Chart } from './charts.js';
import { liveChart, STATUS } from './labels.js';

export class RunPanel {
  constructor(host, job, opts) {
    this.host = host; this.id = job.id; this.opts = opts; // opts: {title, estimate, viewer, onEnd}
    this.charts = {}; this.done = 0; this.total = 0; this.state = job.status || 'queued';
    this.t0 = job.started ? job.started * 1000 : null;
    this.build(); this.connect();
    this.timer = setInterval(() => this.tick(), 250);
  }
  build() {
    clear(this.host);
    this.pill = el('span', { class: 'status queued' }, STATUS.queued);
    this.bar = el('div', { class: 'bar indet', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-label': 'Progress' }, el('i'));
    this.label = el('div', { class: 'run-label', 'aria-live': 'polite' }, 'Starting…');
    this.elapsed = el('span', { class: 'num' }, '0.0 s');
    this.remaining = el('span', { class: 'num' }, this.opts.estimate ? approx(this.opts.estimate) : '–');
    this.stats = el('div', { class: 'stats' });
    // Live charts go to opts.chartHost (the stage beside the viewer) when given, else under the progress bar.
    this.chartGrid = this.opts.chartHost || el('div', { class: 'charts', hidden: true });
    if (this.opts.chartHost) clear(this.chartGrid);
    this.groups = {};
    this.log = el('div', { class: 'log', role: 'log', 'aria-label': 'Log' });
    this.errBox = el('div', { class: 'error-box', role: 'alert', hidden: true });
    this.host.append(
      el('div', { class: 'run-head' }, el('h2', { class: 'section-h' }, this.opts.title || 'Running'), this.pill,
        el('div', { class: 'times' }, el('span', {}, 'Elapsed ', this.elapsed), el('span', {}, 'Remaining ', this.remaining))),
      this.bar,
      el('div', { class: 'run-meta' }, this.label, this.stats),
      this.errBox, ...(this.opts.chartHost ? [] : [this.chartGrid]),
      el('details', { class: 'more log-box' }, el('summary', {}, 'Log'), this.log));
  }
  tick() {
    if (this.finished || this.state !== 'running' || !this.t0) return;
    const s = (Date.now() - this.t0) / 1000;
    this.elapsed.textContent = fmtDuration(s);
    const frac = this.total > 0 ? this.done / this.total : 0;
    if (frac > 0.03 && frac < 1) this.remaining.textContent = approx(s * (1 / frac - 1));
    else if (this.opts.estimate && s < this.opts.estimate) this.remaining.textContent = approx(this.opts.estimate - s);
  }
  connect() {
    const es = this.es = new EventSource(`/api/jobs/${this.id}/events`);
    const on = (name, fn) => es.addEventListener(name, e => { try { fn(JSON.parse(e.data)); } catch (err) { console.error(err); } });
    on('state', d => this.setState(d.status));
    on('status', d => { this.label.textContent = d.text; this.addLog(d.text, true); });
    on('log', d => this.addLog(d.text));
    on('progress', d => this.progress(d));
    on('series', d => this.series(d));
    on('frame', d => { if (d.frame && this.opts.viewer) this.opts.viewer.live(d.frame.positions, d.frame.label, d.frame); });
    on('error', d => { this.error = d; if (d.kind !== 'cancelled') this.showError(d.message); });
    es.addEventListener('end', () => this.finish());
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) this.finish(); };
  }
  setState(s) {
    this.state = s;
    this.pill.className = 'status ' + s; this.pill.textContent = STATUS[s] || s;
    if (s === 'running' && !this.t0) this.t0 = Date.now();
  }
  progress(d) {
    this.done = d.done; this.total = d.total;
    if (d.total > 0) {
      this.bar.classList.remove('indet');
      const pct = Math.min(100, 100 * d.done / d.total);
      this.bar.firstChild.style.width = pct + '%'; this.bar.setAttribute('aria-valuenow', Math.round(pct));
    }
    if (d.label) this.label.textContent = d.label;
    clear(this.stats);
    Object.entries(d.stats || {}).forEach(([k, v]) => {
      if (v !== null && v !== undefined) this.stats.append(el('span', { class: 'stat' }, k, ' ', el('b', { class: 'num' }, typeof v === 'number' ? fmt(v) : String(v))));
    });
  }
  series(d) {
    let c = this.charts[d.chart];
    if (!c) {
      const meta = (this.opts.liveCharts && this.opts.liveCharts[d.chart]) || liveChart(d.chart);
      this.chartGrid.hidden = false;
      // One group per phase (pre-relax, end-point relax, NEB, MD): the newest phase goes on top.
      const prefix = d.chart.replace(/_[^_]+$/, '');
      let g = this.groups[prefix];
      if (!g) { g = this.groups[prefix] = el('div', { class: 'chart-group' }); this.chartGrid.prepend(g); }
      c = this.charts[d.chart] = new Chart(g, { id: d.chart, kind: 'line', compact: !!this.opts.chartHost, ...meta, ...(this.opts.chartMeta || {}), series: [] });
    }
    c.append(d.name, d.x, d.y);
  }
  addLog(text, dim) {
    const near = this.log.scrollTop + this.log.clientHeight >= this.log.scrollHeight - 30;
    this.log.append(el('div', { class: dim ? 'dim' : '' }, text));
    while (this.log.childNodes.length > 500) this.log.removeChild(this.log.firstChild);
    if (near) this.log.scrollTop = this.log.scrollHeight;
  }
  showError(msg) { this.errBox.hidden = false; clear(this.errBox).append(el('strong', {}, 'The job failed. '), String(msg || 'Unknown error')); }
  async cancel() {
    try { await api('POST', `/api/jobs/${this.id}/cancel`); } catch (e) { toast(e.message, 'error'); }
  }
  async finish() {
    if (this.finished) return;
    this.finished = true; this.es.close(); clearInterval(this.timer);
    this.bar.classList.remove('indet');
    let full = null;
    try { full = await api('GET', `/api/jobs/${this.id}`); } catch (e) { toast(e.message, 'error'); }
    if (full) {
      this.setState(full.status);
      this.elapsed.textContent = fmtDuration(full.duration_s);
      this.remaining.textContent = '–';
      if (full.status === 'done') { this.bar.firstChild.style.width = '100%'; this.label.textContent = 'Finished.'; }
      else if (full.status === 'cancelled') this.label.textContent = 'Stopped. Results below cover what was done, if any.';
      if (full.error && full.status !== 'cancelled' && this.errBox.hidden) this.showError(full.error);
    }
    this.opts.onEnd && this.opts.onEnd(full);
  }
  destroy() {
    this.finished = true; if (this.es) this.es.close(); clearInterval(this.timer);
    Object.values(this.charts).forEach(c => c.destroy());
    if (this.opts.chartHost) clear(this.opts.chartHost);
  }
}
