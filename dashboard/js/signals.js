// Signals view: summary sentence, the sentiment "mood line", and the ticker list.

import { esc, money, price as fmtPrice, signed, pctChange, tone, sentence, meter, splitBar } from './format.js';
import { openDetail } from './detail.js';

const state = { recs: [], filter: 'ALL', sort: 'score', settled: false };
const el = (id) => document.getElementById(id);
const st = (r) => r.short_term || {};

const FILTERS = {
  ALL: () => true,
  BUY: (r) => tone(st(r).signal) === 'buy',
  SELL: (r) => tone(st(r).signal) === 'sell',
  HOLD: (r) => (st(r).signal || 'HOLD') === 'HOLD',
};
const SORTS = {
  score: (a, b) => b.composite_score - a.composite_score,
  conf: (a, b) => (st(b).confidence || 0) - (st(a).confidence || 0),
  mentions: (a, b) => b.mention_count - a.mention_count,
};

export function setSignals(data) {
  state.recs = data.recommendations || [];
  renderSummary(data);
  renderMood();
  renderList();
}

/* ── Summary ─────────────────────────────────────────────── */
function renderSummary(d) {
  const r = state.recs;
  const count = (term, t) => r.filter((x) => tone((x[term] || {}).signal) === t).length;
  const n = (v) => `<strong>${(+v || 0).toLocaleString()}</strong>`;
  el('summary').innerHTML =
    `${n(d.total_mentions)} mentions, ${n(d.reddit_mentions)} from Reddit and ${n(d.news_mentions)} from news. ` +
    `Short term: <strong class="t-buy">${count('short_term', 'buy')} buy</strong>, ` +
    `<strong class="t-sell">${count('short_term', 'sell')} sell</strong>. ` +
    `Long term: <strong class="t-buy">${count('long_term', 'buy')} buy</strong>.` +
    (d.portfolio_size ? ` Positions are sized for a ${money(d.portfolio_size)} portfolio.` : '');
  el('countSignals').textContent = r.length;
}

/* ── Mood line ───────────────────────────────────────────── */
function renderMood() {
  const plot = el('moodPlot');
  const recs = state.recs;
  const maxAbs = Math.max(0.2, ...recs.map((r) => Math.abs(r.composite_score || 0)));
  const domain = Math.ceil(maxAbs * 10) / 10;
  const maxMentions = Math.max(1, ...recs.map((r) => r.mention_count || 0));

  el('moodMin').textContent = signed(-domain, 1);
  el('moodMax').textContent = signed(domain, 1);

  plot.innerHTML = recs
    .map((r) => {
      const x = Math.max(1.5, Math.min(98.5, 50 + ((r.composite_score || 0) / domain) * 50));
      const size = 6 + Math.round(Math.sqrt((r.mention_count || 0) / maxMentions) * 12);
      return `<button class="mark t-${tone(st(r).signal)}" data-ticker="${esc(r.ticker)}" data-x="${x}" data-mentions="${r.mention_count || 0}"
        style="--x:${state.settled ? x : 50}%;--dot:${size}px"
        aria-label="${esc(r.ticker)}, score ${signed(r.composite_score, 3)}, ${r.mention_count} mentions">
        <span class="mark-label">${esc(r.ticker)}</span><span class="mark-stem"></span><span class="mark-dot"></span>
      </button>`;
    })
    .join('');

  layoutLanes();
  dimFiltered();
  if (!state.settled) {
    // One orchestrated moment: tickers drift from neutral to their score.
    void plot.offsetWidth; // commit the neutral start position so the move transitions
    plot.querySelectorAll('.mark').forEach((m) => m.style.setProperty('--x', m.dataset.x + '%'));
    state.settled = true;
  }
}

/** Stack labels into lanes so neighbouring tickers never overlap. */
function layoutLanes() {
  const plot = el('moodPlot');
  const width = plot.clientWidth;
  if (!width) return;
  const marks = [...plot.querySelectorAll('.mark')].map((m) => {
    const w = m.querySelector('.mark-label').offsetWidth + 10;
    const center = (m.dataset.x / 100) * width;
    return { m, left: center - w / 2, right: center + w / 2 };
  });
  // Busiest tickers claim label space first; any that cannot fit keep only their dot.
  marks.sort((a, b) => b.m.dataset.mentions - a.m.dataset.mentions);
  const maxLanes = width < 600 ? 9 : 7;
  const lanes = Array.from({ length: maxLanes }, () => []);
  let used = 0;
  for (const item of marks) {
    const lane = lanes.findIndex((spans) => spans.every(([l, r]) => item.right <= l || item.left >= r));
    item.m.classList.toggle('no-label', lane === -1);
    if (lane === -1) { item.m.style.setProperty('--lane', 0); continue; }
    lanes[lane].push([item.left, item.right]);
    used = Math.max(used, lane + 1);
    item.m.style.setProperty('--lane', lane);
  }
  const lanesUsed = used;
  plot.style.setProperty('--lanes', Math.max(1, lanesUsed));
}

function dimFiltered() {
  const match = FILTERS[state.filter];
  el('moodPlot').querySelectorAll('.mark').forEach((m) => {
    const rec = state.recs.find((r) => r.ticker === m.dataset.ticker);
    m.classList.toggle('is-dim', !match(rec));
  });
}

/* ── Ticker list ─────────────────────────────────────────── */
function renderList() {
  const rows = state.recs.filter(FILTERS[state.filter]).sort(SORTS[state.sort]);
  el('signalList').innerHTML = rows.length
    ? rows.map(rowHTML).join('')
    : `<li class="empty">No tickers have a ${state.filter.toLowerCase()} signal right now. Choose All to see every ticker.</li>`;
}

function termCell(p, extra) {
  const t = tone(p.signal || 'HOLD');
  return `<span class="term t-${t}">
    <span class="term-signal">${sentence(p.signal || 'HOLD')}</span>
    <span class="term-conf">${meter(p.confidence, t)}<span>${p.confidence || 0}%</span></span>
    <span class="term-extra">${extra}</span>
  </span>`;
}

function rowHTML(r) {
  const s = st(r), lt = r.long_term || {};
  const t = tone(s.signal || 'HOLD');
  const price = r.price
    ? `<span class="price">${fmtPrice(r.price)}</span>${
        r.change_pct != null ? `<span class="change t-${r.change_pct >= 0 ? 'buy' : 'sell'}">${pctChange(r.change_pct)}</span>` : ''
      }<span class="sub">RSI ${r.rsi || '—'}</span>`
    : `<span class="sub">No price</span>`;
  const stExtra = s.entry ? `$${s.entry} to $${s.take_profit_1}` : '';
  const ltExtra = lt.analyst_target ? `Target $${lt.analyst_target}` : lt.entry ? `$${lt.entry}` : '';

  return `<li><button class="row t-${t}" data-ticker="${esc(r.ticker)}">
    <span class="c-ticker"><span class="ticker">${esc(r.ticker)}</span><span class="sub tone">${esc(s.action_short || sentence(s.signal || 'HOLD'))}</span></span>
    <span class="c-price">${price}</span>
    <span class="c-st"><span class="col-label">Short term</span>${termCell(s, stExtra)}</span>
    <span class="c-lt"><span class="col-label">Long term</span>${termCell(lt, ltExtra)}</span>
    <span class="c-mood">${splitBar(r.bullish_pct, r.neutral_pct, r.bearish_pct)}</span>
    <span class="c-num c-mentions"><span class="col-label">Mentions</span>${r.mention_count}</span>
    <span class="c-num c-position"><span class="col-label">Position</span>${s.position?.shares ? `${s.position.shares} sh` : '—'}</span>
    <span class="c-score tone">${signed(r.composite_score, 3)}</span>
  </button></li>`;
}

/* ── Wiring ──────────────────────────────────────────────── */
export function initSignals() {
  const open = (e) => {
    const btn = e.target.closest('[data-ticker]');
    const rec = btn && state.recs.find((r) => r.ticker === btn.dataset.ticker);
    if (rec) openDetail(rec);
  };
  el('signalList').addEventListener('click', open);
  el('moodPlot').addEventListener('click', open);

  el('filters').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-filter]');
    if (!btn) return;
    state.filter = btn.dataset.filter;
    el('filters').querySelectorAll('[data-filter]').forEach((b) => b.setAttribute('aria-pressed', b === btn));
    renderList();
    dimFiltered();
  });
  el('sort').addEventListener('change', (e) => {
    state.sort = e.target.value;
    renderList();
  });
  let lastWidth = 0;
  new ResizeObserver(([entry]) => {
    const w = Math.round(entry.contentRect.width);
    if (w !== lastWidth) { lastWidth = w; layoutLanes(); }
  }).observe(el('moodPlot'));
}
