// Signals view: summary sentence, the "chatter map" treemap, the ranked
// mover lists, and the full ticker list.

import { esc, money, price as fmtPrice, signed, pctChange, tone, sentence, meter, splitBar } from './format.js';
import { openDetail } from './detail.js';

const state = { recs: [], filter: 'ALL', sort: 'score', developed: false };
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
  renderMap();
  renderMovers();
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

/* ── Chatter map: area = mentions, colour = sentiment ───── */

/** Squarified treemap (Bruls et al.): lays items out as near-square tiles. */
function squarify(items, box) {
  const total = items.reduce((s, i) => s + i.value, 0) || 1;
  const scale = (box.w * box.h) / total;
  const rest = items.map((i) => ({ ...i, area: i.value * scale }));
  const out = [];
  let row = [];
  let { x, y, w, h } = box;

  const worst = (r, side) => {
    const s = r.reduce((a, n) => a + n.area, 0);
    const max = Math.max(...r.map((n) => n.area));
    const min = Math.min(...r.map((n) => n.area));
    return Math.max((side * side * max) / (s * s), (s * s) / (side * side * min));
  };
  const place = (r) => {
    const s = r.reduce((a, n) => a + n.area, 0);
    if (w >= h) {
      const cw = s / h;
      let cy = y;
      for (const n of r) { const nh = n.area / cw; out.push({ ...n, x, y: cy, w: cw, h: nh }); cy += nh; }
      x += cw; w -= cw;
    } else {
      const rh = s / w;
      let cx = x;
      for (const n of r) { const nw = n.area / rh; out.push({ ...n, x: cx, y, w: nw, h: rh }); cx += nw; }
      y += rh; h -= rh;
    }
  };

  while (rest.length) {
    const side = Math.min(w, h);
    if (!row.length || worst([...row, rest[0]], side) <= worst(row, side)) row.push(rest.shift());
    else { place(row); row = []; }
  }
  if (row.length) place(row);
  return out;
}

function renderMap() {
  const map = el('chatterMap');
  const recs = state.recs;
  if (!recs.length) { map.innerHTML = ''; return; }
  const maxMentions = Math.max(1, ...recs.map((r) => r.mention_count || 0));
  const maxAbs = Math.max(0.05, ...recs.map((r) => Math.abs(r.composite_score || 0)));
  const items = [...recs]
    .map((r) => ({ r, value: Math.max(r.mention_count || 0, maxMentions * 0.03) }))
    .sort((a, b) => b.value - a.value);
  const box = { x: 0, y: 0, w: map.clientWidth, h: map.clientHeight };
  if (!box.w || !box.h) return;

  map.innerHTML = squarify(items, box).map(({ r, x, y, w, h }) => {
    const score = r.composite_score || 0;
    const strength = Math.min(1, Math.abs(score) / maxAbs);
    const t = score > 0.005 ? 'buy' : score < -0.005 ? 'sell' : 'flat';
    // Text size follows tile width so tickers never get clipped.
    const size = w * h > 22000 && w > 150 ? 'l' : w * h > 7000 && w > 96 ? 'm' : w > 64 && h > 58 ? 'n' : 's';
    return `<button class="tile t-${t} size-${size}${strength > 0.55 ? ' strong' : ''}" data-ticker="${esc(r.ticker)}"
      style="left:${x}px;top:${y}px;width:${w}px;height:${h}px;--mix:${Math.round(16 + strength * 72)}%"
      aria-label="${esc(r.ticker)}: score ${signed(score, 3)}, ${r.mention_count} mentions, short term ${sentence(st(r).signal || 'HOLD')}">
      <span class="tile-ticker">${esc(r.ticker)}</span>
      <span class="tile-score">${signed(score, 3)}</span>
      <span class="tile-meta">${r.mention_count} mentions</span>
    </button>`;
  }).join('');

  // One orchestrated moment: on first load the colours develop from grey.
  if (!state.developed) {
    map.classList.add('is-undeveloped');
    void map.offsetWidth;
    map.classList.remove('is-undeveloped');
    state.developed = true;
  }
  dimFiltered();
}

function dimFiltered() {
  const match = FILTERS[state.filter];
  el('chatterMap').querySelectorAll('.tile').forEach((tile) => {
    const rec = state.recs.find((r) => r.ticker === tile.dataset.ticker);
    tile.classList.toggle('is-dim', !match(rec));
  });
}

/* ── Movers ──────────────────────────────────────────────── */
function renderMovers() {
  const recs = state.recs;
  const top = (list) => list.slice(0, 5);
  const byScore = [...recs].sort(SORTS.score);
  const lists = {
    moversBull: top(byScore.filter((r) => r.composite_score > 0)),
    moversBear: top(byScore.filter((r) => r.composite_score < 0).reverse()),
    moversTalk: top([...recs].sort(SORTS.mentions)),
  };
  const maxMentions = Math.max(1, ...recs.map((r) => r.mention_count || 0));
  for (const [id, list] of Object.entries(lists)) {
    const talk = id === 'moversTalk';
    el(id).innerHTML = list.length
      ? list.map((r) => `<li><button class="mover" data-ticker="${esc(r.ticker)}">
          <span class="mover-ticker">${esc(r.ticker)}</span>
          ${talk
            ? `<span class="mover-bar">${meter((r.mention_count / maxMentions) * 100)}</span><span class="mover-val">${r.mention_count}</span>`
            : `<span class="mover-sub">${sentence(st(r).signal || 'HOLD')}</span><span class="mover-val t-${tone(r.composite_score > 0 ? 'BUY' : 'SELL')}">${signed(r.composite_score, 3)}</span>`}
        </button></li>`).join('')
      : '<li class="mover-empty">None right now.</li>';
  }
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
  for (const id of ['signalList', 'chatterMap', 'moversBull', 'moversBear', 'moversTalk']) el(id).addEventListener('click', open);

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

  // Re-tile when the map's size changes (window resize, phone rotation).
  let last = '';
  new ResizeObserver(([entry]) => {
    const size = `${Math.round(entry.contentRect.width)}x${Math.round(entry.contentRect.height)}`;
    if (size !== last) { last = size; if (state.recs.length) renderMap(); }
  }).observe(el('chatterMap'));
}
