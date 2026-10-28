// Shared formatting helpers. Everything rendered into HTML goes through esc().

const ENTITIES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ENTITIES[c]);

/** Only allow http(s) links from scraped sources. */
export const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : null);

export const plus = (n) => (n >= 0 ? '+' : '');
export const signed = (n, digits) => plus(n) + (digits == null ? n : n.toFixed(digits));
export const money = (n) => '$' + Number(n || 0).toLocaleString();
export const price = (n) => '$' + Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const signedMoney = (n) => (n < 0 ? '−' : '+') + money(Math.abs(n || 0));
export const pctChange = (n) => `${plus(n)}${n}%`;

/** Map an engine signal (STRONG BUY … STRONG SELL) to a colour tone. */
export function tone(signal) {
  if (signal === 'BUY' || signal === 'STRONG BUY') return 'buy';
  if (signal === 'SELL' || signal === 'STRONG SELL') return 'sell';
  return 'hold';
}
export const upDown = (n, invert = false) => ((n >= 0) !== invert ? 'buy' : 'sell');

/** "STRONG BUY" → "Strong buy" */
export const sentence = (s) => (s ? esc(s.charAt(0) + s.slice(1).toLowerCase()) : '—');

/** Seconds → "4h 18m" */
export const hm = (secs) => `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;

/** One labelled figure. Used for every small stat grid in the app. */
export function metric(label, value, { tone: t, note } = {}) {
  return `<div class="metric${t ? ' t-' + t : ''}"><dt>${esc(label)}</dt><dd>${value ?? '—'}</dd>${
    note ? `<p class="metric-note">${note}</p>` : ''
  }</div>`;
}
export const metrics = (items, cls = '') => `<dl class="metrics ${cls}">${items.join('')}</dl>`;

export const section = (title, body) =>
  `<section class="pane-section"><h3 class="pane-title">${title}</h3>${body}</section>`;

/** Horizontal fill meter, 0–100. */
export const meter = (pct, t) =>
  `<span class="meter${t ? ' t-' + t : ''}"><span style="width:${Math.max(0, Math.min(100, pct || 0))}%"></span></span>`;

/** Bullish / neutral / bearish split bar with labels. */
export function splitBar(bull = 0, neut = 0, bear = 0) {
  return `<span class="split" role="img" aria-label="${bull}% bullish, ${neut}% neutral, ${bear}% bearish">
    <span class="split-bar"><span class="t-buy" style="width:${bull}%"></span><span class="neutral" style="width:${neut}%"></span><span class="t-sell" style="width:${bear}%"></span></span>
    <span class="split-legend"><span class="t-buy">${bull}% bull</span><span>${neut}%</span><span class="t-sell">${bear}% bear</span></span>
  </span>`;
}
