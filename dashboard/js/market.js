// Market Health view: overall score, indices, macro, warnings, drivers, sectors.

import { api } from './api.js';
import { esc, signed, upDown } from './format.js';

const el = (id) => document.getElementById(id);
let loaded = false;

const INDICES = [
  { key: 'spy', name: 'S&P 500', symbol: 'SPY' },
  { key: 'qqq', name: 'Nasdaq 100', symbol: 'QQQ' },
  { key: 'dia', name: 'Dow Jones', symbol: 'DIA' },
  { key: 'iwm', name: 'Russell 2000', symbol: 'IWM' },
];
const MACRO = [
  { key: 'vix', name: 'VIX', label: 'Fear index', invert: true, fmt: (v) => v.toFixed(1) },
  { key: 'tnx', name: '10Y', label: 'Treasury yield', fmt: (v) => `${v.toFixed(2)}%` },
  { key: 'dxy', name: 'DXY', label: 'US dollar', fmt: (v) => v.toFixed(1) },
  { key: 'gld', name: 'GLD', label: 'Gold', fmt: (v) => `$${v.toFixed(0)}` },
  { key: 'uso', name: 'USO', label: 'Oil', fmt: (v) => `$${v.toFixed(1)}` },
];
const REGIME_TONE = { 'Risk-On': 'buy', 'Risk-Off': 'sell' };

export async function loadMarketHealth(force = false) {
  if (loaded && !force) return;
  el('mktStatus').hidden = false;
  el('mktStatus').textContent = 'Fetching market data…';
  el('mktContent').hidden = true;
  try {
    const { market_health } = await api.marketHealth(force);
    render(market_health);
    loaded = true;
    el('mktStatus').hidden = true;
    el('mktContent').hidden = false;
  } catch {
    el('mktStatus').textContent = 'Market data could not load. Check that app.py is running, then open this tab again.';
  }
}

const change = (chg, invert) => `<span class="t-${upDown(chg, invert)}">${signed(chg)}%</span>`;

function render(d) {
  const score = d.health_score || 50;
  const scoreTone = score >= 65 ? 'buy' : score >= 45 ? 'hold' : 'sell';
  const trajTone = d.trajectory === 'Bullish' ? 'buy' : d.trajectory === 'Bearish' ? 'sell' : 'hold';
  const updated = d.fetched_at ? new Date(d.fetched_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';

  el('mktBanner').innerHTML = `
    <div class="mkt-score t-${scoreTone}"><span class="score-num tone">${score}</span><span class="sub">out of 100</span></div>
    <div class="mkt-read">
      <p class="mkt-traj t-${trajTone}">${esc(d.traj_strength)} ${esc(d.trajectory)}</p>
      <p><span class="chip-static t-${REGIME_TONE[d.regime] || 'hold'}">${esc(d.regime || '—')}</span> ${esc(d.regime_desc)}</p>
    </div>
    <div class="mkt-bar t-${scoreTone}"><div class="bar"><span class="tone-bg" style="width:${score}%"></span></div><p class="sub">Updated ${updated}</p></div>`;

  el('mktIndices').innerHTML = INDICES.map(({ key, name, symbol }) => {
    const ix = d.indices?.[key];
    if (!ix) return `<tr><th scope="row">${name} <span class="sub">${symbol}</span></th><td colspan="5" class="muted">—</td></tr>`;
    const above = ix.above_sma50;
    return `<tr>
      <th scope="row">${name} <span class="sub">${symbol}</span></th>
      <td class="num">$${(ix.price || 0).toLocaleString()}</td>
      <td class="num">${change(ix.change_pct || 0)}</td>
      <td class="num">${ix.rsi || '—'}</td>
      <td class="num">${ix.ret_20d != null ? change(ix.ret_20d) : '—'}</td>
      <td class="num">${above == null ? '—' : `<span class="t-${above ? 'buy' : 'sell'}">${above ? 'Above' : 'Below'}</span>`}</td>
    </tr>`;
  }).join('');

  el('mktMacro').innerHTML = MACRO.map((m) => {
    const data = d.macro?.[m.key];
    const value = data?.price ? `<dd>${m.fmt(data.price)} ${change(data.change_pct || 0, m.invert)}</dd>` : '<dd class="muted">—</dd>';
    return `<div class="metric"><dt>${m.name} <span class="sub">${m.label}</span></dt>${value}</div>`;
  }).join('');

  el('mktWarnings').innerHTML = d.warnings?.length
    ? d.warnings.map((w) => `<p class="callout t-hold">${esc(w)}</p>`).join('')
    : '<p class="callout t-buy">No active market warnings.</p>';

  el('mktDrivers').innerHTML = (d.key_drivers || []).map((dr) => `<li>${esc(dr)}</li>`).join('');

  const sectors = d.all_sectors || [];
  const maxAbs = Math.max(1, ...sectors.map((s) => Math.abs(s.ret_5d || 0)));
  el('mktSectors').innerHTML = sectors.map((s) => {
    const ret = s.ret_5d || 0;
    const w = Math.round((Math.abs(ret) / maxAbs) * 100);
    return `<li class="sector t-${upDown(ret)}">
      <span class="sector-name">${esc(s.name)}</span>
      <span class="sector-bar"><span class="neg">${ret < 0 ? `<span class="tone-bg" style="width:${w}%"></span>` : ''}</span><span class="pos">${ret >= 0 ? `<span class="tone-bg" style="width:${w}%"></span>` : ''}</span></span>
      <span class="sector-val tone">${signed(ret)}%</span>
    </li>`;
  }).join('');
}
