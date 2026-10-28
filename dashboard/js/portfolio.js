// Portfolio view: saved holdings (SQLite via the API) and the per-position analysis.

import { api, OwnerCancelled } from './api.js';
import { esc, money, signedMoney, signed, tone, sentence, metric, metrics } from './format.js';

const el = (id) => document.getElementById(id);

/**
 * Reload saved holdings. After a change we always re-analyze; on a plain
 * load we only analyze when there is something to analyze.
 */
export async function loadHoldings({ analyze = 'ifAny' } = {}) {
  let holdings;
  try {
    holdings = (await api.holdings()).holdings || [];
  } catch {
    return; // backend not running: keep the empty state
  }
  el('holdingRows').innerHTML = holdings.map(holdingRow).join('');
  el('countPortfolio').textContent = holdings.length;
  if (analyze === 'always' || holdings.length) analyzePortfolio();
}

function holdingRow(h) {
  const days = h.buy_date ? Math.floor((Date.now() - new Date(h.buy_date).getTime()) / 86400000) : null;
  return `<tr>
    <th scope="row" class="ticker-sm">${esc(h.ticker)}</th>
    <td>${esc(h.shares)}</td>
    <td>$${esc(h.avg_cost)}</td>
    <td class="muted">${days != null ? `${days}d` : '—'}</td>
    <td><button class="icon-btn" data-remove="${esc(h.ticker)}" aria-label="Remove ${esc(h.ticker)}">×</button></td>
  </tr>`;
}

async function saveHolding(form) {
  const data = new FormData(form);
  const ticker = String(data.get('ticker') || '').toUpperCase().trim();
  const shares = parseFloat(data.get('shares')) || 0;
  const avg_cost = parseFloat(data.get('cost')) || 0;
  const buy_date = data.get('date') || null;
  const error = (msg) => { el('addError').textContent = msg; };

  if (!ticker) return error('Enter a ticker symbol.');
  if (shares <= 0) return error('Enter how many shares you hold.');
  if (avg_cost <= 0) return error('Enter your average cost per share.');
  try {
    await api.saveHolding({ ticker, shares, avg_cost, buy_date });
    closeAddForm();
    loadHoldings({ analyze: 'always' });
  } catch (e) {
    error(e instanceof OwnerCancelled
      ? 'Not saved. Saving needs the owner password.'
      : `Could not save: ${e.message}. Make sure app.py is running.`);
  }
}

async function removeHolding(ticker) {
  if (!confirm(`Remove ${ticker} from your portfolio?`)) return;
  try {
    await api.deleteHolding(ticker);
    loadHoldings({ analyze: 'always' });
  } catch (e) {
    if (!(e instanceof OwnerCancelled)) alert(`Could not remove ${ticker}: ${e.message}`);
  }
}

function openAddForm() {
  const form = el('addForm');
  form.hidden = false;
  el('addToggle').hidden = true;
  form.elements.ticker.focus();
}
function closeAddForm() {
  const form = el('addForm');
  form.reset();
  el('addError').textContent = '';
  form.hidden = true;
  el('addToggle').hidden = false;
}

/* ── Analysis ────────────────────────────────────────────── */
const results = (html) => { el('portResults').innerHTML = html; };
const emptyState = (title, body) => `<div class="empty-panel"><p class="empty-title">${title}</p><p>${body}</p></div>`;

export async function analyzePortfolio() {
  const btn = el('analyzeBtn');
  btn.disabled = true;
  btn.textContent = 'Analyzing…';
  try {
    const { analysis } = await api.analyzePortfolio();
    if (!analysis.positions?.length)
      results(emptyState('No holdings saved yet', 'Add a position on the left to see what to do with it.'));
    else
      results(summaryHTML(analysis.summary || {}) + `<ol class="positions">${analysis.positions.map(positionHTML).join('')}</ol>`);
  } catch (e) {
    results(emptyState('Analysis could not load', `${esc(e.message)}. Check that app.py is running, then choose Refresh analysis.`));
  } finally {
    btn.disabled = false;
    btn.textContent = 'Refresh analysis';
  }
}

function summaryHTML(s) {
  const score = s.health_score || 50;
  const t = score >= 70 ? 'buy' : score >= 45 ? 'hold' : 'sell';
  const label = score >= 80 ? 'Healthy' : score >= 60 ? 'Good' : score >= 40 ? 'Mixed' : score >= 20 ? 'At risk' : 'Critical';
  const pnlTone = (s.total_pnl || 0) >= 0 ? 'buy' : 'sell';
  const perf = (p, t2) => metric(t2 === 'buy' ? 'Best' : 'Worst', `${esc(p.ticker)} ${signed(p.pct)}%`, { tone: t2 });

  return `<section class="health t-${t}">
    <div class="health-top">
      <div class="health-score"><span class="score-num tone">${score}</span><div><p class="score-label tone">${label}</p>
        <p class="sub">${s.total_positions || 0} positions, ${s.winners || 0} winners, ${s.losers || 0} losers</p></div></div>
      <div class="health-pnl"><p class="pnl t-${pnlTone}">${signedMoney(s.total_pnl)}</p>
        <p class="sub">${signed(s.total_pnl_pct || 0, 1)}% unrealized P&amp;L on ${money(s.total_value)}</p></div>
    </div>
    <div class="bar"><span class="tone-bg" style="width:${score}%"></span></div>
    ${metrics([
      metric('Urgent', s.high_urgency || 0, { tone: s.high_urgency ? 'sell' : undefined }),
      metric('Watch', s.medium_urgency || 0, { tone: s.medium_urgency ? 'hold' : undefined }),
      s.best_performer ? perf(s.best_performer, 'buy') : '',
      s.worst_performer ? perf(s.worst_performer, 'sell') : '',
    ], 'cols-4')}
  </section>`;
}

function positionHTML(p) {
  const action = p.action || 'HOLD';
  const urgency = { HIGH: 'sell', MEDIUM: 'hold' }[p.urgency] || 'none';
  const actionTone = action.includes('STOP') || action.includes('SELL NOW') ? 'sell'
    : action.includes('ADD MORE') ? 'buy' : p.urgency === 'MEDIUM' ? 'hold' : '';
  const pnlTone = (p.unrealized_pct || 0) >= 0 ? 'buy' : 'sell';
  const facts = [
    p.rsi && `RSI ${p.rsi}`,
    p.atr && `ATR $${p.atr}`,
    p.tp_hit && `<span class="t-buy">20% target reached</span>`,
    p.stop_hit && `<span class="t-sell">Stop level breached</span>`,
    p.mention_count && `${p.mention_count} mentions`,
  ].filter(Boolean);

  return `<li class="position urgency-${urgency}">
    <div class="position-top">
      <div><p class="ticker-md">${esc(p.ticker)}${p.stop_hit ? ' <span class="flag t-sell">Stop hit</span>' : ''}</p>
        <p class="sub">${esc(p.shares)} shares at $${esc(p.avg_cost)}${p.days_held != null ? `, held ${p.days_held}d` : ''}</p></div>
      <div class="position-action"><p class="action${actionTone ? ' t-' + actionTone : ''}">${sentence(action)}</p><p class="prose-sm">${esc(p.rationale)}</p></div>
      <div class="position-price">${p.current_price ? `<p class="price">$${p.current_price}</p>` : ''}
        ${p.unrealized_pct != null ? `<p class="t-${pnlTone}">${signed(p.unrealized_pct, 1)}% <span class="sub">${signedMoney(p.unrealized_pnl)}</span></p>` : ''}</div>
    </div>
    ${metrics([
      metric('Value', p.current_value ? money(p.current_value) : '—'),
      metric('Short term', sentence(p.st_signal), { tone: tone(p.st_signal || 'HOLD') }),
      metric('Long term', sentence(p.lt_signal), { tone: tone(p.lt_signal || 'HOLD') }),
      metric('Entry stop', `$${p.entry_stop || '—'}`, { tone: 'sell' }),
      metric('20% target', `$${p.entry_tp || '—'}`, { tone: 'buy' }),
    ], 'cols-5 compact')}
    ${p.detail ? `<p class="prose-sm position-detail">${esc(p.detail)}</p>` : ''}
    ${facts.length ? `<p class="facts">${facts.map((f) => `<span>${f}</span>`).join('')}</p>` : ''}
  </li>`;
}

export function initPortfolio() {
  el('addToggle').addEventListener('click', openAddForm);
  el('addCancel').addEventListener('click', closeAddForm);
  el('addForm').addEventListener('submit', (e) => { e.preventDefault(); saveHolding(e.target); });
  el('addForm').elements.ticker.addEventListener('input', (e) => { e.target.value = e.target.value.toUpperCase(); });
  el('holdingRows').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-remove]');
    if (btn) removeHolding(btn.dataset.remove);
  });
  el('analyzeBtn').addEventListener('click', analyzePortfolio);
}
