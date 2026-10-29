// Paper-trading view: the agent's simulated account against just holding the S&P 500.

import { api } from './api.js';
import { esc, price, signed, signedMoney, metric, metrics } from './format.js';

const el = (id) => document.getElementById(id);
const pctTone = (n) => (n == null ? undefined : n >= 0 ? 'buy' : 'sell');
const pct = (n) => (n == null ? '—' : `${signed(n, 2)}%`);

export async function loadPaper() {
  el('paperStatus').hidden = false;
  el('paperStatus').textContent = 'Loading paper-trading results…';
  try {
    render(await api.paper());
  } catch (e) {
    el('paperStatus').textContent = `Paper-trading results could not load: ${esc(e.message)}. Check that app.py is running, then open this tab again.`;
  }
}

function render(d) {
  if (!d.started) {
    el('paperStatus').textContent = "The agent hasn't traded yet. It starts with $10,000 and runs each weekday after the market closes.";
    el('paperContent').hidden = true;
    return;
  }
  el('paperStatus').hidden = true;
  el('paperContent').hidden = false;

  const lead = d.return_pct != null && d.benchmark_pct != null ? d.return_pct - d.benchmark_pct : null;
  el('paperStats').innerHTML = metrics([
    metric('Account value', price(d.equity), { note: `Started at ${price(d.start_value)} on ${esc(d.start_date)}` }),
    metric('Agent return', pct(d.return_pct), { tone: pctTone(d.return_pct), note: `Realized ${signedMoney(d.realized_pnl)}` }),
    metric('S&P 500 return', pct(d.benchmark_pct), { tone: pctTone(d.benchmark_pct), note: `Same $10,000 held in SPY: ${d.benchmark ? price(d.benchmark) : '—'}` }),
    metric('Versus the S&P 500', lead == null ? '—' : `${signed(lead, 2)} pts`, {
      tone: pctTone(lead),
      note: d.win_rate == null ? 'No closed trades yet' : `${d.win_rate}% of closed trades won`,
    }),
  ], 'cols-4');

  el('paperChart').innerHTML = chart(d.daily);

  el('paperPositions').innerHTML = d.positions.length
    ? d.positions.map((p) => `<tr>
        <th scope="row">${esc(p.ticker)}</th>
        <td class="num">${p.shares}</td>
        <td class="num">${price(p.avg_cost)}</td>
        <td class="num t-sell">${p.stop_loss ? price(p.stop_loss) : '—'}</td>
        <td class="num t-buy">${p.take_profit ? price(p.take_profit) : '—'}</td>
      </tr>`).join('')
    : '<tr><td colspan="5" class="muted">No open positions. The agent is holding cash.</td></tr>';

  el('paperTrades').innerHTML = d.trades.length
    ? d.trades.map((t) => `<li class="trade">
        <span class="trade-action t-${t.action === 'BUY' ? 'buy' : 'sell'}">${t.action === 'BUY' ? 'Bought' : 'Sold'}</span>
        <span class="trade-what"><strong>${t.shares} ${esc(t.ticker)}</strong> at ${price(t.price)}<span class="sub">${esc(t.reason)}</span></span>
        <span class="trade-when sub">${esc(t.created_at.slice(0, 10))}${t.pnl != null ? `<span class="t-${t.pnl >= 0 ? 'buy' : 'sell'}">${signedMoney(t.pnl)}</span>` : ''}</span>
      </li>`).join('')
    : '<li class="muted">No trades yet.</li>';
}

/** Two-line SVG chart: the agent's account value and the S&P 500 benchmark. */
function chart(daily) {
  const pts = daily.filter((d) => d.equity != null);
  if (!pts.length) return '<p class="muted">The chart starts after the first trading day.</p>';
  const W = 800, H = 240, PAD = 8;
  const values = pts.flatMap((d) => [d.equity, d.benchmark].filter((v) => v != null));
  let lo = Math.min(...values), hi = Math.max(...values);
  if (hi - lo < 1) { lo -= 50; hi += 50; } // flat line on day one: give it room
  const x = (i) => (pts.length === 1 ? W / 2 : PAD + (i / (pts.length - 1)) * (W - 2 * PAD));
  const y = (v) => H - PAD - ((v - lo) / (hi - lo)) * (H - 2 * PAD);
  const line = (key) => pts.map((d, i) => (d[key] == null ? null : `${x(i).toFixed(1)},${y(d[key]).toFixed(1)}`)).filter(Boolean).join(' ');
  const dot = (key, cls) => pts.length === 1 && pts[0][key] != null ? `<circle class="${cls}" cx="${x(0)}" cy="${y(pts[0][key])}" r="5"/>` : '';
  return `<svg class="paper-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
      aria-label="Account value from ${price(pts[0].equity)} to ${price(pts.at(-1).equity)}, against the S&amp;P 500">
      <polyline class="line-spy" points="${line('benchmark')}"/>
      <polyline class="line-agent" points="${line('equity')}"/>
      ${dot('benchmark', 'dot-spy')}${dot('equity', 'dot-agent')}
    </svg>
    <div class="chart-axis"><span>${esc(pts[0].date)}</span><span>${price(lo)} to ${price(hi)}</span><span>${esc(pts.at(-1).date)}</span></div>
    <div class="chart-key"><span class="key-agent">Agent</span><span class="key-spy">S&amp;P 500 (SPY)</span></div>`;
}
