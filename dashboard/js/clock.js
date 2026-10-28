// Market-hours clock in the header. Status arrives with /api/recommendations
// and the countdown ticks locally between refreshes.

import { hm } from './format.js';

export const market = {
  status: 'CLOSED', label: '—', seconds_to_open: 0, seconds_to_close: 0, timestamp_et: '—',
};

const el = (id) => document.getElementById(id);

function render() {
  const s = market;
  const state = s.status === 'OPEN' ? 'open' : s.status === 'PRE_MARKET' || s.status === 'AFTER_HOURS' ? 'pre' : 'closed';
  el('clock').dataset.state = state;
  el('clockLabel').textContent = s.label;
  el('clockTime').textContent = (s.timestamp_et || '').split(' ')[1] || '';

  let countdown = '';
  if (s.status === 'OPEN' && s.seconds_to_close > 0) countdown = `closes in ${hm(s.seconds_to_close)}`;
  else if (s.status !== 'OPEN' && s.seconds_to_open > 0)
    countdown = s.seconds_to_open > 86400 ? `opens ${s.next_open || ''}` : `opens in ${hm(s.seconds_to_open)}`;
  el('clockCountdown').textContent = countdown;
}

export function setMarketStatus(status) {
  Object.assign(market, status);
  el('clock').hidden = false;
  render();
}

/** Plain-language timing note used in the ticker detail panel. */
export function timingNote() {
  const s = market;
  const opensIn = s.seconds_to_open > 0 && s.status !== 'OPEN' ? ` Opens in ${hm(s.seconds_to_open)}.` : '';
  const text = {
    OPEN: 'Market is open, so this signal can be acted on now.',
    PRE_MARKET: 'Pre-market has lower liquidity. Waiting for the 9:30 AM ET open usually gets better fills.',
    AFTER_HOURS: `After hours: set limit orders now or wait for the next open (${s.next_open || '9:30 AM ET'}).`,
  }[s.status] ?? `Market closed. This signal targets the next session open (${s.next_open || 'next trading day'}).`;
  const t = s.status === 'OPEN' ? 'buy' : s.status === 'CLOSED' ? '' : 'hold';
  return { text: text + opensIn, tone: t };
}

export function startClock() {
  render();
  setInterval(() => {
    market.seconds_to_open = Math.max(0, market.seconds_to_open - 1);
    market.seconds_to_close = Math.max(0, market.seconds_to_close - 1);
    render();
  }, 1000);
}
