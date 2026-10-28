// Ticker detail panel (a native <dialog> that slides in from the right).

import { esc, safeUrl, money, signed, tone, sentence, metric, metrics, section, meter } from './format.js';
import { timingNote } from './clock.js';

const TABS = [
  ['st', 'Short term'],
  ['lt', 'Long term'],
  ['tech', 'Technicals'],
  ['fund', 'Fundamentals'],
  ['src', 'Sources'],
];
const el = (id) => document.getElementById(id);
let current = null;

export function openDetail(rec) {
  current = rec;
  const st = rec.short_term || {}, lt = rec.long_term || {};
  const t = tone(st.signal || 'HOLD');
  el('detailTicker').textContent = rec.ticker;
  el('detailTicker').className = `detail-ticker t-${t}`;
  el('detailSub').innerHTML =
    `Short term <strong class="t-${t}">${sentence(st.signal)}</strong>, long term <strong class="t-${tone(lt.signal)}">${sentence(lt.signal)}</strong>. ` +
    `${rec.mention_count} mentions${rec.price ? `, last $${rec.price}` : ''}.`;
  el('detailTabs').innerHTML = TABS.map(
    ([id, name], i) => `<button role="tab" data-pane="${id}" aria-selected="${i === 0}">${name}</button>`,
  ).join('');
  showPane('st');
  el('detail').showModal();
}

function showPane(id) {
  const r = current;
  el('detailTabs').querySelectorAll('[role=tab]').forEach((b) => b.setAttribute('aria-selected', b.dataset.pane === id));
  const body = {
    st: () => predictionPane(r.short_term, 'short'),
    lt: () => predictionPane(r.long_term, 'long'),
    tech: () => technicalsPane(r.technicals, r),
    fund: () => fundamentalsPane(r.fundamentals, r.long_term || {}),
    src: () => sourcesPane(r),
  }[id]();
  el('detailBody').innerHTML = body;
  el('detailBody').scrollTop = 0;
}

const empty = (msg) => `<p class="pane-empty">${msg}</p>`;
const note = (text, t = '') => `<p class="callout${t ? ' t-' + t : ''}">${text}</p>`;
const list = (items) => `<ul class="bullets">${items.map((f) => `<li>${esc(f)}</li>`).join('')}</ul>`;

function predictionPane(p, type) {
  if (!p?.signal) return empty(`No ${type}-term prediction for this ticker yet.`);
  const t = tone(p.signal);
  const isShort = p.signal.includes('SELL');
  const pos = p.position || {};
  let h = `<div class="signal-head t-${t}">
      <div><p class="signal-name">${sentence(p.signal)}</p><p class="sub">${esc(p.action)}</p></div>
      <div class="signal-conf"><span class="big">${p.confidence || 0}%</span><span class="sub">confidence</span>${meter(p.confidence, t)}</div>
    </div>
    <p class="sub">${esc(p.risk_level)} risk, reward ratio ${esc(p.rr_ratio)}. ${esc(p.horizon_note)}</p>`;

  h += section('Rationale', `<p class="prose">${esc(p.rationale || '—')}</p>`);
  if (p.key_factors?.length) h += section('Key factors', list(p.key_factors));

  if (p.entry)
    h += section(
      `Levels <span class="sub">ATR ${p.atr || '—'}, trailing stop ${p.trailing_stop_pct || '—'}%</span>`,
      metrics([
        metric('Entry', `$${p.entry}`),
        metric('Stop loss', `$${p.stop_loss}`, { tone: 'sell', note: isShort ? 'above entry' : 'below entry' }),
        metric('Take profit 1', `$${p.take_profit_1}`, { tone: 'buy' }),
        metric('Take profit 2', `$${p.take_profit_2}`, { tone: 'buy' }),
      ], 'cols-4'),
    );

  if (pos.shares)
    h += section(
      `Position sizing <span class="sub">${money(p.portfolio_size)} portfolio</span>`,
      metrics([
        metric('Shares', pos.shares, { note: isShort ? 'to short' : 'to buy' }),
        metric('Dollar amount', money(pos.dollar_amount), { note: `${pos.pct_of_portfolio || 0}% of portfolio` }),
        metric('Max risk', money(pos.max_risk), { tone: 'sell', note: `$${pos.risk_per_share || 0} per share` }),
      ], 'cols-3'),
    );

  if (p.options_plays?.length)
    h += section(
      'Options and strategies',
      `<dl class="plays">${p.options_plays.map((o) => `<div><dt>${esc(o.name)}</dt><dd>${esc(o.description)}</dd></div>`).join('')}</dl>`,
    );

  if (type === 'long' && p.analyst_target)
    h += section(
      'Analyst consensus',
      metrics([
        metric('Consensus', esc(p.analyst_consensus || '—')),
        metric('Price target', `$${p.analyst_target}`, { tone: 'buy', note: `${signed(p.analyst_upside || 0)}% upside` }),
        metric('Fundamental score', signed(p.fundamental_score || 0, 2)),
      ], 'cols-3'),
    );

  const timing = timingNote();
  h += section('Market timing', note(timing.text, timing.tone));
  return h;
}

function technicalsPane(tech, r) {
  if (!tech?.rsi) return empty('Technical data is unavailable. The backend needs yfinance to compute it.');
  const rsi = tech.rsi || 50;
  const rsiTone = rsi >= 70 ? 'sell' : rsi <= 30 ? 'buy' : 'hold';
  const { macd = {}, bollinger: boll = {}, moving_avgs: mas = {}, volume: vol = {}, trend: tr = {}, support_resistance: sr = {} } = tech;
  const rangePct = tech.week52_high && tech.week52_low && r.price
    ? Math.round(((r.price - tech.week52_low) / (tech.week52_high - tech.week52_low)) * 100)
    : null;
  const pairs = (rows) => `<dl class="pairs">${rows.map(([k, v, t]) => `<div${t ? ` class="t-${t}"` : ''}><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>`;

  let h = section('Price indicators', metrics([
    metric('RSI (14 day)', rsi, {
      tone: rsiTone,
      note: `${meter(rsi, rsiTone)}<span class="scale"><span>Oversold 30</span><span>${esc(tech.rsi_signal)}</span><span>70 overbought</span></span>`,
    }),
    metric('MACD (12/26/9)', macd.bullish ? 'Bullish' : 'Bearish', {
      tone: macd.bullish ? 'buy' : 'sell',
      note: pairs([
        ['Line', macd.macd_line ?? '—'],
        ['Signal', macd.signal_line ?? '—'],
        ['Histogram', macd.histogram ?? '—', (macd.histogram || 0) > 0 ? 'buy' : 'sell'],
        ...(macd.crossover ? [[esc(macd.crossover.replace('_', ' ')), '', macd.crossover.includes('bull') ? 'buy' : 'sell']] : []),
      ]),
    }),
    metric('Bollinger bands', esc(boll.price_vs_bands || '—'), {
      note: pairs([
        ['Upper', `$${boll.upper ?? '—'}`],
        ['Middle', `$${boll.middle ?? '—'}`],
        ['Lower', `$${boll.lower ?? '—'}`],
        ...(boll.squeeze ? [['Squeeze', 'Yes', 'hold']] : []),
      ]),
    }),
  ], 'cols-3'));

  h += section('Moving averages', metrics(['sma20', 'sma50', 'sma200'].map((k) => {
    const v = mas[k], above = mas[`above_${k}`];
    return metric(k.toUpperCase(), v ? `$${v}` : 'N/A', { note: v ? `<span class="t-${above ? 'buy' : 'sell'}">${above ? 'Price above' : 'Price below'}</span>` : '—' });
  }), 'cols-3')
    + (mas.golden_cross ? note('Golden cross: SMA50 is above SMA200, a long-term bullish signal.', 'buy') : '')
    + (mas.death_cross ? note('Death cross: SMA50 is below SMA200, a long-term bearish warning.', 'sell') : ''));

  h += section('Trend, volume and support', metrics([
    metric('Trend (20 day)', esc(`${tr.strength || ''} ${tr.direction || '—'}`), { note: pairs([['Slope per day', `${tr.slope_pct || 0}%`]]) }),
    metric('Volume vs 20-day average', `${vol.volume_ratio || 1}×`, {
      note: pairs([
        ['ATR (14 day)', tech.atr ?? '—'],
        ...(vol.spike ? [[vol.spike_direction === 'up' ? 'Bullish spike' : 'Bearish spike', '', vol.spike_direction === 'up' ? 'buy' : 'sell']] : []),
      ]),
    }),
    metric('Support and resistance', '', {
      note: pairs([
        ['Support', sr.support ? `$${sr.support}` : '—', 'buy'],
        ['Resistance', sr.resistance ? `$${sr.resistance}` : '—', 'sell'],
      ]),
    }),
  ], 'cols-3'));

  if (rangePct !== null)
    h += section('52-week range', `<div class="range">
      <div class="range-track"><span class="range-mark" style="left:${Math.max(0, Math.min(100, rangePct))}%"></span></div>
      <div class="scale"><span class="t-sell">Low $${tech.week52_low}</span><span>$${r.price} is ${rangePct}% of the range</span><span class="t-buy">High $${tech.week52_high}</span></div>
    </div>`);
  return h;
}

function fundamentalsPane(f, lt) {
  if (!f?.pe_ttm && !f?.rev_growth && !f?.recommendation) return empty('Fundamental data is unavailable for this ticker.');
  const fs = f.fundamental_score || 0;
  const fsTone = fs >= 0.3 ? 'buy' : fs <= -0.3 ? 'sell' : 'hold';
  const pct = (v) => (v != null ? `${v}%` : '—');
  const growthTone = (v) => ((v || 0) > 5 ? 'buy' : (v || 0) < 0 ? 'sell' : undefined);

  return section(
    `Valuation <span class="sub">Fundamental score <strong class="t-${fsTone}">${signed(fs, 2)}</strong>, ${esc(lt.analyst_consensus || 'no consensus')}</span>`,
    metrics([
      metric('P/E trailing', f.pe_ttm || '—'),
      metric('P/E forward', f.pe_forward || '—'),
      metric('Price to sales', f.ps_ratio || '—'),
      metric('EV/EBITDA', f.ev_ebitda || '—'),
      metric('Profit margin', pct(f.profit_margin), { tone: (f.profit_margin || 0) > 0 ? 'buy' : 'sell' }),
      metric('Debt to equity', f.debt_equity || '—'),
    ], 'cols-3'),
  ) + section('Growth', metrics([
    metric('Revenue, year on year', pct(f.rev_growth), { tone: growthTone(f.rev_growth) }),
    metric('Earnings, year on year', pct(f.earn_growth), { tone: growthTone(f.earn_growth) }),
    metric('Return on equity', pct(f.roe)),
  ], 'cols-3')) + section(`Analyst ratings <span class="sub">${f.analyst_count || 0} analysts</span>`, metrics([
    metric('Consensus', esc(lt.analyst_consensus || '—')),
    metric('Mean target', `$${f.target_mean || '—'}`, { tone: 'buy', note: f.upside_pct != null ? `${signed(f.upside_pct)}%` : '' }),
    metric('Short interest', pct(f.short_interest_pct), { tone: (f.short_interest_pct || 0) > 15 ? 'sell' : undefined }),
  ], 'cols-3') + (f.next_earnings ? note(`Next earnings on ${esc(f.next_earnings)}. Expect volatility around it.`) : ''));
}

function sourcesPane(r) {
  const sources = r.sources || [];
  const mentions = r.top_mentions || [];
  return section(
    `Sources <span class="sub">${sources.length}</span>`,
    `<ul class="tags">${sources.map((s) => `<li>${esc(s)}</li>`).join('')}</ul>`,
  ) + section(
    `Top mentions <span class="sub">${r.mention_count} total</span>`,
    mentions.length
      ? `<ul class="mentions">${mentions.map((m) => {
          const url = safeUrl(m.url);
          const title = esc(m.title || '(no title)');
          const label = m.sentiment_label === 'bullish' ? 'buy' : m.sentiment_label === 'bearish' ? 'sell' : 'hold';
          return `<li>
            <p class="mention-title">${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${title}</a>` : title}</p>
            <p class="sub"><span class="t-${label}">${esc(m.sentiment_label)}</span> in ${esc(m.source)}, sentiment ${m.sentiment_score.toFixed(3)}, traction ${m.traction.toFixed(3)}</p>
          </li>`;
        }).join('')}</ul>`
      : empty('No individual mentions were kept for this ticker.'),
  );
}

export function initDetail() {
  const dialog = el('detail');
  el('detailTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('[role=tab]');
    if (tab) showPane(tab.dataset.pane);
  });
  el('detailClose').addEventListener('click', () => dialog.close());
  // Clicking the backdrop (outside the panel) closes it.
  dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
}
