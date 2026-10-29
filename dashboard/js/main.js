// Entry point: view switching, data loading and the refresh button.

import { api, OwnerCancelled } from './api.js';
import { startClock, setMarketStatus } from './clock.js';
import { initSignals, setSignals } from './signals.js';
import { initDetail } from './detail.js';
import { initPortfolio, loadHoldings } from './portfolio.js';
import { loadMarketHealth } from './market.js';
import { loadPaper } from './paper.js';
import { demoData } from './demo.js';

const el = (id) => document.getElementById(id);

const LOADING_STEPS = [
  'Scraping Reddit…', 'Fetching news feeds…', 'Running VADER sentiment…', 'Fetching live prices…',
  'Computing RSI, MACD and Bollinger bands…', 'Pulling analyst ratings…', 'Building predictions…',
];

async function loadSignals(refresh = false) {
  const loader = el('loader');
  const refreshBtn = el('refreshBtn');
  loader.hidden = false;
  refreshBtn.disabled = true;
  el('error').hidden = true;
  let step = 0;
  const ticker = setInterval(() => { el('loaderStep').textContent = LOADING_STEPS[step++ % LOADING_STEPS.length]; }, 1100);

  try {
    const data = await api.recommendations(refresh);
    if (data.market_status) setMarketStatus(data.market_status);
    setSignals(data);
  } catch (e) {
    if (e instanceof OwnerCancelled) return; // keep the signals already on screen
    el('error').textContent = `${e.message}. Showing demo data until the server responds.`;
    el('error').hidden = false;
    setSignals(demoData);
  } finally {
    clearInterval(ticker);
    loader.hidden = true;
    refreshBtn.disabled = false;
  }
}

const VIEWS = ['signals', 'portfolio', 'paper', 'market'];
function switchView(view) {
  for (const v of VIEWS) {
    el(`view-${v}`).hidden = v !== view;
    el(`tab-${v}`).setAttribute('aria-current', v === view ? 'page' : 'false');
  }
  if (view === 'portfolio') loadHoldings();
  if (view === 'market') loadMarketHealth();
  if (view === 'paper') loadPaper();
}

el('tabs').addEventListener('click', (e) => {
  const tab = e.target.closest('[data-view]');
  if (tab) switchView(tab.dataset.view);
});
el('refreshBtn').addEventListener('click', () => loadSignals(true));

startClock();
initSignals();
initDetail();
initPortfolio();
loadSignals();
loadHoldings();
