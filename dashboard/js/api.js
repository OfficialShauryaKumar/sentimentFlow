// Thin wrapper around the Flask API. Every endpoint replies { ok, ... }.
// Owner-only calls send the saved password and ask for it when the server refuses.

import { storedPassword, rememberPassword, forgetPassword, askPassword } from './owner.js';

export class OwnerCancelled extends Error {
  name = 'OwnerCancelled';
}

const withPassword = (options, password) =>
  password ? { ...options, headers: { ...options.headers, Authorization: `Bearer ${password}` } } : options;

async function request(path, options = {}, owner = false) {
  let password = owner ? storedPassword() : null;
  let typed = false;
  let res = await fetch(path, withPassword(options, password));

  while (owner && res.status === 401) {
    if (password && !typed) forgetPassword(); // saved password no longer matches
    password = await askPassword({ wrong: typed });
    if (!password) throw new OwnerCancelled('Owner password needed');
    typed = true;
    res = await fetch(path, withPassword(options, password));
  }
  if (typed) rememberPassword(password);

  const body = await res.json().catch(() => {
    throw new Error(`Server answered ${res.status} without data`);
  });
  if (!body.ok) throw new Error(body.error || 'API error');
  return body;
}

const json = (method, data) => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data),
});

export const api = {
  // A plain load is public; forcing a fresh scrape is owner-only.
  recommendations: (refresh) => request(`/api/recommendations${refresh ? '?refresh=true' : ''}`, {}, refresh),
  marketHealth: (refresh) => request(`/api/market-health${refresh ? '?refresh=true' : ''}`, {}, refresh),
  holdings: () => request('/api/portfolio/holdings'),
  saveHolding: (holding) => request('/api/portfolio/holdings', json('POST', holding), true),
  deleteHolding: (ticker) => request(`/api/portfolio/holdings/${encodeURIComponent(ticker)}`, { method: 'DELETE' }, true),
  analyzePortfolio: () => request('/api/portfolio/analyze'),
};
