// Owner password: asked for only when the server answers 401, then remembered
// in this browser so the owner isn't asked again.

const KEY = 'sentimentflow-owner';
const el = (id) => document.getElementById(id);

export function storedPassword() {
  try { return localStorage.getItem(KEY); } catch { return null; }
}
export function rememberPassword(pw) {
  try { localStorage.setItem(KEY, pw); } catch { /* private mode: ask again next time */ }
}
export function forgetPassword() {
  try { localStorage.removeItem(KEY); } catch { /* nothing stored */ }
}

/** Show the password dialog. Resolves with the password, or null if cancelled. */
export function askPassword({ wrong = false } = {}) {
  const dialog = el('owner');
  const form = el('ownerForm');
  form.reset();
  el('ownerError').textContent = wrong ? "That password didn't work. Try again." : '';
  dialog.returnValue = '';
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => {
      resolve(dialog.returnValue === 'unlock' ? form.elements.password.value : null);
    }, { once: true });
  });
}

// Request headers only carry plain ASCII, so catch other characters up front.
el('ownerForm').addEventListener('submit', (e) => {
  if (e.submitter?.value !== 'unlock') return;
  if (/[^\x20-\x7e]/.test(e.target.elements.password.value)) {
    e.preventDefault();
    el('ownerError').textContent = 'Use letters, numbers and standard symbols only.';
  }
});
