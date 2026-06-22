/**
 * JUDGELYTICS — Auth & Session Management
 * JWT storage, user state, and route protection utilities.
 */

const Auth = {
  _KEY_TOKEN: 'jdg_token',
  _KEY_USER:  'jdg_user',

  /* ── Token ───────────────────────────────────────────── */
  getToken() { return localStorage.getItem(this._KEY_TOKEN); },
  setToken(t) { localStorage.setItem(this._KEY_TOKEN, t); },
  clearToken() { localStorage.removeItem(this._KEY_TOKEN); },

  /* ── User ────────────────────────────────────────────── */
  getUser() {
    try { return JSON.parse(localStorage.getItem(this._KEY_USER)); }
    catch { return null; }
  },
  setUser(u) { localStorage.setItem(this._KEY_USER, JSON.stringify(u)); },
  clearUser() { localStorage.removeItem(this._KEY_USER); },

  /* ── State ───────────────────────────────────────────── */
  isLoggedIn() { return !!this.getToken(); },

  /* ── Login helper ────────────────────────────────────── */
  login(data) {
    this.setToken(data.access_token);
    this.setUser({
      uid:   data.uid,
      name:  data.name,
      email: data.email,
      phone: data.phone,
      caseCount: data.case_count ?? 0
    });
  },

  /* ── Logout helper ────────────────────────────────────── */
  logout() {
    this.clearToken();
    this.clearUser();
    window.location.href = 'login.html';
  },

  /* ── Route guards ─────────────────────────────────────── */
  requireAuth(redirectTo = 'login.html') {
    if (!this.isLoggedIn()) {
      window.location.href = redirectTo;
      return false;
    }
    return true;
  },

  requireGuest(redirectTo = 'dashboard.html') {
    if (this.isLoggedIn()) {
      window.location.href = redirectTo;
      return false;
    }
    return true;
  }
};

/* ─── Toast notification system ─────────────────────────────────────── */
const Toast = {
  _container: null,

  _getContainer() {
    if (!this._container) {
      this._container = document.createElement('div');
      this._container.className = 'toast-container';
      document.body.appendChild(this._container);
    }
    return this._container;
  },

  _icons: { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' },

  show(type, title, message = '', duration = 4500) {
    const container = this._getContainer();
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span class="toast-icon">${this._icons[type] || 'ℹ️'}</span>
      <div class="toast-body">
        <div class="toast-title">${title}</div>
        ${message ? `<div class="toast-msg">${message}</div>` : ''}
      </div>
      <button class="toast-close" onclick="Toast._remove(this.parentElement)">✕</button>
    `;
    container.appendChild(toast);

    if (duration > 0) {
      setTimeout(() => this._remove(toast), duration);
    }

    return toast;
  },

  _remove(el) {
    if (!el || el._removing) return;
    el._removing = true;
    el.classList.add('removing');
    setTimeout(() => el.remove(), 300);
  },

  success(title, msg, d) { return this.show('success', title, msg, d); },
  error(title, msg, d)   { return this.show('error',   title, msg, d); },
  info(title, msg, d)    { return this.show('info',    title, msg, d); },
  warning(title, msg, d) { return this.show('warning', title, msg, d); }
};

/* ─── Loading overlay ────────────────────────────────────────────────── */
const Loader = {
  _el: null,

  _get() {
    if (!this._el) {
      this._el = document.createElement('div');
      this._el.className = 'loading-overlay';
      this._el.innerHTML = `
        <div style="text-align:center">
          <div class="spinner spinner-lg"></div>
          <p style="margin-top:16px;color:var(--text-muted);font-size:.88rem">Processing…</p>
        </div>
      `;
      document.body.appendChild(this._el);
    }
    return this._el;
  },

  show(msg = 'Processing…') {
    const el = this._get();
    el.querySelector('p').textContent = msg;
    el.classList.add('active');
  },

  hide() {
    if (this._el) this._el.classList.remove('active');
  }
};

/* ─── Navbar helper ──────────────────────────────────────────────────── */
function initNavbar() {
  const nav = document.querySelector('.navbar');
  if (!nav) return;

  // Scroll shadow
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 10);
  }, { passive: true });

  // Active link
  const current = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.navbar-links a').forEach(a => {
    if (a.getAttribute('href') === current) a.classList.add('active');
  });

  // User info
  const user = Auth.getUser();
  const nameEl = document.getElementById('nav-user-name');
  if (nameEl && user) nameEl.textContent = user.name.split(' ')[0];
}

/* ─── Utility helpers ─────────────────────────────────────────────────── */
const Utils = {
  /* Format currency */
  formatINR(amount) {
    if (!amount) return '₹0';
    return '₹' + Number(amount).toLocaleString('en-IN');
  },

  /* Format date */
  formatDate(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-IN', {
      day: 'numeric', month: 'short', year: 'numeric'
    });
  },

  /* Format datetime */
  formatDateTime(iso) {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('en-IN', {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  },

  /* Outcome badge HTML */
  outcomeBadge(outcome) {
    if (!outcome) return '';
    const map = {
      'Allowed': 'badge-success',
      'Dismissed': 'badge-danger',
      'Partially Allowed': 'badge-accent'
    };
    return `<span class="badge ${map[outcome] || 'badge-muted'}">${outcome}</span>`;
  },

  /* Confidence badge */
  confidenceBadge(c) {
    const map = { high: 'badge-success', medium: 'badge-accent', low: 'badge-danger' };
    return `<span class="badge ${map[c] || 'badge-muted'}">${c} confidence</span>`;
  },

  /* Win probability color */
  probColor(pct) {
    if (pct >= 65) return 'var(--success)';
    if (pct >= 45) return 'var(--accent)';
    return 'var(--danger-light)';
  },

  /* Debounce */
  debounce(fn, delay = 300) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), delay); };
  },

  /* Markdown-ish to HTML (basic) */
  renderMarkdown(text) {
    if (!text) return '';
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/^#{1,3} (.+)$/gm, '<h4>$1</h4>')
      .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
      .replace(/^[-•] (.+)$/gm, '<li>$1</li>')
      .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>')
      .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
      .replace(/\n{2,}/g, '</p><p>')
      .replace(/\n/g, '<br>')
      .replace(/^(.)/m, '<p>$1')
      .replace(/(.*)$/, '$1</p>');
  },

  /* Download blob as file */
  downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
};
