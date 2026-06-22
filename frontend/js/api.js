/**
 * JUDGELYTICS — API Client
 * Axios-based client with automatic JWT injection and error handling.
 * All backend communication goes through this module.
 */

const API_BASE = 'http://localhost:8000/api/v1';

/* ─── Build headers ──────────────────────────────────────────────────── */
function _headers(extra = {}) {
  const token = Auth.getToken();
  const h = { 'Content-Type': 'application/json', ...extra };
  if (token) h['Authorization'] = `Bearer ${token}`;
  return h;
}

/* ─── Generic fetch wrapper ──────────────────────────────────────────── */
async function _request(method, path, body = null, opts = {}) {
  const options = {
    method,
    headers: _headers(opts.headers || {}),
  };

  if (body && !(body instanceof FormData)) {
    options.body = JSON.stringify(body);
  } else if (body instanceof FormData) {
    options.body = body;
    delete options.headers['Content-Type']; // Let browser set multipart boundary
  }

  try {
    const res = await fetch(`${API_BASE}${path}`, options);
    const data = res.headers.get('content-type')?.includes('application/json')
      ? await res.json()
      : await res.blob();

    if (!res.ok) {
      if (res.status === 401 && Auth.isLoggedIn()) {
        Auth.logout();
        throw new ApiError('Your session has expired. Please sign in again.', 401);
      }
      
      let msg = `Error ${res.status}`;
      if (data?.detail) {
        if (Array.isArray(data.detail)) {
          // It's a FastAPI validation error
          msg = data.detail.map(e => {
            const field = e.loc[e.loc.length - 1];
            return `${field}: ${e.msg}`;
          }).join(', ');
        } else {
          msg = data.detail;
        }
      } else if (data?.message) {
        msg = data.message;
      }
      
      throw new ApiError(msg, res.status, data);
    }

    return data;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError('Network error — make sure the backend is running.', 0);
  }
}

class ApiError extends Error {
  constructor(message, status, data = null) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

/* ─── Auth Endpoints ─────────────────────────────────────────────────── */
const AuthAPI = {
  async register(name, email, phone, password) {
    return _request('POST', '/auth/register', { name, email, phone, password });
  },
  async login(email, password) {
    return _request('POST', '/auth/login', { email, password });
  },
  async me() {
    return _request('GET', '/auth/me');
  }
};

/* ─── Case Endpoints ─────────────────────────────────────────────────── */
const CaseAPI = {
  async analyze(caseData) {
    return _request('POST', '/case/analyze', caseData);
  },
  async history(limit = 20, offset = 0) {
    return _request('GET', `/case/history?limit=${limit}&offset=${offset}`);
  },
  async detail(caseId) {
    return _request('GET', `/case/${caseId}`);
  }
};

/* ─── Report Endpoints ───────────────────────────────────────────────── */
const ReportAPI = {
  async generate(caseId) {
    return _request('POST', '/report/generate', { case_id: caseId });
  },
  async download(reportId) {
    const token = Auth.getToken();
    // Direct link for file download
    return `${API_BASE}/report/download/${reportId}`;
  },
  async downloadDirect(reportId) {
    const token = Auth.getToken();
    const res = await fetch(`${API_BASE}/report/download/${reportId}`, {
      headers: { Authorization: `Bearer ${token}` }
    });
    if (!res.ok) throw new ApiError('Failed to download report', res.status);
    return res.blob();
  }
};

/* ─── Chat Endpoints ─────────────────────────────────────────────────── */
const ChatAPI = {
  async sendMessage(content, caseId = null) {
    return _request('POST', '/chat/message', { content, case_id: caseId });
  },
  async history(limit = 50, caseId = null) {
    let url = `/chat/history?limit=${limit}`;
    if (caseId) url += `&case_id=${caseId}`;
    return _request('GET', url);
  }
};

/* ─── Legal Library Endpoints ────────────────────────────────────────── */
const LegalAPI = {
  async sections(query = '', act = '', category = '') {
    let url = '/legal/sections?limit=50';
    if (query) url += `&query=${encodeURIComponent(query)}`;
    if (act) url += `&act=${encodeURIComponent(act)}`;
    if (category) url += `&category=${encodeURIComponent(category)}`;
    return _request('GET', url);
  },
  async section(id) {
    return _request('GET', `/legal/sections/${id}`);
  },
  async judgements(query = '', sector = '', outcome = '') {
    let url = '/legal/judgements?limit=20';
    if (query) url += `&query=${encodeURIComponent(query)}`;
    if (sector) url += `&sector=${encodeURIComponent(sector)}`;
    if (outcome) url += `&outcome=${encodeURIComponent(outcome)}`;
    return _request('GET', url);
  },
  async categories() {
    return _request('GET', '/legal/categories');
  }
};
