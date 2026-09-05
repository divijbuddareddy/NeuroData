// NeuroData Quality AI - Main UI Logic & Settings Manager

function showToast(message, type = 'info') {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = 'position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
    document.body.appendChild(container);
  }
  
  const toast = document.createElement('div');
  const bg = type === 'error' ? '#ef4444' : (type === 'success' ? '#10b981' : '#0284c7');
  toast.style.cssText = `background:${bg};color:#fff;padding:12px 20px;border-radius:8px;font-size:13px;font-weight:700;box-shadow:0 10px 25px rgba(0,0,0,0.25);display:flex;align-items:center;gap:8px;animation:slideIn 0.3s ease;`;
  toast.innerHTML = `<span>${message}</span>`;
  container.appendChild(toast);
  
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

/* ==========================================================================
   Theme Management (Clinical Clean vs Obsidian Dark)
   ========================================================================== */

function updateThemeUI(theme) {
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  if (theme === 'obsidian') {
    if (icon) icon.innerText = '☀️';
    if (label) label.innerText = 'Clinical Light';
  } else {
    if (icon) icon.innerText = '🌙';
    if (label) label.innerText = 'Obsidian Dark';
  }
}

function toggleAppTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  const nextTheme = current === 'obsidian' ? 'light' : 'obsidian';
  document.documentElement.setAttribute('data-theme', nextTheme);
  localStorage.setItem('neurodata_theme', nextTheme);
  updateThemeUI(nextTheme);
}

/* ==========================================================================
   Google AI Studio API Key Manager & Real-Time Q&A Config
   ========================================================================== */

function openApiKeyModal() {
  const modal = document.getElementById('api-key-modal');
  if (modal) modal.classList.add('open');
  const input = document.getElementById('input-gemini-key');
  if (input) {
    const localKey = localStorage.getItem('google_studio_api_key');
    if (localKey) input.value = localKey;
    input.focus();
  }
}

function closeApiKeyModal() {
  const modal = document.getElementById('api-key-modal');
  if (modal) modal.classList.remove('open');
  const statusEl = document.getElementById('modal-key-status');
  if (statusEl) statusEl.style.display = 'none';
}

async function saveApiKey() {
  const input = document.getElementById('input-gemini-key');
  const btn = document.getElementById('btn-save-api-key');
  const statusEl = document.getElementById('modal-key-status');
  if (!input || !btn) return;

  const key = input.value.trim();
  if (!key) {
    showToast("Please enter a valid Google AI Studio API key.", "error");
    return;
  }

  btn.disabled = true;
  btn.innerHTML = "Verifying with Google AI Studio...";

  try {
    const res = await fetch('/api/settings/api-key', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: key })
    });
    const data = await res.json();

    if (res.ok && data.configured) {
      localStorage.setItem('google_studio_api_key', key);
      showToast("Live Google AI Studio Key Verified & Saved!", "success");
      
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.style.background = 'var(--accent-emerald-subtle)';
        statusEl.style.color = 'var(--accent-emerald)';
        statusEl.innerHTML = `<strong>Connected:</strong> ${data.message} (${data.model})`;
      }

      updateApiKeyStatusUI(true, key);
      setTimeout(() => closeApiKeyModal(), 1200);
    } else {
      showToast(data.error || "Failed to verify key with Google AI Studio", "error");
      if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.style.background = 'var(--accent-rose-subtle)';
        statusEl.style.color = 'var(--accent-rose)';
        statusEl.innerHTML = `<strong>Error:</strong> ${data.error || 'Verification failed'}`;
      }
    }
  } catch (err) {
    showToast("Network error verifying API key: " + err.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Verify & Connect Live Key";
  }
}

function updateApiKeyStatusUI(configured, keyStr = "") {
  const dot = document.getElementById('sidebar-gemini-dot');
  const statusTxt = document.getElementById('sidebar-gemini-status');
  const subTxt = document.getElementById('sidebar-gemini-sub');
  const navLabel = document.getElementById('nav-api-key-label');

  if (configured) {
    if (dot) dot.classList.remove('offline');
    if (statusTxt) statusTxt.innerText = "Gemini 2.5 Flash";
    if (subTxt) subTxt.innerText = "Live AI Studio Connected";
    if (navLabel) navLabel.innerText = "Gemini Connected 🟢";
  } else {
    if (dot) dot.classList.add('offline');
    if (statusTxt) statusTxt.innerText = "Google AI Studio";
    if (subTxt) subTxt.innerText = "Click to configure key";
    if (navLabel) navLabel.innerText = "Set Studio API Key";
  }
}

async function checkApiKeyStatus() {
  try {
    const res = await fetch('/api/settings/api-key');
    const data = await res.json();
    if (data.configured) {
      updateApiKeyStatusUI(true);
    } else {
      const localKey = localStorage.getItem('google_studio_api_key');
      if (localKey) {
        // Automatically sync stored local key to backend
        fetch('/api/settings/api-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ api_key: localKey })
        }).then(r => r.json()).then(d => {
          if (d.configured) updateApiKeyStatusUI(true);
        });
      } else {
        updateApiKeyStatusUI(false);
      }
    }
  } catch (e) {
    console.log("Could not fetch API key status", e);
  }
}

/* ==========================================================================
   Demo Workflow Trigger
   ========================================================================== */

async function triggerDemoAnalysis() {
  const btn = document.getElementById('btn-run-demo');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `Running Deep ML Pipeline...`;
  }
  
  try {
    showToast("Generating synthetic MRI cohort & running PyTorch + Scikit-learn models...", "info");
    const res = await fetch('/api/demo/preload', { method: 'POST' });
    const data = await res.json();
    
    if (res.ok && data.dataset_id) {
      showToast("Quality Control Analysis Complete!", "success");
      setTimeout(() => {
        window.location.href = `/scan-quality?dataset_id=${data.dataset_id}`;
      }, 800);
    } else {
      showToast(data.error || "Failed to run analysis", "error");
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = "Run Sample Analysis (15 Scans)";
      }
    }
  } catch (err) {
    showToast("Server connection error: " + err.message, "error");
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = "Run Sample Analysis (15 Scans)";
    }
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
  updateThemeUI(currentTheme);
  checkApiKeyStatus();

  const demoBtn = document.getElementById('btn-run-demo');
  if (demoBtn) {
    demoBtn.addEventListener('click', triggerDemoAnalysis);
  }
});
