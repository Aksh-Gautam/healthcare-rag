/**
 * Healthcare Knowledge Navigator — Client-Side Application
 * Handles API communication, SHAP visualization, confidence gauge,
 * document management, and dynamic UI rendering.
 */

// ============================================================
// State
// ============================================================
const state = {
    shapEnabled: true,
    shapChart: null,
    isLoading: false,
    currentTab: 'search',
};

// ============================================================
// Initialization
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDragDrop();
    checkHealth();
    loadCategories();
    loadDocuments();

    // Enter key submits query
    document.getElementById('query-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });
});

// ============================================================
// Tab Navigation
// ============================================================
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
}

function switchTab(tab) {
    state.currentTab = tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`content-${tab}`).classList.add('active');

    if (tab === 'documents') {
        loadDocuments();
    }
}

// ============================================================
// Health Check & Status
// ============================================================
async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        const dot = document.getElementById('gemini-dot');
        const text = document.getElementById('gemini-status-text');

        if (data.gemini_configured) {
            dot.className = 'status-dot';
            text.textContent = 'Gemini Connected';
        } else {
            dot.className = 'status-dot warning';
            text.textContent = 'Gemini Not Configured';
        }

        document.getElementById('docs-count').textContent = `${data.documents_indexed} docs`;
    } catch (e) {
        const dot = document.getElementById('gemini-dot');
        dot.className = 'status-dot error';
        document.getElementById('gemini-status-text').textContent = 'Server Offline';
    }
}

// ============================================================
// Categories
// ============================================================
async function loadCategories() {
    try {
        const res = await fetch('/api/categories');
        const data = await res.json();
        const select = document.getElementById('category-filter');

        data.categories.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.textContent = cat;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load categories:', e);
    }
}

// ============================================================
// Query Submission
// ============================================================
function useSuggestion(el) {
    document.getElementById('query-input').value = el.textContent;
    submitQuery();
}

function toggleShap() {
    state.shapEnabled = !state.shapEnabled;
    document.getElementById('shap-toggle').classList.toggle('active', state.shapEnabled);
}

async function submitQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query || state.isLoading) return;

    state.isLoading = true;
    showLoading();

    const category = document.getElementById('category-filter').value;

    try {
        const res = await fetch('/api/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                top_k: 5,
                category_filter: category || null,
                enable_shap: state.shapEnabled,
            }),
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        const data = await res.json();
        renderResults(data);
    } catch (e) {
        console.error('Query failed:', e);
        showToast('Failed to get response. Is the server running?', 'error');
        hideLoading();
    } finally {
        state.isLoading = false;
    }
}

// ============================================================
// Loading States
// ============================================================
const loadingMessages = [
    'Retrieving clinical evidence...',
    'Searching knowledge base...',
    'Analyzing medical literature...',
    'Computing SHAP attributions...',
    'Synthesizing evidence-based answer...',
    'Evaluating source confidence...',
];

let loadingMsgIndex = 0;
let loadingInterval = null;

function showLoading() {
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('loading-state').classList.remove('hidden');
    document.getElementById('search-btn').disabled = true;

    loadingMsgIndex = 0;
    document.getElementById('loading-text').textContent = loadingMessages[0];

    loadingInterval = setInterval(() => {
        loadingMsgIndex = (loadingMsgIndex + 1) % loadingMessages.length;
        document.getElementById('loading-text').textContent = loadingMessages[loadingMsgIndex];
    }, 2500);
}

function hideLoading() {
    document.getElementById('loading-state').classList.add('hidden');
    document.getElementById('search-btn').disabled = false;
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

// ============================================================
// Results Rendering
// ============================================================
function renderResults(data) {
    hideLoading();

    document.getElementById('results-container').classList.remove('hidden');
    document.getElementById('empty-state').classList.add('hidden');

    renderAnswer(data.answer);
    renderDisclaimer(data.disclaimer);
    renderConfidence(data.confidence);
    renderCitations(data.citations);
    renderSources(data.retrieved_chunks);

    if (data.shap_explanation && state.shapEnabled) {
        renderShapChart(data.shap_explanation);
    } else {
        document.getElementById('shap-panel').classList.add('hidden');
    }
}

function renderAnswer(answer) {
    const el = document.getElementById('answer-content');
    // Convert [Source N] markers to styled badges
    let formatted = escapeHtml(answer);
    formatted = formatted.replace(
        /\[Source (\d+)\]/g,
        '<span class="citation-badge" onclick="scrollToSource($1)" title="Click to see source">Source $1</span>'
    );
    // Convert markdown-like formatting
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/\n- /g, '\n• ');
    formatted = formatted.replace(/\n#{1,3} (.*)/g, '\n<strong>$1</strong>');
    el.innerHTML = formatted;
}

function renderDisclaimer(text) {
    document.getElementById('disclaimer').textContent = text;
}

// ============================================================
// Confidence Gauge (SVG Arc)
// ============================================================
function renderConfidence(confidence) {
    const score = confidence.overall_score;
    const percentage = Math.round(score * 100);

    // Animate gauge
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeValue = document.getElementById('gauge-value');

    // Calculate arc: the path goes from left (20,80) to right (140,80) in a semicircle
    // Total arc length ≈ π * 60 ≈ 188.5
    const arcLength = Math.PI * 60;
    const fillLength = arcLength * score;

    gaugeFill.style.strokeDasharray = `${arcLength}`;
    gaugeFill.style.strokeDashoffset = `${arcLength}`; // Start empty

    // Color based on score
    let color;
    if (score >= 0.7) color = '#10b981';
    else if (score >= 0.4) color = '#f59e0b';
    else color = '#ef4444';

    gaugeFill.style.stroke = color;

    // Animate
    requestAnimationFrame(() => {
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = `${arcLength - fillLength}`;
        }, 100);
    });

    // Animate counter
    animateCounter(gaugeValue, 0, percentage, 1200);

    // Breakdown bars
    const breakdown = document.getElementById('confidence-breakdown');
    const items = [
        { label: 'Relevance', value: confidence.retrieval_relevance },
        { label: 'Source Agreement', value: confidence.source_agreement },
        { label: 'Evidence Quality', value: confidence.evidence_quality },
        { label: 'Coverage', value: confidence.coverage },
    ];

    breakdown.innerHTML = items.map(item => `
        <div class="breakdown-item">
            <span class="breakdown-label">${item.label}</span>
            <div class="breakdown-bar-track">
                <div class="breakdown-bar-fill" style="width: 0%;" data-target="${item.value * 100}"></div>
            </div>
            <span class="breakdown-value">${Math.round(item.value * 100)}%</span>
        </div>
    `).join('');

    // Animate bars
    setTimeout(() => {
        breakdown.querySelectorAll('.breakdown-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.target + '%';
        });
    }, 200);
}

function animateCounter(el, from, to, duration) {
    const start = performance.now();
    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // Ease out cubic
        const current = Math.round(from + (to - from) * eased);
        el.textContent = current + '%';
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ============================================================
// SHAP Chart (Chart.js Horizontal Bar)
// ============================================================
function renderShapChart(shapData) {
    const panel = document.getElementById('shap-panel');
    panel.classList.remove('hidden');

    // Destroy previous chart
    if (state.shapChart) {
        state.shapChart.destroy();
    }

    const ctx = document.getElementById('shap-chart').getContext('2d');
    const labels = shapData.chunk_labels.map(l => truncate(l, 45));
    const values = shapData.shap_values;

    // Color: positive = green (supports), negative = red (contradicts)
    const colors = values.map(v =>
        v >= 0 ? 'rgba(16, 185, 129, 0.8)' : 'rgba(239, 68, 68, 0.8)'
    );
    const borderColors = values.map(v =>
        v >= 0 ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'
    );

    state.shapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (contribution to answer quality)',
                data: values,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.7,
            }],
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1200,
                easing: 'easeOutQuart',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(148, 163, 184, 0.2)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => {
                            const val = ctx.raw;
                            const direction = val >= 0 ? '↑ Supports' : '↓ Contradicts';
                            return `${direction} answer: ${Math.abs(val).toFixed(4)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'SHAP Value (Contribution)',
                        color: '#64748b',
                        font: { size: 11, family: 'Inter' },
                    },
                    grid: { color: 'rgba(148, 163, 184, 0.08)' },
                    ticks: { color: '#64748b', font: { size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        color: '#94a3b8',
                        font: { size: 11, family: 'Inter' },
                    },
                },
            },
        },
    });

    // Render explanation text
    const explEl = document.getElementById('shap-explanation');
    let explText = shapData.explanation_text || '';
    explText = explText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    explText = explText.replace(/\n/g, '<br>');
    explEl.innerHTML = explText;
}

// ============================================================
// Citations
// ============================================================
function renderCitations(citations) {
    const list = document.getElementById('citations-list');
    if (!citations || citations.length === 0) {
        list.innerHTML = '<p style="font-size:13px;color:var(--text-tertiary);">No citations extracted.</p>';
        return;
    }

    list.innerHTML = citations.map(c => `
        <div class="source-card" onclick="scrollToSource(${c.index})" style="cursor:pointer;">
            <div class="source-card-header">
                <span class="source-title">[${c.index}] ${escapeHtml(c.title)}</span>
                <span class="source-relevance ${getRelevanceClass(c.relevance)}">${Math.round(c.relevance * 100)}%</span>
            </div>
            <div class="source-meta">
                <span class="source-tag">${escapeHtml(c.source)}</span>
            </div>
        </div>
    `).join('');
}

// ============================================================
// Sources
// ============================================================
function renderSources(chunks) {
    const list = document.getElementById('sources-list');
    const count = document.getElementById('sources-count');
    count.textContent = `${chunks.length} sources`;

    list.innerHTML = chunks.map((c, i) => `
        <div class="source-card" id="source-${i + 1}" onclick="toggleSource(this)">
            <div class="source-card-header">
                <span class="source-title">[Source ${i + 1}] ${escapeHtml(c.title)}</span>
                <span class="source-relevance ${getRelevanceClass(c.relevance_score)}">${Math.round(c.relevance_score * 100)}%</span>
            </div>
            <div class="source-meta">
                <span class="source-tag">${escapeHtml(c.category)}</span>
                <span class="source-tag">${escapeHtml(c.evidence_level)}</span>
                <span class="source-tag">${escapeHtml(c.specialty)}</span>
                <span class="source-tag">${escapeHtml(c.date)}</span>
            </div>
            <p class="source-expand-hint">Click to expand source content</p>
            <div class="source-content">${escapeHtml(c.content)}</div>
        </div>
    `).join('');
}

function toggleSource(el) {
    el.classList.toggle('expanded');
    const hint = el.querySelector('.source-expand-hint');
    if (hint) {
        hint.textContent = el.classList.contains('expanded') ? 'Click to collapse' : 'Click to expand source content';
    }
}

function scrollToSource(index) {
    const el = document.getElementById(`source-${index}`);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('expanded');
        // Flash effect
        el.style.borderColor = 'var(--accent-primary)';
        el.style.boxShadow = 'var(--shadow-glow)';
        setTimeout(() => {
            el.style.borderColor = '';
            el.style.boxShadow = '';
        }, 2000);
    }
}

// ============================================================
// Document Manager
// ============================================================
async function loadDocuments() {
    try {
        const res = await fetch('/api/documents');
        const data = await res.json();

        document.getElementById('kb-count').textContent = `${data.total_documents} documents · ${data.total_chunks} chunks`;

        const grid = document.getElementById('doc-grid');
        grid.innerHTML = data.documents.map(doc => `
            <div class="doc-card">
                <div class="doc-card-title">${escapeHtml(doc.title)}</div>
                <div class="doc-card-meta">
                    <span><span class="label">Source:</span> ${escapeHtml(doc.source)}</span>
                    <span><span class="label">Category:</span> ${escapeHtml(doc.category)}</span>
                    <span><span class="label">Evidence:</span> ${escapeHtml(doc.evidence_level)}</span>
                    <span><span class="label">Specialty:</span> ${escapeHtml(doc.specialty)}</span>
                    <span><span class="label">Chunks:</span> ${doc.chunk_count}</span>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load documents:', e);
    }
}

// ============================================================
// File Upload
// ============================================================
function initDragDrop() {
    const area = document.getElementById('upload-area');
    if (!area) return;

    ['dragenter', 'dragover'].forEach(evt => {
        area.addEventListener(evt, (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        area.addEventListener(evt, (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
        });
    });

    area.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) uploadFile(files[0]);
    });
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) uploadFile(file);
    event.target.value = ''; // Reset
}

async function uploadFile(file) {
    if (!file.name.endsWith('.txt')) {
        showToast('Only .txt files are supported.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showToast('Uploading and indexing document...', 'info');
        const res = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error('Upload failed');

        const data = await res.json();
        showToast(data.message, 'success');
        loadDocuments();
        loadCategories();
        checkHealth();
    } catch (e) {
        console.error('Upload failed:', e);
        showToast('Failed to upload document.', 'error');
    }
}

// ============================================================
// Toast Notifications
// ============================================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(20px)';
        toast.style.transition = 'all 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================================
// Utility Functions
// ============================================================
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.slice(0, len - 3) + '...' : str;
}

function getRelevanceClass(score) {
    if (score >= 0.7) return 'high';
    if (score >= 0.4) return 'medium';
    return 'low';
}
