function getToken() { return localStorage.getItem('verifai_token'); }
function getUser()  { try { return JSON.parse(localStorage.getItem('verifai_user')); } catch { return null; } }

function authFetch(url, opts = {}) {
    const token = getToken();
    if (token) {
        opts.headers = opts.headers || {};
        if (opts.headers instanceof Headers) {
            opts.headers.set('Authorization', 'Bearer ' + token);
        } else {
            opts.headers['Authorization'] = 'Bearer ' + token;
        }
    }
    return fetch(url, opts).then(res => {
        if (res.status === 401) {
            localStorage.removeItem('verifai_token');
            localStorage.removeItem('verifai_user');
            window.location.href = '/login';
        }
        return res;
    });
}

function logout() {
    localStorage.removeItem('verifai_token');
    localStorage.removeItem('verifai_user');
    window.location.href = '/login';
}

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function show(el) { if (typeof el === 'string') el = $(el); if (el) el.style.display = ''; }
function hide(el) { if (typeof el === 'string') el = $(el); if (el) el.style.display = 'none'; }

function spinner(on, text) {
    const s = $('#spinner');
    if (on) { $('#spinner-text').textContent = text || 'Processing...'; show(s); }
    else hide(s);
}

function toast(msg, type = 'success') {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3500);
}

function confBadge(val) {
    const cls = val >= 0.7 ? 'conf-high' : val >= 0.5 ? 'conf-med' : 'conf-low';
    return `<span class="conf-badge ${cls}">${val.toFixed(3)}</span>`;
}

function metricCard(label, value) {
    return `<div class="metric-card"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function formatTime(ms) {
    if (ms >= 60000) return `${(ms / 60000).toFixed(1)}m`;
    if (ms >= 1000) return `${(ms / 1000).toFixed(1)}s`;
    return `${ms.toFixed(0)}ms`;
}

const STEP_ICONS = { encode: 'E', retrieve: 'R', reason: 'G', evaluate: 'V', calibrate: 'C', fetch_new: 'F', done: '✓' };

function applyChartTheme() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    Chart.defaults.color = isDark ? '#6b7280' : '#6b7280';
    Chart.defaults.borderColor = isDark ? '#1e2330' : '#e2e5ea';
    Chart.defaults.font.family = "'Inter', sans-serif";
}
applyChartTheme();

function gridColor() {
    return document.documentElement.getAttribute('data-theme') !== 'light' ? '#1e2330' : '#e2e5ea';
}

let chartInstances = {};
function getChart(id) { return chartInstances[id]; }
function destroyChart(id) { if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; } }

let appConfig = null;

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('verifai-theme', theme);
    const icon = $('#theme-toggle i');
    if (icon) {
        icon.setAttribute('data-lucide', theme === 'dark' ? 'moon' : 'sun');
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }
    applyChartTheme();
}

(function() {
    const saved = localStorage.getItem('verifai-theme') || 'dark';
    setTheme(saved);
})();

$('#theme-toggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
});

const sidebar = $('#sidebar');
const sidebarOverlay = $('#sidebar-overlay');

function isMobile() { return window.innerWidth <= 900; }

$('#sidebar-toggle').addEventListener('click', () => {
    if (isMobile()) {
        sidebar.classList.toggle('mobile-open');
        sidebarOverlay.classList.toggle('active');
    } else {
        sidebar.classList.toggle('collapsed');
        document.body.classList.toggle('sidebar-collapsed');
    }
});

sidebarOverlay.addEventListener('click', () => {
    sidebar.classList.remove('mobile-open');
    sidebarOverlay.classList.remove('active');
});

function navigateTo(page) {
    $$('.page').forEach(p => p.classList.remove('active'));
    $$('.nav-link').forEach(l => l.classList.remove('active'));
    const target = $(`#page-${page}`);
    const link = $(`.nav-link[data-page="${page}"]`);
    if (target) target.classList.add('active');
    if (link) link.classList.add('active');

    if (isMobile()) {
        sidebar.classList.remove('mobile-open');
        sidebarOverlay.classList.remove('active');
    }

    if (page === 'dashboard') loadDashboard();
    if (page === 'kb') loadKB();
    if (page === 'settings') loadSettings();
}

$$('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo(link.dataset.page);
    });
});

$$('.feature-card[data-goto]').forEach(card => {
    card.addEventListener('click', () => navigateTo(card.dataset.goto));
});

async function loadDashboard() {
    try {
        const res = await authFetch('/api/stats');
        const data = await res.json();
        const kb = data.kb || {};
        const hist = data.history || [];

        $('#dash-metrics').innerHTML = [
            metricCard('Total Queries', data.queries),
            metricCard('Avg Confidence', data.avg_confidence.toFixed(2)),
            metricCard('KB Chunks', kb.total_chunks || 0),
            metricCard('Documents', kb.total_documents || 0),
            metricCard('Fact Checks', data.fact_checks),
        ].join('');

        if (hist.length > 0) {
            const labels = hist.map((_, i) => `Q${i + 1}`);
            const gc = gridColor();

            show('#dash-conf-card');
            destroyChart('dashConf');
            const confVals = hist.map(h => h.confidence);
            chartInstances['dashConf'] = new Chart($('#dash-conf-chart'), {
                type: 'line',
                data: { labels, datasets: [{
                    label: 'Confidence (C_f)',
                    data: confVals,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,.1)',
                    fill: true, tension: .3, pointRadius: 4,
                    pointBackgroundColor: confVals.map(v => v >= 0.7 ? '#22c55e' : v >= 0.5 ? '#eab308' : '#ef4444'),
                }]},
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        annotation: { annotations: {
                            threshold: { type: 'line', yMin: 0.7, yMax: 0.7, borderColor: 'rgba(234,179,8,.5)', borderDash: [6, 3], borderWidth: 1,
                                label: { display: true, content: 'τ = 0.70', position: 'end', font: { size: 10 } }
                            }
                        }}
                    },
                    scales: { y: { min: 0, max: 1.05, grid: { color: gc } }, x: { grid: { display: false } } },
                },
            });

            show('#dash-scores-card');
            destroyChart('dashScores');
            chartInstances['dashScores'] = new Chart($('#dash-scores-chart'), {
                type: 'bar',
                data: { labels, datasets: [
                    { label: 'Semantic', data: hist.map(h => h.s_c || 0), backgroundColor: '#3b82f6', borderRadius: 3 },
                    { label: 'Factual', data: hist.map(h => h.f_c || 0), backgroundColor: '#f97316', borderRadius: 3 },
                ]},
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 12 } } },
                    scales: { y: { min: 0, max: 1.05, grid: { color: gc } }, x: { grid: { display: false } } },
                },
            });

            show('#dash-time-card');
            destroyChart('dashTime');
            const timesRaw = hist.map(h => h.time_ms || 0);
            const timesSeconds = timesRaw.map(t => t / 1000);
            chartInstances['dashTime'] = new Chart($('#dash-time-chart'), {
                type: 'bar',
                data: { labels, datasets: [{
                    label: 'Time (seconds)',
                    data: timesSeconds,
                    backgroundColor: timesRaw.map(t => t > 60000 ? 'rgba(239,68,68,.6)' : t > 30000 ? 'rgba(234,179,8,.6)' : 'rgba(34,197,94,.6)'),
                    borderRadius: 3,
                }]},
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { grid: { color: gc }, title: { display: true, text: 'Seconds' } }, x: { grid: { display: false } } },
                },
            });

            show('#dash-model-card');
            destroyChart('dashModel');
            const engineNameMap = { 'gemini-2.0-flash': 'Primary Engine', 'llama-3.3-70b-versatile': 'Fallback Engine A', 'llama-3.1-8b-instant': 'Fallback Engine B', 'gemma2-9b-it': 'Fallback Engine C', 'mixtral-8x7b-32768': 'Fallback Engine D', 'demo_cache': 'Cache Engine' };
            const modelCounts = {};
            hist.forEach(h => { const m = engineNameMap[h.model] || h.model || 'Unknown'; modelCounts[m] = (modelCounts[m] || 0) + 1; });
            const modelLabels = Object.keys(modelCounts);
            const modelData = Object.values(modelCounts);
            const modelColors = ['#3b82f6', '#f97316', '#22c55e', '#ef4444', '#a855f7'];
            chartInstances['dashModel'] = new Chart($('#dash-model-chart'), {
                type: 'doughnut',
                data: { labels: modelLabels, datasets: [{ data: modelData, backgroundColor: modelColors.slice(0, modelLabels.length) }] },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 10, font: { size: 11 } } } },
                },
            });

            show('#dash-recent-card');
            $('#dash-recent-list').innerHTML = hist.slice().reverse().slice(0, 10).map(h => {
                const confCls = h.confidence >= 0.7 ? 'color:var(--green)' : h.confidence >= 0.5 ? 'color:var(--yellow)' : 'color:var(--red)';
                const truncQ = h.query.length > 70 ? h.query.substring(0, 70) + '...' : h.query;
                return `<div class="dash-query-item">
                    <span class="dq-text">${truncQ}</span>
                    <span class="dq-conf" style="${confCls}">${h.confidence.toFixed(3)}</span>
                    <span style="font-size:.75rem;color:var(--text-muted)">${formatTime(h.time_ms || 0)}</span>
                </div>`;
            }).join('');
        } else {
            hide('#dash-conf-card');
            hide('#dash-scores-card');
            hide('#dash-time-card');
            hide('#dash-model-card');
            hide('#dash-recent-card');
        }
    } catch (e) {
        console.error('Dashboard load failed:', e);
    }
}

function loadRecentSearches() {
    authFetch('/api/stats').then(r => r.json()).then(data => {
        const hist = data.history || [];
        const el = $('#qa-recent');
        if (!el || hist.length === 0) { if (el) el.innerHTML = ''; return; }
        const recent = hist.slice().reverse().slice(0, 6);
        el.innerHTML = '<span style="font-size:.72rem;color:var(--text-muted);margin-right:.25rem">Recent:</span>' +
            recent.map(h => {
                const q = h.query;
                const display = q.length > 40 ? q.substring(0, 40) + '...' : q;
                return `<span class="recent-chip" data-query="${q.replace(/"/g, '&quot;')}" title="${q.replace(/"/g, '&quot;')}"><i data-lucide="clock"></i>${display}</span>`;
            }).join('');
        $$('.recent-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                $('#qa-input').value = chip.dataset.query;
                runQA();
            });
        });
        if (window.lucide) lucide.createIcons();
    }).catch(() => {});
}

$('#qa-btn').addEventListener('click', runQA);
$('#qa-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') runQA(); });

async function runQA() {
    const query = $('#qa-input').value.trim();
    if (!query) return;

    $('#qa-score').innerHTML = '';
    $('#qa-metrics').innerHTML = '';
    $('#qa-answer').innerHTML = '';
    $('#qa-model-info').textContent = '';
    hide('#qa-sources-wrap');
    $('#qa-timeline').innerHTML = '';

    show('#qa-output');
    $('#qa-answer').innerHTML = `
        <div class="qa-loading">
            <div class="spinner-ring"></div>
            <div class="qa-loading-text">Running AutoRAG++ pipeline...</div>
        </div>`;

    $('#qa-btn').disabled = true;
    show('#qa-spinner');

    try {
        const token = getToken();
        const tokenParam = token ? `&token=${encodeURIComponent(token)}` : '';
        const evtSource = new EventSource(`/api/qa?query=${encodeURIComponent(query)}${tokenParam}`);
        let steps = [];

        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                evtSource.close();
                $('#qa-btn').disabled = false;
                hide('#qa-spinner');
                return;
            }
            const msg = JSON.parse(e.data);

            if (msg.type === 'step') {
                steps.push(msg);
                renderTimeline(steps);
            }

            if (msg.type === 'result') {
                renderQAResult(msg);
                loadRecentSearches();
            }
        };

        evtSource.onerror = () => {
            evtSource.close();
            $('#qa-btn').disabled = false;
            hide('#qa-spinner');
            $('#qa-answer').innerHTML = '';
            toast('Pipeline request failed', 'error');
        };
    } catch (e) {
        $('#qa-btn').disabled = false;
        hide('#qa-spinner');
        $('#qa-answer').innerHTML = '';
        toast('Pipeline request failed', 'error');
    }
}

function renderTimeline(steps) {
    const html = steps.map((s, i) => {
        const isLast = i === steps.length - 1;
        const dotClass = s.step_type === 'done' ? 'done-step' : s.step_type === 'calibrate' ? 'warn' : '';
        const icon = STEP_ICONS[s.step_type] || '•';
        const label = s.step_type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        const cycle = s.cycle > 0 ? ` · Cycle ${s.cycle}` : '';
        const time = s.duration_ms > 0 ? `${s.duration_ms.toFixed(0)}ms` : '';

        let calCompare = '';
        if (s.step_type === 'calibrate' && s.data) {
            const d = s.data;
            calCompare = `
                <div class="cal-compare">
                    <div class="cal-col">
                        <div class="cal-heading">Before</div>
                        <div class="cal-row">k = <span class="val">${d.k_before}</span></div>
                        <div class="cal-row">λ₁ = <span class="val">${d.l1_before}</span></div>
                        <div class="cal-row">λ₂ = <span class="val">${d.l2_before}</span></div>
                    </div>
                    <div class="cal-arrow">→</div>
                    <div class="cal-col">
                        <div class="cal-heading">After</div>
                        <div class="cal-row">k = <span class="val">${d.k_after}</span></div>
                        <div class="cal-row">λ₁ = <span class="val">${d.l1_after}</span></div>
                        <div class="cal-row">λ₂ = <span class="val">${d.l2_after}</span></div>
                    </div>
                </div>`;
        }

        let evalCompare = '';
        if (s.step_type === 'evaluate' && s.cycle > 0 && s.data) {
            const prevEval = steps.filter(p => p.step_type === 'evaluate' && p.cycle === s.cycle - 1)[0];
            if (prevEval && prevEval.data) {
                const prev = prevEval.data;
                const curr = s.data;
                const arrow = (val, prev) => val > prev ? '↑' : val < prev ? '↓' : '=';
                const diffCls = (val, prev) => val > prev ? 'color:var(--green)' : val < prev ? 'color:var(--red)' : '';
                evalCompare = `
                    <div class="cal-compare">
                        <div class="cal-col">
                            <div class="cal-heading">Cycle ${s.cycle - 1}</div>
                            <div class="cal-row">Semantic: <span class="val">${prev.s_c.toFixed(3)}</span></div>
                            <div class="cal-row">Factual: <span class="val">${prev.f_c.toFixed(3)}</span></div>
                            <div class="cal-row">Confidence: <span class="val">${prev.c_f.toFixed(3)}</span></div>
                        </div>
                        <div class="cal-arrow">→</div>
                        <div class="cal-col">
                            <div class="cal-heading">Cycle ${s.cycle}</div>
                            <div class="cal-row">Semantic: <span class="val" style="${diffCls(curr.s_c, prev.s_c)}">${curr.s_c.toFixed(3)} ${arrow(curr.s_c, prev.s_c)}</span></div>
                            <div class="cal-row">Factual: <span class="val" style="${diffCls(curr.f_c, prev.f_c)}">${curr.f_c.toFixed(3)} ${arrow(curr.f_c, prev.f_c)}</span></div>
                            <div class="cal-row">Confidence: <span class="val" style="${diffCls(curr.c_f, prev.c_f)}">${curr.c_f.toFixed(3)} ${arrow(curr.c_f, prev.c_f)}</span></div>
                        </div>
                    </div>`;
            }
        }

        return `
            <div class="timeline-step">
                <div class="step-dot ${dotClass}">${icon}</div>
                <div class="step-body">
                    <div class="step-label">${label}${cycle}</div>
                    <div class="step-detail">${s.detail}</div>
                    ${calCompare}
                    ${evalCompare}
                </div>
                <div class="step-time">${time}</div>
            </div>
            ${!isLast ? '<div class="step-connector"></div>' : ''}
        `;
    }).join('');
    $('#qa-timeline').innerHTML = html;
}

function renderQAResult(r) {
    if (r.error) {
        $('#qa-score').innerHTML = '';
        $('#qa-metrics').innerHTML = '';
        $('#qa-answer').innerHTML = `<span style="color:var(--red)">${r.error}</span>`;
        $('#qa-model-info').textContent = '';
        hide('#qa-sources-wrap');
        return;
    }

    const warning = r.confidence < 0.7
        ? `<div class="low-conf-warning"><i data-lucide="alert-triangle"></i> Low confidence — this answer may not be fully supported by the available knowledge base.</div>`
        : '';

    $('#qa-score').innerHTML = `<strong>Answer</strong> ${confBadge(r.confidence)}`;

    $('#qa-metrics').innerHTML = [
        metricCard('Semantic Score', r.semantic_consistency.toFixed(3)),
        metricCard('Factual Score', r.factual_correctness.toFixed(3)),
        metricCard('Confidence', r.confidence.toFixed(3)),
        metricCard('Total Time', formatTime(r.total_time_ms)),
    ].join('');

    $('#qa-answer').innerHTML = warning + r.answer;
    $('#qa-model-info').textContent = `${r.total_sources_used} sources · ${r.calibration_cycles} calibration cycle(s)`;

    if (r.confidence < 0.7 && window.lucide) lucide.createIcons();

    if (r.sources && r.sources.length > 0) {
        show('#qa-sources-wrap');
        $('#qa-src-count').textContent = r.sources.length;
        $('#qa-sources').innerHTML = r.sources.map(s =>
            `<div class="source-item">
                <span class="s-name">[${s.rank}] ${s.doc_name}</span>
                <span class="s-score"> · ${s.hybrid_score.toFixed(4)}</span>
                <div class="s-text">${s.text.substring(0, 200)}...</div>
            </div>`
        ).join('');
    } else {
        hide('#qa-sources-wrap');
    }
}

$('#fv-btn').addEventListener('click', runVerify);

$$('.fv-example').forEach(btn => {
    btn.addEventListener('click', () => {
        if (!appConfig) return;
        const key = btn.dataset.key;
        const text = appConfig.verify_examples[key];
        if (text) {
            $('#fv-input').value = text;
            runVerify();
        }
    });
});

async function runVerify() {
    const text = $('#fv-input').value.trim();
    if (!text) return;

    show('#fv-output');
    $('#fv-trust-badge').innerHTML = '';
    $('#fv-summary').innerHTML = '';
    $('#fv-metrics').innerHTML = '';
    $('#fv-sources').innerHTML = '';
    $('#fv-highlight').innerHTML = '';
    $('#fv-claims').innerHTML = '';
    destroyChart('fv-pie');
    destroyChart('fv-bar');

    const stepDefs = [
        { icon: 'scissors', label: 'Extract Claims' },
        { icon: 'database', label: 'Retrieve Evidence' },
        { icon: 'brain', label: 'Entailment Scoring' },
        { icon: 'scale', label: 'Classify Verdicts' },
    ];

    function renderProgressPipeline(activeIndex, completedSteps) {
        const stepsHtml = stepDefs.map((s, i) => {
            const completed = completedSteps[i];
            const isActive = i === activeIndex;
            const isPending = i > activeIndex && !completed;
            const iconClass = isActive ? 'vp-icon vp-loading' : (completed ? 'vp-icon vp-done' : 'vp-icon');
            const opacity = isPending ? 'opacity:.3' : '';
            const labelOpacity = isPending ? 'opacity:.4' : '';
            const timeText = completed ? formatTime(completed.time_ms) : (isActive ? '…' : 'waiting');
            const detailText = completed ? completed.detail : '';
            return `<div class="vp-step" style="${isPending ? '' : ''}">
                <div class="${iconClass}" style="${opacity}"><i data-lucide="${s.icon}"></i></div>
                <div class="vp-label" style="${labelOpacity}">${s.label}</div>
                <div class="vp-time">${timeText}</div>
                ${detailText ? `<div class="vp-detail">${detailText}</div>` : ''}
            </div>`;
        }).join('');
        $('#fv-pipeline').innerHTML = `<div class="vp-total">Analyzing text…</div><div class="verify-pipeline">${stepsHtml}</div>`;
        if (typeof lucide !== 'undefined') lucide.createIcons();
    }

    const completedSteps = {};
    renderProgressPipeline(0, completedSteps);

    const token = getToken();
    const url = `/api/verify-stream?text=${encodeURIComponent(text)}&token=${encodeURIComponent(token || '')}`;
    const es = new EventSource(url);
    let finalData = null;

    es.onmessage = (e) => {
        if (e.data === '[DONE]') {
            es.close();
            return;
        }
        try {
            const msg = JSON.parse(e.data);
            if (msg.type === 'step') {
                completedSteps[msg.index] = { time_ms: msg.time_ms, detail: msg.detail };
                const nextActive = msg.index + 1 < stepDefs.length ? msg.index + 1 : -1;
                renderProgressPipeline(nextActive, completedSteps);
            } else if (msg.type === 'result') {
                finalData = msg;
                finalData.pipeline = stepDefs.map((s, i) => ({
                    step: s.label, icon: s.icon,
                    time_ms: completedSteps[i] ? completedSteps[i].time_ms : 0,
                    detail: completedSteps[i] ? completedSteps[i].detail : '',
                }));
                renderVerifyResults(finalData);
                show('#fv-output');
                es.close();
            } else if (msg.type === 'error') {
                $('#fv-pipeline').innerHTML = '';
                toast(msg.error, 'error');
                es.close();
            }
        } catch (err) {
            console.error('Verify SSE parse error:', err);
        }
    };
    es.onerror = () => {
        es.close();
        if (!finalData) {
            $('#fv-pipeline').innerHTML = '';
            toast('Verification failed', 'error');
        }
    };
}

function renderVerifyResults(data) {
    const claims = data.claims;
    const t = data.thresholds;
    const verified = claims.filter(c => c.verdict === 'VERIFIED').length;
    const contradicted = claims.filter(c => c.verdict === 'CONTRADICTED').length;
    const unverifiable = claims.filter(c => c.verdict === 'UNVERIFIABLE').length;
    const avg = claims.reduce((a, c) => a + c.nli_score, 0) / claims.length;

    const pipeEl = $('#fv-pipeline');
    if (data.pipeline && data.pipeline.length) {
        pipeEl.innerHTML =
            (data.total_ms ? `<div class="vp-total">Total verification time: <span>${formatTime(data.total_ms)}</span></div>` : '') +
            '<div class="verify-pipeline">' +
            data.pipeline.map(s => `
                <div class="vp-step">
                    <div class="vp-icon"><i data-lucide="${s.icon}"></i></div>
                    <div class="vp-label">${s.step}</div>
                    <div class="vp-time">${formatTime(s.time_ms)}</div>
                    <div class="vp-detail">${s.detail}</div>
                </div>
            `).join('') +
            '</div>';
    } else {
        pipeEl.innerHTML = '';
    }

    const offTopic = claims.every(c => c.nli_score === 0 && !c.evidence);

    const verifiedPct = claims.length ? (verified / claims.length) * 100 : 0;
    let trustLevel, trustLabel, trustIcon;
    if (offTopic) {
        trustLevel = 'low'; trustLabel = 'Outside Knowledge Base Domain'; trustIcon = 'alert-triangle';
    } else if (verifiedPct >= 70) {
        trustLevel = 'high'; trustLabel = 'High Reliability'; trustIcon = 'shield-check';
    } else if (verifiedPct >= 40) {
        trustLevel = 'moderate'; trustLabel = 'Moderate Reliability'; trustIcon = 'shield-alert';
    } else {
        trustLevel = 'low'; trustLabel = 'Low Reliability'; trustIcon = 'shield-x';
    }
    $('#fv-trust-badge').innerHTML = `
        <div class="trust-badge ${trustLevel}">
            <i data-lucide="${trustIcon}"></i>
            ${offTopic ? 'The input text does not relate to any content in the Knowledge Base' : `${trustLabel} — ${verified} of ${claims.length} claims verified (${verifiedPct.toFixed(0)}%)`}
        </div>`;

    if (offTopic) {
        $('#fv-summary').innerHTML = `<p class="fv-summary-text">
            The input text was decomposed into <strong>${claims.length}</strong> atomic claims,
            but none of them could be matched to content in the Knowledge Base.
            Please verify text that relates to the uploaded documents.
        </p>`;
    } else {
        const parts = [];
        if (verified) parts.push(`<strong style="color:var(--green)">${verified}</strong> verified`);
        if (contradicted) parts.push(`<strong style="color:var(--red)">${contradicted}</strong> contradicted`);
        if (unverifiable) parts.push(`<strong style="color:var(--yellow)">${unverifiable}</strong> unverifiable`);
        $('#fv-summary').innerHTML = `<p class="fv-summary-text">
            The input text was decomposed into <strong>${claims.length}</strong> atomic claims.
            Of these, ${parts.join(', ')}.
            Average entailment score: <strong>${avg.toFixed(3)}</strong>
        </p>`;
    }

    if (offTopic) {
        $('#fv-metrics').innerHTML = [
            metricCard('Claims', claims.length),
            metricCard('Relevance', '<span style="color:var(--red)">None</span>'),
        ].join('');
        $('#fv-sources').innerHTML = '';
        $('#fv-highlight').innerHTML = '';
        $('#fv-claims').innerHTML = claims.map((c, i) => `
            <div class="claim-item">
                <div class="claim-header">
                    <div class="claim-dot unverifiable"></div>
                    <div class="claim-title">Claim ${i + 1}: ${c.claim.substring(0, 80)}${c.claim.length > 80 ? '...' : ''}</div>
                    <div class="claim-score" style="color:var(--yellow)">NO MATCH</div>
                </div>
            </div>
        `).join('');
        if (typeof lucide !== 'undefined') lucide.createIcons();
        return;
    }

    $('#fv-metrics').innerHTML = [
        metricCard('Claims', claims.length),
        metricCard('Verified', `<span style="color:var(--green)">${verified}</span>`),
        metricCard('Contradicted', `<span style="color:var(--red)">${contradicted}</span>`),
        metricCard('Unverifiable', `<span style="color:var(--yellow)">${unverifiable}</span>`),
        metricCard('Avg Score', avg.toFixed(3)),
    ].join('');

    const srcMap = {};
    claims.forEach(c => {
        if (c.evidence_source) srcMap[c.evidence_source] = (srcMap[c.evidence_source] || 0) + 1;
    });
    const srcEntries = Object.entries(srcMap).sort((a, b) => b[1] - a[1]);
    if (srcEntries.length) {
        $('#fv-sources').innerHTML = '<div class="fv-sources">' +
            srcEntries.map(([name, cnt]) =>
                `<span class="fv-src-chip"><i data-lucide="file-text"></i>${name} (${cnt} claim${cnt > 1 ? 's' : ''})</span>`
            ).join('') + '</div>';
    } else {
        $('#fv-sources').innerHTML = '';
    }

    destroyChart('fv-pie');
    chartInstances['fv-pie'] = new Chart($('#fv-pie'), {
        type: 'doughnut',
        data: {
            labels: ['Verified', 'Contradicted', 'Unverifiable'],
            datasets: [{ data: [verified, contradicted, unverifiable], backgroundColor: ['#22c55e', '#ef4444', '#f59e0b'], borderWidth: 2, borderColor: ['#16a34a', '#dc2626', '#d97706'] }],
        },
        options: { responsive: true, cutout: '55%', plugins: { legend: { position: 'bottom', labels: { padding: 15 } } } },
    });

    destroyChart('fv-bar');
    const barColors = claims.map(c => c.verdict === 'VERIFIED' ? '#22c55e' : c.verdict === 'CONTRADICTED' ? '#ef4444' : '#f59e0b');
    const barBorders = claims.map(c => c.verdict === 'VERIFIED' ? '#16a34a' : c.verdict === 'CONTRADICTED' ? '#dc2626' : '#d97706');
    chartInstances['fv-bar'] = new Chart($('#fv-bar'), {
        type: 'bar',
        data: {
            labels: claims.map((_, i) => `C${i + 1}`),
            datasets: [{ data: claims.map(c => c.nli_score), backgroundColor: barColors, borderColor: barBorders, borderWidth: 1.5, borderRadius: 3 }],
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                annotation: {
                    annotations: {
                        verifiedLine: {
                            type: 'line', yMin: t.verified, yMax: t.verified,
                            borderColor: '#16a34a', borderWidth: 2, borderDash: [8, 4],
                            label: { display: true, content: '✓ Verified ≥ ' + t.verified, position: 'end', backgroundColor: 'rgba(22,163,74,.2)', color: '#22c55e', font: { size: 11, weight: 'bold' } }
                        },
                        contradictedLine: {
                            type: 'line', yMin: t.contradicted, yMax: t.contradicted,
                            borderColor: '#dc2626', borderWidth: 2, borderDash: [8, 4],
                            label: { display: true, content: '✗ Contradicted ≤ ' + t.contradicted, position: 'end', backgroundColor: 'rgba(220,38,38,.2)', color: '#ef4444', font: { size: 11, weight: 'bold' } }
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        title: (ctx) => {
                            const i = ctx[0].dataIndex;
                            return claims[i].claim.length > 60 ? claims[i].claim.substring(0, 60) + '...' : claims[i].claim;
                        },
                        label: (ctx) => {
                            const i = ctx.dataIndex;
                            const c = claims[i];
                            return `${c.verdict} · Score: ${c.nli_score.toFixed(3)}`;
                        },
                        afterLabel: (ctx) => {
                            const i = ctx.dataIndex;
                            const c = claims[i];
                            if (c.evidence) {
                                const ev = c.evidence.length > 80 ? c.evidence.substring(0, 80) + '...' : c.evidence;
                                return `Evidence: ${ev}`;
                            }
                            return '';
                        }
                    }
                }
            },
            scales: {
                y: { min: 0, max: 1.1, grid: { color: gridColor() } },
                x: { grid: { display: false } },
            },
        },
    });

    const inputText = $('#fv-input').value;
    if (inputText && claims.length) {
        let highlighted = inputText;
        const claimMap = claims.map(c => ({ text: c.claim, verdict: c.verdict.toLowerCase(), score: c.nli_score }));
        const sorted = claimMap
            .map(cm => ({ ...cm, idx: inputText.toLowerCase().indexOf(cm.text.toLowerCase().substring(0, 40)) }))
            .filter(cm => cm.idx >= 0)
            .sort((a, b) => b.idx - a.idx);

        sorted.forEach(cm => {
            const start = cm.idx;
            let end = start + cm.text.length;
            const nextPeriod = inputText.indexOf('.', end - 5);
            if (nextPeriod > 0 && nextPeriod < end + 10) end = nextPeriod + 1;
            else end = Math.min(end + 1, inputText.length);

            const original = inputText.substring(start, end);
            const tipText = `${cm.verdict.toUpperCase()} · Score: ${cm.score.toFixed(3)}`;
            const replacement = `<span class="fv-hl ${cm.verdict}"><span class="fv-hl-tip">${tipText}</span>${original}</span>`;
            highlighted = highlighted.substring(0, start) + replacement + highlighted.substring(end);
        });

        $('#fv-highlight').innerHTML = `<div class="fv-highlight-box">
            <span class="hl-label"><i data-lucide="highlighter" style="width:13px;height:13px;display:inline;vertical-align:middle"></i> Claim Annotations</span>
            ${highlighted}
        </div>`;
    } else {
        $('#fv-highlight').innerHTML = '';
    }

    $('#fv-claims').innerHTML = claims.map((c, i) => {
        const cls = c.verdict.toLowerCase();
        const fillColor = cls === 'verified' ? 'var(--green)' : cls === 'contradicted' ? 'var(--red)' : 'var(--yellow)';
        const expanded = c.verdict !== 'VERIFIED';
        return `
            <div class="claim-item">
                <div class="claim-header" onclick="this.nextElementSibling.classList.toggle('open')">
                    <div class="claim-dot ${cls}"></div>
                    <div class="claim-title">Claim ${i + 1}: ${c.claim.substring(0, 80)}${c.claim.length > 80 ? '...' : ''}</div>
                    <div class="claim-score">${c.verdict} · ${(c.nli_score * 100).toFixed(1)}%</div>
                </div>
                <div class="claim-body ${expanded ? 'open' : ''}">
                    <p>${c.claim}</p>
                    <div class="nli-bar"><div class="nli-fill" style="width:${Math.min(c.nli_score * 100, 100)}%;background:${fillColor}"></div></div>
                    ${c.evidence ? `<p class="caption" style="margin-top:.5rem"><strong>Evidence</strong> (${c.evidence_source}): ${c.evidence}</p>` : ''}
                </div>
            </div>
        `;
    }).join('');

    if (typeof lucide !== 'undefined') lucide.createIcons();
}

$('#xray-btn').addEventListener('click', runXRay);
$('#xray-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') runXRay(); });

async function runXRay() {
    const query = $('#xray-input').value.trim();
    if (!query) return;

    hide('#xray-output');
    show('#xray-progress');
    $('#xray-pbar').style.width = '100%';
    $('#xray-ptext').textContent = 'Running AutoRAG++ pipeline...';

    const stages = [
        [3000, 'Running Basic RAG pipeline...'],
        [6000, 'Running LLM-Only baseline...'],
        [9000, 'Computing comparison metrics...'],
    ];
    const timers = stages.map(([ms, text]) => setTimeout(() => {
        if ($('#xray-progress').style.display !== 'none') $('#xray-ptext').textContent = text;
    }, ms));

    try {
        const res = await authFetch(`/api/xray?query=${encodeURIComponent(query)}`);
        const data = await res.json();
        timers.forEach(clearTimeout);
        hide('#xray-progress');
        renderXRay(data);
        show('#xray-output');
    } catch (e) {
        timers.forEach(clearTimeout);
        hide('#xray-progress');
        toast('Comparison failed', 'error');
    }
}

function renderXRay(data) {
    const modes = ['AutoRAG++', 'Basic RAG', 'LLM-Only'];
    const modeColors = ['#6366f1', '#3b82f6', '#94a3b8'];
    const modeBorders = ['#4f46e5', '#2563eb', '#64748b'];
    const results = [data.autorag, data.basic_rag, data.llm_only];

    const winnerIdx = results.reduce((best, r, i) => r.confidence > results[best].confidence ? i : best, 0);

    destroyChart('xray-scores');
    chartInstances['xray-scores'] = new Chart($('#xray-scores'), {
        type: 'bar',
        data: {
            labels: modes,
            datasets: [
                { label: 'S_c', data: results.map(r => r.semantic_consistency), backgroundColor: '#3b82f6', borderColor: '#2563eb', borderWidth: 1, borderRadius: 3 },
                { label: 'F_c', data: results.map(r => r.factual_correctness), backgroundColor: '#f97316', borderColor: '#ea580c', borderWidth: 1, borderRadius: 3 },
                { label: 'C_f', data: results.map(r => r.confidence), backgroundColor: '#10b981', borderColor: '#059669', borderWidth: 1, borderRadius: 3 },
            ],
        },
        options: {
            responsive: true,
            scales: { y: { min: 0, max: 1.15, grid: { color: gridColor() } }, x: { grid: { display: false } } },
            plugins: { legend: { position: 'bottom', labels: { padding: 12 } } },
        },
    });

    destroyChart('xray-time');
    chartInstances['xray-time'] = new Chart($('#xray-time'), {
        type: 'bar',
        data: {
            labels: modes,
            datasets: [{ label: 'Time (s)', data: results.map(r => r.total_time_ms / 1000), backgroundColor: modeColors, borderColor: modeBorders, borderWidth: 1.5, borderRadius: 3 }],
        },
        options: {
            responsive: true,
            scales: { y: { grid: { color: gridColor() }, title: { display: true, text: 'Seconds' } }, x: { grid: { display: false } } },
            plugins: { legend: { display: false } },
        },
    });

    $('#xray-answers').innerHTML = modes.map((mode, i) => {
        const r = results[i];
        const isWinner = i === winnerIdx && r.confidence > 0;
        const winnerBorder = isWinner ? 'border: 2px solid var(--green); box-shadow: 0 0 12px rgba(34,197,94,.15);' : '';
        const winnerTag = isWinner ? '<span style="color:var(--green);font-size:.7rem;font-weight:600;margin-left:.5rem">★ BEST</span>' : '';
        return `
            <div class="card" style="${winnerBorder}">
                <h3 style="display:flex;align-items:center;gap:.3rem">${mode} ${confBadge(r.confidence)}${winnerTag}</h3>
                <div class="metrics-row" style="margin:.5rem 0">
                    ${metricCard('Sources', r.total_sources_used)}
                    ${metricCard('Time', formatTime(r.total_time_ms))}
                    ${metricCard('Cycles', r.calibration_cycles)}
                </div>
                <div class="answer-box">${r.error ? `<span style="color:var(--red)">${r.error}</span>` : r.answer}</div>
            </div>
        `;
    }).join('');

    const metrics = [
        ['S_c (Semantic)', r => r.semantic_consistency, v => v.toFixed(3)],
        ['F_c (Factual)', r => r.factual_correctness, v => v.toFixed(3)],
        ['C_f (Confidence)', r => r.confidence, v => v.toFixed(3)],
        ['Sources Used', r => r.total_sources_used, v => v],
        ['Calibration Cycles', r => r.calibration_cycles, v => v],
        ['Response Time', r => r.total_time_ms, v => formatTime(v)],
    ];
    $('#xray-table').innerHTML = `
        <thead><tr><th>Metric</th>${modes.map(m => `<th>${m}</th>`).join('')}</tr></thead>
        <tbody>${metrics.map(([name, getter, fmt]) => {
            const vals = results.map(getter);
            const isTime = name === 'Response Time';
            const bestVal = isTime ? Math.min(...vals.filter(v => v > 0)) : Math.max(...vals);
            return `<tr><td><strong>${name}</strong></td>${vals.map(v =>
                `<td style="${v === bestVal && bestVal > 0 ? 'color:var(--green);font-weight:600' : ''}">${fmt(v)}</td>`
            ).join('')}</tr>`;
        }).join('')}</tbody>
    `;

    if (typeof lucide !== 'undefined') lucide.createIcons();
}

$$('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        $$('.tab').forEach(t => t.classList.remove('active'));
        $$('.tab-content').forEach(tc => tc.classList.remove('active'));
        tab.classList.add('active');
        $(`#${tab.dataset.tab}`).classList.add('active');
    });
});

async function loadKB() {
    try {
        const statsRes = await authFetch('/api/stats');
        const stats = (await statsRes.json()).kb || {};
        $('#kb-metrics').innerHTML = [
            metricCard('Chunks', stats.total_chunks || 0),
            metricCard('Documents', stats.total_documents || 0),
            metricCard('Seed', stats.seed_docs || 0),
            metricCard('Auto-fetched', stats.auto_fetched_docs || 0),
            metricCard('Uploaded', stats.uploaded_docs || 0),
        ].join('');

        const docsRes = await authFetch('/api/kb/documents');
        const docs = await docsRes.json();

        if (docs.length === 0) {
            hide('#kb-doc-table');
            hide('#kb-delete-row');
            show('#kb-empty');
        } else {
            show('#kb-doc-table');
            hide('#kb-empty');
            show('#kb-delete-row');

            $('#kb-doc-table').innerHTML = `
                <thead><tr><th>Document</th><th>Source</th><th>Chunks</th></tr></thead>
                <tbody>${docs.map(d =>
                    `<tr><td>${d.doc_name}</td><td>${d.source}</td><td>${d.chunk_count}</td></tr>`
                ).join('')}</tbody>
            `;

            const sel = $('#kb-del-select');
            sel.innerHTML = docs.map(d => `<option value="${d.doc_name}">${d.doc_name}</option>`).join('');
        }
    } catch (e) {
        console.error('KB load failed:', e);
    }
}

$('#kb-del-btn').addEventListener('click', async () => {
    const name = $('#kb-del-select').value;
    if (!name) return;
    spinner(true, 'Deleting...');
    try {
        await authFetch(`/api/kb/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
        spinner(false);
        toast(`Deleted ${name}`);
        loadKB();
    } catch (e) {
        spinner(false);
        toast('Delete failed', 'error');
    }
});

const uploadArea = $('#kb-upload-area');
const uploadFile = $('#kb-file');

uploadArea.addEventListener('click', () => uploadFile.click());
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.style.borderColor = 'var(--accent)'; });
uploadArea.addEventListener('dragleave', () => { uploadArea.style.borderColor = ''; });
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.borderColor = '';
    if (e.dataTransfer.files.length) {
        uploadFile.files = e.dataTransfer.files;
        handleFileSelect();
    }
});
uploadFile.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    const file = uploadFile.files[0];
    if (file) {
        $('#kb-file-name').textContent = file.name;
        show('#kb-upload-btn');
    }
}

$('#kb-upload-btn').addEventListener('click', async () => {
    const file = uploadFile.files[0];
    if (!file) return;
    spinner(true, 'Indexing document...');
    try {
        const form = new FormData();
        form.append('file', file);
        const res = await authFetch('/api/kb/upload', { method: 'POST', body: form });
        const data = await res.json();
        spinner(false);
        if (!res.ok) {
            toast(data.detail || 'Upload failed', 'error');
            return;
        }
        toast(`Indexed ${data.doc_name}: ${data.chunks_added} chunks`);
        uploadFile.value = '';
        $('#kb-file-name').textContent = '';
        hide('#kb-upload-btn');
        loadKB();
    } catch (e) {
        spinner(false);
        toast('Upload failed', 'error');
    }
});

$('#kb-search-btn').addEventListener('click', async () => {
    const query = $('#kb-q').value.trim();
    const k = parseInt($('#kb-k').value) || 5;
    if (!query) return;

    spinner(true, 'Searching...');
    try {
        const res = await authFetch('/api/kb/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, k }),
        });
        const results = await res.json();
        spinner(false);

        if (results.length === 0) {
            $('#kb-results').innerHTML = '<div class="empty-state"><p>No results found.</p></div>';
            return;
        }

        $('#kb-results').innerHTML = results.map((r, i) => {
            const src = r.metadata?.source || 'unknown';
            const badge = src === 'uploaded' ? '<span class="badge" style="background:var(--accent);font-size:.65rem;padding:2px 6px;margin-left:6px">Uploaded</span>'
                : src === 'auto_fetched' ? '<span class="badge" style="background:var(--warning);color:#000;font-size:.65rem;padding:2px 6px;margin-left:6px">Wikipedia</span>'
                : src === 'web' ? '<span class="badge" style="background:#10b981;color:#fff;font-size:.65rem;padding:2px 6px;margin-left:6px">Web</span>'
                : src === 'seed' ? '<span class="badge" style="background:var(--muted);font-size:.65rem;padding:2px 6px;margin-left:6px">Seed</span>' : '';
            return `<div class="source-item">
                <span class="s-name">#${i + 1} ${r.metadata?.doc_name || '?'}${badge}</span>
                <span class="s-score"> · similarity: ${r.similarity.toFixed(4)}</span>
                <div class="s-text">${r.text}</div>
            </div>`;
        }).join('');
    } catch (e) {
        spinner(false);
        toast('Search failed', 'error');
    }
});

$('#kb-seed-btn').addEventListener('click', async () => {
    const topic = $('#kb-seed-topic').value.trim();
    spinner(true, topic ? `Searching Wikipedia + Web for "${topic}"...` : 'Auto-detecting topics from your documents...');
    $('#kb-seed-status').textContent = '';
    try {
        const res = await authFetch('/api/kb/seed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic }),
        });
        const data = await res.json();
        spinner(false);
        if (data.chunks_added > 0) {
            const parts = [];
            if (data.wiki) parts.push(`${data.wiki} from Wikipedia`);
            if (data.web) parts.push(`${data.web} from Web`);
            const detail = parts.join(', ');
            toast(`Added ${data.chunks_added} chunks (${detail})`);
            $('#kb-seed-status').textContent = `Added ${data.chunks_added} chunks: ${detail}.`;
        } else {
            toast('No new articles found', 'warning');
            $('#kb-seed-status').textContent = topic
                ? 'No relevant articles found for this topic.'
                : 'No uploaded documents to derive topics from. Upload a PDF first or enter a topic manually.';
        }
        loadKB();
    } catch (e) {
        spinner(false);
        toast('Augmentation failed', 'error');
    }
});

function initSettingsSliders() {
    const sliders = [
        { id: 'sv-verified', valId: 'sv-verified-val', fmt: v => parseFloat(v).toFixed(2) },
        { id: 'sv-contradicted', valId: 'sv-contradicted-val', fmt: v => parseFloat(v).toFixed(2) },
        { id: 'sv-confidence', valId: 'sv-confidence-val', fmt: v => parseFloat(v).toFixed(2) },
        { id: 'sv-cycles', valId: 'sv-cycles-val', fmt: v => v },
    ];
    sliders.forEach(s => {
        const el = document.getElementById(s.id);
        const valEl = document.getElementById(s.valId);
        if (el && valEl) {
            el.addEventListener('input', () => { valEl.textContent = s.fmt(el.value); });
        }
    });
}
initSettingsSliders();

async function loadSettings() {
    try {
        const res = await authFetch('/api/settings');
        const data = await res.json();
        $('#settings-provider').value = data.preferred_provider || 'auto';

        if (data.verified_threshold != null) {
            $('#sv-verified').value = data.verified_threshold;
            $('#sv-verified-val').textContent = data.verified_threshold.toFixed(2);
        }
        if (data.contradicted_threshold != null) {
            $('#sv-contradicted').value = data.contradicted_threshold;
            $('#sv-contradicted-val').textContent = data.contradicted_threshold.toFixed(2);
        }
        if (data.confidence_threshold != null) {
            $('#sv-confidence').value = data.confidence_threshold;
            $('#sv-confidence-val').textContent = data.confidence_threshold.toFixed(2);
        }
        if (data.max_cycles != null) {
            $('#sv-cycles').value = data.max_cycles;
            $('#sv-cycles-val').textContent = data.max_cycles;
        }

        const avail = data.available || {};
        $('#settings-metrics').innerHTML = [
            metricCard('Gemini', avail.gemini ? '<span style="color:var(--green)">✓ Ready</span>' : '<span style="color:var(--red)">✗ No Key</span>'),
            metricCard('Groq', avail.groq ? '<span style="color:var(--green)">✓ Ready</span>' : '<span style="color:var(--red)">✗ No Key</span>'),
            metricCard('KB Chunks', avail.kb_chunks || 0),
            metricCard('Verification', 'Entailment Engine'),
        ].join('');
    } catch (e) {
        console.error('Settings load failed:', e);
    }
}

$('#settings-save').addEventListener('click', async () => {
    const payload = {
        preferred_provider: $('#settings-provider').value,
        verified_threshold: parseFloat($('#sv-verified').value),
        contradicted_threshold: parseFloat($('#sv-contradicted').value),
        confidence_threshold: parseFloat($('#sv-confidence').value),
        max_cycles: parseInt($('#sv-cycles').value),
    };
    try {
        const res = await authFetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (res.ok) {
            toast('Settings saved successfully');
            const changesEl = $('#settings-changes');
            if (data.changes && data.changes.length) {
                changesEl.innerHTML = data.changes.map(c =>
                    `<div class="change-item"><i data-lucide="check-circle"></i> ${c}</div>`
                ).join('');
                if (typeof lucide !== 'undefined') lucide.createIcons();
            } else {
                changesEl.innerHTML = '<div class="change-item"><i data-lucide="check-circle"></i> Provider preference updated</div>';
                if (typeof lucide !== 'undefined') lucide.createIcons();
            }
            setTimeout(() => { changesEl.innerHTML = ''; }, 5000);
        } else {
            toast('Save failed: ' + (data.detail || 'Unknown error'), 'error');
        }
    } catch (e) {
        toast('Save failed', 'error');
    }
});

async function init() {
    if (typeof lucide !== 'undefined') lucide.createIcons();

    try {
        const statusRes = await fetch('/api/auth/status');
        const statusData = await statusRes.json();
        if (statusData.auth_enabled) {
            const token = getToken();
            if (!token) { window.location.href = '/login'; return; }
            const meRes = await fetch('/api/auth/me', {
                headers: { 'Authorization': 'Bearer ' + token }
            });
            if (!meRes.ok) {
                localStorage.removeItem('verifai_token');
                localStorage.removeItem('verifai_user');
                window.location.href = '/login';
                return;
            }
        }
    } catch (e) {
        console.warn('Auth check failed:', e);
    }

    const user = getUser();
    if (user && user.name) {
        const avatar = $('.header-avatar');
        if (avatar) {
            avatar.title = user.name;
            avatar.innerHTML = `<span style="font-size:.8rem;font-weight:600">${user.name.charAt(0).toUpperCase()}</span>`;
            avatar.style.cursor = 'pointer';
            avatar.addEventListener('click', () => {
                if (confirm('Sign out?')) logout();
            });
        }
    }

    try {
        const res = await authFetch('/api/config');
        appConfig = await res.json();

        loadRecentSearches();

        if (appConfig.xray_examples) {
            $('#xray-examples').innerHTML = appConfig.xray_examples.map(q =>
                `<span class="example-chip" data-query="${q.replace(/"/g, '&quot;')}">${q.length > 45 ? q.substring(0, 45) + '...' : q}</span>`
            ).join('');
            $$('#xray-examples .example-chip').forEach(chip => {
                chip.addEventListener('click', () => {
                    $('#xray-input').value = chip.dataset.query;
                    runXRay();
                });
            });
        }
    } catch (e) {
        console.error('Config load failed:', e);
    }

    loadDashboard();
}

init();
