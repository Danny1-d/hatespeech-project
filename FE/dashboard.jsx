import { useState, useEffect, useCallback } from "react";

// ── Config ─────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";   // FastAPI backend URL

// ── Colours ────────────────────────────────────────────────────────────────
const C = {
  bg: "#0d0f14", card: "#141720", border: "#1e2330",
  hate: "#e84855", safe: "#3bb273", accent: "#2e86ab",
  yellow: "#ffc857", purple: "#6a4c93", muted: "#8892a4",
  text: "#e8ecf4", subtext: "#a8b2c4",
};

const MODEL_COLORS = { transformer: C.safe, deep: C.accent, classical: C.yellow };

// ── Tiny helpers ───────────────────────────────────────────────────────────
function Badge({ children, color }) {
  return (
    <span style={{
      background: color + "22", color,
      border: `1px solid ${color}44`,
      borderRadius: 4, padding: "2px 8px",
      fontSize: 11, fontWeight: 600,
      letterSpacing: "0.04em", textTransform: "uppercase",
    }}>{children}</span>
  );
}

function Bar({ value, color, height = 8 }) {
  return (
    <div style={{ background: C.border, borderRadius: 4, height, flex: 1 }}>
      <div style={{
        width: `${Math.min(value * 100, 100)}%`, height: "100%",
        background: color, borderRadius: 4, transition: "width 0.5s ease",
      }} />
    </div>
  );
}

function Spinner() {
  return (
    <span style={{
      display: "inline-block", width: 16, height: 16,
      border: `2px solid ${C.border}`,
      borderTop: `2px solid ${C.accent}`,
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
    }} />
  );
}

function StatusDot({ ok }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8,
      borderRadius: "50%", background: ok ? C.safe : C.hate,
      boxShadow: `0 0 6px ${ok ? C.safe : C.hate}`,
      marginRight: 6,
    }} />
  );
}

const TABS = ["Overview", "Models", "Live Detector", "Batch Test"];
const EXAMPLES = [
  "Chukwu gozie gị! I love my Igbo culture and people so much 🙏",
  "Kill all those useless people! Gbuo ha niile from this country!",
  "Nna, the market today was full of life. Ahịa na-atọ ụtọ nke ọma.",
  "These dirty criminals from that tribe should be removed. Trash people!",
  "Just got promoted at work! Obi ụtọ nke ukwuu. God is truly faithful.",
  "All women are useless in business. Nwanyị adịghị mma at all!",
];

// ── API helpers ────────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ══════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab]           = useState("Live Detector");
  const [health, setHealth]     = useState(null);
  const [stats, setStats]       = useState(null);
  const [serverUp, setServerUp] = useState(null);  // null=checking, true/false

  // Live Detector state
  const [inputText, setInputText]     = useState("");
  const [prediction, setPrediction]   = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError]     = useState(null);

  // Batch state
  const [batchInput, setBatchInput]   = useState(EXAMPLES.join("\n"));
  const [batchResult, setBatchResult] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);

  // Re-train
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainMsg, setTrainMsg]         = useState(null);

  // ── Poll backend health on mount ────────────────────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const h = await apiFetch("/health");
      setHealth(h);
      setServerUp(true);
    } catch {
      setServerUp(false);
      setHealth(null);
    }
  }, []);

  const fetchStats = useCallback(async () => {
    try {
      const s = await apiFetch("/stats");
      setStats(s);
    } catch {}
  }, []);

  useEffect(() => {
    checkHealth();
    fetchStats();
    const id = setInterval(checkHealth, 15000);
    return () => clearInterval(id);
  }, [checkHealth, fetchStats]);

  // ── Single prediction ───────────────────────────────────────────────────
  async function handlePredict() {
    if (!inputText.trim()) return;
    setPredLoading(true);
    setPredError(null);
    setPrediction(null);
    try {
      const result = await apiFetch("/predict", {
        method: "POST",
        body: JSON.stringify({ text: inputText }),
      });
      setPrediction(result);
    } catch (e) {
      setPredError(e.message);
    } finally {
      setPredLoading(false);
    }
  }

  // ── Batch prediction ────────────────────────────────────────────────────
  async function handleBatch() {
    const texts = batchInput.split("\n").map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;
    setBatchLoading(true);
    setBatchResult(null);
    try {
      const result = await apiFetch("/predict/batch", {
        method: "POST",
        body: JSON.stringify({ texts }),
      });
      setBatchResult(result);
    } catch (e) {
      alert("Batch error: " + e.message);
    } finally {
      setBatchLoading(false);
    }
  }

  // ── Re-train ────────────────────────────────────────────────────────────
  async function handleRetrain() {
    setTrainLoading(true);
    setTrainMsg(null);
    try {
      const r = await apiFetch("/train", { method: "POST" });
      setTrainMsg(`✓ Retrained! F1 = ${(r.test_f1 * 100).toFixed(1)}% on ${r.train_size} samples`);
      await checkHealth();
      await fetchStats();
    } catch (e) {
      setTrainMsg(`✗ ${e.message}`);
    } finally {
      setTrainLoading(false);
    }
  }

  // ── Shared styles ────────────────────────────────────────────────────────
  const card = {
    background: C.card, border: `1px solid ${C.border}`,
    borderRadius: 10, padding: 16,
  };

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "20px 16px",
    }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
        .fade-in { animation: fadeIn 0.3s ease; }
        textarea:focus, input:focus { outline: none; border-color: ${C.accent} !important; }
        button:hover { opacity: 0.88; }
      `}</style>

      {/* ── Header ── */}
      <div style={{ maxWidth: 900, margin: "0 auto 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 10 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{
              background: `linear-gradient(135deg, ${C.hate}, ${C.purple})`,
              borderRadius: 10, padding: "8px 12px", fontSize: 22,
            }}>🛡️</div>
            <div>
              <div style={{ fontSize: 16, fontWeight: 800 }}>Hate Speech Detector</div>
              <div style={{ color: C.subtext, fontSize: 11, marginTop: 2 }}>English–Igbo Code-Mixed · Powered by Logistic Regression + TF-IDF</div>
            </div>
          </div>

          {/* Server status badge */}
          <div style={{
            ...card, padding: "8px 14px",
            display: "flex", alignItems: "center", gap: 8, fontSize: 12,
          }}>
            {serverUp === null ? <Spinner /> : <StatusDot ok={serverUp} />}
            {serverUp === null && <span style={{ color: C.muted }}>Connecting…</span>}
            {serverUp === true && (
              <span style={{ color: C.safe }}>
                API Online
                {health?.test_f1 > 0 && <span style={{ color: C.muted, marginLeft: 6 }}>· F1 {(health.test_f1 * 100).toFixed(0)}%</span>}
              </span>
            )}
            {serverUp === false && (
              <span style={{ color: C.hate }}>
                API Offline — run: <code style={{ background: C.bg, padding: "1px 5px", borderRadius: 3 }}>python 07_api_server.py</code>
              </span>
            )}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, borderBottom: `1px solid ${C.border}`, marginTop: 18 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? C.accent : "transparent",
              color: tab === t ? "#fff" : C.muted,
              border: "none", borderRadius: "6px 6px 0 0",
              padding: "7px 14px", cursor: "pointer",
              fontSize: 12, fontFamily: "inherit",
              fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 900, margin: "0 auto" }}>

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* OVERVIEW TAB                                                    */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {tab === "Overview" && (
          <div className="fade-in">
            {/* Key metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 16 }}>
              {[
                { icon: "🏆", label: "Best Model F1", value: "91.1%", color: C.safe, sub: "AfriBERTa" },
                { icon: "📦", label: "Dataset Size",  value: stats?.dataset?.total || "1,234", color: C.accent, sub: "annotated posts", raw: true },
                { icon: "⚠️", label: "Hate Speech",   value: `${stats?.dataset?.hate || 592}`, color: C.hate,  sub: "labelled posts", raw: true },
                { icon: "🤖", label: "Models Tested", value: "7",     color: C.yellow, sub: "benchmarked", raw: true },
              ].map(m => (
                <div key={m.label} style={{ ...card, textAlign: "center" }}>
                  <div style={{ fontSize: 20 }}>{m.icon}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: m.color, marginTop: 4 }}>{m.value}</div>
                  <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{m.label}</div>
                  <div style={{ fontSize: 9, color: C.subtext }}>{m.sub}</div>
                </div>
              ))}
            </div>

            {/* Active model info */}
            <div style={{
              ...card, marginBottom: 12,
              background: `linear-gradient(135deg,${C.safe}12,${C.accent}12)`,
              border: `1px solid ${C.safe}44`,
            }}>
              <div style={{ fontSize: 10, color: C.safe, fontWeight: 700, marginBottom: 6, letterSpacing: "0.07em" }}>
                ★ ACTIVE MODEL (connected to API)
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                <div>
                  <div style={{ fontWeight: 800, fontSize: 15 }}>
                    {health?.model || "Logistic Regression (TF-IDF)"}
                  </div>
                  <div style={{ color: C.subtext, fontSize: 11, marginTop: 3 }}>
                    Trained: {health?.trained_at || "—"} · Samples: {health?.train_size || "—"}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                  <div style={{ textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: C.safe }}>
                      {health?.test_f1 > 0 ? `${(health.test_f1*100).toFixed(1)}%` : "—"}
                    </div>
                    <div style={{ fontSize: 10, color: C.muted }}>Test F1</div>
                  </div>
                  <button onClick={handleRetrain} disabled={trainLoading || !serverUp} style={{
                    background: C.accent, color: "#fff", border: "none",
                    borderRadius: 7, padding: "7px 14px", cursor: "pointer",
                    fontSize: 11, fontFamily: "inherit", fontWeight: 700,
                    opacity: !serverUp ? 0.4 : 1,
                    display: "flex", alignItems: "center", gap: 6,
                  }}>
                    {trainLoading ? <><Spinner /> Training…</> : "↺ Retrain"}
                  </button>
                </div>
              </div>
              {trainMsg && (
                <div style={{ marginTop: 8, fontSize: 12, color: trainMsg.startsWith("✓") ? C.safe : C.hate }}>
                  {trainMsg}
                </div>
              )}
            </div>

            {/* Architecture flow */}
            <div style={{ ...card }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 10 }}>⚙️ Full Stack Architecture</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center", fontSize: 11 }}>
                {[
                  ["React Dashboard", C.accent],
                  ["→", null],
                  ["HTTP POST /predict", C.yellow],
                  ["→", null],
                  ["FastAPI Server :8000", C.purple],
                  ["→", null],
                  ["Preprocessor", C.muted],
                  ["→", null],
                  ["TF-IDF Features", C.muted],
                  ["→", null],
                  ["LR Model", C.safe],
                  ["→", null],
                  ["JSON Response", C.hate],
                ].map(([label, color], i) => (
                  <span key={i} style={{
                    color: color || C.muted,
                    background: color && color !== C.muted ? color + "18" : "transparent",
                    border: color && color !== C.muted ? `1px solid ${color}44` : "none",
                    borderRadius: 5, padding: color && color !== C.muted ? "3px 8px" : 0,
                  }}>{label}</span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* MODELS TAB                                                      */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {tab === "Models" && (
          <div className="fade-in">
            <p style={{ color: C.subtext, fontSize: 12, marginBottom: 14 }}>
              Benchmark results from the research paper. The active API uses Logistic Regression.
              To use AfriBERTa, fine-tune Module 03 and update the model path in 07_api_server.py.
            </p>
            {(stats?.benchmark || []).map((m, i) => (
              <div key={m.name} style={{
                ...card, marginBottom: 10,
                border: `1px solid ${i === 0 ? C.safe + "66" : C.border}`,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    {i === 0 && <span style={{ color: C.safe }}>★</span>}
                    <span style={{ fontWeight: 700, fontSize: 13 }}>{m.name}</span>
                    <Badge color={MODEL_COLORS[m.type]}>{m.type}</Badge>
                  </div>
                  <div style={{ display: "flex", gap: 14 }}>
                    {[["Acc", m.acc, C.accent], ["F1", m.f1w, C.safe], ["F1 Hate", m.f1h, C.hate]].map(([l, v, c]) => (
                      <div key={l} style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 15, fontWeight: 800, color: c }}>{(v * 100).toFixed(1)}%</div>
                        <div style={{ fontSize: 10, color: C.muted }}>{l}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ fontSize: 10, color: C.muted, minWidth: 44 }}>F1</span>
                  <Bar value={m.f1w} color={MODEL_COLORS[m.type]} />
                  <span style={{ fontSize: 11, color: C.subtext, minWidth: 36, textAlign: "right" }}>
                    {(m.f1w * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* LIVE DETECTOR TAB — fully connected to backend                 */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {tab === "Live Detector" && (
          <div className="fade-in">

            {/* Offline banner */}
            {serverUp === false && (
              <div style={{
                background: C.hate + "18", border: `1px solid ${C.hate}55`,
                borderRadius: 8, padding: "10px 14px", marginBottom: 14, fontSize: 12,
              }}>
                <strong style={{ color: C.hate }}>⚠️ Backend Offline</strong>
                <span style={{ color: C.subtext, marginLeft: 8 }}>
                  Start the server first:
                </span>
                <code style={{ background: C.bg, borderRadius: 4, padding: "2px 8px", marginLeft: 6, color: C.yellow }}>
                  python 07_api_server.py
                </code>
              </div>
            )}

            {/* Input area */}
            <div style={{ ...card, marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>
                ✍️ Enter English-Igbo Text
              </div>
              <textarea
                value={inputText}
                onChange={e => { setInputText(e.target.value); setPrediction(null); setPredError(null); }}
                onKeyDown={e => { if (e.key === "Enter" && e.ctrlKey) handlePredict(); }}
                placeholder="Type or paste an English-Igbo code-mixed post here… (Ctrl+Enter to submit)"
                style={{
                  width: "100%", minHeight: 80,
                  background: C.bg, border: `1px solid ${C.border}`,
                  borderRadius: 7, color: C.text, fontFamily: "inherit",
                  fontSize: 13, padding: "10px 12px", resize: "vertical",
                  boxSizing: "border-box",
                }}
              />
              <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
                <button onClick={handlePredict}
                  disabled={predLoading || !inputText.trim() || !serverUp}
                  style={{
                    background: serverUp ? C.accent : C.border,
                    color: "#fff", border: "none", borderRadius: 7,
                    padding: "8px 18px", cursor: serverUp ? "pointer" : "not-allowed",
                    fontSize: 12, fontFamily: "inherit", fontWeight: 700,
                    display: "flex", alignItems: "center", gap: 6,
                  }}>
                  {predLoading ? <><Spinner /> Analysing…</> : "Analyse →"}
                </button>
                {EXAMPLES.slice(0, 3).map((ex, i) => (
                  <button key={i} onClick={() => { setInputText(ex); setPrediction(null); setPredError(null); }}
                    style={{
                      background: "transparent", color: C.muted,
                      border: `1px solid ${C.border}`, borderRadius: 7,
                      padding: "6px 10px", cursor: "pointer",
                      fontSize: 11, fontFamily: "inherit",
                    }}>
                    Example {i + 1}
                  </button>
                ))}
              </div>
            </div>

            {/* Error */}
            {predError && (
              <div style={{
                ...card, marginBottom: 12,
                background: C.hate + "15", border: `1px solid ${C.hate}55`,
              }} className="fade-in">
                <span style={{ color: C.hate }}>✗ Error: </span>
                <span style={{ color: C.subtext, fontSize: 12 }}>{predError}</span>
              </div>
            )}

            {/* Prediction result — real API response */}
            {prediction && (
              <div className="fade-in" style={{
                ...card,
                background: prediction.label === 1 ? C.hate + "15" : C.safe + "15",
                border: `1px solid ${prediction.label === 1 ? C.hate : C.safe}55`,
              }}>
                <div style={{ display: "flex", gap: 14, alignItems: "flex-start", marginBottom: 14 }}>
                  <div style={{ fontSize: 38 }}>{prediction.label === 1 ? "⚠️" : "✅"}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontSize: 20, fontWeight: 800,
                      color: prediction.label === 1 ? C.hate : C.safe,
                    }}>
                      {prediction.label_name.toUpperCase()}
                    </div>
                    <div style={{ color: C.subtext, fontSize: 11, marginTop: 3 }}>
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                      &nbsp;·&nbsp; Model: {prediction.model_used}
                      &nbsp;·&nbsp; Latency: {prediction.latency_ms}ms
                    </div>
                  </div>
                </div>

                {/* Probability bars */}
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
                  {[
                    ["Hate Speech",     prediction.prob_hate, C.hate],
                    ["Not Hate Speech", prediction.prob_safe, C.safe],
                  ].map(([label, val, color]) => (
                    <div key={label}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ fontSize: 11, color: C.muted }}>{label}</span>
                        <span style={{ fontSize: 11, color, fontWeight: 700 }}>{(val * 100).toFixed(1)}%</span>
                      </div>
                      <Bar value={val} color={color} height={10} />
                    </div>
                  ))}
                </div>

                {/* Metadata chips */}
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <Badge color={C.yellow}>Igbo ratio: {(prediction.igbo_ratio * 100).toFixed(0)}%</Badge>
                  <Badge color={prediction.is_code_mixed ? C.accent : C.muted}>
                    {prediction.is_code_mixed ? "Code-Mixed ✓" : "Not Code-Mixed"}
                  </Badge>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* BATCH TEST TAB                                                  */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        {tab === "Batch Test" && (
          <div className="fade-in">
            <p style={{ color: C.subtext, fontSize: 12, marginBottom: 12 }}>
              Paste multiple posts (one per line) and classify them all at once via the real API.
            </p>

            <div style={{ ...card, marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>📋 Posts (one per line, max 50)</div>
              <textarea
                value={batchInput}
                onChange={e => setBatchInput(e.target.value)}
                style={{
                  width: "100%", minHeight: 160, background: C.bg,
                  border: `1px solid ${C.border}`, borderRadius: 7,
                  color: C.text, fontFamily: "inherit", fontSize: 12,
                  padding: "10px 12px", resize: "vertical", boxSizing: "border-box",
                }}
              />
              <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center" }}>
                <button onClick={handleBatch}
                  disabled={batchLoading || !serverUp}
                  style={{
                    background: serverUp ? C.accent : C.border,
                    color: "#fff", border: "none", borderRadius: 7,
                    padding: "8px 18px", cursor: serverUp ? "pointer" : "not-allowed",
                    fontSize: 12, fontFamily: "inherit", fontWeight: 700,
                    display: "flex", alignItems: "center", gap: 6,
                  }}>
                  {batchLoading ? <><Spinner /> Running…</> : `Run Batch (${batchInput.split("\n").filter(t => t.trim()).length} posts)`}
                </button>
                <span style={{ fontSize: 11, color: C.muted }}>
                  Calls <code style={{ color: C.yellow }}>POST /predict/batch</code>
                </span>
              </div>
            </div>

            {/* Batch results */}
            {batchResult && (
              <div className="fade-in">
                {/* Summary */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 12 }}>
                  {[
                    { label: "Total",      value: batchResult.total,      color: C.accent },
                    { label: "Hate",       value: batchResult.hate_count, color: C.hate   },
                    { label: "Safe",       value: batchResult.safe_count, color: C.safe   },
                    { label: "Latency",    value: `${batchResult.latency_ms}ms`, color: C.yellow, raw: true },
                  ].map(m => (
                    <div key={m.label} style={{ ...card, textAlign: "center" }}>
                      <div style={{ fontSize: 20, fontWeight: 800, color: m.color }}>
                        {m.raw ? m.value : m.value}
                      </div>
                      <div style={{ fontSize: 11, color: C.muted }}>{m.label}</div>
                    </div>
                  ))}
                </div>

                {/* Hate ratio bar */}
                <div style={{ ...card, marginBottom: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 700 }}>Hate Speech Rate</span>
                    <span style={{ color: C.hate, fontWeight: 700, fontSize: 13 }}>
                      {batchResult.total > 0 ? ((batchResult.hate_count / batchResult.total) * 100).toFixed(1) : 0}%
                    </span>
                  </div>
                  <Bar
                    value={batchResult.total > 0 ? batchResult.hate_count / batchResult.total : 0}
                    color={C.hate} height={12}
                  />
                </div>

                {/* Individual results */}
                {batchResult.results.map((r, i) => (
                  <div key={i} style={{
                    ...card, marginBottom: 8,
                    borderLeft: `3px solid ${r.label === 1 ? C.hate : C.safe}`,
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
                      <div style={{ fontSize: 12, color: C.text, flex: 1, lineHeight: 1.5 }}>
                        {r.text.length > 90 ? r.text.slice(0, 90) + "…" : r.text}
                      </div>
                      <div style={{ display: "flex", gap: 6, flexShrink: 0, alignItems: "center" }}>
                        <Badge color={r.label === 1 ? C.hate : C.safe}>
                          {r.label === 1 ? "Hate" : "Safe"}
                        </Badge>
                        <span style={{ fontSize: 10, color: C.muted }}>
                          {(r.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}