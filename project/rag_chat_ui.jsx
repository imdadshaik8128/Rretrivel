import { useState, useRef, useEffect, useCallback } from "react";

// ── API config ────────────────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

async function callRAGAPI(query, subject, sessionId) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, subject, session_id: sessionId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Constants ─────────────────────────────────────────────────────────────────
const SUBJECTS = [
  "Biology","Economics","Geography","History",
  "Maths_sem_1","Maths_sem_2","Physics","Social_political",
];
const SUBJECT_LABELS = {
  Biology:"Biology", Economics:"Economics", Geography:"Geography",
  History:"History", Maths_sem_1:"Maths Sem 1", Maths_sem_2:"Maths Sem 2",
  Physics:"Physics", Social_political:"Social & Political",
};
const SUBJECT_ICONS = {
  Biology:"🧬", Economics:"📈", Geography:"🌍", History:"📜",
  Maths_sem_1:"∑", Maths_sem_2:"∫", Physics:"⚛", Social_political:"🏛",
};
const SUBJECT_COLORS = {
  Biology:"#4ade80", Economics:"#fb923c", Geography:"#38bdf8",
  History:"#f59e0b", Maths_sem_1:"#a78bfa", Maths_sem_2:"#c084fc",
  Physics:"#67e8f9", Social_political:"#f87171",
};

function genId() { return Math.random().toString(36).substr(2,9); }
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function newSession(subject) {
  return { id: genId(), subject, title: "New Chat", messages: [], messageCount: 0, createdAt: new Date() };
}

// ══════════════════════════════════════════════════════════════════════════════
export default function RAGChatUI() {
  const initial = newSession("Biology");
  const [sessions, setSessions] = useState([initial]);
  const [activeId, setActiveId] = useState(initial.id);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState("");
  const [collapsed, setCollapsed] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const endRef = useRef(null);
  const inputRef = useRef(null);

  const active = sessions.find(s => s.id === activeId);
  const color = SUBJECT_COLORS[active?.subject] || "#94a3b8";

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json()).then(d => setApiStatus(d.status === "ok" ? "ok" : "error"))
      .catch(() => setApiStatus("error"));
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages, loading]);

  const patchSession = useCallback((id, fn) => {
    setSessions(prev => prev.map(s => s.id === id ? fn(s) : s));
  }, []);

  const createNew = useCallback((subject) => {
    const s = newSession(subject || active?.subject || "Biology");
    setSessions(prev => [s, ...prev]);
    setActiveId(s.id);
    setInput("");
  }, [active]);

  const switchSubject = useCallback((subj) => {
    if (!active || active.subject === subj) return;
    if (active.messages.length > 0) {
      patchSession(activeId, s => ({
        ...s,
        messages: [...s.messages, {
          id: genId(), role: "system", timestamp: new Date(),
          content: `SUBJECT CHANGE DETECTED: A new session has been initialized to maintain memory integrity for the '${SUBJECT_LABELS[subj]}' discussion.`,
        }],
      }));
      setTimeout(() => {
        const ns = newSession(subj);
        ns.title = `${SUBJECT_LABELS[subj]} — Session`;
        setSessions(prev => [ns, ...prev]);
        setActiveId(ns.id);
      }, 500);
    } else {
      patchSession(activeId, s => ({ ...s, subject: subj, title: "New Chat" }));
    }
  }, [active, activeId, patchSession]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || !active || loading) return;
    const sid = activeId;
    const userMsg = { id: genId(), role: "user", content: text, timestamp: new Date() };
    patchSession(sid, s => ({
      ...s,
      messages: [...s.messages, userMsg],
      messageCount: s.messageCount + 1,
      title: s.messageCount === 0 ? text.slice(0,36) + (text.length > 36 ? "…" : "") : s.title,
    }));
    setInput("");
    setLoading(true);
    setStage("parsing");
    await delay(150);
    setStage("retrieving");
    try {
      const data = await callRAGAPI(text, active.subject, sid);
      setStage("generating");
      await delay(200);
      const aiMsg = {
        id: genId(), role: "assistant", timestamp: new Date(),
        answerType: data.answer_type,
        displayMarkdown: data.display_answer_markdown,
        spokenAnswer: data.spoken_answer,
        citations: data.citations,
        confidencePct: data.confidence_pct,
        lowConfidenceWarning: data.low_confidence_warning,
        filterPath: data.filter_path,
        parsedQuery: data.parsed_query,
        timings: { retrieval: data.retrieval_ms, generation: data.generation_ms, total: data.total_ms },
      };
      patchSession(sid, s => ({ ...s, messages: [...s.messages, aiMsg], messageCount: s.messageCount + 1 }));
    } catch (e) {
      patchSession(sid, s => ({
        ...s,
        messages: [...s.messages, { id: genId(), role: "error", content: e.message, timestamp: new Date() }],
      }));
    } finally { setLoading(false); setStage(""); }
  }, [input, active, activeId, loading, patchSession]);

  const onKey = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };
  const lbl = SUBJECT_LABELS[active?.subject] || active?.subject || "—";

  return (
    <div style={S.root}>
      {/* SIDEBAR */}
      <aside style={{ ...S.sidebar, width: collapsed ? 62 : 264 }}>
        <div style={S.sbHeader}>
          {!collapsed && <span style={S.brand}>TextbookAI</span>}
          <button style={S.iconBtn} onClick={() => setCollapsed(v => !v)} title="Toggle">
            {collapsed ? "›" : "‹"}
          </button>
        </div>

        <button
          style={{ ...S.newBtn, justifyContent: collapsed ? "center" : "flex-start" }}
          onClick={() => createNew()} title="New Chat"
        >
          <span style={{ fontSize: 18, color: "#60a5fa", lineHeight: 1 }}>＋</span>
          {!collapsed && <span style={{ marginLeft: 8, fontSize: 13 }}>New Chat</span>}
        </button>

        {!collapsed ? (
          <div style={S.subjSection}>
            <p style={S.sectionLbl}>SUBJECT</p>
            {SUBJECTS.map(subj => {
              const act = active?.subject === subj;
              const c = SUBJECT_COLORS[subj];
              return (
                <button key={subj} style={{ ...S.subjBtn, background: act ? `${c}18` : "transparent", borderLeft: `3px solid ${act ? c : "transparent"}`, color: act ? c : "#64748b" }} onClick={() => switchSubject(subj)}>
                  <span style={{ fontSize: 14 }}>{SUBJECT_ICONS[subj]}</span>
                  <span style={{ marginLeft: 8, fontSize: 12, fontWeight: act ? 700 : 400 }}>{SUBJECT_LABELS[subj]}</span>
                </button>
              );
            })}
          </div>
        ) : (
          <div style={{ padding: "6px 0", display: "flex", flexDirection: "column", alignItems: "center", gap: 3 }}>
            {SUBJECTS.map(subj => {
              const act = active?.subject === subj;
              const c = SUBJECT_COLORS[subj];
              return (
                <button key={subj} title={SUBJECT_LABELS[subj]} style={{ width: 38, height: 38, borderRadius: 9, border: `2px solid ${act ? c : "transparent"}`, background: act ? `${c}22` : "transparent", cursor: "pointer", fontSize: 16, display: "flex", alignItems: "center", justifyContent: "center" }} onClick={() => switchSubject(subj)}>
                  {SUBJECT_ICONS[subj]}
                </button>
              );
            })}
          </div>
        )}

        {!collapsed && (
          <div style={S.histSection}>
            <p style={S.sectionLbl}>RECENT</p>
            <div style={{ overflowY: "auto", flex: 1, display: "flex", flexDirection: "column", gap: 1 }}>
              {sessions.map(s => {
                const isAct = s.id === activeId;
                const c = SUBJECT_COLORS[s.subject] || "#94a3b8";
                return (
                  <button key={s.id} style={{ ...S.sessBtn, background: isAct ? "#0d2137" : "transparent", borderLeft: `3px solid ${isAct ? c : "transparent"}` }} onClick={() => { setActiveId(s.id); setInput(""); }}>
                    <span style={{ fontSize: 11, flexShrink: 0 }}>{SUBJECT_ICONS[s.subject]}</span>
                    <div style={{ flex: 1, minWidth: 0, marginLeft: 7 }}>
                      <p style={S.sessTitle}>{s.title}</p>
                      <p style={S.sessMeta}>{s.messageCount > 0 ? `${s.messageCount} msgs` : "Empty"} · {SUBJECT_LABELS[s.subject]}</p>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div style={S.statusRow}>
          <div style={{ ...S.dot, background: apiStatus === "ok" ? "#4ade80" : apiStatus === "error" ? "#f87171" : "#fbbf24", boxShadow: `0 0 5px ${apiStatus === "ok" ? "#4ade80" : apiStatus === "error" ? "#f87171" : "#fbbf24"}` }} />
          {!collapsed && <span style={{ fontSize: 10, color: "#1e3a5f" }}>{apiStatus === "ok" ? "API Connected" : apiStatus === "error" ? "API Offline" : "Checking…"}</span>}
        </div>
      </aside>

      {/* MAIN */}
      <main style={S.main}>
        {/* Topbar */}
        <header style={S.topbar}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ ...S.pill, background: `${color}1a`, border: `1px solid ${color}55`, color }}>
              <span style={{ marginRight: 5 }}>{SUBJECT_ICONS[active?.subject]}</span>{lbl}
            </div>
            <span style={{ fontSize: 12, color: "#334155" }}>{active?.title}</span>
          </div>
          <div style={S.memBadge}>
            <div style={{ ...S.dot, background: "#4ade80", boxShadow: "0 0 5px #4ade80" }} />
            {active?.messageCount || 0} messages in memory
          </div>
        </header>

        {/* Messages */}
        <div style={S.msgArea}>
          {!active || active.messages.length === 0 ? (
            <EmptyState subject={active?.subject} color={color} label={lbl} apiStatus={apiStatus}
              onSuggest={s => { setInput(s); inputRef.current?.focus(); }} />
          ) : (
            <div style={S.msgList}>
              {active.messages.map(msg => <Bubble key={msg.id} msg={msg} color={color} />)}
              {loading && <LoadingBubble stage={stage} color={color} />}
              <div ref={endRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div style={S.inputArea}>
          {apiStatus === "error" && (
            <div style={S.apiBanner}>
              ⚠ Backend offline — run: <code style={{ background: "#0a1628", padding: "1px 5px", borderRadius: 3 }}>uvicorn api:app --reload</code>
            </div>
          )}
          <div style={{ ...S.inputWrap, borderColor: input ? `${color}88` : "#1e3a5f", boxShadow: input ? `0 0 0 3px ${color}12` : "none" }}>
            <textarea ref={inputRef} value={input} onChange={e => setInput(e.target.value)} onKeyDown={onKey}
              placeholder={`Ask about ${lbl}…`} style={S.textarea} rows={1}
              disabled={loading || apiStatus === "error"} />
            <button
              style={{ ...S.sendBtn, background: input.trim() && !loading ? color : "#0d2137", color: input.trim() && !loading ? "#0a1628" : "#334155", cursor: input.trim() && !loading ? "pointer" : "not-allowed" }}
              onClick={send} disabled={!input.trim() || loading || apiStatus === "error"}
            >
              {loading ? <Spin /> : "↑"}
            </button>
          </div>
          <p style={S.hint}>Enter to send · Shift+Enter for newline · Switching subject mid-conversation starts a new memory session</p>
        </div>
      </main>
    </div>
  );
}

// ── Message Bubble ─────────────────────────────────────────────────────────────
function Bubble({ msg, color }) {
  const [dbg, setDbg] = useState(false);
  const fmt = d => new Date(d).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  if (msg.role === "system") return (
    <div style={S.sysNotice}>
      <span style={{ flexShrink: 0 }}>⚡</span>
      <p style={{ margin: 0, fontSize: 12 }}>{msg.content}</p>
    </div>
  );

  if (msg.role === "error") return (
    <div style={{ ...S.sysNotice, borderColor: "#f8717155", color: "#f87171" }}>
      <span style={{ flexShrink: 0 }}>✗</span>
      <p style={{ margin: 0, fontSize: 12 }}>{msg.content}</p>
    </div>
  );

  const isUser = msg.role === "user";
  return (
    <div style={{ alignSelf: isUser ? "flex-end" : "flex-start", maxWidth: isUser ? "66%" : "88%", display: "flex", flexDirection: "column", gap: 4 }}>
      {!isUser && (
        <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 1 }}>
          <div style={{ width: 24, height: 24, borderRadius: 6, background: `${color}1a`, border: `1px solid ${color}44`, color, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 800, flexShrink: 0 }}>AI</div>
          {msg.answerType && (
            <span style={{ fontSize: 9, fontWeight: 800, padding: "2px 7px", borderRadius: 10, letterSpacing: "0.4px", background: msg.answerType === "reference" ? "#4ade8018" : "#60a5fa18", color: msg.answerType === "reference" ? "#4ade80" : "#60a5fa", border: `1px solid ${msg.answerType === "reference" ? "#4ade8044" : "#60a5fa44"}` }}>
              {msg.answerType === "reference" ? "● REFERENCE" : "● CONCEPT"}
            </span>
          )}
          {msg.confidencePct !== undefined && <ConfBar pct={msg.confidencePct} />}
        </div>
      )}

      {msg.lowConfidenceWarning && (
        <div style={{ padding: "5px 10px", background: "#fbbf2410", border: "1px solid #fbbf2433", borderRadius: 7, color: "#fbbf24", fontSize: 11 }}>
          ⚠ {msg.lowConfidenceWarning}
        </div>
      )}

      <div style={{ padding: "11px 15px", background: isUser ? `${color}18` : "#0d1f38", border: `1px solid ${isUser ? `${color}44` : "#1e3a5f"}`, borderRadius: isUser ? "16px 16px 4px 16px" : "4px 16px 16px 16px", lineHeight: 1.65 }}>
        {isUser
          ? <span style={{ fontSize: 14, color: "#e2e8f0" }}>{msg.content}</span>
          : <MD content={msg.displayMarkdown || ""} />}
      </div>

      {!isUser && msg.citations?.length > 0 && (
        <div style={{ padding: "7px 11px", background: "#06111f", border: "1px solid #0d2137", borderRadius: 8 }}>
          <p style={{ margin: "0 0 5px", fontSize: 10, fontWeight: 700, color: "#334155", letterSpacing: "0.5px" }}>📚 SOURCES</p>
          {msg.citations.map((c, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 3 }}>
              <span style={{ fontSize: 9, fontWeight: 700, padding: "1px 5px", borderRadius: 6, textTransform: "uppercase", background: c.chunk_type === "activity" ? "#4ade8018" : c.chunk_type === "exercise" ? "#60a5fa18" : "#94a3b818", color: c.chunk_type === "activity" ? "#4ade80" : c.chunk_type === "exercise" ? "#60a5fa" : "#94a3b8" }}>
                {c.chunk_type}
              </span>
              <span style={{ fontSize: 11, color: "#64748b" }}>
                Ch.{c.chapter_number}{c.chapter_title ? ` — ${c.chapter_title}` : ""}
                {c.activity_number && c.activity_number !== "None" && c.activity_number.trim() ? ` · Activity ${c.activity_number}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      {!isUser && msg.filterPath && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ fontSize: 10, color: "#1e3a5f", fontFamily: "monospace" }}>🔎 {msg.filterPath}</span>
          {msg.timings && <span style={{ fontSize: 9, color: "#1e3a5f", fontFamily: "monospace" }}>ret:{msg.timings.retrieval}ms gen:{msg.timings.generation}ms</span>}
          {msg.parsedQuery && <button style={{ background: "none", border: "none", color: "#334155", fontSize: 9, cursor: "pointer", textDecoration: "underline", padding: 0 }} onClick={() => setDbg(v => !v)}>{dbg ? "hide debug" : "debug ▾"}</button>}
        </div>
      )}

      {dbg && msg.parsedQuery && (
        <pre style={{ background: "#06111f", border: "1px solid #0d2137", borderRadius: 8, padding: "8px 12px", fontSize: 10, color: "#475569", overflow: "auto", fontFamily: "monospace", margin: 0, maxHeight: 140 }}>
          {JSON.stringify(msg.parsedQuery, null, 2)}
        </pre>
      )}

      <span style={{ fontSize: 9, color: "#1e3a5f", alignSelf: "flex-end" }}>{fmt(msg.timestamp)}</span>
    </div>
  );
}

function ConfBar({ pct }) {
  const c = pct >= 75 ? "#4ade80" : pct >= 50 ? "#fbbf24" : "#f87171";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      <div style={{ width: 52, height: 4, background: "#0d2137", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: c, borderRadius: 3, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 9, color: c, fontWeight: 700 }}>{pct}%</span>
    </div>
  );
}

function LoadingBubble({ stage, color }) {
  const labels = { parsing: "Parsing query…", retrieving: "Retrieving chunks…", generating: "Generating answer…" };
  return (
    <div style={{ alignSelf: "flex-start" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "11px 15px", background: "#0d1f38", border: "1px solid #1e3a5f", borderRadius: "4px 16px 16px 16px" }}>
        <Dots color={color} />
        <span style={{ fontSize: 12, color: "#475569" }}>{labels[stage] || "Processing…"}</span>
      </div>
    </div>
  );
}

function Dots({ color }) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {[0,1,2].map(i => (
        <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: color, animation: `b 1.2s ease-in-out ${i*0.2}s infinite` }} />
      ))}
      <style>{`@keyframes b{0%,60%,100%{transform:translateY(0);opacity:.35}30%{transform:translateY(-5px);opacity:1}}`}</style>
    </div>
  );
}

function Spin() {
  return <>
    <div style={{ width: 13, height: 13, border: "2px solid #1e3a5f", borderTop: "2px solid #94a3b8", borderRadius: "50%", animation: "sp 0.7s linear infinite" }} />
    <style>{`@keyframes sp{to{transform:rotate(360deg)}}`}</style>
  </>;
}

function EmptyState({ subject, color, label, onSuggest, apiStatus }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", textAlign: "center", padding: "40px 24px" }}>
      <div style={{ fontSize: 44, width: 90, height: 90, borderRadius: 22, border: `2px solid ${color}44`, boxShadow: `0 0 36px ${color}18`, display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 18 }}>
        {SUBJECT_ICONS[subject] || "📚"}
      </div>
      <h2 style={{ fontSize: 20, fontWeight: 700, margin: "0 0 8px", color: "#f1f5f9" }}>
        Ask anything about <span style={{ color }}>{label}</span>
      </h2>
      <p style={{ fontSize: 13, color: "#334155", maxWidth: 360, margin: "0 0 24px", lineHeight: 1.6 }}>
        The RAG pipeline deterministically retrieves textbook chunks and generates a structured answer using Ollama.
      </p>
      {apiStatus === "error" && (
        <div style={{ padding: "8px 14px", background: "#f59e0b12", border: "1px solid #f59e0b44", borderRadius: 8, color: "#fbbf24", fontSize: 11, marginBottom: 16 }}>
          ⚠ Backend offline — run: <code style={{ background: "#06111f", padding: "1px 5px", borderRadius: 3 }}>uvicorn api:app --reload</code>
        </div>
      )}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxWidth: 480, width: "100%" }}>
        {["Explain Activity 2 from Chapter 1","What is photosynthesis?","Solve Exercise 3.1 Chapter 4","How does osmosis work?"].map(s => (
          <button key={s} style={{ padding: "10px 13px", background: "#0d2137", border: "1px solid #1e3a5f", borderRadius: 9, color: "#475569", cursor: "pointer", fontSize: 12, textAlign: "left", lineHeight: 1.4 }} onClick={() => onSuggest(s)}>{s}</button>
        ))}
      </div>
    </div>
  );
}

// ── Minimal markdown renderer ─────────────────────────────────────────────────
function MD({ content }) {
  return (
    <div style={{ fontSize: 14, color: "#e2e8f0", lineHeight: 1.7 }}>
      {content.split("\n").map((line, i) => {
        if (line.startsWith("### ")) return <p key={i} style={{ margin: "7px 0 3px", fontWeight: 700, fontSize: 13, color: "#f1f5f9" }}>{line.slice(4)}</p>;
        if (line.startsWith("## "))  return <p key={i} style={{ margin: "9px 0 3px", fontWeight: 700, fontSize: 14, color: "#f8fafc" }}>{line.slice(3)}</p>;
        if (line.startsWith("# "))   return <p key={i} style={{ margin: "9px 0 3px", fontWeight: 800, fontSize: 15, color: "#f8fafc" }}>{line.slice(2)}</p>;
        if (line.startsWith("- "))   return <p key={i} style={{ margin: "2px 0 2px 12px", color: "#cbd5e1" }}>• {bold(line.slice(2))}</p>;
        if (line.startsWith("*") && line.endsWith("*") && !line.startsWith("**")) return <p key={i} style={{ margin: "5px 0", fontStyle: "italic", color: "#475569", fontSize: 12 }}>{line.slice(1,-1)}</p>;
        if (line === "") return <br key={i} />;
        return <p key={i} style={{ margin: "2px 0" }}>{bold(line)}</p>;
      })}
    </div>
  );
}
function bold(t) {
  return t.split(/\*\*(.*?)\*\*/g).map((p,i) => i%2===1 ? <strong key={i} style={{ color:"#f1f5f9" }}>{p}</strong> : p);
}

// ── Styles ─────────────────────────────────────────────────────────────────────
const S = {
  root: { display:"flex", height:"100vh", background:"#060d1a", fontFamily:"'DM Sans','Segoe UI',system-ui,sans-serif", color:"#e2e8f0", overflow:"hidden" },
  sidebar: { background:"#04101e", borderRight:"1px solid #0a1f35", display:"flex", flexDirection:"column", overflow:"hidden", flexShrink:0, transition:"width 0.22s cubic-bezier(0.4,0,0.2,1)" },
  sbHeader: { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"13px 11px", borderBottom:"1px solid #0a1f35", minHeight:50, flexShrink:0 },
  brand: { fontWeight:800, fontSize:14, background:"linear-gradient(135deg,#38bdf8,#818cf8)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent", letterSpacing:"-0.5px", whiteSpace:"nowrap" },
  iconBtn: { background:"transparent", border:"none", color:"#334155", cursor:"pointer", fontSize:17, width:28, height:28, borderRadius:6, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 },
  newBtn: { display:"flex", alignItems:"center", margin:"9px 8px 5px", padding:"8px 11px", background:"#0a1f35", border:"1px solid #1e3a5f", borderRadius:9, color:"#94a3b8", cursor:"pointer", fontSize:13, fontWeight:600, flexShrink:0, gap:0 },
  subjSection: { padding:"3px 8px 5px", overflowY:"auto", flexShrink:0 },
  sectionLbl: { fontSize:9, fontWeight:700, letterSpacing:"1.4px", color:"#0a1f35", margin:"7px 3px 4px" },
  subjBtn: { display:"flex", alignItems:"center", width:"100%", padding:"6px 9px", border:"none", borderRadius:"0 7px 7px 0", cursor:"pointer", textAlign:"left", transition:"all 0.12s", whiteSpace:"nowrap" },
  histSection: { flex:1, overflow:"hidden", display:"flex", flexDirection:"column", padding:"3px 8px 5px", borderTop:"1px solid #0a1f35", marginTop:3 },
  sessBtn: { display:"flex", alignItems:"center", width:"100%", padding:"7px 9px", border:"none", borderRadius:"0 7px 7px 0", cursor:"pointer", transition:"all 0.12s", textAlign:"left" },
  sessTitle: { margin:0, fontSize:11, fontWeight:500, color:"#64748b", whiteSpace:"nowrap", overflow:"hidden", textOverflow:"ellipsis", maxWidth:170 },
  sessMeta: { margin:0, fontSize:9, color:"#0a1f35" },
  statusRow: { display:"flex", alignItems:"center", gap:6, padding:"9px 13px", borderTop:"1px solid #0a1f35", flexShrink:0 },
  dot: { width:6, height:6, borderRadius:"50%", flexShrink:0 },
  main: { flex:1, display:"flex", flexDirection:"column", overflow:"hidden" },
  topbar: { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"0 20px", height:50, borderBottom:"1px solid #0a1f35", background:"#04101e", flexShrink:0 },
  pill: { display:"flex", alignItems:"center", padding:"3px 10px", borderRadius:18, fontSize:12, fontWeight:600 },
  memBadge: { display:"flex", alignItems:"center", gap:5, fontSize:10, color:"#334155", padding:"3px 9px", background:"#0a1f35", borderRadius:5, border:"1px solid #1e3a5f" },
  msgArea: { flex:1, overflowY:"auto", padding:"22px 26px", scrollbarWidth:"thin", scrollbarColor:"#0a1f35 transparent" },
  msgList: { display:"flex", flexDirection:"column", gap:16, maxWidth:860, margin:"0 auto" },
  sysNotice: { display:"flex", alignItems:"flex-start", gap:9, padding:"9px 13px", background:"#0a1f35", border:"1px solid #f59e0b44", borderRadius:9, color:"#fbbf24", fontSize:12 },
  inputArea: { padding:"13px 20px 16px", borderTop:"1px solid #0a1f35", flexShrink:0 },
  apiBanner: { padding:"7px 12px", background:"#f59e0b0e", border:"1px solid #f59e0b44", borderRadius:7, color:"#fbbf24", fontSize:11, marginBottom:9, maxWidth:860, margin:"0 auto 9px" },
  inputWrap: { display:"flex", alignItems:"flex-end", gap:9, background:"#0a1f35", border:"1px solid #1e3a5f", borderRadius:13, padding:"9px 9px 9px 14px", transition:"all 0.18s", maxWidth:860, margin:"0 auto" },
  textarea: { flex:1, background:"transparent", border:"none", outline:"none", color:"#e2e8f0", fontSize:13, resize:"none", fontFamily:"'DM Sans',sans-serif", lineHeight:1.5, minHeight:20, maxHeight:110, overflowY:"auto" },
  sendBtn: { width:32, height:32, borderRadius:8, border:"none", fontSize:16, fontWeight:700, display:"flex", alignItems:"center", justifyContent:"center", transition:"all 0.15s", flexShrink:0 },
  hint: { textAlign:"center", fontSize:9, color:"#0a1f35", margin:"6px auto 0", maxWidth:860 },
};
