import { useState, useRef, useEffect } from "react";

const SYSTEM_PROMPT = `You are a senior equity research analyst. Your job is to answer questions about SEC filings (10-K, 10-Q) with the precision and format expected by institutional investors.

RESPONSE FORMAT RULES:
1. Lead with the direct numerical answer (dollar figure, %, or ratio) in the first sentence.
2. Provide year-over-year or quarter-over-quarter comparison whenever the data is available.
3. Highlight key drivers of the change (volume, price, mix, geography, product segment).
4. Flag any risks, one-time items, or non-GAAP adjustments that affect comparability.
5. If the exact figure is not in the context, say so clearly and state what CAN be inferred.
6. Structure your answer with these labeled sections:
   ▸ KEY FIGURE — the headline number
   ▸ BREAKDOWN — segment / product / geo detail  
   ▸ DRIVERS — what caused the change
   ▸ COMPARABLES — prior period or peer context if available
   ▸ WATCH ITEMS — risks, non-recurring items, guidance

Use concise analyst prose. No filler. No unnecessary hedging. DO NOT fabricate numbers.`;

const EXAMPLE_QUERIES = [
  "What was total revenue in FY2024 and YoY growth?",
  "What is the gross margin trend over the last 4 quarters?",
  "What is free cash flow vs net income?",
  "What guidance did management provide for next fiscal year?",
  "What are the top risk factors impacting revenue?",
  "What drove operating income change YoY?",
];

function formatAnalystResponse(text) {
  // Highlight section headers
  return text
    .replace(/▸\s*(KEY FIGURE[^\n]*)/g, '<span class="section-header key">▸ $1</span>')
    .replace(/▸\s*(BREAKDOWN[^\n]*)/g, '<span class="section-header breakdown">▸ $1</span>')
    .replace(/▸\s*(DRIVERS[^\n]*)/g, '<span class="section-header drivers">▸ $1</span>')
    .replace(/▸\s*(COMPARABLES[^\n]*)/g, '<span class="section-header comp">▸ $1</span>')
    .replace(/▸\s*(WATCH ITEMS[^\n]*)/g, '<span class="section-header watch">▸ $1</span>')
    // Highlight dollar amounts
    .replace(/(\$[\d,\.]+\s*(?:billion|million|thousand|[BMK])?)/gi, '<span class="num-dollar">$1</span>')
    // Highlight percentages
    .replace(/([\+\-]?\d+\.?\d*%)/g, '<span class="num-pct">$1</span>')
    // Newlines to breaks
    .replace(/\n/g, '<br/>');
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ticker, setTicker] = useState("FRSH");
  const [filingType, setFilingType] = useState("10-K");
  const [year, setYear] = useState("2024");
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const buildContext = () =>
    `Analyzing ${ticker} ${filingType} for fiscal year ${year}. ` +
    `Provide analyst-grade answers using data from this filing.`;

  const sendQuery = async (query) => {
    if (!query.trim() || loading) return;

    const userMsg = { role: "user", text: query, ts: new Date() };
    setMessages((m) => [...m, userMsg]);
    setInput("");
    setLoading(true);

    const history = messages.map((m) => ({
      role: m.role === "user" ? "user" : "assistant",
      content: m.role === "user" ? m.text : m.rawText || m.text,
    }));

    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          system: SYSTEM_PROMPT + "\n\nFiling context: " + buildContext(),
          messages: [
            ...history,
            { role: "user", content: query },
          ],
        }),
      });
      const data = await res.json();
      const raw = data.content?.[0]?.text || "No response received.";
      setMessages((m) => [
        ...m,
        { role: "assistant", text: formatAnalystResponse(raw), rawText: raw, ts: new Date() },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `<span class="error">ERROR: ${e.message}</span>`, ts: new Date() },
      ]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendQuery(input);
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #0a0c0f;
          --surface: #0f1318;
          --border: #1e2430;
          --border-bright: #2a3545;
          --text: #c8d4e0;
          --text-dim: #5a6a80;
          --text-muted: #3a4a5a;
          --accent: #f5a623;
          --accent-dim: rgba(245,166,35,0.12);
          --green: #00d084;
          --green-dim: rgba(0,208,132,0.12);
          --red: #ff4757;
          --red-dim: rgba(255,71,87,0.12);
          --blue: #4dabf7;
          --blue-dim: rgba(77,171,247,0.1);
          --cyan: #63e6be;
          --mono: 'IBM Plex Mono', monospace;
          --sans: 'IBM Plex Sans', sans-serif;
        }

        html, body, #root {
          height: 100%;
          background: var(--bg);
          color: var(--text);
          font-family: var(--mono);
          font-size: 13px;
          line-height: 1.6;
        }

        .app {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 1100px;
          margin: 0 auto;
        }

        /* TOP BAR */
        .topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 10px 20px;
          border-bottom: 1px solid var(--border);
          background: var(--surface);
          flex-shrink: 0;
        }

        .brand {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .brand-tag {
          background: var(--accent);
          color: #0a0c0f;
          font-weight: 600;
          font-size: 10px;
          padding: 2px 7px;
          letter-spacing: 0.12em;
        }

        .brand-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--text);
          letter-spacing: 0.05em;
        }

        .controls {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .ctrl-label {
          color: var(--text-dim);
          font-size: 10px;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .ctrl-input {
          background: var(--bg);
          border: 1px solid var(--border-bright);
          color: var(--accent);
          font-family: var(--mono);
          font-size: 12px;
          font-weight: 600;
          padding: 4px 8px;
          width: 70px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          outline: none;
        }

        .ctrl-input:focus { border-color: var(--accent); }

        .ctrl-select {
          background: var(--bg);
          border: 1px solid var(--border-bright);
          color: var(--text);
          font-family: var(--mono);
          font-size: 12px;
          padding: 4px 8px;
          outline: none;
        }

        .status-dot {
          width: 6px; height: 6px;
          border-radius: 50%;
          background: var(--green);
          box-shadow: 0 0 6px var(--green);
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }

        /* EXAMPLE QUERIES */
        .examples-bar {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 8px 20px;
          border-bottom: 1px solid var(--border);
          background: var(--surface);
          overflow-x: auto;
          flex-shrink: 0;
          scrollbar-width: none;
        }

        .examples-bar::-webkit-scrollbar { display: none; }

        .ex-label {
          color: var(--text-muted);
          font-size: 10px;
          white-space: nowrap;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          flex-shrink: 0;
        }

        .ex-chip {
          background: none;
          border: 1px solid var(--border-bright);
          color: var(--text-dim);
          font-family: var(--mono);
          font-size: 10px;
          padding: 3px 10px;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.15s;
          flex-shrink: 0;
        }

        .ex-chip:hover {
          border-color: var(--accent);
          color: var(--accent);
          background: var(--accent-dim);
        }

        /* MESSAGES */
        .messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 16px;
          scrollbar-width: thin;
          scrollbar-color: var(--border-bright) transparent;
        }

        .msg {
          display: flex;
          gap: 14px;
          animation: fadeIn 0.2s ease;
        }

        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(4px); }
          to   { opacity: 1; transform: none; }
        }

        .msg-icon {
          flex-shrink: 0;
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.05em;
          margin-top: 2px;
        }

        .msg-icon.user {
          background: var(--accent-dim);
          color: var(--accent);
          border: 1px solid rgba(245,166,35,0.3);
        }

        .msg-icon.assistant {
          background: var(--green-dim);
          color: var(--green);
          border: 1px solid rgba(0,208,132,0.3);
        }

        .msg-body { flex: 1; min-width: 0; }

        .msg-header {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 6px;
        }

        .msg-who {
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .msg-who.user { color: var(--accent); }
        .msg-who.assistant { color: var(--green); }

        .msg-time {
          font-size: 10px;
          color: var(--text-muted);
        }

        .msg-text {
          color: var(--text);
          line-height: 1.75;
          font-size: 13px;
        }

        .msg-text.user {
          color: var(--text-dim);
          font-family: var(--sans);
        }

        /* Analyst response styling */
        .msg-text :global(.section-header) {
          display: block;
          margin: 12px 0 4px;
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .msg-text :global(.section-header.key)      { color: var(--accent); }
        .msg-text :global(.section-header.breakdown) { color: var(--blue); }
        .msg-text :global(.section-header.drivers)   { color: var(--cyan); }
        .msg-text :global(.section-header.comp)      { color: var(--text-dim); }
        .msg-text :global(.section-header.watch)     { color: var(--red); }

        .msg-text :global(.num-dollar) {
          color: var(--accent);
          font-weight: 600;
        }

        .msg-text :global(.num-pct) {
          color: var(--cyan);
          font-weight: 500;
        }

        .msg-text :global(.error) {
          color: var(--red);
        }

        /* RESPONSE BOX */
        .response-box {
          background: var(--surface);
          border: 1px solid var(--border);
          border-left: 2px solid var(--green);
          padding: 14px 16px;
        }

        /* LOADING */
        .loading-row {
          display: flex;
          align-items: center;
          gap: 14px;
          padding: 0 0 4px;
        }

        .loading-icon {
          width: 28px;
          height: 28px;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--green-dim);
          color: var(--green);
          border: 1px solid rgba(0,208,132,0.3);
          font-size: 10px;
          font-weight: 600;
        }

        .loading-text {
          font-size: 11px;
          color: var(--text-dim);
          letter-spacing: 0.08em;
        }

        .dots span {
          animation: blink 1.2s infinite;
          color: var(--green);
        }

        .dots span:nth-child(2) { animation-delay: 0.2s; }
        .dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes blink {
          0%, 80%, 100% { opacity: 0.2; }
          40% { opacity: 1; }
        }

        /* EMPTY STATE */
        .empty {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 12px;
          opacity: 0.4;
          pointer-events: none;
        }

        .empty-grid {
          display: grid;
          grid-template-columns: repeat(3, 1px);
          gap: 20px;
          margin-bottom: 16px;
        }

        .empty-dot {
          width: 3px;
          height: 3px;
          background: var(--text-dim);
          border-radius: 50%;
        }

        .empty-title {
          font-size: 12px;
          color: var(--text-dim);
          letter-spacing: 0.15em;
          text-transform: uppercase;
        }

        .empty-sub {
          font-size: 11px;
          color: var(--text-muted);
        }

        /* INPUT BAR */
        .input-bar {
          border-top: 1px solid var(--border);
          padding: 14px 20px;
          background: var(--surface);
          flex-shrink: 0;
        }

        .input-row {
          display: flex;
          gap: 10px;
          align-items: flex-end;
        }

        .input-prompt {
          color: var(--accent);
          font-weight: 600;
          font-size: 13px;
          padding-bottom: 9px;
          flex-shrink: 0;
        }

        .input-field {
          flex: 1;
          background: var(--bg);
          border: 1px solid var(--border-bright);
          color: var(--text);
          font-family: var(--mono);
          font-size: 13px;
          padding: 8px 12px;
          resize: none;
          min-height: 38px;
          max-height: 120px;
          outline: none;
          line-height: 1.5;
          transition: border-color 0.15s;
        }

        .input-field:focus { border-color: var(--accent); }
        .input-field::placeholder { color: var(--text-muted); }

        .send-btn {
          background: var(--accent);
          color: #0a0c0f;
          border: none;
          font-family: var(--mono);
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.1em;
          padding: 0 16px;
          height: 38px;
          cursor: pointer;
          transition: opacity 0.15s;
          text-transform: uppercase;
          flex-shrink: 0;
        }

        .send-btn:hover:not(:disabled) { opacity: 0.85; }
        .send-btn:disabled { opacity: 0.4; cursor: default; }

        .input-hint {
          font-size: 10px;
          color: var(--text-muted);
          margin-top: 6px;
          letter-spacing: 0.05em;
        }

        /* DIVIDER */
        .divider {
          display: flex;
          align-items: center;
          gap: 10px;
          font-size: 10px;
          color: var(--text-muted);
          letter-spacing: 0.1em;
          margin: 4px 0;
        }

        .divider::before, .divider::after {
          content: '';
          flex: 1;
          height: 1px;
          background: var(--border);
        }
      `}</style>

      <div className="app">
        {/* TOP BAR */}
        <div className="topbar">
          <div className="brand">
            <span className="brand-tag">LAD-RAG</span>
            <span className="brand-name">SEC Filing Analyst</span>
          </div>
          <div className="controls">
            <span className="ctrl-label">Ticker</span>
            <input
              className="ctrl-input"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              maxLength={5}
            />
            <select
              className="ctrl-select"
              value={filingType}
              onChange={(e) => setFilingType(e.target.value)}
            >
              <option>10-K</option>
              <option>10-Q</option>
              <option>8-K</option>
            </select>
            <input
              className="ctrl-input"
              value={year}
              onChange={(e) => setYear(e.target.value)}
              maxLength={4}
              style={{ width: 56 }}
            />
            <div className="status-dot" title="Model ready" />
          </div>
        </div>

        {/* EXAMPLE QUERIES */}
        <div className="examples-bar">
          <span className="ex-label">Quick&nbsp;queries</span>
          {EXAMPLE_QUERIES.map((q, i) => (
            <button key={i} className="ex-chip" onClick={() => sendQuery(q)}>
              {q}
            </button>
          ))}
        </div>

        {/* MESSAGES */}
        <div className="messages">
          {messages.length === 0 && !loading && (
            <div className="empty">
              <div className="empty-grid">
                {[...Array(9)].map((_, i) => <div key={i} className="empty-dot" />)}
              </div>
              <div className="empty-title">Awaiting analyst query</div>
              <div className="empty-sub">Configure ticker · filing type · year above</div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className="msg">
              <div className={`msg-icon ${m.role}`}>
                {m.role === "user" ? "YOU" : "AI"}
              </div>
              <div className="msg-body">
                <div className="msg-header">
                  <span className={`msg-who ${m.role}`}>
                    {m.role === "user" ? "Analyst Query" : `${ticker} Analysis`}
                  </span>
                  <span className="msg-time">
                    {m.ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                  </span>
                </div>
                {m.role === "assistant" ? (
                  <div className="response-box">
                    <div
                      className="msg-text"
                      dangerouslySetInnerHTML={{ __html: m.text }}
                    />
                  </div>
                ) : (
                  <div className="msg-text user">{m.text}</div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="loading-row">
              <div className="loading-icon">AI</div>
              <span className="loading-text">
                Processing filing data
                <span className="dots">
                  <span>.</span><span>.</span><span>.</span>
                </span>
              </span>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* INPUT BAR */}
        <div className="input-bar">
          <div className="input-row">
            <span className="input-prompt">&gt;</span>
            <textarea
              ref={inputRef}
              className="input-field"
              placeholder={`Query ${ticker} ${filingType} ${year} financial data...`}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
            />
            <button
              className="send-btn"
              onClick={() => sendQuery(input)}
              disabled={!input.trim() || loading}
            >
              {loading ? "..." : "EXECUTE"}
            </button>
          </div>
          <div className="input-hint">↵ Enter to send · Shift+↵ new line · Click quick queries above for examples</div>
        </div>
      </div>
    </>
  );
}
