import { useState, useRef, useEffect } from "react";

// ─────────────────────────────────────────────────
// CONFIG — paste your Groq key here
// ─────────────────────────────────────────────────
const GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE";
const GROQ_MODEL   = "llama-3.3-70b-versatile";

// ─────────────────────────────────────────────────
// FINANCE DICTIONARY (mirrors navigator.py)
// ─────────────────────────────────────────────────
const FINANCE_DICT = {
  revenue:  ["net revenue","total revenue","sales","top line"],
  profit:   ["net income","net earnings","bottom line"],
  margin:   ["gross margin","operating margin","net margin","ebitda margin"],
  growth:   ["year over year","yoy","qoq","increase","decrease"],
  cash:     ["free cash flow","fcf","operating cash flow"],
  guidance: ["outlook","forecast","full year","next quarter"],
  earnings: ["eps","earnings per share","diluted eps"],
};

function expandQuery(query) {
  const q = query.toLowerCase();
  const expansions = [];
  for (const [term, syns] of Object.entries(FINANCE_DICT)) {
    if (q.includes(term)) expansions.push(...syns);
    if (syns.some(s => q.includes(s))) expansions.push(term);
  }
  return expansions.length ? query + " " + [...new Set(expansions)].join(" ") : query;
}

// ─────────────────────────────────────────────────
// SYSTEM PROMPT
// ─────────────────────────────────────────────────
const SYSTEM_PROMPT = `You are a senior equity research analyst. Answer questions about SEC filings with institutional-grade precision.

RESPONSE STRUCTURE — always use these exact section labels:
▸ KEY FIGURE    — lead with the headline dollar/% number in sentence 1
▸ BREAKDOWN     — by segment, product, geography if data is available
▸ DRIVERS       — causes of change (volume, price, mix, FX, acquisitions)
▸ COMPARABLES   — prior period for YoY or QoQ context
▸ WATCH ITEMS   — risks, non-recurring items, non-GAAP flags, guidance

CALCULATOR: For any math, show it inline like: (593.4 / 521.8 - 1) × 100 = 13.7% growth

CITATION RULE: Reference data as "See Table 2 on Page 45" when citing tables.

STRICT RULES:
• Always lead with exact numbers — never start with "Based on..."
• Never fabricate figures. If not in context: "Not disclosed in retrieved pages."
• Show YoY/QoQ comparison whenever prior period is mentioned.
• Flag non-GAAP items explicitly.
• Concise analyst prose — no filler.`;

// ─────────────────────────────────────────────────
// FORMAT RESPONSE — highlight financial markup
// ─────────────────────────────────────────────────
function formatResponse(text) {
  return text
    .replace(/▸\s*(KEY FIGURE[^\n]*)/g,  '<span class="sh key">▸ $1</span>')
    .replace(/▸\s*(BREAKDOWN[^\n]*)/g,   '<span class="sh bd">▸ $1</span>')
    .replace(/▸\s*(DRIVERS[^\n]*)/g,     '<span class="sh dr">▸ $1</span>')
    .replace(/▸\s*(COMPARABLES[^\n]*)/g, '<span class="sh cp">▸ $1</span>')
    .replace(/▸\s*(WATCH ITEMS[^\n]*)/g, '<span class="sh wt">▸ $1</span>')
    .replace(/(\$[\d,\.]+\s*(?:billion|million|[BMK])?)/gi, '<span class="n-dollar">$1</span>')
    .replace(/([\+\-]?\d+\.?\d*\s?%)/g,  '<span class="n-pct">$1</span>')
    .replace(/(See Table \d+ on Page \d+)/gi, '<span class="n-cite">$1</span>')
    .replace(/\n/g, "<br/>");
}

const EXAMPLES = [
  "What was total revenue FY2024 and YoY growth?",
  "What is the gross margin trend?",
  "Free cash flow vs net income — any divergence?",
  "What guidance did management give for next year?",
  "Top 3 risk factors impacting revenue?",
  "Break down operating expenses YoY.",
  "What drove operating income change?",
  "Diluted EPS vs prior year?",
];

// ─────────────────────────────────────────────────
// APP
// ─────────────────────────────────────────────────
export default function App() {
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [loading,     setLoading]     = useState(false);
  const [ticker,      setTicker]      = useState("FRSH");
  const [filingType,  setFilingType]  = useState("10-K");
  const [year,        setYear]        = useState("2024");
  const [queryCount,  setQueryCount]  = useState(0);
  const chatEndRef = useRef(null);
  const inputRef   = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  const buildContext = () =>
    `Analyzing ${ticker} ${filingType} for fiscal year ${year}.`;

  const sendQuery = async (rawQuery) => {
    if (!rawQuery.trim() || loading) return;

    const expanded  = expandQuery(rawQuery);
    const userMsg   = { role: "user", text: rawQuery, ts: new Date() };
    setMessages(m => [...m, userMsg]);
    setInput("");
    setLoading(true);

    const history = messages.map(m => ({
      role:    m.role === "user" ? "user" : "assistant",
      content: m.rawText || m.text,
    }));

    try {
      const res = await fetch("https://api.groq.com/openai/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type":  "application/json",
          "Authorization": `Bearer ${GROQ_API_KEY}`,
        },
        body: JSON.stringify({
          model:      GROQ_MODEL,
          max_tokens: 1200,
          messages: [
            { role: "system", content: SYSTEM_PROMPT + "\n\nFiling context: " + buildContext() },
            ...history,
            { role: "user", content: expanded },
          ],
        }),
      });

      const data   = await res.json();
      const raw    = data.choices?.[0]?.message?.content || data.error?.message || "No response.";
      setQueryCount(c => c + 1);
      setMessages(m => [...m, {
        role: "assistant", text: formatResponse(raw), rawText: raw, ts: new Date(),
        expanded: expanded !== rawQuery ? expanded : null,
      }]);
    } catch (e) {
      setMessages(m => [...m, {
        role: "assistant",
        text: `<span style="color:#ff4757">ERROR: ${e.message}</span>`,
        ts: new Date(),
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(input); }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
        *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
        :root{
          --bg:#080b0f;--surface:#0d1117;--surface2:#111820;
          --border:#1a2332;--border2:#243040;
          --text:#b8ccd8;--dim:#4a6070;--muted:#2a3a48;
          --amber:#f5a623;--amber-d:rgba(245,166,35,.1);
          --green:#00d47e;--green-d:rgba(0,212,126,.08);
          --red:#ff4757;--blue:#4da6ff;--cyan:#5ce6c8;--purple:#a78bfa;
          --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
        }
        html,body,#root{height:100%;background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;line-height:1.6}

        /* LAYOUT */
        .wrap{display:flex;flex-direction:column;height:100vh;max-width:1080px;margin:0 auto}

        /* TOPBAR */
        .top{display:flex;align-items:center;justify-content:space-between;padding:10px 18px;
          border-bottom:1px solid var(--border);background:var(--surface);flex-shrink:0}
        .brand{display:flex;align-items:center;gap:10px}
        .badge{background:var(--amber);color:#080b0f;font-weight:700;font-size:9px;
          padding:2px 7px;letter-spacing:.15em;text-transform:uppercase}
        .brand-name{font-size:13px;font-weight:500;letter-spacing:.04em}
        .brand-sub{font-size:10px;color:var(--dim);letter-spacing:.08em}
        .ctrl-row{display:flex;align-items:center;gap:8px}
        .cl{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
        .ci{background:var(--bg);border:1px solid var(--border2);color:var(--amber);
          font-family:var(--mono);font-size:12px;font-weight:600;padding:3px 8px;
          width:68px;text-transform:uppercase;letter-spacing:.08em;outline:none}
        .ci:focus{border-color:var(--amber)}
        .cs{background:var(--bg);border:1px solid var(--border2);color:var(--text);
          font-family:var(--mono);font-size:12px;padding:3px 8px;outline:none}
        .stat{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim)}
        .dot{width:6px;height:6px;border-radius:50%;background:var(--green);
          box-shadow:0 0 6px var(--green);animation:pulse 2.5s infinite}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

        /* METRICS BAR */
        .metrics{display:flex;gap:0;border-bottom:1px solid var(--border);
          background:var(--surface);flex-shrink:0}
        .metric{flex:1;padding:8px 16px;border-right:1px solid var(--border);text-align:center}
        .metric:last-child{border-right:none}
        .mv{font-size:16px;font-weight:600;color:var(--amber);letter-spacing:-.01em}
        .ml{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:1px}

        /* EXAMPLES */
        .exbar{display:flex;align-items:center;gap:6px;padding:7px 18px;
          border-bottom:1px solid var(--border);background:var(--surface);
          overflow-x:auto;flex-shrink:0;scrollbar-width:none}
        .exbar::-webkit-scrollbar{display:none}
        .exlbl{font-size:9px;color:var(--muted);white-space:nowrap;letter-spacing:.1em;text-transform:uppercase;flex-shrink:0}
        .exchip{background:none;border:1px solid var(--border2);color:var(--dim);
          font-family:var(--mono);font-size:10px;padding:3px 10px;cursor:pointer;
          white-space:nowrap;transition:all .15s;flex-shrink:0}
        .exchip:hover{border-color:var(--amber);color:var(--amber);background:var(--amber-d)}

        /* MESSAGES */
        .msgs{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;
          gap:14px;scrollbar-width:thin;scrollbar-color:var(--border2) transparent}
        .msg{display:flex;gap:12px;animation:fadein .2s ease}
        @keyframes fadein{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
        .ico{flex-shrink:0;width:26px;height:26px;display:flex;align-items:center;
          justify-content:center;font-size:9px;font-weight:700;letter-spacing:.05em;margin-top:2px}
        .ico.user{background:var(--amber-d);color:var(--amber);border:1px solid rgba(245,166,35,.25)}
        .ico.ai{background:var(--green-d);color:var(--green);border:1px solid rgba(0,212,126,.2)}
        .mbody{flex:1;min-width:0}
        .mhdr{display:flex;align-items:center;gap:8px;margin-bottom:5px}
        .mwho{font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
        .mwho.user{color:var(--amber)} .mwho.ai{color:var(--green)}
        .mts{font-size:9px;color:var(--muted)}
        .mexp{font-size:9px;color:var(--dim);margin-bottom:4px;font-style:italic}
        .mtext{color:var(--text);line-height:1.8;font-size:13px}
        .mtext.user{color:var(--dim);font-family:var(--sans)}

        /* RESPONSE BOX */
        .rbox{background:var(--surface2);border:1px solid var(--border);
          border-left:2px solid var(--green);padding:14px 16px}

        /* SECTION HEADERS */
        .sh{display:block;margin:10px 0 3px;font-size:9px;font-weight:700;
          letter-spacing:.14em;text-transform:uppercase}
        .sh.key{color:var(--amber)} .sh.bd{color:var(--blue)}
        .sh.dr{color:var(--cyan)}   .sh.cp{color:var(--dim)}
        .sh.wt{color:var(--red)}

        /* NUMBERS */
        .n-dollar{color:var(--amber);font-weight:600}
        .n-pct{color:var(--cyan);font-weight:500}
        .n-cite{color:var(--blue);font-style:italic;text-decoration:underline;text-underline-offset:2px}

        /* LOADING */
        .loading{display:flex;align-items:center;gap:12px}
        .lico{width:26px;height:26px;display:flex;align-items:center;justify-content:center;
          background:var(--green-d);color:var(--green);border:1px solid rgba(0,212,126,.2);font-size:9px;font-weight:700}
        .ltxt{font-size:11px;color:var(--dim);letter-spacing:.06em}
        .dots span{animation:blink 1.2s infinite;color:var(--green)}
        .dots span:nth-child(2){animation-delay:.2s}
        .dots span:nth-child(3){animation-delay:.4s}
        @keyframes blink{0%,80%,100%{opacity:.15}40%{opacity:1}}

        /* EMPTY */
        .empty{flex:1;display:flex;flex-direction:column;align-items:center;
          justify-content:center;gap:10px;opacity:.3;pointer-events:none}
        .egrid{display:grid;grid-template-columns:repeat(5,3px);gap:14px;margin-bottom:12px}
        .edot{width:3px;height:3px;background:var(--dim);border-radius:50%}
        .etitle{font-size:11px;color:var(--dim);letter-spacing:.15em;text-transform:uppercase}
        .esub{font-size:10px;color:var(--muted)}

        /* INPUT */
        .inputbar{border-top:1px solid var(--border);padding:12px 18px;
          background:var(--surface);flex-shrink:0}
        .irow{display:flex;gap:8px;align-items:flex-end}
        .iprompt{color:var(--amber);font-weight:700;font-size:14px;padding-bottom:8px;flex-shrink:0}
        .ifield{flex:1;background:var(--bg);border:1px solid var(--border2);color:var(--text);
          font-family:var(--mono);font-size:13px;padding:7px 12px;resize:none;
          min-height:36px;max-height:110px;outline:none;line-height:1.5;transition:border-color .15s}
        .ifield:focus{border-color:var(--amber)}
        .ifield::placeholder{color:var(--muted)}
        .sbtn{background:var(--amber);color:#080b0f;border:none;font-family:var(--mono);
          font-size:10px;font-weight:700;letter-spacing:.1em;padding:0 16px;height:36px;
          cursor:pointer;transition:opacity .15s;text-transform:uppercase;flex-shrink:0}
        .sbtn:hover:not(:disabled){opacity:.8}
        .sbtn:disabled{opacity:.35;cursor:default}
        .ihint{font-size:9px;color:var(--muted);margin-top:5px;letter-spacing:.04em}
      `}</style>

      <div className="wrap">

        {/* TOP BAR */}
        <div className="top">
          <div className="brand">
            <span className="badge">LAD-RAG</span>
            <div>
              <div className="brand-name">SEC Filing Analyst</div>
              <div className="brand-sub">Groq · Llama-3.3-70B · Layout-Aware RAG</div>
            </div>
          </div>
          <div className="ctrl-row">
            <span className="cl">Ticker</span>
            <input className="ci" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} maxLength={5} />
            <select className="cs" value={filingType} onChange={e => setFilingType(e.target.value)}>
              <option>10-K</option><option>10-Q</option><option>8-K</option>
            </select>
            <input className="ci" value={year} onChange={e => setYear(e.target.value)} maxLength={4} style={{width:54}} />
            <div className="stat"><div className="dot"/><span>LIVE</span></div>
          </div>
        </div>

        {/* METRICS BAR */}
        <div className="metrics">
          {[
            { v: ticker,      l: "Ticker"    },
            { v: filingType,  l: "Filing"    },
            { v: year,        l: "Fiscal Yr" },
            { v: queryCount,  l: "Queries"   },
            { v: messages.filter(m=>m.role==="assistant").length, l: "Responses" },
          ].map((m,i) => (
            <div className="metric" key={i}>
              <div className="mv">{m.v}</div>
              <div className="ml">{m.l}</div>
            </div>
          ))}
        </div>

        {/* EXAMPLE CHIPS */}
        <div className="exbar">
          <span className="exlbl">Quick queries</span>
          {EXAMPLES.map((q,i) => (
            <button key={i} className="exchip" onClick={() => sendQuery(q)}>{q}</button>
          ))}
        </div>

        {/* MESSAGES */}
        <div className="msgs">
          {messages.length === 0 && !loading && (
            <div className="empty">
              <div className="egrid">{[...Array(15)].map((_,i)=><div key={i} className="edot"/>)}</div>
              <div className="etitle">Awaiting analyst query</div>
              <div className="esub">Configure ticker · filing type · year above, then ask</div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className="msg">
              <div className={`ico ${m.role === "user" ? "user" : "ai"}`}>
                {m.role === "user" ? "YOU" : "AI"}
              </div>
              <div className="mbody">
                <div className="mhdr">
                  <span className={`mwho ${m.role === "user" ? "user" : "ai"}`}>
                    {m.role === "user" ? "Analyst Query" : `${ticker} · ${filingType} ${year}`}
                  </span>
                  <span className="mts">
                    {m.ts.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"})}
                  </span>
                </div>
                {m.expanded && (
                  <div className="mexp">↳ Expanded: {m.expanded.slice(0, 80)}…</div>
                )}
                {m.role === "assistant" ? (
                  <div className="rbox">
                    <div className="mtext" dangerouslySetInnerHTML={{__html: m.text}} />
                  </div>
                ) : (
                  <div className="mtext user">{m.text}</div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="loading">
              <div className="lico">AI</div>
              <span className="ltxt">
                Querying {ticker} {filingType} {year} data
                <span className="dots"><span>.</span><span>.</span><span>.</span></span>
              </span>
            </div>
          )}
          <div ref={chatEndRef}/>
        </div>

        {/* INPUT BAR */}
        <div className="inputbar">
          <div className="irow">
            <span className="iprompt">&gt;</span>
            <textarea
              ref={inputRef}
              className="ifield"
              placeholder={`Query ${ticker} ${filingType} ${year} financial data...`}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              rows={1}
            />
            <button className="sbtn" onClick={() => sendQuery(input)} disabled={!input.trim() || loading}>
              {loading ? "..." : "EXECUTE"}
            </button>
          </div>
          <div className="ihint">↵ Enter to send · Shift+↵ new line · Click chips above for example queries</div>
        </div>

      </div>
    </>
  );
}
