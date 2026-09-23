import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  CircleAlert,
  Download,
  FileJson,
  Link2,
  LoaderCircle,
  Mail,
  Network,
  ScanSearch,
  Trash2,
  Upload,
} from "lucide-react";

const DETECTORS = [
  { name: "email", label: "Email", path: "/api/v1/email/analyze", Icon: Mail },
  { name: "url", label: "URL", path: "/api/v1/url/analyze", Icon: Link2 },
  { name: "network", label: "Network", path: "/api/v1/network/analyze", Icon: Network },
];

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = typeof body.detail === "string"
      ? body.detail
      : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return body;
}

function parseNetworkFlow(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("Network flow must be valid JSON.");
  }

  const features = parsed && Object.hasOwn(parsed, "features")
    ? parsed.features
    : parsed;
  if (features === null || typeof features !== "object") {
    throw new Error("Network flow must contain named features or an ordered list.");
  }
  if (Object.keys(features).length !== 78) {
    throw new Error("Network flow must contain exactly 78 features.");
  }
  return { features };
}

function buildRequests(email, url, networkFlow) {
  const requests = [];
  if (email.trim()) {
    requests.push({ name: "email", path: DETECTORS[0].path, payload: { text: email.trim() } });
  }
  if (url.trim()) {
    requests.push({ name: "url", path: DETECTORS[1].path, payload: { url: url.trim() } });
  }
  if (networkFlow.trim()) {
    requests.push({ name: "network", path: DETECTORS[2].path, payload: parseNetworkFlow(networkFlow) });
  }
  if (requests.length === 0) {
    throw new Error("Enter an email, URL, or network flow to analyze.");
  }
  return requests;
}

function Finding({ detector, result }) {
  const score = detector.name === "email" ? result.phishing_probability
    : detector.name === "url" ? result.malicious_probability
    : result.anomaly_probability;
  // The network value is a normalized anomaly score, not a probability,
  // so it is shown on a 0-1 scale rather than as a percentage.
  const isNetwork = detector.name === "network";
  const scoreLabel = isNetwork ? "Anomaly score" : "Threat probability";
  const scoreText = isNetwork ? score.toFixed(2) : `${Math.round(score * 100)}%`;
  const Icon = detector.Icon;

  return (
    <div className="finding">
      <div className="finding-icon"><Icon size={17} strokeWidth={1.9} /></div>
      <div className="finding-copy">
        <strong>{detector.label}</strong>
        <span>{result.prediction} <span className="separator">/</span> {result.model_version}</span>
      </div>
      <div className="finding-score">
        <strong>{scoreText}</strong>
        <span>{scoreLabel}</span>
      </div>
      <span className={`severity-tag ${result.severity.toLowerCase()}`}>{result.severity}</span>
    </div>
  );
}

function App() {
  const [email, setEmail] = useState("");
  const [url, setUrl] = useState("");
  const [networkFlow, setNetworkFlow] = useState("");
  const [fileName, setFileName] = useState("");
  const [apiStatus, setApiStatus] = useState("checking");
  const [formError, setFormError] = useState("");
  const [report, setReport] = useState(null);
  const [runState, setRunState] = useState("idle");

  useEffect(() => {
    function checkApi() {
      fetch("/health")
        .then((response) => setApiStatus(response.ok ? "online" : "offline"))
        .catch(() => setApiStatus("offline"));
    }

    checkApi();
    const interval = window.setInterval(checkApi, 10_000);
    return () => window.clearInterval(interval);
  }, []);

  async function analyzeIncident(event) {
    event.preventDefault();
    setFormError("");

    let requests;
    try {
      requests = buildRequests(email, url, networkFlow);
    } catch (error) {
      setFormError(error.message);
      return;
    }

    setRunState("running");
    const outcomes = await Promise.allSettled(
      requests.map((request) => postJson(request.path, request.payload))
    );
    const nextReport = {
      analyzed_at: new Date().toISOString(),
      detections: {},
      errors: {},
      triage: null,
      triage_error: null,
    };

    outcomes.forEach((outcome, index) => {
      const moduleName = requests[index].name;
      if (outcome.status === "fulfilled") {
        nextReport.detections[moduleName] = outcome.value;
      } else {
        nextReport.errors[moduleName] = outcome.reason.message;
      }
    });

    if (Object.keys(nextReport.detections).length > 0) {
      try {
        nextReport.triage = await postJson(
          "/api/v1/orchestration/analyze",
          nextReport.detections
        );
      } catch (error) {
        nextReport.triage_error = error.message;
      }
    } else {
      nextReport.triage_error = "No detector returned a result.";
    }

    setReport(nextReport);
    setRunState("done");
  }

  function clearForm() {
    setEmail("");
    setUrl("");
    setNetworkFlow("");
    setFileName("");
    setFormError("");
    setReport(null);
    setRunState("idle");
  }

  async function loadNetworkFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setNetworkFlow(await file.text());
    setFormError("");
  }

  function downloadReport() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = "triage-report.json";
    link.click();
    URL.revokeObjectURL(downloadUrl);
  }

  const detectorCount = report ? Object.keys(report.detections).length : 0;
  const errorCount = report ? Object.keys(report.errors).length : 0;
  const resultState = runState === "running" ? "Analyzing"
    : !report ? "Awaiting analysis"
    : errorCount > 0 ? "Partial analysis"
    : report.triage ? "Analysis complete" : "Detection complete";

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={19} strokeWidth={2.2} /></div>
          <div className="brand-name"><strong>Cybersec Triage</strong><span>CM3020 final project</span></div>
        </div>
        <nav aria-label="Primary navigation">
          <a className="nav-link active" href="/" aria-current="page"><ScanSearch size={17} />Analyze</a>
          <a className="nav-link" href="/docs" target="_blank" rel="noopener noreferrer"><FileJson size={17} />API reference<ArrowUpRight size={13} className="nav-arrow" /></a>
        </nav>
        <div className="sidebar-status">
          <span className={`status-dot ${apiStatus}`} aria-hidden="true" />
          <span>{apiStatus === "online" ? "API connected" : apiStatus === "offline" ? "API unavailable" : "Checking API"}</span>
        </div>
      </aside>

      <main className="main">
        <header className="page-header">
          <h1>Threat triage</h1>
          <span className="header-meta">Email <i>/</i> URL <i>/</i> Network</span>
        </header>

        <div className="workspace">
          <section className="input-pane" aria-labelledby="input-heading">
            <div className="section-heading"><h2 id="input-heading">Threat artifacts</h2></div>
            <form onSubmit={analyzeIncident} noValidate>
              <div className="input-group">
                <label className="field-label" htmlFor="email-text">Email text</label>
                <textarea id="email-text" rows="6" maxLength="50000" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Paste the email content" />
              </div>
              <div className="input-group">
                <label className="field-label" htmlFor="url-input">URL</label>
                <input id="url-input" type="text" maxLength="2048" inputMode="url" autoComplete="off" value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com/path" />
              </div>
              <div className="input-group">
                <label className="field-label" htmlFor="network-json">Network flow JSON</label>
                <textarea id="network-json" className="json-input" rows="6" spellCheck="false" value={networkFlow} onChange={(event) => setNetworkFlow(event.target.value)} placeholder="Paste a network flow or select a JSON file" />
                <div className="file-row">
                  <label className="file-control" htmlFor="network-file"><Upload size={14} />Choose JSON file</label>
                  <input id="network-file" type="file" accept=".json,application/json" onChange={loadNetworkFile} />
                  <span className="file-name">{fileName || "No file selected"}</span>
                </div>
              </div>
              {formError && <p className="form-error" role="alert"><CircleAlert size={16} />{formError}</p>}
              <div className="form-actions">
                <button className="primary-button" type="submit" disabled={runState === "running"}>
                  {runState === "running" ? <LoaderCircle size={16} className="spin" /> : <ScanSearch size={16} />}
                  {runState === "running" ? "Analyzing..." : "Analyze incident"}
                </button>
                <button className="icon-button" type="button" onClick={clearForm} disabled={runState === "running"} title="Clear inputs and results" aria-label="Clear inputs and results"><Trash2 size={17} /></button>
              </div>
            </form>
          </section>

          <section className="result-pane" aria-labelledby="result-heading" aria-live="polite">
            <div className="section-heading"><h2 id="result-heading">Triage report</h2><span className="run-state">{resultState}</span></div>
            {!report && runState !== "running" && <div className="empty-state"><div className="empty-mark"><ScanSearch size={25} strokeWidth={1.4} /></div><h3>No analysis yet</h3><p>Results will appear here after an incident is analyzed.</p></div>}
            {runState === "running" && <div className="empty-state"><LoaderCircle size={26} className="spin" /><h3>Analyzing incident</h3></div>}
            {report && runState === "done" && <>
              <div className="summary-band"><div><span className="result-label">OVERALL SEVERITY</span><strong className={`overall-severity ${report.triage?.severity?.toLowerCase() || ""}`}>{report.triage?.severity || "Incomplete"}</strong></div><span className="module-count">{detectorCount} detector{detectorCount === 1 ? "" : "s"} completed</span></div>
              {report.triage && <div className="triage-content"><div className="result-block"><h3>Evidence summary</h3><p>{report.triage.evidence_summary}</p></div><div className="result-block"><h3>Analyst recommendation</h3><p>{report.triage.analyst_recommendation}</p></div><div className="result-block"><h3>Based on</h3><div className="module-tags">{DETECTORS.filter((detector) => report.triage.contributing_modules.includes(detector.name)).map((detector) => <span key={detector.name} className="module-tag"><detector.Icon size={13} />{detector.label}</span>)}</div></div></div>}
              {report.triage_error && <div className="notice" role="status"><CircleAlert size={17} /><span>Combined triage unavailable: {report.triage_error}</span></div>}
              <div className="findings-heading"><h3>Detector findings</h3><span>{detectorCount} result{detectorCount === 1 ? "" : "s"}</span></div>
              <div className="findings">{DETECTORS.filter((detector) => report.detections[detector.name]).map((detector) => <Finding key={detector.name} detector={detector} result={report.detections[detector.name]} />)}</div>
              {errorCount > 0 && <div className="module-errors">{Object.entries(report.errors).map(([module, detail]) => <p key={module}><CircleAlert size={15} /><span>{module.toUpperCase()}: {detail}</span></p>)}</div>}
              <div className="result-actions"><button className="secondary-button" type="button" onClick={downloadReport}><Download size={15} />Download JSON</button></div>
            </>}
          </section>
        </div>
      </main>
    </div>
  );
}

export default App;
