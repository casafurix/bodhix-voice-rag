const TEXT_BUDGET_MS = 200;
// Mirrors api/config.py's voice_default_budget_ms — voice requests run a
// slower NVIDIA embedding + LLM path and are budgeted very differently
// from text, not just "the same 200ms plus STT". Showing 200ms for both
// would make every voice request look like it's blowing its budget by
// 30-40x when it's actually well within its own real one.
const VOICE_BUDGET_MS = 35000;

const STAGE_LABELS: Record<string, string> = {
  normalise: 'Normalise',
  guard_in: 'Guard-in',
  embed: 'Embed',
  retrieve: 'Vector search',
  fuse: 'Fuse (RRF)',
  coverage_gate: 'Coverage gate',
  assemble: 'Assemble',
  answer: 'Answer',
  guard_out: 'Guard-out',
};

export function LatencyHUD({ timings }: { timings: Record<string, number> }) {
  const entries = Object.entries(timings);
  if (!entries.length) return null;

  const stt = timings['stt'];
  const isVoice = typeof stt === 'number';
  const budgetMs = isVoice ? VOICE_BUDGET_MS : TEXT_BUDGET_MS;
  const stages = entries.filter(([k]) => k !== 'stt' && STAGE_LABELS[k]);
  const core = stages.reduce((s, [, v]) => s + v, 0);
  const max = Math.max(...stages.map(([, v]) => v), 20);
  const within = core <= budgetMs;

  return (
    <details className="latency">
      <summary>
        Latency — {isVoice ? 'voice' : 'text'} core{' '}
        <b className={within ? 'ok' : 'over'}>
          {core.toFixed(1)}ms / {budgetMs.toLocaleString()}ms
        </b>
      </summary>
      {stages.map(([k, v]) => (
        <div key={k} className="lat-row">
          <span className="lat-name">{STAGE_LABELS[k] ?? k}</span>
          <span className="lat-bar-track">
            <span className="lat-bar" style={{ width: `${Math.max(2, (v / max) * 100)}%` }} />
          </span>
          <span className="lat-ms">{v.toFixed(1)}ms</span>
        </div>
      ))}
      {typeof stt === 'number' && (
        <>
          <div className="lat-note">outside the budget — speech recognition is network-bound:</div>
          <div className="lat-row">
            <span className="lat-name">STT (Sarvam)</span>
            <span className="lat-bar-track">
              <span className="lat-bar lat-bar-ext" style={{ width: '60%' }} />
            </span>
            <span className="lat-ms">{stt.toFixed(1)}ms</span>
          </div>
        </>
      )}
    </details>
  );
}

export default LatencyHUD;
