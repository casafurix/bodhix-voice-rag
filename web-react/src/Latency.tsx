const BUDGET_MS = 200;

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
  const stages = entries.filter(([k]) => k !== 'stt' && STAGE_LABELS[k]);
  const core = stages.reduce((s, [, v]) => s + v, 0);
  const max = Math.max(...stages.map(([, v]) => v), BUDGET_MS * 0.3);
  const within = core <= BUDGET_MS;

  return (
    <details className="latency">
      <summary>
        Latency — retrieval core{' '}
        <b className={within ? 'ok' : 'over'}>
          {core.toFixed(1)}ms / {BUDGET_MS}ms
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
