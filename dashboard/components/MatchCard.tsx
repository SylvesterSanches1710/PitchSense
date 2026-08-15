import { UpcomingMatchPrediction } from "@/lib/api";

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatKickoff(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    weekday: "short", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

const confidenceColor: Record<string, string> = {
  High: "text-emerald-400",
  Medium: "text-amber-400",
  Low: "text-slate-400",
};

export default function MatchCard({ match }: { match: UpcomingMatchPrediction }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold text-slate-100">
          {match.home_team} <span className="text-slate-500">vs</span> {match.away_team}
        </h3>
        <span className="text-sm text-slate-500">{formatKickoff(match.kickoff_utc)}</span>
      </div>

      <div className="mt-1 flex items-center gap-3 text-sm">
        <span className={confidenceColor[match.confidence_label]}>
          {match.confidence_label} confidence
        </span>
        {match.stake_market_margin_pct !== null && (
          <span className="text-slate-500">
            Market margin: {match.stake_market_margin_pct.toFixed(1)}%
          </span>
        )}
      </div>

      {match.low_data_warnings.length > 0 && (
        <div className="mt-3 rounded-lg border border-amber-900 bg-amber-950/50 p-3 text-sm text-amber-300">
          {match.low_data_warnings.map((warning, i) => (
            <p key={i}>⚠ {warning}</p>
          ))}
        </div>
      )}

      <table className="mt-4 w-full text-sm">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="pb-2 font-normal">Outcome</th>
            <th className="pb-2 font-normal text-right">Model</th>
            <th className="pb-2 font-normal text-right">Stake Odds</th>
            <th className="pb-2 font-normal text-right">EV</th>
          </tr>
        </thead>
        <tbody>
          {match.outcomes.map((outcome) => (
            <tr
              key={outcome.outcome}
              className={outcome.is_positive_ev ? "text-emerald-400" : "text-slate-300"}
            >
              <td className="py-1">{outcome.label}</td>
              <td className="py-1 text-right">{formatPercent(outcome.model_probability)}</td>
              <td className="py-1 text-right">
                {outcome.stake_odds !== null ? outcome.stake_odds.toFixed(2) : "—"}
              </td>
              <td className="py-1 text-right font-medium">
                {outcome.ev !== null ? `${outcome.ev >= 0 ? "+" : ""}${(outcome.ev * 100).toFixed(1)}%` : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}