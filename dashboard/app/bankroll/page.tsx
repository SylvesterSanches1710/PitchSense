"use client";

import { useEffect, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";

interface BetRecord {
  id: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  outcome: string;
  stake: number;
  odds_taken: number;
  status: string;
  profit_loss: number | null;
  placed_at: string;
}

interface BetSummary {
  total_bets: number;
  pending: number;
  settled: number;
  won: number;
  win_rate: number | null;
  total_staked: number;
  total_profit_loss: number;
  roi_pct: number | null;
}

interface CumulativePLPoint {
  bet_id: number;
  settled_at: string;
  cumulative_pl: number;
}

interface BetHistory {
  bets: BetRecord[];
  summary: BetSummary;
  cumulative_pl: CumulativePLPoint[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL;

const outcomeLabel: Record<string, string> = { H: "Home", D: "Draw", A: "Away" };

const statusColor: Record<string, string> = {
  won: "text-emerald-400",
  lost: "text-red-400",
  pending: "text-amber-400",
  void: "text-slate-500",
};

function StatCard({ label, value, valueColor }: { label: string; value: string; valueColor?: string }) {
  return (
    <div className="rounded-lg bg-slate-800 p-4">
      <p className="text-sm text-slate-400">{label}</p>
      <p className={`mt-1 text-xl font-bold ${valueColor ?? "text-slate-100"}`}>{value}</p>
    </div>
  );
}

export default function BankrollPage() {
  const [data, setData] = useState<BetHistory | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/betting/history`)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);
        return res.json();
      })
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
        <p className="text-red-400">Error loading bet history: {error}</p>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-10 text-slate-400">
        Loading...
      </main>
    );
  }

  const { bets, summary, cumulative_pl } = data;
  const plColor = summary.total_profit_loss >= 0 ? "text-emerald-400" : "text-red-400";

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-bold text-slate-100">Bankroll & Betting History</h1>
        <p className="mt-1 text-slate-500">Real bets logged, settled against actual results</p>

        <div className="mt-8 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="Total Bets" value={String(summary.total_bets)} />
          <StatCard
            label="Win Rate"
            value={summary.win_rate !== null ? `${(summary.win_rate * 100).toFixed(1)}%` : "—"}
          />
          <StatCard
            label="Total P/L"
            value={`${summary.total_profit_loss >= 0 ? "+" : ""}${summary.total_profit_loss.toFixed(2)}`}
            valueColor={plColor}
          />
          <StatCard
            label="ROI"
            value={summary.roi_pct !== null ? `${summary.roi_pct >= 0 ? "+" : ""}${summary.roi_pct.toFixed(1)}%` : "—"}
            valueColor={summary.roi_pct !== null && summary.roi_pct >= 0 ? "text-emerald-400" : "text-red-400"}
          />
        </div>

        {summary.settled < 20 && (
          <p className="mt-3 text-xs text-slate-600">
            Small sample size ({summary.settled} settled bet{summary.settled === 1 ? "" : "s"}) —
            win rate and ROI are noisy at this volume and shouldn&apos;t be read as reliable
            long-run performance yet.
          </p>
        )}

        {cumulative_pl.length > 1 && (
          <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold text-slate-100">Cumulative P/L</h2>
            <div className="mt-4 h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={cumulative_pl}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="bet_id" stroke="#64748b" fontSize={12} />
                  <YAxis stroke="#64748b" fontSize={12} />
                  <ReferenceLine y={0} stroke="#475569" />
                  <Tooltip
                    contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }}
                    labelStyle={{ color: "#94a3b8" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="cumulative_pl"
                    stroke="#34d399"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-100">All Bets</h2>
          <table className="mt-4 w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="pb-2 font-normal">Match</th>
                <th className="pb-2 font-normal">Bet</th>
                <th className="pb-2 font-normal text-right">Stake</th>
                <th className="pb-2 font-normal text-right">Odds</th>
                <th className="pb-2 font-normal">Status</th>
                <th className="pb-2 font-normal text-right">P/L</th>
              </tr>
            </thead>
            <tbody>
              {bets.map((bet) => (
                <tr key={bet.id} className="text-slate-300">
                  <td className="py-2">{bet.home_team} vs {bet.away_team}</td>
                  <td className="py-2">{outcomeLabel[bet.outcome]}</td>
                  <td className="py-2 text-right">{bet.stake.toFixed(2)}</td>
                  <td className="py-2 text-right">{bet.odds_taken.toFixed(2)}</td>
                  <td className={`py-2 ${statusColor[bet.status]}`}>{bet.status}</td>
                  <td className={`py-2 text-right ${bet.profit_loss !== null ? (bet.profit_loss >= 0 ? "text-emerald-400" : "text-red-400") : "text-slate-500"}`}>
                    {bet.profit_loss !== null ? `${bet.profit_loss >= 0 ? "+" : ""}${bet.profit_loss.toFixed(2)}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}