export interface OutcomeAnalysis {
  outcome: string;
  label: string;
  model_probability: number;
  stake_fair_probability: number | null;
  stake_odds: number | null;
  ev: number | null;
  kelly_stake_fraction: number | null;
  is_positive_ev: boolean;
}

export interface UpcomingMatchPrediction {
  match_id: number;
  kickoff_utc: string;
  home_team: string;
  away_team: string;
  confidence_label: string;
  confidence_margin: number;
  stake_market_margin_pct: number | null;
  low_data_warnings: string[];
  outcomes: OutcomeAnalysis[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export async function getUpcomingMatches(): Promise<UpcomingMatchPrediction[]> {
  const res = await fetch(`${API_URL}/matches/upcoming`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch upcoming matches: ${res.status}`);
  }
  return res.json();
}