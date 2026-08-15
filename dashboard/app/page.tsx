import { getUpcomingMatches } from "@/lib/api";
import MatchCard from "@/components/MatchCard";

export default async function DashboardPage() {
  const matches = await getUpcomingMatches();

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-bold text-slate-100">PitchSense</h1>
        <p className="mt-1 text-slate-500">Upcoming Premier League predictions</p>

        <div className="mt-8 space-y-4">
          {matches.map((match) => (
            <MatchCard key={match.match_id} match={match} />
          ))}
        </div>
      </div>
    </main>
  );
}