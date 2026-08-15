interface ModelCVResult {
  model_name: string;
  log_loss_mean: number;
  log_loss_std: number;
  brier_mean: number;
  brier_std: number;
  accuracy_mean: number;
  accuracy_std: number;
}

interface ModelPerformance {
  computed_at: string;
  test_period: string;
  cross_validation_folds: number;
  cross_validation: ModelCVResult[];
  calibration_ece: Record<string, number>;
  final_model: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL;

async function getModelPerformance(): Promise<ModelPerformance> {
  const res = await fetch(`${API_URL}/model/performance`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to fetch model performance: ${res.status}`);
  }
  return res.json();
}

export default async function ModelPerformancePage() {
  const perf = await getModelPerformance();

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-10">
      <div className="mx-auto max-w-3xl">
        <h1 className="text-2xl font-bold text-slate-100">Model Performance</h1>
        <p className="mt-1 text-slate-500">
          {perf.cross_validation_folds}-fold time-series cross-validation, tested on {perf.test_period}
        </p>

        <div className="mt-8 rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-100">Model Comparison</h2>
          <p className="mt-1 text-sm text-slate-500">
            Sorted by log loss (lower is better) — mean ± std across {perf.cross_validation_folds} folds
          </p>

          <table className="mt-4 w-full text-sm">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="pb-2 font-normal">Model</th>
                <th className="pb-2 font-normal text-right">Log Loss</th>
                <th className="pb-2 font-normal text-right">Brier</th>
                <th className="pb-2 font-normal text-right">Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {perf.cross_validation.map((r, i) => (
                <tr
                  key={r.model_name}
                  className={i === 0 ? "font-semibold text-emerald-400" : "text-slate-300"}
                >
                  <td className="py-2">
                    {r.model_name}
                    {r.model_name === perf.final_model && (
                      <span className="ml-2 rounded bg-emerald-950 px-2 py-0.5 text-xs text-emerald-400">
                        selected
                      </span>
                    )}
                  </td>
                  <td className="py-2 text-right">
                    {r.log_loss_mean.toFixed(3)} ± {r.log_loss_std.toFixed(3)}
                  </td>
                  <td className="py-2 text-right">
                    {r.brier_mean.toFixed(3)} ± {r.brier_std.toFixed(3)}
                  </td>
                  <td className="py-2 text-right">
                    {(r.accuracy_mean * 100).toFixed(1)}% ± {(r.accuracy_std * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="mt-4 text-xs text-slate-600">
            Random guessing baseline: log loss ≈ 1.099, Brier ≈ 0.667. A gap between models
            smaller than roughly one standard deviation shouldn&apos;t be read as a meaningful
            difference.
          </p>
        </div>

        <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-100">
            Calibration — {perf.final_model}
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Expected Calibration Error per outcome (lower is better; 0 = perfectly calibrated)
          </p>

          <div className="mt-4 grid grid-cols-3 gap-4">
            {Object.entries(perf.calibration_ece).map(([label, ece]) => (
              <div key={label} className="rounded-lg bg-slate-800 p-4 text-center">
                <p className="text-sm text-slate-400">{label}</p>
                <p className="mt-1 text-2xl font-bold text-slate-100">{ece.toFixed(3)}</p>
              </div>
            ))}
          </div>

          <p className="mt-4 text-xs text-slate-600">
            High-probability Away Win predictions have been found to run somewhat
            overconfident — treat those specifically with extra caution beyond what the
            ECE number alone conveys.
          </p>
        </div>

        <p className="mt-6 text-xs text-slate-600">
          Last computed: {new Date(perf.computed_at).toLocaleString()}
        </p>
      </div>
    </main>
  );
}