import pandas as pd
import re

from typing import Iterable, List, Dict, Union
from pathlib import Path
from evaluation.evaluator import Evaluator

# Accepts with or without the optional "_run" suffix before ".csv"
FILE_REGEX = re.compile(
    r"logs_(?P<timestamp>[\d\-]+_[\d\-]+)_"    # e.g. 2025-08-07_14-52
    r"op-(?P<model>[\w\-]+)_"                 # e.g. rule-based
    r"freq-(?P<freq>\d+)_"                    # e.g. 60
    r"building-(?P<building>[^.]+)\_run"   # e.g. SFH3
)

PathLike = Union[Path, str]


class MultiRunEvaluator:
    """
    Aggregate results from multiple run CSVs (different MPC frequencies, models, buildings).

    New columns added:
      - solver_fail_count: number of steps with solver_ok == False
      - solver_fail_pct:   solver_fail_count / total rows
      - solver_top_errors: top 3 'solver_error' values with counts (semicolon-separated)
      - solver_status_mix: counts of 'solver_status' for failed steps (semicolon-separated)

    Notes:
      - If a CSV lacks the columns 'solver_ok'/'solver_error'/'solver_status',
        the failure metrics default to 0 / empty strings.
    """

    def __init__(self, run_paths: Iterable[PathLike]):
        # Normalize/flatten run_paths (works with generators from Path.glob)
        self.run_paths: List[Path] = []
        for p in run_paths:
            if isinstance(p, (list, tuple)):
                self.run_paths.extend(Path(x) for x in p)
            else:
                self.run_paths.append(Path(p))

        records: List[Dict] = []
        for path in self.run_paths:
            if not path.is_file():
                print(f"⚠️  skipped '{path}': not a file")
                continue

            m = FILE_REGEX.match(path.name)
            if not m:
                print(f"⚠️  skipped '{path.name}': does not match expected file naming convention")
                continue

            meta = m.groupdict()
            # Ensure types
            try:
                meta["freq"] = int(meta["freq"])
            except Exception:
                pass

            # Costs & energy summaries via your Evaluator
            ev = Evaluator(path)
            costs_summary = ev.get_costs()

            # Read CSV to compute solver failure metrics
            df = pd.read_csv(path)
            n = len(df)

            if "solver_ok" in df.columns:
                fail_mask = ~df["solver_ok"].astype(bool)
                n_fail = int(fail_mask.sum())
                fail_pct = (n_fail / n) if n else 0.0

                # Top 3 error messages (human text)
                if "solver_error" in df.columns:
                    top_errors = (
                        df.loc[fail_mask, "solver_error"]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .head(3)
                    )
                    top_errors_str = "; ".join([f"{k}:{v}" for k, v in top_errors.items()])
                else:
                    top_errors_str = ""

                # Status category mix (compact categorical summary)
                if "solver_status" in df.columns:
                    status_counts = (
                        df.loc[fail_mask, "solver_status"]
                        .dropna()
                        .astype(str)
                        .value_counts()
                    )
                    status_mix_str = "; ".join([f"{k}:{v}" for k, v in status_counts.items()])
                else:
                    status_mix_str = ""
            else:
                n_fail = 0
                fail_pct = 0.0
                top_errors_str = ""
                status_mix_str = ""

            # Assemble a record for this run
            record = {
                "path": str(path),
                **meta,
                "t_start": ev.t_start,
                "t_end": ev.t_end,
                "total_timespan": ev.total_timespan,
                **costs_summary,
                "e_end": ev.e_end,
                "pg_import_total": ev.pg_import_total,
                "pg_export_total": ev.pg_export_total,
                "steps": n,
                "solver_fails": f"{n_fail}/{n} ({round((n_fail / n)*100, 2)}%)",
                "solver_fail_count": n_fail,
                "solver_fail_pct": fail_pct,
                "solver_top_errors": top_errors_str,
                "solver_status_mix": status_mix_str,
            }
            records.append(record)

        if not records:
            raise ValueError("No valid run files found.")

        self.df = pd.DataFrame(records)

    # -----------------------------
    # Public views
    # -----------------------------

    def leaderboard(self, by: str = "net_cost", per_building: bool = True) -> pd.DataFrame:
        """
        Returns a leaderboard sorted by the metric 'by'.
        If per_building=True, prints a small table per building (like before).
        Otherwise, returns a single DataFrame you can print or display.
        """
        cols_base = [
            "model", "building", "freq", "t_start", "t_end", "solver_fails",
            "pg_import_total", "import_cost", "pg_export_total", "export_revenue", "e_end", by
        ]
        cols = [c for c in cols_base if c in self.df.columns]

        if not per_building:
            return self.df[cols].sort_values(by, ignore_index=True)

        # per-building display (kept from your original behavior)
        from IPython.display import display
        for b in self.df["building"].unique():
            df_building = (
                self.df[self.df["building"] == b][cols]
                .sort_values(by, ignore_index=True)
            )
            print(f"\n🏢 Building: {b}")
            display(df_building)

    def pivot(self, value: str = "net_cost") -> pd.DataFrame:
        """
        Matrix: rows=building, cols=(model,freq), filled with *value* metric.
        """
        return self.df.pivot_table(
            index="building",
            columns=["model", "freq"],
            values=value,
        )

    def failure_table(self) -> pd.DataFrame:
        """
        Tabular overview of solver failure rates per run.
        """
        cols = [
            "model", "building", "freq", "t_start", "t_end", "steps",
            "solver_fail_count", "solver_fail_pct",
            "solver_status_mix", "solver_top_errors", "path"
        ]
        cols = [c for c in cols if c in self.df.columns]
        return self.df[cols].sort_values(["solver_fail_pct", "solver_fail_count"], ascending=False).reset_index(drop=True)

    def summary_by(self, group: List[str], agg_value: str = "net_cost") -> pd.DataFrame:
        """
        Quick grouped summary (mean costs + failure rates).
        Example: summary_by(["model","freq"])
        """
        df = self.df.copy()
        # Aggregate both performance and reliability
        out = df.groupby(group).agg(
            mean_cost=(agg_value, "mean"),
            runs=("path", "count"),
            total_steps=("steps", "sum"),
            total_failures=("solver_fail_count", "sum"),
        )
        out["fail_pct_overall"] = out["total_failures"] / out["total_steps"].replace(0, pd.NA)
        return out.reset_index()
