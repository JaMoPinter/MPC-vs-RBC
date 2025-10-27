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
            df = pd.read_csv(path, low_memory=False)
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
                "e_export_total": ev.e_export_total,
                "e_import_total": ev.e_import_total,
                "e_throughput": ev.e_throughput_total,
                "e_battery_deg_costs": ev.battery_deg_costs,
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

    def leaderboard(self, by: str = "net_cost_final", per_building: bool = True, cols_to_show=None) -> pd.DataFrame:
        """
        Returns a leaderboard sorted by the metric 'by'.
        If per_building=True, prints a small table per building (like before).
        Otherwise, returns a single DataFrame you can print or display.
        """
        if cols_to_show is not None:
            cols_base = cols_to_show
        else:
            cols_base = [
                "model", "building", "freq", "t_start", "t_end", "solver_fails",
                "e_import_total", "import_cost", "e_export_total", "export_revenue", "e_throughput", "e_discharged_total", "e_battery_deg_costs", "e_end", "net_cost", "net_cost_adj",
                by
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


    def leaderboard_mean(
        self,
        group=("model",),             # e.g. ("model",) or ("model","freq")
        sort_by="net_cost_final",           # which metric to sort on (defaults to net_cost_final)
        rank_by: str | None = None,   # which metric to rank on (defaults to sort_by or net_cost/net_cost_adj)
        numeric_agg: str = "mean",    # "mean" or "median" for numeric aggregation
        round_ndigits: int | None = 2 # round numeric outputs (None = no rounding)
    ) -> pd.DataFrame:
        """
        Aggregate across buildings/runs and return one row per group with:
        - numeric means/medians of all numeric columns,
        - reliability rollups (runs, buildings, total_steps, total_failures, solver_fail_pct_overall),
        - avg_rank: average per-building rank for `rank_by` (smaller is better),
        - rank_n_buildings: #buildings contributing to avg_rank.

        Ranking procedure:
        1) For each building, aggregate `rank_by` over duplicate rows of `group` (e.g., multiple runs) using mean.
        2) Rank groups within the building (ascending).
        3) Average ranks across buildings where the group is present.

        Notes:
        - If a group is missing for some buildings, those buildings are skipped for that group's avg_rank.
        - Choose `rank_by` explicitly if you don't want it to follow `sort_by`.
        """
        cols_base = ["path",
                "model", "building", "freq", "t_start", "t_end", "solver_fails",
                "e_import_total", "import_cost", "e_export_total", "export_revenue", "e_throughput", "e_discharged_total", "e_battery_deg_costs", "e_end", "net_cost", "net_cost_adj", 
                "net_cost_final"
            ]
        cols = [c for c in cols_base if c in self.df.columns]

        df = self.df[cols].copy()
        gcols = list(group)

        # choose rank_by default smartly
        if rank_by is None:
            rank_by = sort_by if sort_by in df.columns else ("net_cost_final" if "net_cost_final" in df.columns else "net_cost_adj")
        if rank_by not in df.columns:
            raise ValueError(f"`rank_by`='{rank_by}' not found in columns.")

        # ---- reliability rollup (as before)
        rollup = df.groupby(gcols, dropna=False).agg(
            runs=("path", "count"),
            buildings=("building", pd.Series.nunique)
        )

        # ---- numeric aggregation (mean/median)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        agg_fn = "mean" if numeric_agg == "mean" else "median"
        num_agg = df.groupby(gcols, dropna=False)[num_cols].agg(agg_fn)

        out = rollup.join(num_agg, how="left").reset_index()

        # ---- compute average rank across buildings for `rank_by`
        # 1) collapse to one value per (building, group) for ranking
        rank_base = (
            df.groupby(["building", *gcols], dropna=False)[rank_by]
            .mean()
            .reset_index()
        )
        # 2) within each building, rank ascending (best = 1.0); method='average' handles ties fairly
        rank_base["rank"] = rank_base.groupby("building")[rank_by].rank(method="average", ascending=True)
        # 3) average ranks across buildings for each group
        avg_rank = (
            rank_base.groupby(gcols, dropna=False)
                    .agg(avg_rank=("rank", "mean"),
                        rank_n_buildings=("building", "nunique"))
                    .reset_index()
        )[['model', 'avg_rank']]
        out = out.merge(avg_rank, on=gcols, how="left")

        # ---- sorting
        if sort_by not in out.columns:
            for cand in ["avg_rank", "net_cost_adj", "net_cost", "import_cost", "export_revenue"]:
                if cand in out.columns:
                    sort_by = cand
                    break
            else:
                sort_by = None
        if sort_by is not None:
            out = out.sort_values(sort_by, ascending=True, ignore_index=True)

        # ---- rounding
        if round_ndigits is not None:
            for c in out.columns:
                if pd.api.types.is_float_dtype(out[c]):
                    out[c] = out[c].round(round_ndigits)

        return out


    def pivot(self, models: List[str], value: str = "net_cost") -> pd.DataFrame:
        """
        Matrix: rows=building, cols=(model,freq), filled with *value* metric.
        """

        return self.df[self.df["model"].isin(models)].pivot_table(
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
