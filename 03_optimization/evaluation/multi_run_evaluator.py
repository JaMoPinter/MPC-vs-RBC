
import pandas as pd
import re

from typing import Iterable, List, Dict
from pathlib import Path
from evaluation.evaluator import Evaluator

FILE_REGEX = re.compile(
    r"logs_(?P<timestamp>[\d\-]+_[\d\-]+)_"   # e.g. 2025-08-07_14-52
    r"op-(?P<model>[\w\-]+)_"                # e.g. rule-based
    r"freq-(?P<freq>\d+)_"                   # e.g. 60
    r"building-(?P<building>[^.]+)\.csv"     # e.g. SFH3
)
class MultiRunEvaluator:
    """ 
    Aggregate results from multiple runs (e.g., different MPC frequencies, models or buildings)
    """

    def __init__(self, run_paths: Iterable[Path | str]):
        self.run_paths = [Path(path) for path in run_paths]


        records: List[Dict] = []
        for path in self.run_paths:
            m = FILE_REGEX.match(path.name)
            if not m:
                print(f"⚠️  skipped '{path.name}': does not match expected file naming convention")
                continue

            meta = m.groupdict()
            ev = Evaluator(path)

            costs_summary = ev.get_costs()
            records.append({'t_start': ev.t_start, 't_end': ev.t_end, 'total_timespan': ev.total_timespan, **meta, **costs_summary, 'e_end': ev.e_end, 'pg_import_total': ev.pg_import_total, 'pg_export_total': ev.pg_export_total})

            # ev has a class variable called total timespan. I want to include in in the records

        print("self.run_paths:", self.run_paths)
        if not records:
            raise ValueError("No valid run files found.")
        
        self.df = pd.DataFrame(records)
        #self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        #self.df.set_index('timestamp', inplace=True)


    def leaderboard(self, by="net_cost") -> pd.DataFrame:
        """ Returns a leaderboard of the runs sorted by the specified metric. """

        cols = ["model", "building", "freq", "t_start", "t_end", "pg_import_total", "pg_export_total", "e_end", by]

        return self.df[cols].sort_values(by, ignore_index=True)
    

    def pivot(self, value="net_cost") -> pd.DataFrame:
        """
        Matrix: rows=building, cols=model/freq, filled with *value* metric.
        """
        return self.df.pivot_table(
            index="building",
            columns=["model", "freq"],
            values=value,
        )
