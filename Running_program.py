# Running_program.py
"""
Experiment runner that:
 - creates a main run folder per grid run
 - writes the param_grid and copies input CSVs into main_run/inputs/
 - ensures each run's outdir is saved under the main run folder
 - flattens signal + trading params into the summary dataframe
 - prints best run by sharpe and its annualized return
"""

import os
import itertools
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pprint import pformat
from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Data_Ingestion as di
import Data_processing as dp
import globals

# -------------------------
# Config 
# -------------------------
INGEST_NEW_DATA = True
DAYS_AGO = 15 * 365
TOP_N = 102

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
globals.create_check_folder("data", PROJECT_ROOT)
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TC_BPS = 5
DAYS_PER_YEAR = 252
VOL_WINDOW = 20 # giving default value for vol_window for convenience
EPS = 1e-8

# -------------------------
@dataclass
class StrategyConfig:
    signal_params: Dict[str, Any]
    trading_params: Dict[str, Any]
    meta: Dict[str, Any] = None

    def to_dict(self):
        return {"signal_params": self.signal_params, "trading_params": self.trading_params, "meta": self.meta or {}}

# -------------------------
def expand_grid(param_grid: Dict[str, Iterable]) -> Iterable[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values_list = [globals.listify(param_grid[k]) for k in keys]
    for combo in itertools.product(*values_list):
        yield dict(zip(keys, combo))

# -------------------------
class ExperimentRunner:
    def __init__(self,
                 base_data_dir: str = BASE_DATA_DIR,
                 ingest_new_data: bool = INGEST_NEW_DATA,
                 days_ago: int = DAYS_AGO,
                 top_n: int = TOP_N,
                 eps: float = EPS):
        self.base_data_dir = base_data_dir
        self.ingest_new_data = ingest_new_data
        self.days_ago = days_ago
        self.top_n = top_n
        self.eps = eps
        os.makedirs(self.base_data_dir, exist_ok=True)

    def _make_main_run_dir(self, name_hint: str = None):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name_comp = f"{ts}"
        if name_hint:
            safe_hint = "".join(c for c in name_hint if c.isalnum() or c in ("_", "-")).rstrip()
            name_comp = f"{name_comp}_{safe_hint}"
        main_dir = os.path.join(self.base_data_dir, "experiments", name_comp)
        os.makedirs(main_dir, exist_ok=True)
        return main_dir

    def load_or_ingest(self):
        if self.ingest_new_data:
            start_date = di.get_date_n_days_ago_str(self.days_ago)
            raw_data = di.ingest_equity_data(start_date, top_n=self.top_n, features_list=None, tickers_list=None, back_date=0)
            raw_path = os.path.join(self.base_data_dir, "1_raw_data.csv")
            raw_data.to_csv(raw_path)
            return raw_data, raw_path
        else:
            raw_path = os.path.join(self.base_data_dir, "1_raw_data.csv")
            raw_data = pd.read_csv(raw_path, header=[0, 1], index_col=0, parse_dates=True)
            return raw_data, raw_path

    def process_data(self, raw_data: pd.DataFrame, vol_window: int, eps: float, default: bool = False):
        processed = dp.compute_returns(raw_data.dropna(axis=1, how="all"), vol_window=vol_window, eps=eps)
        processed_path = os.path.join(self.base_data_dir, "2_processed_data.csv")
        if default:
            processed_path = os.path.join(self.base_data_dir, "2_processed_data_default.csv")
        processed.to_csv(processed_path)
        return processed, processed_path

    def make_signals(self, processed_data, strategy_kind, lookback, vol_window, default: bool = False):
        signals = dp.cross_sectional_signal(processed_data, kind=strategy_kind, lookback=lookback, vol_window=vol_window)
        # do not overwrite user's base signals file each run to avoid accidental clobber;
        # but keep a default copy under base_data_dir for reproducibility
        signals_path = os.path.join(self.base_data_dir, f"3_signals_lookback_{int(lookback)}_volwindow_{vol_window}.csv")
        if default:
            signals_path = os.path.join(self.base_data_dir, "3_signals_default.csv")
        try:
            signals.to_csv(signals_path)
        except Exception:
            # best-effort: if write fails, ignore (signals still returned)
            pass
        return signals, signals_path

    def _save_grid_inputs(self, main_run_dir: str, param_grid: dict, input_paths: Dict[str, str]):
        inputs_dir = os.path.join(main_run_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)
        with open(os.path.join(inputs_dir, "param_grid.json"), "w", encoding="utf-8") as f:
            json.dump(param_grid, f, indent=2)
        for label, path in input_paths.items():
            if path and os.path.exists(path):
                try:
                    shutil.copy2(path, os.path.join(inputs_dir, os.path.basename(path)))
                except Exception as e:
                    print(f"Warning: could not copy {label} at {path} -> {e}")

    def run_grid(self,
                param_grid: Dict[str, Iterable],
                strategy_kind: str = "mean_reversion",
                signal_param_names: Iterable[str] = ("lookback",),
                main_run_name_hint: str = None,
                max_runs: int = None):
        """
        Run a grid of experiments defined by param_grid, where each key is a parameter name and each value is an iterable of possible values.
        For each combination of parameters, runs a backtest and records results in a summary dataframe.
        """
        # helper to canonicalize numpy scalars to native python types
        def _canonicalize_params(d: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            for k, v in d.items():
                if isinstance(v, (np.floating,)):
                    out[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                else:
                    out[k] = v
            return out

        # Prepare main-run folder (timestamped)
        main_run_dir = self._make_main_run_dir(main_run_name_hint)
        print(f"Main run dir: {main_run_dir}")

        # Load/process data once and capture input paths
        raw_data, raw_path = self.load_or_ingest()

        # Determine a safe default lookback
        if "lookback" in param_grid:
            default_lookback = int(globals.listify(param_grid["lookback"])[0])
        else:
            default_lookback = 10


        # compute a default processed dataset (used for default signals + to save as an "input")
        default_vol_window = int(globals.listify(param_grid.get("vol_window", [VOL_WINDOW]))[0]) if "vol_window" in param_grid else VOL_WINDOW
        processed_default, processed_path = self.process_data(raw_data, vol_window=default_vol_window, eps=self.eps, default = True)

        # Create a default signals file for convenience (saved under base_data_dir)
        signals_default, signals_path = self.make_signals(processed_default, strategy_kind=strategy_kind, lookback=default_lookback, vol_window=default_vol_window, default=True)

        # Save param grid + input copies into main_run_dir/inputs/
        input_paths = {"raw": raw_path, "processed": processed_path, "signals_default": signals_path}
        self._save_grid_inputs(main_run_dir, param_grid, input_paths)


        combos = list(expand_grid(param_grid))
        if max_runs is not None:
            combos = combos[:max_runs]
        print(f"Prepared {len(combos)} parameter combinations to run...")

        summary_rows = []
        for run_index, combo in enumerate(combos, start=1):
            print("\n" + "=" * 60)
            print(f"Starting run {run_index}/{len(combos)}")
            print(pformat(combo))

            # canonicalize combo and extract per-run strategy_kind if present
            combo_canon = _canonicalize_params(combo)
            per_run_strategy = combo_canon.pop("strategy_kind", strategy_kind)

            # split params into signal vs trading
            signal_params = {k: combo_canon[k] for k in combo_canon if k in signal_param_names}
            trading_params = {k: combo_canon[k] for k in combo_canon if k not in signal_param_names}

            config = StrategyConfig(signal_params=signal_params, trading_params=trading_params, meta={"strategy_kind": per_run_strategy})

            # Determine lookback (guaranteed integer)
            lookback = int(signal_params.get("lookback", default_lookback))
            vol_window = int(combo_canon.get("vol_window", VOL_WINDOW))

            # Create run-specific signals
            processed, processed_path = self.process_data(raw_data, vol_window=vol_window, eps=self.eps)
            signals, signals_path_run = self.make_signals(processed, strategy_kind=per_run_strategy, lookback=lookback, vol_window=vol_window)

            # Run backtest: pass the TOP-LEVEL main_run_dir so dp.run_and_record creates per-run folders under it
            summary_row, outdir = dp.run_and_record(
                run_index,
                trading_params,
                signals,
                processed,
                tc_bps=TC_BPS,
                days_per_year=DAYS_PER_YEAR,
                strategy_kind=per_run_strategy,
                data_dir=main_run_dir
            )

            # Flatten all params into the summary_row for easy analysis
            # (signal params)
            for k, v in signal_params.items():
                summary_row[k] = v
            summary_row["vol_window"] = vol_window
            # (trading params)
            for k, v in trading_params.items():
                summary_row[k] = v

            # include strategy_kind explicitly
            summary_row["strategy_kind"] = per_run_strategy

            # Keep reproducible config + timestamp
            summary_row["config"] = config.to_dict()
            summary_row["run_time"] = datetime.now().isoformat()
            summary_rows.append(summary_row)

            # write full_config.json inside the run folder if outdir was returned
            try:
                if outdir:
                    with open(os.path.join(outdir, "full_config.json"), "w", encoding="utf-8") as f:
                        json.dump(config.to_dict(), f, indent=2)
            except Exception as e:
                print("Warning: failed to write full_config.json:", e)

        # save grid summary at main_run_dir level
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(main_run_dir, f"grid_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved grid summary to {summary_csv}")

        # Identify best run by Sharpe and plot its cumulative index
        if not summary_df.empty and "sharpe" in summary_df.columns:
            sharpe_series = summary_df["sharpe"].replace([None], pd.NA).dropna()
            if not sharpe_series.empty:
                best_idx = sharpe_series.idxmax()
                best_row = summary_df.loc[best_idx]

                best_run_name = best_row.get("run_name", f"run_{int(best_idx) + 1}")
                best_sharpe = best_row.get("sharpe", None)
                ann_ret = best_row.get("ann_return", None)
                best_outdir = best_row.get("outdir", None)

                # Print best-run summary
                try:
                    if ann_ret is not None:
                        ann_ret_pct = float(ann_ret) * 100.0
                        print(f"\nBest run by Sharpe: {best_run_name} (sharpe={best_sharpe:.3f}) — annualized return = {ann_ret_pct:.2f}%")
                    else:
                        print(f"\nBest run by Sharpe: {best_run_name} (sharpe={best_sharpe:.3f}) — annualized return: N/A")
                except Exception:
                    print(f"\nBest run by Sharpe: {best_run_name} (sharpe={best_sharpe}) — annualized return: {ann_ret}")

                # Plot cumulative return of best run if outdir exists and cum_index.csv is present
                try:
                    if best_outdir:
                        cum_index_path = os.path.join(best_outdir, "cum_index.csv")
                        if os.path.exists(cum_index_path):
                            ci_df = pd.read_csv(cum_index_path, index_col=0, parse_dates=True)

                            # pick sensible numeric series/column
                            if isinstance(ci_df, pd.DataFrame):
                                if ci_df.shape[1] == 1:
                                    ci = ci_df.iloc[:, 0]
                                else:
                                    numeric_cols = ci_df.select_dtypes(include="number").columns
                                    ci = ci_df[numeric_cols[0]] if len(numeric_cols) > 0 else ci_df.iloc[:, 0]
                            else:
                                ci = pd.Series(ci_df)

                            plt.figure(figsize=(10, 6))
                            ci.plot()
                            plt.title(f"Best Run: {best_run_name}\nSharpe={best_sharpe:.3f}, Ann Return={ann_ret_pct:.2f}%" if ann_ret is not None else f"Best Run: {best_run_name}\nSharpe={best_sharpe:.3f}")
                            plt.xlabel("Date")
                            plt.ylabel("Cumulative Return")
                            plt.grid(True)
                            plt.tight_layout()

                            png_path = os.path.join(main_run_dir, "best_run_cumulative_return.png")
                            plt.savefig(png_path)
                            plt.close()
                            print(f"Saved best run cumulative return plot to:\n{png_path}")
                        else:
                            print("Could not find cum_index.csv for best run (expected at: {}).".format(cum_index_path))
                    else:
                        print("Best run 'outdir' field is empty; cannot locate cum_index.csv.")
                except Exception as e:
                    print("Failed to generate best run plot:", e)
            else:
                print("No valid sharpe values found in summary.")
        else:
            print("Summary is empty or 'sharpe' column missing; cannot determine best run.")

        return summary_df, main_run_dir



# -------------------------
if __name__ == "__main__":
    param_grid = {
        "strategy_kind": ["momentum"], # "momentum"
        "lookback": [5, 20, 80, 252], # 5, 20, 80, 
        "rebalance_every_n_days": [1],
        "partial_rebalance": list(np.linspace(0.1, 0.5, 5)),     # np.logspace(np.log10(0.01), np.log10(1.0), 20)), # 10 # list(np.linspace(0.1, 0.5, 5))
        "vol_window": [5, 10, 20], # 10, 20, 60
        "no_trade_band": [0.0],
        "min_trade_size": list(np.logspace(np.log10(0.001), np.log10(0.01), 10))
    }


    runner = ExperimentRunner()
    summary_df, main_dir = runner.run_grid(
            param_grid,
            strategy_kind="momentum",
            signal_param_names=("lookback", "vol_window"), 
            main_run_name_hint="strategy_comparison"
        )
    print("Main run folder:", main_dir)
    print(summary_df.head())
