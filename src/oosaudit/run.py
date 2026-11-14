"""Train one (model, protocol, seed) cell with Qlib's official configuration.

Only four things differ from the official benchmark workflow:

* the data come from the cached raw Alpha158 store (``data.py``) instead of
  being recomputed for every run;
* the handler, the processor fitting window and the segments follow the
  chosen protocol (``PROTOCOLS``), and the test segment runs to the end of the
  sample instead of stopping at 2020-08-01;
* thread counts are capped at the machine's cores;
* stochastic models get an explicit seed.

Each run writes its test-set scores to
``results/preds/<protocol>/<model>/seed<k>.parquet`` and a JSON sidecar with
timings and settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import pandas as pd

from . import data

# Qlib records runs through MLflow's file store; keep it local and quiet.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "qlib"
PRED_DIR = Path(os.environ.get("OOSAUDIT_RESULTS", ROOT / "results")) / "preds"

PROTOCOLS: dict[str, dict[str, tuple[str, str]]] = {
    # The official segments; the test segment is extended to the end of the data.
    "pub": {
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", data.END),
    },
    # The same window lengths, moved forward six years.
    "refit": {
        "train": ("2014-01-01", "2020-12-31"),
        "valid": ("2021-01-01", "2022-12-31"),
        "test": ("2023-01-01", data.END),
    },
}

# Annual refits: for test year Y, train on [Y-9, Y-3] and validate on [Y-2, Y-1].
ROLL_YEARS = tuple(range(2017, 2027))


def roll_segments(year: int) -> dict[str, tuple[str, str]]:
    last = data.END if year == int(data.END[:4]) else f"{year}-12-31"
    return {
        "train": (f"{year - 9}-01-01", f"{year - 3}-12-31"),
        "valid": (f"{year - 2}-01-01", f"{year - 1}-12-31"),
        "test": (f"{year}-01-01", last),
    }


CONFIGS = {
    "linear": "workflow_config_linear_Alpha158.yaml",
    "lightgbm": "workflow_config_lightgbm_Alpha158.yaml",
    "xgboost": "workflow_config_xgboost_Alpha158.yaml",
    "catboost": "workflow_config_catboost_Alpha158.yaml",
    "doubleensemble": "workflow_config_doubleensemble_Alpha158.yaml",
    "mlp": "workflow_config_mlp_Alpha158.yaml",
    "gru": "workflow_config_gru_Alpha158.yaml",
    "lstm": "workflow_config_lstm_Alpha158.yaml",
    "alstm": "workflow_config_alstm_Alpha158.yaml",
    "transformer": "workflow_config_transformer_Alpha158.yaml",
    "localformer": "workflow_config_localformer_Alpha158.yaml",
    "tcn": "workflow_config_tcn_Alpha158.yaml",
}

# The models whose output does not depend on a seed.
DETERMINISTIC = {"linear"}

_DEFAULT_LEARN = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

# Keys that set threads or devices, per model class; values are capped below.
_THREAD_KEYS = ("num_threads", "nthread", "thread_count", "n_jobs")


# Qlib-side processors swapped for semantically identical, leaner versions.
_REPLACED_PROCESSORS = {"CSZFillna": "oosaudit.processors"}


def load_official(model: str) -> dict:
    """Parse an official config the way ``qrun`` does (ruamel, YAML 1.2)."""
    from ruamel.yaml import YAML

    with open(CONFIG_DIR / CONFIGS[model]) as f:
        return YAML(typ="safe", pure=True).load(f)


def _swap_processors(procs: list) -> list:
    out = []
    for p in procs:
        name = p if isinstance(p, str) else p.get("class")
        if name in _REPLACED_PROCESSORS:
            p = {
                "class": name,
                "module_path": _REPLACED_PROCESSORS[name],
                "kwargs": {} if isinstance(p, str) else dict(p.get("kwargs", {})),
            }
        out.append(p)
    return out


def model_config(model: str, seed: int, threads: int) -> dict:
    """The official ``task.model`` block with a seed and a thread cap."""
    cfg = copy.deepcopy(load_official(model)["task"]["model"])
    kw = cfg.setdefault("kwargs", {})
    for key in _THREAD_KEYS:
        if key in kw:
            kw[key] = threads
    if model in DETERMINISTIC:
        return cfg
    cls = cfg["class"]
    if cls in ("LGBModel", "DEnsembleModel"):
        kw["seed"] = seed
        kw.setdefault("verbosity", -1)
    elif cls == "XGBModel":
        kw["seed"] = seed
    elif cls == "CatBoostModel":
        kw["random_seed"] = seed
        # Poisson bootstrap exists only on GPU; Bernoulli keeps row subsampling on CPU.
        if kw.get("bootstrap_type") == "Poisson":
            kw["bootstrap_type"] = "Bernoulli"
    else:  # Qlib's PyTorch models
        kw["seed"] = seed
    return cfg


def segments_for(protocol: str, year: int | None = None) -> dict[str, tuple[str, str]]:
    if protocol == "roll":
        assert year is not None
        return roll_segments(year)
    return PROTOCOLS[protocol]


def handler_config(model: str, protocol: str, year: int | None = None) -> dict:
    """A DataHandlerLP over the cached store with the official processors."""
    from qlib.contrib.data.handler import check_transform_proc

    official = load_official(model)["data_handler_config"]
    seg = segments_for(protocol, year)
    fit_start, fit_end = seg["train"]
    infer = _swap_processors(official.get("infer_processors", []))
    learn = _swap_processors(official.get("learn_processors", _DEFAULT_LEARN))
    kwargs = {
        "instruments": None,
        "start_time": fit_start,
        "end_time": seg["test"][1],
        "data_loader": {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {"config": str(data.raw_store_path())},
        },
        "infer_processors": check_transform_proc(copy.deepcopy(infer), fit_start, fit_end),
        "learn_processors": check_transform_proc(copy.deepcopy(learn), fit_start, fit_end),
    }
    if "process_type" in official:
        kwargs["process_type"] = official["process_type"]
    return {"class": "DataHandlerLP", "module_path": "qlib.data.dataset.handler", "kwargs": kwargs}


def dataset_config(model: str, protocol: str, year: int | None = None) -> dict:
    official = load_official(model)["task"]["dataset"]
    seg = segments_for(protocol, year)
    kwargs = {
        "handler": handler_config(model, protocol, year),
        "segments": {k: list(v) for k, v in seg.items()},
    }
    if "step_len" in official.get("kwargs", {}):
        kwargs["step_len"] = official["kwargs"]["step_len"]
    return {"class": official["class"], "module_path": "qlib.data.dataset", "kwargs": kwargs}


def pred_path(model: str, protocol: str, seed: int) -> Path:
    return PRED_DIR / protocol / model / f"seed{seed}.parquet"


def _torch_compat() -> None:
    """Accept the ``verbose`` argument Qlib passes to ``ReduceLROnPlateau``.

    Newer PyTorch removed it; it only toggled log messages.
    """
    import inspect

    import torch

    cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    if "verbose" in inspect.signature(cls.__init__).parameters or getattr(cls, "_oosaudit_patched", False):
        return
    init = cls.__init__

    def patched(self, *args, verbose=None, **kwargs):
        init(self, *args, **kwargs)

    cls.__init__ = patched
    cls._oosaudit_patched = True


def fit_predict(model: str, seed: int, threads: int, dataset, experiment: str) -> tuple[pd.Series, dict, float]:
    """Fit the official model on ``dataset`` and score its test segment."""
    import torch
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    _torch_compat()
    torch.set_num_threads(threads)
    mcfg = model_config(model, seed, threads)
    m = init_instance_by_config(mcfg)
    t0 = time.time()
    with R.start(experiment_name=experiment, uri=str(data.CACHE_DIR / "mlruns")):
        m.fit(dataset)
        fit_s = time.time() - t0
        pred = m.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    return pred.rename("score").astype("float32"), mcfg, fit_s


def run_roll(model: str, seeds: list[int], threads: int) -> list[Path]:
    """Annual refits over ``ROLL_YEARS``; one concatenated file per seed."""
    from qlib.utils import init_instance_by_config

    parts: dict[int, list[pd.Series]] = {s: [] for s in seeds}
    fit_s: dict[int, float] = {s: 0.0 for s in seeds}
    mcfgs: dict[int, dict] = {}
    for year in ROLL_YEARS:
        ds = init_instance_by_config(dataset_config(model, "roll", year))
        for s in seeds:
            pred, mcfgs[s], sec = fit_predict(model, s, threads, ds, f"roll-{model}")
            parts[s].append(pred)
            fit_s[s] += sec
        print(f"{model} roll {year} done", flush=True)
    outs = []
    for s in seeds:
        out = pred_path(model, "roll", s)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred = pd.concat(parts[s]).sort_index()
        pred.to_frame().to_parquet(out)
        meta = {
            "model": model,
            "protocol": "roll",
            "seed": s,
            "model_config": mcfgs[s],
            "segments": {y: roll_segments(y) for y in ROLL_YEARS},
            "rows": int(len(pred)),
            "seconds_fit": round(fit_s[s], 1),
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
        outs.append(out)
    return outs


def run_one(model: str, protocol: str, seed: int, threads: int = 2, dataset=None) -> Path:
    from qlib.utils import init_instance_by_config

    out = pred_path(model, protocol, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if dataset is None:
        dataset = init_instance_by_config(dataset_config(model, protocol))
    pred, mcfg, fit_s = fit_predict(model, seed, threads, dataset, f"{protocol}-{model}")
    pred.to_frame().to_parquet(out)
    meta = {
        "model": model,
        "protocol": protocol,
        "seed": seed,
        "model_config": mcfg,
        "segments": PROTOCOLS[protocol],
        "rows": int(len(pred)),
        "seconds_fit": round(fit_s, 1),
        "seconds_total": round(time.time() - t0, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=sorted(CONFIGS))
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS) + ["roll"], default="pub")
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from qlib.utils import init_instance_by_config

    data.init_qlib(1)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.model in DETERMINISTIC:
        seeds = seeds[:1]
    todo = [s for s in seeds if args.force or not pred_path(args.model, args.protocol, s).exists()]
    if not todo:
        return
    if args.protocol == "roll":
        for p in run_roll(args.model, todo, args.threads):
            meta = json.loads(p.with_suffix(".json").read_text())
            print(f"{args.model} roll seed={meta['seed']} fit={meta['seconds_fit']}s rows={meta['rows']}", flush=True)
        return
    dataset = init_instance_by_config(dataset_config(args.model, args.protocol))
    for s in todo:
        p = run_one(args.model, args.protocol, s, args.threads, dataset=dataset)
        meta = json.loads(p.with_suffix(".json").read_text())
        print(f"{args.model} {args.protocol} seed={s} fit={meta['seconds_fit']}s rows={meta['rows']}", flush=True)


if __name__ == "__main__":
    main()
