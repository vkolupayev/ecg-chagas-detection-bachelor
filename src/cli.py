import argparse
import sys
from pathlib import Path
from typing import Any
import yaml
import logging

logger = logging.getLogger(__name__)

_ALL_KEYS: frozenset[str] = frozenset(
    {
        "train_mode",
        "model",
        # implemented only for ecg_founder modifications
        "use_bn",
        "use_do",
        "do_prob",
        "use_dlr",
        "use_de",
        # data augmentations
        "tm_aug",
        "lm_aug",
        "gn_aug",
        "ms_aug",
        # non-FTS
        "ls",
        "rls",
        # core training
        "bs",
        "lr",
        "weight_decay",
        "epochs",
        "early_stop_patience",
        "loss_pos_weight",
        "loss_alpha",
        "loss_gamma",
        "test_run",
        "run_name",
        "display_run_name",
    }
)

_BOOL_YAML_KEYS: frozenset[str] = frozenset(
    {
        "use_bn",
        "use_do",
        "use_dlr",
        "use_de",
        "tm_aug",
        "lm_aug",
        "gn_aug",
        "ms_aug",
        "ls",
        "rls",
        "test_run",
    }
)

_INT_YAML_KEYS: frozenset[str] = frozenset(
    {
        "bs",
        "epochs",
        "early_stop_patience",
    }
)

_FLOAT_YAML_KEYS: frozenset[str] = frozenset(
    {
        "lr",
        "weight_decay",
        "do_prob",
        "loss_pos_weight",
        "loss_alpha",
        "loss_gamma",
    }
)


def parse_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        _die(f"Config file not found: {path}")
    if not config_path.is_file():
        _die(f"Config path is not a file: {path}")

    try:
        with config_path.open() as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        _die(f"Failed to parse YAML config: {exc}")

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        _die("YAML config must be a mapping (key: value) at the top level.")

    unknown = set(raw.keys()) - _ALL_KEYS
    if unknown:
        _die(f"Unknown key(s) in config file: {', '.join(sorted(unknown))}")

    for key in _BOOL_YAML_KEYS:
        if key in raw:
            raw[key] = _coerce_bool(key, raw[key])

    for key in _INT_YAML_KEYS:
        if key in raw:
            raw[key] = _coerce_int(key, raw[key])

    for key in _FLOAT_YAML_KEYS:
        if key in raw:
            raw[key] = _coerce_float(key, raw[key])

    # run_name must be a non-empty string if provided
    if "run_name" in raw:
        if not isinstance(raw["run_name"], str) or not raw["run_name"].strip():
            _die("Config key 'run_name' must be a non-empty string.")
        raw["run_name"] = raw["run_name"].strip()

    if "display_run_name" in raw:
        if (
            not isinstance(raw["display_run_name"], str)
            or not raw["display_run_name"].strip()
        ):
            _die("Config key 'display_run_name' must be a non-empty string.")
        raw["display_run_name"] = raw["display_run_name"].strip()

    return raw


def _coerce_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False
    _die(f"Config key '{key}' must be a boolean (true/false), got: {value!r}")


def _coerce_int(key: str, value: Any) -> int:
    if isinstance(value, bool):
        _die(f"Config key '{key}' must be an integer, got a boolean: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    _die(f"Config key '{key}' must be an integer, got: {value!r}")


def _coerce_float(key: str, value: Any) -> float:
    if isinstance(value, bool):
        _die(f"Config key '{key}' must be a float, got a boolean: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    _die(f"Config key '{key}' must be a float, got: {value!r}")


def resolve_run_names(
    merged: dict[str, Any],
    config_path: str | None,
) -> tuple[str | None, str | None]:
    run_name = None
    display_run_name = None
    # 1. Explicitly provided via CLI or YAML
    if merged.get("run_name"):
        run_name = merged["run_name"]
    if merged.get("display_run_name"):
        display_run_name = merged["display_run_name"]

    # 2. Derive from the config filename stem
    if (config_path) and (run_name is None):
        stem = Path(config_path).stem.strip()
        if stem:
            run_name = stem
        if display_run_name is None:
            display_run_name = run_name

    return run_name, display_run_name


VALID_TRAIN_MODES = ("FT", "LP", "ST", "FTS")
VALID_MODELS = ("ecg_founder", "hubert_ecg")
ECG_FOUNDER_ONLY = frozenset({"use_bn", "use_do", "do_prob", "use_dlr", "use_de"})
NON_FTS_ONLY = frozenset({"ls", "rls"})


def validate_args(args: dict[str, Any], provided_keys: set[str]) -> None:
    for required in ("train_mode", "model"):
        if args.get(required) is None:
            _die(f"Required argument missing: --{required}")

    train_mode: str = args["train_mode"]
    model: str = args["model"]

    if train_mode not in VALID_TRAIN_MODES:
        _die(f"--train_mode must be one of {VALID_TRAIN_MODES}, got: {train_mode!r}")

    if model not in VALID_MODELS:
        _die(f"--model must be one of {VALID_MODELS}, got: {model!r}")

    if model != "ecg_founder":
        provided_exclusive = [
            f"--{k}" for k in sorted(ECG_FOUNDER_ONLY) if k in provided_keys
        ]
        if provided_exclusive:
            _die(
                f"The following options are only valid when --model ecg_founder: "
                f"{', '.join(provided_exclusive)}"
            )
    else:
        use_do = args.get("use_do")
        do_prob = args.get("do_prob")

        if do_prob is not None and not use_do:
            _die("--do_prob requires --use_do to be set.")

        if use_do:
            if do_prob is None:
                _die("--use_do requires --do_prob (a float in [0.1, 0.5]).")
            if not (0.1 <= do_prob <= 0.5):
                _die(f"--do_prob must be between [0.1, 0.5], " f"got: {do_prob}")

    if train_mode == "FTS":
        provided_non_fts = [
            f"--{k}" for k in sorted(NON_FTS_ONLY) if k in provided_keys
        ]
        if provided_non_fts:
            _die(
                f"The following options are not valid when --train_mode FTS: "
                f"{', '.join(provided_non_fts)}"
            )

    lr = args.get("lr")
    if lr is not None and lr < 0.0:
        _die(f"--lr must be > 0, got: {lr}")

    wd = args.get("weight_decay")
    if wd is not None and wd < 0.0:
        _die(f"--weight_decay must be >= 0, got: {wd}")

    epochs = args.get("epochs")
    if epochs is not None and epochs < 1:
        _die(f"--epochs must be >= 1, got: {epochs}")

    esp = args.get("early_stop_patience")
    if esp is not None and esp < 1:
        _die(f"--early_stop_patience must be >= 1, got: {esp}")

    lpw = args.get("loss_pos_weight")
    if lpw is not None and lpw < 1.0:
        _die(f"--loss_pos_weight must be >= 1.0, got: {lpw}")

    la = args.get("loss_alpha")
    if la is not None and (0.0 < la < 1.0):
        _die(f"--loss_alpha must be in [0, 1], got: {la}")

    lg = args.get("loss_gamma")
    if lg is not None and lg < 0.0:
        _die(f"--loss_gamma must be >= 0, got: {lg}")

    if not args.get("run_name"):
        _die(
            "--run_name is required. Provide it explicitly or via --config "
            "(the config filename stem will be used as the run name)."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_cli",
        description="Training configuration CLI with optional YAML config support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Fully via CLI
        python experiment_run.py --train_mode FT --model ecg_founder --use_bn --tm_aug --ls \\
                            --lr 1e-4 --epochs 30 --run_name founder_fine_tune_bn_tm \\
                            --display_run_name "ECG Founder FT Batch Norm Temporal Mask"

        # ecg_founder with dropout
        python experiment_run.py --train_mode FT --model ecg_founder --use_do --do_prob 0.5 \\
                            --lr 1e-4 --epochs 30 --run_name founder_ft_do_50 \\
                            --display_run_name "ECG Founder FT 50% Dropout"

        # Via YAML config (run_name defaults to config filename stem)
        python experiment_run.py --config experiments/experiment_example.yaml

        # YAML base with CLI override
        python experiment_run.py --config config.yaml --gn_aug --run_name gn_override
        """,
    )

    parser.add_argument(
        "--config",
        metavar="PATH",
        help=(
            "Path to a YAML config file. CLI flags override config values. "
            "The config filename stem is used as run_name if not set explicitly."
        ),
    )

    core = parser.add_argument_group("core arguments")
    core.add_argument(
        "--train_mode",
        choices=VALID_TRAIN_MODES,
        help=f"Training mode. Choices: {', '.join(VALID_TRAIN_MODES)}.",
    )
    core.add_argument(
        "--model",
        choices=VALID_MODELS,
        help=f"Model architecture. Choices: {', '.join(VALID_MODELS)}.",
    )
    core.add_argument(
        "--run_name",
        help=(
            "Human-readable identifier for this run. "
            "Defaults to the config filename stem when --config is used."
        ),
    )
    core.add_argument(
        "--display_run_name",
        help=(
            "Human-readable identifier for this run. "
            "Defaults to the config filename stem when --config is used."
        ),
    )
    core.add_argument(
        "--test_run",
        action="store_true",
        default=False,
        help="Dry-run mode: minimal epochs/steps to verify the pipeline end-to-end.",
    )

    hyper = parser.add_argument_group("training hyper-parameters")
    hyper.add_argument(
        "--bs",
        type=int,
        default=256,
        help="Batch size (must be >= 1).",
    )
    hyper.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (must be > 0).",
    )
    hyper.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Optimiser weight decay / L2 penalty (must be >= 0).",
    )
    hyper.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Total number of training epochs (must be >= 1).",
    )
    hyper.add_argument(
        "--early_stop_patience",
        type=int,
        default=6,
        help="Epochs without improvement before early stopping (must be >= 1).",
    )

    loss = parser.add_argument_group("loss hyper-parameters")
    loss.add_argument(
        "--loss_pos_weight",
        type=float,
        default=45.0,
        help="Positive-class weight for the loss (must be > 1.0 if specified).",
    )
    loss.add_argument(
        "--loss_alpha",
        type=float,
        default=0.0,
        help="Focal loss alpha: class-balancing factor, in [0, 1].",
    )
    loss.add_argument(
        "--loss_gamma",
        type=float,
        default=1.5,
        help="Focal loss gamma: gamma term (must be >= 0).",
    )

    ecg = parser.add_argument_group(
        "ecg_founder options",
        "Only valid when --model ecg_founder.",
    )
    ecg.add_argument(
        "--use_bn",
        action="store_true",
        default=False,
        help="Enable Batch Normalisation.",
    )
    ecg.add_argument(
        "--use_do", action="store_true", default=False, help="Enable Dropout."
    )
    ecg.add_argument(
        "--do_prob",
        type=float,
        help="Dropout Probability [0.1 - 0.5]. Requires --use_do.",
    )
    ecg.add_argument(
        "--use_dlr",
        action="store_true",
        default=False,
        help="Enable Discriminative Learning Rates.",
    )
    ecg.add_argument(
        "--use_de",
        action="store_true",
        default=False,
        help="Enable Demographic Encoder.",
    )

    aug = parser.add_argument_group("data augmentation flags")
    aug.add_argument(
        "--tm_aug",
        action="store_true",
        default=False,
        help="Enable Random Temporal Masking augmentation.",
    )
    aug.add_argument(
        "--lm_aug",
        action="store_true",
        default=False,
        help="Enable Random Lead Masking augmentation.",
    )
    aug.add_argument(
        "--gn_aug",
        action="store_true",
        default=False,
        help="Enable Random Gaussian Noise augmentation.",
    )
    aug.add_argument(
        "--ms_aug",
        action="store_true",
        default=False,
        help="Enable Random Magnitude Scaling augmentation.",
    )

    reg = parser.add_argument_group(
        "Label smoothing options",
        "Not valid when --train_mode FTS.",
    )
    reg.add_argument(
        "--ls",
        action="store_true",
        default=False,
        help="Enable Static Label Smoothing.",
    )
    reg.add_argument(
        "--rls",
        action="store_true",
        default=False,
        help="Enable Refined Label Smoothing.",
    )

    return parser


def merge_config_and_cli(
    yaml_config: dict[str, Any],
    cli_args: argparse.Namespace,
    explicit_keys: set[str],
) -> dict[str, Any]:
    cli_dict = vars(cli_args)

    # Start from argparse defaults as the baseline.
    merged: dict[str, Any] = {k: v for k, v in cli_dict.items() if k != "config"}

    # YAML overrides defaults for any key the user did NOT explicitly pass.
    for key, yaml_value in yaml_config.items():
        if key not in explicit_keys:
            merged[key] = yaml_value

    # Explicit CLI args win over everything, including the config.
    for key in explicit_keys:
        merged[key] = cli_dict[key]

    # Ensure every expected key is present for uniform downstream access.
    for key in _ALL_KEYS:
        merged.setdefault(key, None)

    return merged


def log_config(args: dict[str, Any]) -> None:
    WIDTH = 48
    logger.info("-" * WIDTH)
    logger.info(" Resolved training configuration")
    logger.info("-" * WIDTH)

    sections: list[tuple[str, list[str]]] = [
        ("Run", ["run_name", "display_run_name", "test_run"]),
        ("Core", ["train_mode", "model"]),
        (
            "Hyper-parameters",
            ["bs", "lr", "weight_decay", "epochs", "early_stop_patience"],
        ),
        ("Loss", ["loss_pos_weight", "loss_alpha", "loss_gamma"]),
        (
            "ECG Founder Modifications",
            ["use_bn", "use_do", "do_prob", "use_dlr", "use_de"],
        ),
        ("Augmentation", ["tm_aug", "lm_aug", "gn_aug", "ms_aug"]),
        ("Regularisation", ["ls", "rls"]),
    ]
    for section_name, keys in sections:
        relevant = {k: args[k] for k in keys if args.get(k) is not None}
        if relevant:
            logger.info(f"[{section_name}]")
            for k, v in relevant.items():
                logger.info(f"    {k:<22} {v}")

    logger.info("-" * WIDTH)


def _die(message: str) -> None:
    logger.error(f"error: {message}")
    sys.exit(2)
