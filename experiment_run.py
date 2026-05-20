import os
from argparse import SUPPRESS
from src.logger import setup_logger

setup_logger(f"{os.path.splitext(os.path.basename(__file__))[0]}")
from src.cli import (
    build_parser,
    parse_yaml_config,
    merge_config_and_cli,
    validate_args,
    log_config,
    resolve_run_names,
)
from src.experiment import run_experiment


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    cli_args = parser.parse_args(argv)

    # parse a second time with all defaults suppressed
    # args specified vs deffault
    shadow_parser = build_parser()
    for action in shadow_parser._actions:
        action.default = SUPPRESS
    explicit_keys: set[str] = set(vars(shadow_parser.parse_args(argv)).keys())
    explicit_keys.discard("config")

    yaml_config = {}
    if cli_args.config:
        yaml_config = parse_yaml_config(cli_args.config)
    merged_args = merge_config_and_cli(yaml_config, cli_args, explicit_keys)

    merged_args["run_name"], merged_args["display_run_name"] = resolve_run_names(
        merged_args, cli_args.config
    )

    provided_keys = explicit_keys | yaml_config.keys()
    validate_args(merged_args, provided_keys)

    log_config(merged_args)
    if merged_args["test_run"]:
        merged_args["epochs"] = 1
    run_experiment(merged_args)


if __name__ == "__main__":
    main()
