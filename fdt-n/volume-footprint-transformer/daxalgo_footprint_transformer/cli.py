from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .backtest import run_profitability_study
from .config import EncodingConfig, ModelConfig
from .data import (
    EncodedFootprintSeries,
    build_footprint_bars,
    download_binance_trades,
    iter_binance_trades,
    write_dataset_manifest,
)
from .export import export_onnx
from .screening import evaluate_screen_candidate, prepare_screen_datasets
from .training import TrainConfig, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="daxalgo-footprint-transformer")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="download and encode footprint bars")
    prepare.add_argument("--symbol", default="BTCUSDT")
    prepare.add_argument("--start", type=date.fromisoformat, required=True)
    prepare.add_argument("--end", type=date.fromisoformat, required=True)
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--row-size", type=float, default=2.5)
    prepare.add_argument("--interval-seconds", type=int, default=60)
    prepare.add_argument("--lookback", type=int, default=64)
    prepare.add_argument("--horizon", type=int, default=8)
    prepare.add_argument("--max-rows", type=int, default=1024)
    prepare.add_argument(
        "--aggregate-trades",
        action="store_true",
        help="use aggregate-trade archives instead of live-parity individual trades",
    )

    prepare_screen = sub.add_parser(
        "prepare-screen", help="build declared multi-timeframe development datasets"
    )
    prepare_screen.add_argument("--protocol", type=Path, required=True)
    prepare_screen.add_argument("--source-manifests", type=Path, nargs="+", required=True)
    prepare_screen.add_argument("--output-dir", type=Path, required=True)
    prepare_screen.add_argument("--candidates", nargs="+", required=True)

    train = sub.add_parser("train", help="train with purged chronological splits")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--bootstrap-resamples", type=int, default=2_000)
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument(
        "--selection-only",
        action="store_true",
        help="fit and compare on validation without evaluating the reserved screen partition",
    )
    train.add_argument("--device")
    model = train.add_mutually_exclusive_group()
    model.add_argument(
        "--model-preset",
        choices=("fdt-n", "fdt-n-tiny"),
        default="fdt-n",
    )
    model.add_argument(
        "--d-model",
        type=int,
        help="use the default FDT-N architecture with a custom model width",
    )

    backtest = sub.add_parser(
        "backtest", help="run the frozen fresh-holdout profitability study"
    )
    backtest.add_argument("--preregistration", type=Path, required=True)
    backtest.add_argument("--development-dataset", type=Path, required=True)
    backtest.add_argument("--dataset", type=Path, required=True)
    backtest.add_argument("--dataset-manifest", type=Path, required=True)
    backtest.add_argument("--development-manifest", type=Path, required=True)
    backtest.add_argument("--checkpoints", type=Path, nargs=3, required=True)
    backtest.add_argument("--training-results", type=Path, nargs=3, required=True)
    backtest.add_argument("--ridge", type=Path, required=True)
    backtest.add_argument("--run-dir", type=Path, required=True)
    backtest.add_argument("--device")
    backtest.add_argument("--batch-size", type=int, default=32)

    screen = sub.add_parser(
        "screen-candidate", help="evaluate one promoted candidate on the reserved development slice"
    )
    screen.add_argument("--protocol", type=Path, required=True)
    screen.add_argument("--candidate", required=True)
    screen.add_argument("--dataset", type=Path, required=True)
    screen.add_argument("--dataset-manifest", type=Path, required=True)
    screen.add_argument("--checkpoint", type=Path, required=True)
    screen.add_argument("--selection-result", type=Path, required=True)
    screen.add_argument("--run-dir", type=Path, required=True)
    screen.add_argument("--device")
    screen.add_argument("--batch-size", type=int, default=32)

    export = sub.add_parser("export", help="export a checkpoint to ONNX")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--dataset", type=Path)
    export.add_argument("--dataset-manifest", type=Path)
    export.add_argument("--training-results", type=Path)
    export.add_argument("--instrument-key")
    export.add_argument("--source")

    report = sub.add_parser("report", help="render the frozen multi-seed research PDF")
    report.add_argument("--results", type=Path, nargs="+", required=True)
    report.add_argument("--dataset-manifest", type=Path, required=True)
    report.add_argument("--dataset", type=Path, required=True)
    report.add_argument("--model-metadata", type=Path, required=True)
    report.add_argument(
        "--paper",
        type=Path,
        default=Path("docs/research/volume-footprint-transformer/research-paper.md"),
    )
    report.add_argument(
        "--evidence-ledger",
        type=Path,
        default=Path("docs/research/volume-footprint-transformer/evidence-ledger.md"),
    )
    report.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/fdt-n-volume-footprint-transformer-research-paper.pdf"),
    )
    report.add_argument(
        "--workdir",
        type=Path,
        default=Path("tmp/pdfs/fdt-n-volume-footprint-transformer"),
    )

    publication = sub.add_parser(
        "publication",
        help="render the concise anonymous academic manuscript",
    )
    publication.add_argument("--results", type=Path, nargs="+", required=True)
    publication.add_argument("--dataset-manifest", type=Path, required=True)
    publication.add_argument("--dataset", type=Path, required=True)
    publication.add_argument("--model-metadata", type=Path, required=True)
    publication.add_argument(
        "--manuscript",
        type=Path,
        default=Path("docs/research/volume-footprint-transformer/publication-manuscript.md"),
    )
    publication.add_argument(
        "--evidence-ledger",
        type=Path,
        default=Path("docs/research/volume-footprint-transformer/evidence-ledger.md"),
    )
    publication.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/fdt-n-volume-footprint-transformer-publication-manuscript.pdf"),
    )
    publication.add_argument(
        "--workdir",
        type=Path,
        default=Path("tmp/pdfs/fdt-n-volume-footprint-transformer-publication"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        config = EncodingConfig(
            lookback_bars=args.lookback,
            horizon_bars=args.horizon,
            max_rows=args.max_rows,
            row_size=args.row_size,
            interval_seconds=args.interval_seconds,
        )
        csv_paths = download_binance_trades(
            args.symbol,
            args.start,
            args.end,
            args.data_dir,
            aggregate=args.aggregate_trades,
        )
        bars = build_footprint_bars(iter_binance_trades(csv_paths, config.quantity_scale), config)
        series = EncodedFootprintSeries.from_bars(bars, config)
        series.save(args.output)
        source_name = (
            "Binance Data Vision spot daily aggregate trades"
            if args.aggregate_trades
            else "Binance Data Vision spot daily individual trades"
        )
        write_dataset_manifest(
            args.output.with_suffix(".manifest.json"),
            csv_paths,
            series,
            source_name,
        )
        print(json.dumps({"bars": len(series), "output": str(args.output)}, sort_keys=True))
        return 0
    if args.command == "prepare-screen":
        result = prepare_screen_datasets(
            args.protocol,
            args.source_manifests,
            args.output_dir,
            args.candidates,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.command == "train":
        series = EncodedFootprintSeries.load(args.dataset)
        model_config = ModelConfig.from_preset(args.model_preset)
        if args.d_model is not None:
            model_config = ModelConfig(**{**model_config.to_dict(), "d_model": args.d_model})
        result = train_model(
            series,
            args.run_dir,
            TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=args.seed,
                bootstrap_resamples=args.bootstrap_resamples,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                selection_only=args.selection_only,
            ),
            model_config,
            args.device,
        )
        print(json.dumps({"best_epoch": result["best_epoch"], "run": str(args.run_dir)}))
        return 0
    if args.command == "backtest":
        result = run_profitability_study(
            preregistration_path=args.preregistration,
            development_dataset_path=args.development_dataset,
            dataset_path=args.dataset,
            dataset_manifest_path=args.dataset_manifest,
            development_manifest_path=args.development_manifest,
            checkpoint_paths=args.checkpoints,
            training_results_paths=args.training_results,
            ridge_path=args.ridge,
            output_dir=args.run_dir,
            batch_size=args.batch_size,
            device_name=args.device,
        )
        primary = result["financial_metrics"]["fdt_n_tiny_ensemble_primary"]
        print(
            json.dumps(
                {
                    "net_return": primary["net_return"],
                    "run": str(args.run_dir),
                    "trades": primary["trades"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "screen-candidate":
        result = evaluate_screen_candidate(
            args.protocol,
            args.candidate,
            args.dataset,
            args.dataset_manifest,
            args.checkpoint,
            args.selection_result,
            args.run_dir,
            batch_size=args.batch_size,
            device_name=args.device,
        )
        primary = result["financial_metrics"]["primary"]
        print(
            json.dumps(
                {
                    "candidate": args.candidate,
                    "net_return": primary["net_return"],
                    "status": result["status"],
                    "trades": primary["trades"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "export":
        series = EncodedFootprintSeries.load(args.dataset) if args.dataset else None
        metadata = export_onnx(
            args.checkpoint,
            args.output_dir,
            series,
            instrument_key=args.instrument_key,
            source=args.source,
            dataset_path=args.dataset,
            dataset_manifest=args.dataset_manifest,
            training_results=args.training_results,
        )
        print(json.dumps(metadata, sort_keys=True))
        return 0
    if args.command == "report":
        from .report import PendingResultsError, generate_pdf_report

        try:
            metadata = generate_pdf_report(
                args.results,
                args.dataset_manifest,
                args.dataset,
                args.paper,
                args.evidence_ledger,
                args.model_metadata,
                args.output,
                args.workdir,
            )
        except (PendingResultsError, FileNotFoundError, ValueError) as exception:
            print(str(exception), file=sys.stderr)
            return 2
        print(json.dumps(metadata, sort_keys=True))
        return 0
    if args.command == "publication":
        from .publication import generate_publication_pdf
        from .report import PendingResultsError

        try:
            metadata = generate_publication_pdf(
                args.results,
                args.dataset_manifest,
                args.dataset,
                args.manuscript,
                args.evidence_ledger,
                args.model_metadata,
                args.output,
                args.workdir,
            )
        except (
            PendingResultsError,
            FileExistsError,
            FileNotFoundError,
            RuntimeError,
            ValueError,
        ) as exception:
            print(str(exception), file=sys.stderr)
            return 2
        print(json.dumps(metadata, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
