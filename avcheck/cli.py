"""avcheck command-line entry point."""

import argparse
import csv
import sys

import librosa

from avcheck.audio.integrity import (
    compare_chromagram,
    compute_dynamic_range_delta,
    compute_loudness_delta,
    detect_clipping,
    detect_silence_dropouts,
)
from avcheck.evaluate.evaluation import evaluate_variants, write_evaluation_csv
from avcheck.inject.engine import inject_all_defects
from avcheck.video.plotting import plot_quality_curve
from avcheck.video.quality import score_video


def _run_audio_checks(ref_path: str, test_path: str) -> dict:
    ref, sr = librosa.load(ref_path, sr=None, mono=True)
    test, _ = librosa.load(test_path, sr=sr, mono=True)

    return {
        "clipping_ref": detect_clipping(ref, sr),
        "clipping_test": detect_clipping(test, sr),
        "loudness": compute_loudness_delta(ref, test, sr),
        "dropouts": detect_silence_dropouts(ref, test, sr),
        "dynamic_range": compute_dynamic_range_delta(ref, test, sr),
        "chromagram": compare_chromagram(ref, test, sr),
    }


def _write_csv(results: dict, output_path: str) -> None:
    loudness = results["loudness"]
    worst_time, worst_delta = loudness["worst_window"] if loudness["worst_window"] else ("", "")

    rows = [
        ("clipping_ref", "num_clipped_samples", results["clipping_ref"]["num_clipped_samples"]),
        ("clipping_ref", "clipped_ratio", results["clipping_ref"]["clipped_ratio"]),
        ("clipping_ref", "num_clipped_regions", results["clipping_ref"]["num_clipped_regions"]),
        ("clipping_test", "num_clipped_samples", results["clipping_test"]["num_clipped_samples"]),
        ("clipping_test", "clipped_ratio", results["clipping_test"]["clipped_ratio"]),
        ("clipping_test", "num_clipped_regions", results["clipping_test"]["num_clipped_regions"]),
        ("loudness", "global_delta_db", loudness["global_delta_db"]),
        ("loudness", "worst_window_time_sec", worst_time),
        ("loudness", "worst_window_delta_db", worst_delta),
        ("dropouts", "num_dropout_regions", results["dropouts"]["num_dropout_regions"]),
        ("dynamic_range", "ref_dynamic_range_db", results["dynamic_range"]["ref_dynamic_range_db"]),
        ("dynamic_range", "test_dynamic_range_db", results["dynamic_range"]["test_dynamic_range_db"]),
        ("dynamic_range", "dynamic_range_delta_db", results["dynamic_range"]["dynamic_range_delta_db"]),
        ("chromagram", "mean_chroma_similarity", results["chromagram"]["mean_chroma_similarity"]),
        ("chromagram", "worst_frame_similarity", results["chromagram"]["worst_frame_similarity"]),
        ("chromagram", "worst_frame_time_sec", results["chromagram"]["worst_frame_time_sec"]),
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "metric", "value"])
        writer.writerows(rows)
        for label, key in (("clipping_ref", "clipping_ref"), ("clipping_test", "clipping_test")):
            for start, end in results[key]["clipped_regions"]:
                writer.writerow([label, "clipped_region", f"{start:.3f}-{end:.3f}"])
        for start, end in results["dropouts"]["dropout_regions"]:
            writer.writerow(["dropouts", "dropout_region", f"{start:.3f}-{end:.3f}"])


def _print_summary(results: dict) -> None:
    dr = results["dynamic_range"]
    chroma = results["chromagram"]
    print("=== AVCheck Audio Report ===")
    print(
        f"Clipping (ref):  {results['clipping_ref']['num_clipped_regions']} region(s), "
        f"{results['clipping_ref']['clipped_ratio'] * 100:.3f}% of samples"
    )
    print(
        f"Clipping (test): {results['clipping_test']['num_clipped_regions']} region(s), "
        f"{results['clipping_test']['clipped_ratio'] * 100:.3f}% of samples"
    )
    print(f"Loudness delta:  {results['loudness']['global_delta_db']:+.2f} dB")
    print(f"Dropout regions: {results['dropouts']['num_dropout_regions']}")
    print(
        f"Dynamic range:   ref {dr['ref_dynamic_range_db']:.2f} dB -> "
        f"test {dr['test_dynamic_range_db']:.2f} dB ({dr['dynamic_range_delta_db']:+.2f} dB)"
    )
    print(
        f"Chroma similarity: mean {chroma['mean_chroma_similarity']:.3f}, "
        f"worst {chroma['worst_frame_similarity']:.3f} @ {chroma['worst_frame_time_sec']:.2f}s"
    )


def _write_video_csv(per_frame: list, output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timestamp_sec", "psnr_db", "ssim"])
        for row in per_frame:
            writer.writerow([row["frame"], f"{row['timestamp']:.4f}", row["psnr"], row["ssim"]])


def _print_video_summary(summary: dict) -> None:
    print("=== AVCheck Video Report ===")
    print(f"Frames scored:   {summary['num_frames']}")
    print(f"Mean PSNR:       {summary['mean_psnr']:.2f} dB")
    print(f"Min PSNR:        {summary['min_psnr']:.2f} dB")
    print(f"Mean SSIM:       {summary['mean_ssim']:.4f}")
    if summary["worst_frame_index"] is not None:
        print(
            f"Min SSIM:        {summary['min_ssim']:.4f} @ frame {summary['worst_frame_index']} "
            f"({summary['worst_frame_timestamp']:.2f}s)"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avcheck", description="Automated audio/video validation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("audio", help="Run audio integrity checks")
    audio_parser.add_argument("reference", help="Path to reference audio file")
    audio_parser.add_argument("test", help="Path to processed/degraded audio file")
    audio_parser.add_argument("-o", "--output", default="avcheck_audio_report.csv", help="CSV output path")

    video_parser = subparsers.add_parser("video", help="Run video quality scoring (PSNR/SSIM)")
    video_parser.add_argument("reference", help="Path to reference video file")
    video_parser.add_argument("test", help="Path to processed/degraded video file")
    video_parser.add_argument("-o", "--output", default="avcheck_video_report.csv", help="Per-frame CSV output path")
    video_parser.add_argument("--plot", default="avcheck_quality_curve.png", help="Quality-curve plot output path")

    inject_parser = subparsers.add_parser("inject", help="Generate ground-truth-labeled defect variants")
    inject_parser.add_argument("reference_video", help="Path to reference video file")
    inject_parser.add_argument("reference_audio", help="Path to reference audio file")
    inject_parser.add_argument("-o", "--output-dir", default="avcheck_variants", help="Output directory for variants")
    inject_parser.add_argument("--seed", type=int, default=0, help="Random seed for defect placement")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run all detectors against all injected variants")
    evaluate_parser.add_argument("manifest", help="Path to manifest.json produced by `avcheck inject`")
    evaluate_parser.add_argument("reference_video", help="Path to reference video file")
    evaluate_parser.add_argument("reference_audio", help="Path to reference audio file")
    evaluate_parser.add_argument("-o", "--output", default="avcheck_evaluation.csv", help="CSV output path")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "audio":
        results = _run_audio_checks(args.reference, args.test)
        _write_csv(results, args.output)
        _print_summary(results)
        print(f"\nCSV report written to {args.output}")
        return 0

    if args.command == "video":
        result = score_video(args.reference, args.test)
        _write_video_csv(result["per_frame"], args.output)
        plot_quality_curve(result["per_frame"], args.plot)
        _print_video_summary(result["summary"])
        print(f"\nPer-frame CSV written to {args.output}")
        print(f"Quality-curve plot written to {args.plot}")
        return 0

    if args.command == "inject":
        manifest = inject_all_defects(args.reference_video, args.reference_audio, args.output_dir, seed=args.seed)
        print(f"=== AVCheck Injection Report ===\nGenerated {len(manifest['variants'])} labeled variants in {args.output_dir}:")
        for variant in manifest["variants"]:
            print(f"  {variant['name']:<16} ({variant['media']}) -> {variant['file']}")
        print(f"\nManifest written to {manifest['manifest_path']}")
        return 0

    if args.command == "evaluate":
        report = evaluate_variants(args.manifest, args.reference_video, args.reference_audio)
        write_evaluation_csv(report, args.output)
        print("=== AVCheck Evaluation Report ===")
        print(f"{'defect_class':<16}{'precision':>10}{'recall':>10}{'tp':>6}{'fp':>6}{'fn':>6}")
        for name, r in report.items():
            print(f"{name:<16}{r['precision']:>10.3f}{r['recall']:>10.3f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")
        print(f"\nCSV report written to {args.output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
