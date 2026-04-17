"""Smoke test for curated V1 vs V2 regression benchmark (not full benchmarks)."""

from __future__ import annotations

from experiments.benchmark_v1_v2_regression import run_benchmark


def test_regression_benchmark_all_contracts_pass():
    report = run_benchmark()
    assert report["cases_passed_contracts"] == report["cases_total"]
