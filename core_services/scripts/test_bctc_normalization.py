#!/usr/bin/env python3
"""Test BCTC normalization"""

import asyncio
import sys
from pathlib import Path

# Add project root to path (must be before local imports)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core_services.data_ingestion.data_normalization import normalize_financial_report  # noqa: E402
from core_services.data_ingestion.source_adapters import VNStockAdapter  # noqa: E402


async def test():
    print("Testing BCTC Normalization")
    print("=" * 80)

    adapter = VNStockAdapter()
    result = await adapter.fetch_financial_report("VNM", year=2024, quarter=2)

    if not result.success:
        print(f"Failed: {result.error}")
        return False

    print("✅ Fetched raw data")
    raw_data = result.data

    # Normalize
    normalized = normalize_financial_report(raw_data)

    print("✅ Normalized data")
    print("\nKey metrics:")
    metrics = ["symbol", "fiscal_year", "fiscal_quarter", "revenue",
               "net_profit", "total_assets", "eps", "roe"]
    for k in metrics:
        if k in normalized:
            v = normalized[k]
            if isinstance(v, float):
                if k in ["roe", "eps"]:
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v:,.2f}")
            else:
                print(f"  {k}: {v}")

    # Validate
    core_metrics = ["revenue", "net_profit", "total_assets"]
    checks = [
        ("Core metrics", all(normalized.get(m) for m in core_metrics)),
        ("Normalized flag", normalized.get("_normalized") is True),
        ("Data type", normalized.get("_data_type") == "financial_report"),
    ]

    print("\nValidation:")
    all_ok = all(ok for _, ok in checks)
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    return all_ok


if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
