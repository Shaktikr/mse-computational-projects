"""Write the two SYNTHETIC demo datasets to data/synthetic_*/ as CSV files.

    python scripts/generate_synthetic_data.py

The CSV headers carry the 'true' parameters so you can check what the analysis recovers.
Replace these folders with your own test files (same format) to analyse real data.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from creepkit.io import save_test  # noqa: E402
from creepkit.synthetic import COMPOSITE, COMPOSITE_MATRIX, TEST_MATRIX, SyntheticAlloy, generate_dataset  # noqa: E402


def main():
    for folder, alloy, matrix in [
        ("synthetic_single_phase", SyntheticAlloy(), TEST_MATRIX),
        ("synthetic_composite", COMPOSITE, COMPOSITE_MATRIX),
    ]:
        out = ROOT / "data" / folder
        out.mkdir(parents=True, exist_ok=True)
        for t in generate_dataset(alloy, matrix):
            save_test(t, out / f"{t.specimen_id}.csv")
        print(f"wrote {len(list(out.glob('*.csv')))} curves to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
