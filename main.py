"""
main.py

Runs the complete tensile-test workflow:
    1. analyse all specimens
    2. write outputs/summary.csv
    3. write the stress-strain plots and the yield-strength boxplot
    4. print the material-wise averages
"""

from pathlib import Path

import pandas as pd

from campaign import TestCampaign


def main():

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "tensile_public_data" / "data"
    output_dir = base_dir / "outputs"

    campaign = TestCampaign(data_dir, output_dir)

    print(f"Analysing {len(campaign.specimens)} specimens...")
    campaign.run()

    summary_path = campaign.save_summary()
    campaign.plot_stress_strain_by_material()
    campaign.plot_yield_boxplot()

    print("\nPER-TEST SUMMARY")
    print("-" * 60)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(campaign.summary[[
            "test_id", "material", "youngs_modulus_GPa", "yield_strength_MPa",
            "uts_MPa", "elongation_percent",
        ]].round(2).to_string(index=False))

    print("\nMATERIAL-WISE AVERAGES")
    print("-" * 60)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(campaign.material_averages())

    if campaign.failures:
        print("\nTESTS THAT FAILED (not in the averages)")
        print("-" * 60)
        for test_id, message in campaign.failures.items():
            print(test_id, "->", message)

    print("\nFiles written to:", output_dir)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
