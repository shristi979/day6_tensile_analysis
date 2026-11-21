"""
campaign.py

TestCampaign runs the full analysis on every specimen listed in specimens.csv,
then writes outputs/summary.csv and the plots.

No test code lives in this file. Run main.py to execute the workflow.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from specimen import Specimen, TensileTest


class TestCampaign:
    # Stop pytest from trying to collect this class as a test case
    __test__ = False

    def __init__(self, data_dir, output_dir):

        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.specimens = self.load_specimens()

        self.summary = pd.DataFrame()   # one row per successful test
        self.curves = {}                # test_id -> stress-strain DataFrame
        self.failures = {}              # test_id -> error message

    
    # Loading
    

    def load_specimens(self):
        """Read specimens.csv and build a list of Specimen objects."""

        meta = pd.read_csv(self.data_dir / "specimens.csv")

        specimens = []
        for _, row in meta.iterrows():
            specimens.append(
                Specimen(
                    test_id=row["test_id"],
                    material=row["material"],
                    area=row["area_mm2"],
                    gauge_length=row["gauge_length_mm"],
                )
            )
        return specimens

   
    # Running the analysis
   

    def run(self):
        """
        Analyse every specimen. A test that fails is recorded in
        self.failures with its error message and does not stop the others.
        """

        rows = []
        self.curves = {}
        self.failures = {}

        for sp in self.specimens:

            file_path = self.raw_dir / f"{sp.test_id}.csv"

            try:
                test = TensileTest(sp, file_path, verbose=False)
                props = test.calculate_all_properties()

                # Extra columns so the cleaning and the fracture definition
                # are visible in the summary
                props["duplicates_removed"] = test.cleaning_report.get("duplicates_removed")
                props["missing_rows_removed"] = test.cleaning_report.get("missing_rows_removed")
                props["spikes_removed"] = test.cleaning_report.get("spikes_removed")
                props["slack_rows_removed"] = test.cleaning_report.get("slack_rows_removed")
                props["fracture_method"] = test.fracture_info.get("method")

                rows.append(props)
                self.curves[sp.test_id] = test.calculate_stress_strain()

            except Exception as err:
                self.failures[sp.test_id] = str(err)

        self.summary = pd.DataFrame(rows)
        return self.summary

   
    # Outputs
   

    def save_summary(self, filename="summary.csv"):
        """Write the per-test results to outputs/summary.csv."""

        path = self.output_dir / filename
        self.summary.to_csv(path, index=False)
        return path

    def material_averages(self):
        """Mean of each property per material, with the number of tests."""

        cols = [
            "youngs_modulus_GPa",
            "yield_strength_MPa",
            "uts_MPa",
            "elongation_percent",
        ]

        grouped = self.summary.groupby("material")
        averages = grouped[cols].mean().round(2)
        averages.insert(0, "n_tests", grouped.size())
        return averages

    def plot_stress_strain_by_material(self):
        """One figure per material with all of its stress-strain curves."""

        paths = []

        for material, group in self.summary.groupby("material"):

            fig, ax = plt.subplots(figsize=(10, 6))

            for test_id in group["test_id"]:
                curve = self.curves[test_id]
                ax.plot(curve["strain"], curve["stress_MPa"],
                        linewidth=1, label=test_id)

            ax.set_xlabel("Engineering Strain")
            ax.set_ylabel("Engineering Stress (MPa)")
            ax.set_title(f"Stress-Strain Curves - {material}")
            ax.grid(True)
            ax.legend(fontsize=7, ncol=2)

            path = self.output_dir / f"stress_strain_{material}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

        return paths

    def plot_yield_boxplot(self):
        """Boxplot of 0.2% offset yield strength for each material."""

        materials = sorted(self.summary["material"].unique())
        data = [
            self.summary.loc[self.summary["material"] == m, "yield_strength_MPa"]
            for m in materials
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data)
        ax.set_xticks(range(1, len(materials) + 1))
        ax.set_xticklabels(materials)
        ax.set_xlabel("Material")
        ax.set_ylabel("0.2% Offset Yield Strength (MPa)")
        ax.set_title("Yield Strength by Material")
        ax.grid(True, axis="y")

        path = self.output_dir / "yield_strength_boxplot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
