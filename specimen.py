
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


#SPECIMEN DATACLASS


@dataclass
class Specimen:
    test_id: str
    material: str
    area: float            # cross-section area, mm^2
    gauge_length: float    # mm

#TENSILE TEST CLASS


class TensileTest:

    def __init__(self, specimen, file_path, verbose=True):

        self.specimen = specimen
        self.file_path = Path(file_path)
        self.verbose = verbose

        # Working data (modified by the cleaning methods)
        self.data = self.load_data()

        # Bookkeeping
        self.cleaning_report = {"original_rows": len(self.data)}
        self.removed_spikes = pd.DataFrame()
        self.modulus_info = {}
        self.yield_info = {}
        self.fracture_info = {}
        self._cleaned = False

    
    # Loading
   

    def load_data(self):
        
        return pd.read_csv(self.file_path)

   
    # Cleaning
 
    def clean_basic_data(self):


        df = self.data.copy()
        n_start = len(df)

        df = df.drop_duplicates()
        duplicates_removed = n_start - len(df)

        n_before_na = len(df)
        df = df.dropna()
        missing_removed = n_before_na - len(df)

        df = df.reset_index(drop=True)
        self.data = df

        self.cleaning_report["duplicates_removed"] = duplicates_removed
        self.cleaning_report["missing_rows_removed"] = missing_removed

        if self.verbose:
            print("\nDATA CLEANING REPORT -", self.specimen.test_id)
            print("-" * 40)
            print("Original rows:", n_start)
            print("Duplicates removed:", duplicates_removed)
            print("Rows with missing values removed:", missing_removed)
            print("Remaining rows:", len(df))

        return df

    def standardize_displacement(self):


        df = self.data.copy()

        if "displacement_mm" in df.columns:
            pass
        elif "displacement_m" in df.columns:
            df["displacement_mm"] = df["displacement_m"] * 1000.0
        else:
            raise ValueError(
                f"{self.specimen.test_id}: no displacement column found "
                "(expected 'displacement_mm' or 'displacement_m')."
            )

        # Keep a single displacement column so the units are never ambiguous
        df = df.drop(columns=["displacement_m"], errors="ignore")

        self.data = df
        return df

    def remove_sensor_spikes(self, window=11, n_sigma=6.0):

        df = self.data.copy()

        rolling_median = df["force_N"].rolling(
            window=window, center=True, min_periods=window
        ).median()

        residual = df["force_N"] - rolling_median
        valid = residual.notna()

        if valid.sum() == 0:
            return df

        res_valid = residual[valid]
        mad = np.median(np.abs(res_valid - np.median(res_valid)))

        if mad == 0:
            spike_mask = pd.Series(False, index=df.index)
        else:
            robust_sigma = 1.4826 * mad
            spike_mask = valid & (residual.abs() > n_sigma * robust_sigma)

        self.removed_spikes = df.loc[spike_mask].copy()

        df = df.loc[~spike_mask].reset_index(drop=True)
        self.data = df

        self.cleaning_report["spikes_removed"] = int(spike_mask.sum())

        if self.verbose:
            print("\nSENSOR SPIKE REPORT -", self.specimen.test_id)
            print("-" * 40)
            print("Spikes removed:", int(spike_mask.sum()))
            print("Remaining rows:", len(df))

        return df

    def remove_initial_slack(self, threshold_frac=0.02):

        df = self.data.copy()

        threshold = threshold_frac * df["force_N"].max()
        above = df.index[df["force_N"] >= threshold]

        if len(above) == 0:
            n_removed = 0
        else:
            first_valid = above[0]
            n_removed = int(first_valid)  # index is 0..n-1 after reset
            df = df.iloc[first_valid:].reset_index(drop=True)

        self.data = df
        self.cleaning_report["slack_rows_removed"] = n_removed

        if self.verbose:
            print("\nINITIAL SLACK REPORT -", self.specimen.test_id)
            print("-" * 40)
            print("Force threshold (N):", round(threshold, 2))
            print("Leading rows removed:", n_removed)
            print("Remaining rows:", len(df))

        return df

    def clean(self):
       

        self.clean_basic_data()
        self.standardize_displacement()
        self.remove_sensor_spikes()
        self.remove_initial_slack()

        self.cleaning_report["final_rows"] = len(self.data)
        self._cleaned = True

        return self.data

  
    # Stress and strain
   

    def calculate_stress_strain(self):

        if not self._cleaned:
            self.clean()

        df = self.data.copy()

        df["stress_MPa"] = df["force_N"] / self.specimen.area
        df["strain"] = df["displacement_mm"] / self.specimen.gauge_length

        return df

    def _get_curve(self, df):
        """Return a stress-strain DataFrame with a clean 0..n-1 index."""
        if df is None:
            df = self.calculate_stress_strain()
        return df.reset_index(drop=True)

   
    # Mechanical properties
   

    def calculate_youngs_modulus(self, df=None, lower_frac=0.15, upper_frac=0.40):


        df = self._get_curve(df)

        max_stress = df["stress_MPa"].max()
        lo = lower_frac * max_stress
        hi = upper_frac * max_stress

        reached = df.index[df["stress_MPa"] >= hi]
        if len(reached) == 0:
            raise ValueError("Stress never reaches the upper bound of the band.")
        first_hi = reached[0]

        loading = df.loc[:first_hi]
        band = loading[
            (loading["stress_MPa"] >= lo) & (loading["stress_MPa"] <= hi)
        ]

        if len(band) < 5:
            raise ValueError(
                f"{self.specimen.test_id}: only {len(band)} points in the "
                "fitting band (need at least 5)."
            )

        x = band["strain"].to_numpy()
        y = band["stress_MPa"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)

        y_fit = slope * x + intercept
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        self.modulus_info = {
            "band_MPa": (round(lo, 1), round(hi, 1)),
            "n_points": len(x),
            "R2": r2,
            "slope_MPa": slope,
            "intercept_MPa": intercept,
            "end_index": int(first_hi),
        }

        return slope / 1000.0

    def calculate_yield_strength(self, df=None, offset=0.002):
  

        df = self._get_curve(df)

        self.calculate_youngs_modulus(df)

        slope = self.modulus_info["slope_MPa"]
        intercept = self.modulus_info["intercept_MPa"]
        start = self.modulus_info["end_index"]

        strain = df["strain"].to_numpy()
        stress = df["stress_MPa"].to_numpy()

        # Positive while the curve is above the offset line, <= 0 after yield
        offset_line = slope * (strain - offset) + intercept
        diff = stress - offset_line

        after = np.where(diff[start:] <= 0)[0]
        if len(after) == 0:
            raise ValueError(
                f"{self.specimen.test_id}: curve never crosses the "
                f"{offset * 100:.1f}% offset line (no yield found)."
            )

        j = start + after[0]
        if j == start:
            raise ValueError("Crossing found at the start of the search; check the data.")

        # Linear interpolation between j-1 (above the line) and j (below it)
        frac = diff[j - 1] / (diff[j - 1] - diff[j])
        yield_strain = strain[j - 1] + frac * (strain[j] - strain[j - 1])
        yield_stress = stress[j - 1] + frac * (stress[j] - stress[j - 1])

        self.yield_info = {
            "offset": offset,
            "yield_strain": yield_strain,
            "yield_stress_MPa": yield_stress,
            "crossing_index": int(j),
        }

        return yield_stress

    def calculate_uts(self, df=None):
        
        df = self._get_curve(df)
        return df["stress_MPa"].max()

    def calculate_fracture_strain(self, df=None, drop_frac=0.10):
      
        df = self._get_curve(df)

        i_uts = df["stress_MPa"].idxmax()
        post = df.loc[i_uts:]

        threshold = drop_frac * df["force_N"].max()
        dropped = post.index[post["force_N"] < threshold]

        if len(dropped) > 0:
            i_frac = max(int(dropped[0]) - 1, int(i_uts))
            method = "last point before load drop"
        else:
            i_frac = int(df.index[-1])
            method = "last data point (no load drop in record)"

        fracture_strain = df.loc[i_frac, "strain"]

        self.fracture_info = {
            "index": i_frac,
            "method": method,
            "stress_at_fracture_MPa": df.loc[i_frac, "stress_MPa"],
        }

        return fracture_strain

    def calculate_all_properties(self):
     
       
        df = self.calculate_stress_strain()

        E_GPa = self.calculate_youngs_modulus(df)
        yield_MPa = self.calculate_yield_strength(df)
        uts_MPa = self.calculate_uts(df)
        fracture_strain = self.calculate_fracture_strain(df)

        return {
            "test_id": self.specimen.test_id,
            "material": self.specimen.material,
            "youngs_modulus_GPa": E_GPa,
            "yield_strength_MPa": yield_MPa,
            "uts_MPa": uts_MPa,
            "fracture_strain": fracture_strain,
            "elongation_percent": fracture_strain * 100.0,
            "modulus_R2": self.modulus_info["R2"],
            "modulus_n_points": self.modulus_info["n_points"],
        }
