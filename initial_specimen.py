
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path



# 1. SPECIMEN DATACLASS

@dataclass
class Specimen:

    test_id: str
    material: str
    area: float
    gauge_length: float



# 2. TENSILE TEST CLASS


class TensileTest:

    def __init__(self, specimen, file_path):

        self.specimen = specimen
        self.file_path = Path(file_path)

        # Load experimental data
        self.data = self.load_data()


  
    # Load CSV file
    

    def load_data(self):

        df = pd.read_csv(self.file_path)

        return df


   
    # Basic data cleaning
    

    def clean_basic_data(self):

        # Work on a copy of the original data
        df = self.data.copy()

        # Initial number of rows
        initial_rows = len(df)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Count duplicates removed
        duplicates_removed = initial_rows - len(df)

        # Count missing values
        missing_before = df.isnull().sum().sum()

        # Remove rows containing missing values
        df = df.dropna()

        # Count missing values removed
        missing_removed = missing_before

        # Reset index
        df = df.reset_index(drop=True)

        # Update working dataset
        self.data = df

        # Display cleaning report
        print("\nDATA CLEANING REPORT")
        print("-" * 40)

        print("Original rows:", initial_rows)
        print("Duplicates removed:", duplicates_removed)
        print("Missing values removed:", missing_removed)
        print("Remaining rows:", len(df))

        return df

   
  
    # Remove sensor spikes
   

    def remove_sensor_spikes(self):

        df = self.data.copy()

        # Calculate local rolling median
        rolling_median = df["force_N"].rolling(
            window=11,
            center=True,
            min_periods=3
        ).median()

        # Calculate deviation from local median
        residual = df["force_N"] - rolling_median

        # Calculate Median Absolute Deviation (MAD)
        mad = np.median(
            np.abs(residual - np.median(residual))
        )

        # Avoid division by zero / zero threshold
        if mad == 0:

            print("No spikes detected: MAD is zero.")
            return df

        # Robust standard deviation
        robust_sigma = 1.4826 * mad

        # Identify potential spikes
        threshold = 6 * robust_sigma

        spike_mask = np.abs(residual) > threshold

        # Count detected spikes
        spikes_removed = spike_mask.sum()

        # Remove detected spikes
        df = df.loc[~spike_mask].copy()

        # Reset index
        df = df.reset_index(drop=True)

        # Update working dataset
        self.data = df

        print("\nSENSOR SPIKE REPORT")
        print("-" * 40)
        print("Potential spikes detected:", spikes_removed)
        print("Remaining rows:", len(df))

        return df



    # Calculate stress and strain
   

    def calculate_stress_strain(self):

        df = self.data.copy()

        # Identify displacement column
        if "displacement_mm" in df.columns:

            displacement = df["displacement_mm"]

        elif "displacement_m" in df.columns:

            displacement = df["displacement_m"] * 1000

        else:

            raise ValueError(
                "Displacement column not found!"
            )

        # Extract force
        force = df["force_N"]

        # Calculate engineering stress
        stress = force / self.specimen.area

        # Calculate engineering strain
        strain = displacement / self.specimen.gauge_length

        # Store calculated values
        df["displacement_mm"] = displacement
        df["stress_MPa"] = stress
        df["strain"] = strain

        return df
    
    def calculate_youngs_modulus(self, df=None, lower_frac=0.15, upper_frac=0.40):

        if df is None:
            df = self.calculate_stress_strain()

        df = df.reset_index(drop=True)

        # Stress band for the fit, as fractions of the maximum stress
        max_stress = df["stress_MPa"].max()
        lo = lower_frac * max_stress
        hi = upper_frac * max_stress

        # Loading part only: rows up to the first point that reaches 'hi'
        first_hi = df.index[df["stress_MPa"] >= hi][0]
        loading = df.loc[:first_hi]

        # Points inside the band
        band = loading[
            (loading["stress_MPa"] >= lo) & (loading["stress_MPa"] <= hi)
        ]

        if len(band) < 5:
            raise ValueError("Too few points in the fitting band.")

        x = band["strain"].to_numpy()
        y = band["stress_MPa"].to_numpy()

        slope, intercept = np.polyfit(x, y, 1)

        # R-squared of the straight-line fit
        y_fit = slope * x + intercept
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # Keep the details so we can inspect them later
        self.modulus_info = {
            "band_MPa": (round(lo, 1), round(hi, 1)),
            "n_points": len(x),
            "R2": r2,
        }

        return slope / 1000
    
# 3. TESTING A


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "tensile_public_data" / "data"

    # Read specimen information
    specimens_df = pd.read_csv(DATA_DIR / "specimens.csv")

    # Select test_01
    row = specimens_df.loc[
        specimens_df["test_id"] == "test_01"
    ].iloc[0]

    # Create Specimen object
    specimen = Specimen(
        test_id=row["test_id"],
        material=row["material"],
        area=row["area_mm2"],
        gauge_length=row["gauge_length_mm"]
    )

    # Create TensileTest object
    test = TensileTest(
        specimen=specimen,
        file_path=DATA_DIR / "raw" / "test_01.csv"
    )

    # Calculate stress and strain
    results = test.calculate_stress_strain()

    # Display results
    print("\nSPECIMEN INFORMATION")
    print(specimen)

    print("\nSTRESS AND STRAIN DATA")
    print(results.head())

    print("\nDATA SHAPE:", results.shape)

# Testing B 

#  BASIC DATA CLEANING


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent

    DATA_DIR = BASE_DIR / "tensile_public_data" / "data"

    specimens_df = pd.read_csv(DATA_DIR / "specimens.csv")

    row = specimens_df.loc[
        specimens_df["test_id"] == "test_01"
    ].iloc[0]

    specimen = Specimen(
        test_id=row["test_id"],
        material=row["material"],
        area=row["area_mm2"],
        gauge_length=row["gauge_length_mm"]
    )

    test = TensileTest(
        specimen=specimen,
        file_path=DATA_DIR / "raw" / "test_01.csv"
    )

    cleaned_data = test.clean_basic_data()

    print("\nCLEANED DATA")
    print(cleaned_data.head())

    print("\nFINAL DATA SHAPE:", cleaned_data.shape)



# TEST: SENSOR SPIKE DETECTION

# First perform basic cleaning
test.clean_basic_data()

# Remove potential force spikes
spike_cleaned_data = test.remove_sensor_spikes()

print("\nSPIKE-CLEANED DATA")
print(spike_cleaned_data.head())

print("\nFINAL SHAPE:", spike_cleaned_data.shape)


if __name__ == "__main__":

    # Your existing specimen and test object creation code
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "tensile_public_data" / "data"

    specimens_df = pd.read_csv(DATA_DIR / "specimens.csv")

    row = specimens_df.loc[
        specimens_df["test_id"] == "test_01"
    ].iloc[0]

    specimen = Specimen(
        test_id=row["test_id"],
        material=row["material"],
        area=row["area_mm2"],
        gauge_length=row["gauge_length_mm"]
    )

    # TEST: SENSOR SPIKE DETECTION + PLOT
    test = TensileTest(
        specimen,
        DATA_DIR / "raw" / "test_01.csv"
    )

    test.clean_basic_data()
    before_spike_removal = test.data.copy()

    after_spike_removal = test.remove_sensor_spikes()

    # Plot comparison
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    plt.plot(
        before_spike_removal["time_s"],
        before_spike_removal["force_N"],
        label="Before spike removal",
        alpha=0.7
    )

    plt.plot(
        after_spike_removal["time_s"],
        after_spike_removal["force_N"],
        label="After spike removal",
        linewidth=1.5
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title("Force Data Before and After Spike Removal")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # TEST: IDENTIFY INITIAL SLACK

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.plot(
    after_spike_removal["displacement_m"],
    after_spike_removal["force_N"]
)

plt.xlabel("Displacement (m)")
plt.ylabel("Force (N)")
plt.title("Identifying Initial Slack - Test 01")
plt.grid(True)
plt.show()

# TEST: CHECK FORCE RANGE

print("\nFORCE RANGE")
print("-" * 40)

print("Minimum force:", after_spike_removal["force_N"].min(), "N")
print("Maximum force:", after_spike_removal["force_N"].max(), "N")


# CHECK INITIAL FORCE READINGS

print("\nFIRST 10 FORCE READINGS")
print(after_spike_removal["force_N"].head(10).to_list())

# TEST: ENGINEERING STRESS AND STRAIN

stress_strain_data = test.calculate_stress_strain()

print("\nENGINEERING STRESS-STRAIN DATA")
print("-" * 50)

print(
    stress_strain_data[
        ["displacement_mm", "force_N", "stress_MPa", "strain"]
    ].head(10)
)

# Plot stress-strain curve
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    stress_strain_data["strain"],
    stress_strain_data["stress_MPa"]
)

plt.xlabel("Engineering Strain")
plt.ylabel("Engineering Stress (MPa)")
plt.title("Engineering Stress-Strain Curve - Test 01")
plt.grid(True)
plt.show()


# TEST: YOUNG'S MODULUS

E = test.calculate_youngs_modulus(stress_strain_data)

print("\nYOUNG'S MODULUS")
print("-" * 40)
print("Young's Modulus:", round(E, 3), "GPa")

# VISUALIZE YOUNG'S MODULUS FITTING REGION

# max_strain = stress_strain_data["strain"].max()

# elastic_data = stress_strain_data[
#     stress_strain_data["strain"] <= 0.10 * max_strain
# ]

# plt.figure(figsize=(10, 6))

# plt.plot(
#     stress_strain_data["strain"],
#     stress_strain_data["stress_MPa"],
#     label="Full stress-strain curve"
# )

# plt.scatter(
#     elastic_data["strain"],
#     elastic_data["stress_MPa"],
#     color="red",
#     s=10,
#     label="Data used for modulus calculation"
# )

# plt.xlabel("Engineering Strain")
# plt.ylabel("Engineering Stress (MPa)")
# plt.title("Young's Modulus: Fitting Region")
# plt.legend()
# plt.grid(True)
# plt.show()

# ZOOM ON THE ELASTIC REGION (read-only)

zoom = stress_strain_data[stress_strain_data["stress_MPa"] <= 400]
zoom = zoom[zoom["strain"] <= 0.005]

plt.figure(figsize=(10, 6))
plt.plot(zoom["strain"], zoom["stress_MPa"], ".-", markersize=3)
plt.xlabel("Engineering Strain")
plt.ylabel("Engineering Stress (MPa)")
plt.title("Zoom: start of curve (strain <= 0.005)")
plt.grid(True)
plt.show()

print("Points in this zoom:", len(zoom))
print("First row where stress >= 350 MPa:")
print(stress_strain_data[stress_strain_data["stress_MPa"] >= 350].head(1)[["strain", "stress_MPa"]])

# YOUNG'S MODULUS: SENSITIVITY TO THE FITTING BAND (read-only)

# Only use the loading part of the curve: rows up to the first point at/above 350 MPa
i_end = stress_strain_data[stress_strain_data["stress_MPa"] >= 350].index[0]
loading = stress_strain_data.loc[:i_end]

# Candidate stress bands (MPa), chosen by eye from the zoom plot of test_01
bands = [(50, 300), (75, 300), (100, 300), (100, 330), (150, 330), (50, 350)]

results_rows = []

for lo, hi in bands:

    band_data = loading[
        (loading["stress_MPa"] >= lo) & (loading["stress_MPa"] <= hi)
    ]

    x = band_data["strain"].to_numpy()
    y = band_data["stress_MPa"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)

    # R-squared: how well the straight line explains the data (1.0 = perfect)
    y_fit = slope * x + intercept
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    results_rows.append({
        "band_MPa": f"{lo}-{hi}",
        "n_points": len(x),
        "E_GPa": round(slope / 1000, 2),
        "R2": round(r2, 5),
    })

print("\nYOUNG'S MODULUS: BAND SENSITIVITY")
print("-" * 40)
print(pd.DataFrame(results_rows).to_string(index=False))

# TEST: NEW YOUNG'S MODULUS METHOD

E_new = test.calculate_youngs_modulus(stress_strain_data)

print("\nYOUNG'S MODULUS (stress-band method)")
print("-" * 40)
print("E:", round(E_new, 2), "GPa")
print("Details:", test.modulus_info)

# LIST MATERIALS (read-only)

print("\nSPECIMEN LIST")
print("-" * 40)
print(specimens_df[["test_id", "material", "area_mm2", "gauge_length_mm"]].to_string(index=False))

# CHECK THE METHOD ON ALUMINIUM (test_02) AND COPPER (test_03)

for tid in ["test_02", "test_03"]:

    r = specimens_df.loc[specimens_df["test_id"] == tid].iloc[0]

    sp = Specimen(
        test_id=r["test_id"],
        material=r["material"],
        area=r["area_mm2"],
        gauge_length=r["gauge_length_mm"]
    )

    tt = TensileTest(sp, DATA_DIR / "raw" / f"{tid}.csv")
    tt.clean_basic_data()
    tt.remove_sensor_spikes()
    ss = tt.calculate_stress_strain()

    print("\n" + "=" * 50)
    print(tid, "-", sp.material)
    print("Max stress (MPa):", round(ss["stress_MPa"].max(), 1))
    print("Max strain:", round(ss["strain"].max(), 4))

    try:
        E_mat = tt.calculate_youngs_modulus(ss)
        print("E:", round(E_mat, 2), "GPa")
        print("Details:", tt.modulus_info)
        lo, hi = tt.modulus_info["band_MPa"]
        in_band = ss[(ss["stress_MPa"] >= lo) & (ss["stress_MPa"] <= hi)
                     & (ss["strain"] <= ss.loc[ss["stress_MPa"] >= hi].iloc[0]["strain"])]
    except ValueError as err:
        print("Method failed:", err)
        in_band = ss.iloc[0:0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: full curve
    axes[0].plot(ss["strain"], ss["stress_MPa"])
    axes[0].scatter(in_band["strain"], in_band["stress_MPa"], color="red", s=10)
    axes[0].set_title(f"{tid} ({sp.material}): full curve")

    # Right: zoom on the start of the curve
    zoom_x = ss[ss["strain"] <= 0.01]
    axes[1].plot(zoom_x["strain"], zoom_x["stress_MPa"], ".-", markersize=3)
    axes[1].scatter(in_band["strain"], in_band["stress_MPa"], color="red", s=12,
                    label="fitting band")
    axes[1].set_title("Zoom: strain <= 0.01")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("Engineering Strain")
        ax.set_ylabel("Engineering Stress (MPa)")
        ax.grid(True)

    plt.show()
    
    # RUN YOUNG'S MODULUS ON ALL 30 TESTS (read-only check)

import io
from contextlib import redirect_stdout

check_rows = []

for _, r in specimens_df.iterrows():

    sp = Specimen(
        test_id=r["test_id"],
        material=r["material"],
        area=r["area_mm2"],
        gauge_length=r["gauge_length_mm"]
    )

    tt = TensileTest(sp, DATA_DIR / "raw" / f"{r['test_id']}.csv")

    # Silence the cleaning reports so the table stays readable
    with redirect_stdout(io.StringIO()):
        tt.clean_basic_data()
        tt.remove_sensor_spikes()

    ss = tt.calculate_stress_strain()

    try:
        E_val = tt.calculate_youngs_modulus(ss)
        check_rows.append({
            "test_id": r["test_id"],
            "material": r["material"],
            "E_GPa": round(E_val, 1),
            "R2": round(tt.modulus_info["R2"], 5),
            "n_points": tt.modulus_info["n_points"],
            "band_MPa": tt.modulus_info["band_MPa"],
        })
    except Exception as err:
        check_rows.append({
            "test_id": r["test_id"],
            "material": r["material"],
            "E_GPa": None,
            "R2": None,
            "n_points": None,
            "band_MPa": str(err),
        })

check_table = pd.DataFrame(check_rows)

print("\nYOUNG'S MODULUS: ALL TESTS")
print("-" * 60)
print(check_table.to_string(index=False))

print("\nPER-MATERIAL SUMMARY")
print(check_table.groupby("material")["E_GPa"].agg(["count", "mean", "std", "min", "max"]).round(1))

# INSPECT THE LOW-E STEEL TESTS (read-only)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for tid, colour in [("test_01", "gray"), ("test_28", "red"), ("test_15", "blue")]:

    r = specimens_df.loc[specimens_df["test_id"] == tid].iloc[0]

    sp = Specimen(r["test_id"], r["material"], r["area_mm2"], r["gauge_length_mm"])
    tt = TensileTest(sp, DATA_DIR / "raw" / f"{tid}.csv")

    with redirect_stdout(io.StringIO()):
        tt.clean_basic_data()
        tt.remove_sensor_spikes()

    ss = tt.calculate_stress_strain()
    zoom_ss = ss[ss["strain"] <= 0.004]

    axes[0].plot(zoom_ss["strain"], zoom_ss["stress_MPa"], ".-", markersize=3,
                 color=colour, label=tid)

    E_val = tt.calculate_youngs_modulus(ss)

    print(tid, "E =", round(E_val, 1), "GPa | first stress (MPa):",
          round(ss["stress_MPa"].iloc[0], 1),
          "| first strain:", round(ss["strain"].iloc[0], 6),
          "| max stress (MPa):", round(ss["stress_MPa"].max(), 1))

    # Local modulus between consecutive points in the first 25 rows
    local_E = (ss["stress_MPa"].diff() / ss["strain"].diff()).iloc[1:26] / 1000
    axes[1].plot(range(1, 26), local_E.to_numpy(), "o-", markersize=3,
                 color=colour, label=tid)

axes[0].set_title("Start of curve, strain <= 0.004")
axes[0].set_xlabel("Engineering Strain")
axes[0].set_ylabel("Engineering Stress (MPa)")
axes[1].set_title("Point-to-point slope, first 25 rows")
axes[1].set_xlabel("Row index")
axes[1].set_ylabel("Local E (GPa)")

for ax in axes:
    ax.grid(True)
    ax.legend()

plt.show()

# STEEL: SENSITIVITY OF E TO THE FITTING BAND (read-only)

band_settings = [(0.15, 0.40), (0.20, 0.60), (0.30, 0.65)]

steel_rows = []

for _, r in specimens_df[specimens_df["material"] == "steel"].iterrows():

    sp = Specimen(r["test_id"], r["material"], r["area_mm2"], r["gauge_length_mm"])
    tt = TensileTest(sp, DATA_DIR / "raw" / f"{r['test_id']}.csv")

    with redirect_stdout(io.StringIO()):
        tt.clean_basic_data()
        tt.remove_sensor_spikes()

    ss = tt.calculate_stress_strain()

    row_out = {"test_id": r["test_id"]}

    for lf, uf in band_settings:
        E_val = tt.calculate_youngs_modulus(ss, lower_frac=lf, upper_frac=uf)
        row_out[f"E {lf:.2f}-{uf:.2f}"] = round(E_val, 1)
        row_out[f"n {lf:.2f}-{uf:.2f}"] = tt.modulus_info["n_points"]

    steel_rows.append(row_out)

steel_table = pd.DataFrame(steel_rows)

print("\nSTEEL: E vs FITTING BAND")
print("-" * 60)
print(steel_table.to_string(index=False))

print("\nMEAN AND STD OF E FOR EACH BAND")
print(steel_table.filter(like="E ").agg(["mean", "std"]).round(1))