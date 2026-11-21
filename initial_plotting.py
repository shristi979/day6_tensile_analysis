
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# 1. Define directories


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "tensile_public_data" / "data"

RAW_DIR = DATA_DIR / "raw"

OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# 2. Select two test files


test_files = ["test_01.csv", "test_02.csv"]



# 3. Create subplots


fig, axes = plt.subplots(1, 2, figsize=(13, 5))



# 4. Read and plot data


for ax, filename in zip(axes, test_files):

    file_path = RAW_DIR / filename

    df = pd.read_csv(file_path)

    # Identify displacement column
    if "displacement_mm" in df.columns:

        displacement = df["displacement_mm"]

    elif "displacement_m" in df.columns:

        displacement = df["displacement_m"] * 1000

    else:
        raise ValueError(f"Invalid displacement column: {filename}")

    force = df["force_N"]

    # Plot force vs displacement
    ax.plot(displacement, force, linewidth=1.2)

    ax.set_title(filename.replace(".csv", ""))
    ax.set_xlabel("Displacement (mm)")
    ax.set_ylabel("Force (N)")

    ax.grid(True, linestyle="--", alpha=0.6)



# 5. Format figure


fig.suptitle(
    "Initial Tensile Test: Force vs Displacement",
    fontsize=14
)

fig.tight_layout()



# 6. Save PNG


output_path = OUTPUT_DIR / "initial_force_displacement.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlotting completed successfully!")
print("Saved at:", output_path)
