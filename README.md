# Tensile Test Data Analysis

Python project that analyses 30 tensile tests (steel, aluminium and copper). It cleans the raw data, calculates mechanical properties, and produces a summary table and plots.

## Folder structure

```
day6_tensile_analysis/
├── tensile_public_data/
│   └── data/
│       ├── raw/                 test_01.csv ... test_30.csv
│       └── specimens.csv        test_id, material, area_mm2, gauge_length_mm
├── outputs/                     created automatically
├── initial_plotting.py          force vs displacement for test_01 and test_02
├── specimen.py                  Specimen dataclass and TensileTest class
├── campaign.py                  TestCampaign class (runs all tests, makes outputs)
├── main.py                      runs the complete workflow
├── findings.md                  short written findings
└── README.md
```

## How to run

1. Install the requirements: `pip install pandas numpy matplotlib`
2. Keep the folder structure above.
3. Run `main.py` (in Spyder press `F5`, or in a terminal use `python main.py`).

## What the code does

**`specimen.py`**

- `Specimen`: holds test_id, material, area (mm²) and gauge length (mm).
- `TensileTest`: loads one test and cleans it by removing duplicate rows, removing rows with missing values, converting displacement to mm, removing sensor spikes and removing initial slack.
- It then calculates:
  - Young's modulus (GPa)
  - 0.2% offset yield strength (MPa)
  - ultimate tensile strength (MPa)
  - fracture strain and elongation (%)

**`campaign.py`**

- `TestCampaign`: runs `TensileTest` on every specimen in `specimens.csv`, writes the summary table, makes the plots and calculates the material-wise averages. If one test fails, the others still run and the failure is listed.

**`main.py`**

- Runs everything and prints the per-test results and the material averages.

## Outputs (in `outputs/`)

- `summary.csv`: one row per test with all properties and cleaning counts
- `stress_strain_steel.png`, `stress_strain_aluminium.png`, `stress_strain_copper.png`
- `yield_strength_boxplot.png`

## Method notes

- **Stress and strain:** stress = force / area; strain = displacement / gauge length.
- **Young's modulus:** a straight line is fitted to the part of the curve between 15% and 40% of the maximum stress. This skips the curved start and stays below yielding.
- **Yield strength:** the 0.2% offset method. A line parallel to the elastic line is shifted by 0.002 strain, and the yield strength is where the curve crosses it.
- **Sensor spikes:** a point is removed if its force is far from the median of its neighbours. The first and last few points of a test are never tested.
- **Initial slack:** leading rows with force below 2% of the maximum force are removed.
- **Fracture strain:** the last point before the force drops below 10% of maximum. If the record has no such drop, the last data point is used (stated in the `fracture_method` column of `summary.csv`).

## Limitations

- Several tests end at a much lower elongation than others of the same material (test_02, test_03, test_26), so their fracture point should be checked.
- Displacement measurement was not checked against an extensometer, so the modulus is the value the recorded data gives.
