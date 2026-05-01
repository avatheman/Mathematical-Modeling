# 🌡️ Numerical Simulation of Transient Heat Flow in a One-Dimensional Conductor
### with Temperature-Dependent Conductivity

> A computational study using the **Finite Difference Method (FDM)** to simulate how heat travels through a 1D rod over time — where the material's ability to conduct heat changes with temperature.

---

## 📌 Overview

This project models the **transient heat equation** in a one-dimensional conductor, with the added complexity of **temperature-dependent thermal conductivity** — meaning the material doesn't conduct heat the same way at all temperatures, which is closer to real-world behavior.

The simulation is implemented in **Python** and cross-validated against **MATLAB**, with results compared to analytical solutions and statistically evaluated using industry-standard error metrics.

This work was developed as part of a **Mathematical Modeling Internship** at **Sacred Heart College, Chalakkudy**.

---

## 🧠 What Problem Are We Solving?

Imagine a metal rod. One end is hot, the other is cold. Heat flows from hot to cold — but *how fast*, and *how does the temperature change at each point of the rod over time*?

That's the **heat equation**. Now add one more twist: the rod conducts heat *differently* depending on how hot it already is. That's **temperature-dependent conductivity**, and it makes the problem non-linear — which means we can't just solve it by hand easily.

That's where **numerical simulation** comes in.

---

## ✨ Features

- 📐 Solves the **1D transient heat equation** numerically using FDM
- 🔁 Handles **non-linear (temperature-dependent) thermal conductivity** `k(T)`
- 📊 Generates **Temperature vs. Time** plots at different positions along the rod
- 📈 Generates **Temperature vs. Position** plots at different time steps
- 🔬 Compares **numerical results vs. analytical solution**
- 📉 Validates accuracy using statistical metrics:
  - **RMSE** — Root Mean Square Error
  - **R²** — Coefficient of Determination
  - **MAPE** — Mean Absolute Percentage Error
  - **AIC** — Akaike Information Criterion
- 🔄 Cross-verified with **MATLAB** implementation for consistency

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core implementation |
| NumPy | Numerical arrays and matrix operations |
| SciPy | Scientific computations and ODE utilities |
| Matplotlib | Plotting and visualization |
| MATLAB | Independent cross-validation and comparison |

---

## 📁 Project Structure

```
heat-flow-simulation/
│
├── python/
│   ├── main.py               # Entry point — runs the full simulation
│   ├── fdm_solver.py         # FDM scheme implementation
│   ├── conductivity.py       # Temperature-dependent k(T) definition
│   ├── analytical.py         # Analytical solution (for comparison)
│   ├── validation.py         # RMSE, R², MAPE, AIC calculations
│   └── plots.py              # All visualization functions
│
├── matlab/
│   └── heat_simulation.m     # MATLAB equivalent for cross-validation
│
├── results/
│   ├── temp_vs_time.png
│   ├── temp_vs_position.png
│   └── validation_table.csv
│
├── report/                   # Internship report (if applicable)
│
├── requirements.txt
└── README.md
```

> ⚠️ Adjust the folder names above to match your actual project structure.

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heat-flow-simulation.git
cd heat-flow-simulation
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
scipy
matplotlib
```

### 3. Run the simulation
```bash
python python/main.py
```

---

## 📊 Results

### Temperature vs. Time
Shows how the temperature at specific points on the rod evolves over time.

### Temperature vs. Position
Shows the temperature profile across the entire rod at specific moments in time.

### Analytical vs. Numerical Comparison
Plots both solutions on the same graph — the closer they are, the more accurate the simulation.

---

## ✅ Statistical Validation

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **RMSE** | Average error magnitude | Closer to 0 |
| **R²** | How well numerical fits analytical | Closer to 1 |
| **MAPE** | Percentage-based error | Closer to 0% |
| **AIC** | Model quality with complexity penalty | Lower is better |

> Results are saved to `results/validation_table.csv`.

---

## 📚 Mathematical Background

The governing equation is:

```
ρ·Cp · ∂T/∂t = ∂/∂x [ k(T) · ∂T/∂x ]
```

Where:
- `T` — Temperature (K or °C)
- `t` — Time (s)
- `x` — Position along the rod (m)
- `k(T)` — Thermal conductivity (temperature-dependent) (W/m·K)
- `ρ` — Density (kg/m³)
- `Cp` — Specific heat capacity (J/kg·K)

The FDM discretizes this equation over a grid in both space and time, turning it into a system of algebraic equations that can be solved step by step.

---

## 🏫 Acknowledgements

This project was developed as part of a **Mathematical Modeling Internship** at:

**Sacred Heart College, Chalakkudy**
Department of Mathematics

Special thanks to the faculty and mentors who guided this work.

---

## 👤 Author

**Abhishek P G**
Mathematical Modeling Intern
Sacred Heart College, Chalakkudy

---

## 📄 License

This project is for academic and educational purposes.
