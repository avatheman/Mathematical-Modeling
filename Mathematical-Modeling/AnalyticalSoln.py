# ================================================================
#  ANALYTICAL SOLUTION -- 1D HEAT EQUATION (OOP Version)
#  Candle Problem: One end heated, one end cold
#
#  PDE  : dT/dt = alpha * d2T/dx2
#  IC   : T(x,0) = T_initial  (rod at room temperature)
#  BC   : T(0,t) = T_left     (candle end -- hot)
#         T(L,t) = T_right    (open end   -- cold)
#
#  Method : Fourier Series
#
#  Author : Abhishek P G
#  Guide  : Dr. Vijitha Mukundan
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ================================================================
#  CLASS DEFINITION
# ================================================================

class HeatEquation1D:
    """
    Solves the 1D transient heat equation analytically
    using the Fourier Series method.

    Physical Setup:
        - A rod of length L
        - Left end held at T_left  (candle end)
        - Right end held at T_right (cold end)
        - Rod initially at uniform T_initial (room temperature)
    """

    # ── Constructor ─────────────────────────────────────────────
    def __init__(self, L, alpha, T_left, T_right, T_initial,
                 N=50, nx=100):
        """
        Initialize the heat equation problem.

        Parameters:
            L         : rod length (m)
            alpha     : thermal diffusivity (m^2/s)
            T_left    : left boundary temperature (C)
            T_right   : right boundary temperature (C)
            T_initial : initial uniform temperature (C)
            N         : number of Fourier terms
            nx        : number of spatial grid points
        """
        self.L         = L
        self.alpha     = alpha
        self.T_left    = T_left
        self.T_right   = T_right
        self.T_initial = T_initial
        self.N         = N
        self.nx        = nx

        # Build spatial grid
        self.x = np.linspace(0, L, nx)

        # Compute steady state once -- reused in every solution
        self.T_steady = self._compute_steady_state()

        print("HeatEquation1D object created!")
        print(f"  Rod length       : {self.L} m")
        print(f"  Thermal diff.    : {self.alpha} m^2/s")
        print(f"  Left BC          : {self.T_left} C")
        print(f"  Right BC         : {self.T_right} C")
        print(f"  Initial temp     : {self.T_initial} C")
        print(f"  Fourier terms    : {self.N}")
        print(f"  Spatial points   : {self.nx}")

    # ── Private Method: Steady State ────────────────────────────
    def _compute_steady_state(self):
        """
        Compute the steady state temperature profile.
        This is the straight line from T_left to T_right.
        T_steady(x) = T_left + (T_right - T_left) * x/L
        """
        return self.T_left + (self.T_right - self.T_left) \
               * self.x / self.L

    # ── Private Method: Fourier Coefficient ─────────────────────
    def _fourier_coefficient(self, n):
        """
        Compute the nth Fourier coefficient Bn.
        Bn = (2/L) * integral of (T_initial - T_steady)
                                  * sin(n*pi*x/L) dx
        """
        integrand = (self.T_initial - self.T_steady) * np.sin(n * np.pi * self.x / self.L)
        return (2 / self.L) * np.trapezoid(integrand, self.x)

    # ── Public Method: Solve at time t ──────────────────────────
    def solve(self, t):
        """
        Compute the full Fourier Series solution at time t.

        T(x,t) = T_steady(x)
               + SUM Bn * sin(n*pi*x/L) * exp(-alpha*(n*pi/L)^2*t)
        """
        T = self.T_steady.copy()

        for n in range(1, self.N + 1):
            Bn    = self._fourier_coefficient(n)
            decay = np.exp(-self.alpha * (n * np.pi / self.L)**2 * t)
            T    += Bn * decay * np.sin(n * np.pi * self.x / self.L)

        return T

    # ── Public Method: Print Summary ────────────────────────────
    def print_summary(self, t_list):
        """
        Print temperature values at key positions for each time.
        """
        print("\nComputing analytical solution...")
        print(f"{'':->50}")
        print(f"{'Time (s)':<12} {'T(x=0)':<12}"
              f" {'T(x=L/2)':<12} {'T(x=L)':<12}")
        print(f"{'':->50}")

        for t in t_list:
            T   = self.solve(t)
            mid = self.nx // 2
            print(f"{t:<12.0f} {T[0]:<12.4f}"
                  f" {T[mid]:<12.4f} {T[-1]:<12.4f}")

        print(f"{'':->50}")

    # ── Public Method: Plot ──────────────────────────────────────
    def plot(self, t_list, save_as='heat_analytical_oop.png'):
        """
        Generate two plots:
        1. Temperature profiles at different times
        2. Temperature heatmap (x vs t)
        """
        solutions = [self.solve(t) for t in t_list]
        colors    = plt.cm.plasma(np.linspace(0.1, 0.9, len(t_list)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            'Analytical Solution: 1D Heat Equation (OOP)\n'
            r'Candle Problem - Iron Rod'
            f' (alpha = {self.alpha:.2e} m^2/s)',
            fontsize=13, fontweight='bold'
        )

        # ── Left plot: profiles ──────────────────────────────────
        ax1 = axes[0]
        for i, (t, T) in enumerate(zip(t_list, solutions)):
            label = (f't = 0 s (initial)' if t == 0
                     else f't = {t/3600:.1f} hr' if t >= 3600
                     else f't = {t} s')
            ax1.plot(self.x, T, color=colors[i],
                     linewidth=2, label=label)

        ax1.plot(self.x, self.T_steady, 'k--',
                 linewidth=1.5, label='Steady State (t->inf)')
        ax1.set_xlabel('Position x (m)', fontsize=11)
        ax1.set_ylabel('Temperature T (C)', fontsize=11)
        ax1.set_title('Temperature Distribution Over Time', fontsize=11)
        ax1.legend(fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, self.L])
        ax1.set_ylim([-5, self.T_left + 10])
        ax1.annotate('Candle\n(100 C)', xy=(0, 100),
                     xytext=(0.05, 88), fontsize=8, color='red')
        ax1.annotate('Open end\n(0 C)', xy=(1, 0),
                     xytext=(0.75, 15), fontsize=8, color='blue')

        # ── Right plot: heatmap ──────────────────────────────────
        ax2       = axes[1]
        t_fine    = np.linspace(0, max(t_list), 200)
        X_g, T_g  = np.meshgrid(self.x, t_fine)
        U_g       = np.array([self.solve(t) for t in t_fine])

        contour = ax2.contourf(X_g, T_g, U_g, levels=20, cmap='plasma')
        plt.colorbar(contour, ax=ax2, label='Temperature (C)')
        ax2.set_xlabel('Position x (m)', fontsize=11)
        ax2.set_ylabel('Time t (seconds)', fontsize=11)
        ax2.set_title('Temperature Heatmap (x vs t)', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as {save_as}")

    # ── Public Method: Export CSV ────────────────────────────────
    def export_csv(self, t_list, filename='heat_analytical_oop.csv'):
        """
        Export solution data to CSV file.
        Columns: x (m), t (s), T_analytical (C)
        """
        rows = []
        for t in t_list:
            T = self.solve(t)
            for xi, Ti in zip(self.x, T):
                rows.append({
                    'x (m)'            : round(float(xi), 4),
                    't (s)'            : t,
                    'T_analytical (C)' : round(float(Ti), 6)
                })

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"\nCSV saved as {filename}")
        print(f"Total data points : {len(df)}")
        print(f"\nFirst few rows:")
        print(df.head(6).to_string(index=False))
        return df


# ================================================================
#  MAIN -- Run the simulation
# ================================================================

if __name__ == "__main__":

    # ── Step 1: Create the problem object ───────────────────────
    rod = HeatEquation1D(
        L         = 1.0,
        alpha     = 2.3e-5,
        T_left    = 100.0,
        T_right   = 0.0,
        T_initial = 25.0,
        N         = 50,
        nx        = 100
    )

    # ── Step 2: Define time steps ────────────────────────────────
    t_list = [0, 100, 500, 1000, 3000, 6000, 10000]

    # ── Step 3: Print summary table ──────────────────────────────
    rod.print_summary(t_list)

    # ── Step 4: Generate plots ───────────────────────────────────
    rod.plot(t_list, save_as='heat_analytical_oop.png')

    # ── Step 5: Export CSV ───────────────────────────────────────
    rod.export_csv(t_list, filename='heat_analytical_oop.csv')

    print("\nDone!")
