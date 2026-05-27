# ============================================================
# ================ FDM - Heat Equation =======================
#
# Author : Abhishek P G
# Guide  : Dr Vijitha Mukundan
# Method : Crank Nicholson Method
# ============================================================
# ============================================================

# Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class HeatRod:
    """
    This class represents our iron rod and everything
    we need to simulate how heat moves through it.

    We divide the rod into small pieces (grid points)
    and track the temperature at each piece over time.
    """

    def __init__(self, length, num_points, diffusivity,
                 time_step, num_steps):
        """
        Set up the rod with all its physical properties.

        length       : how long is the rod? (metres)
        num_points   : how many points we divide the rod into
        diffusivity  : how fast does heat spread? (alpha)
        time_step    : how big is each time jump? (seconds)
        num_steps    : how many time jumps do we take?
        """

        # Physical properties of the rod
        self.length      = length
        self.num_points  = num_points
        self.diffusivity = diffusivity
        self.time_step   = time_step
        self.num_steps   = num_steps

        # How far apart are the grid points?
        self.spacing = length / (num_points - 1)

        # The mesh ratio r = alpha * dt / dx^2
        # This controls how heat spreads per time step
        self.r = diffusivity * time_step / self.spacing**2

        # Create the spatial grid (positions along the rod)
        self.positions = np.linspace(0, length, num_points)

        # Start the rod at room temperature everywhere
        self.temperature = np.full(num_points, 25.0)

        # Apply boundary conditions
        self.temperature[0]  = 100.0   # candle end -- always hot
        self.temperature[-1] = 0.0     # cold end   -- always cold

        # Where we store temperature profiles at different times
        self.saved_profiles = {}
        self.saved_times    = []

        print("Rod is ready!")
        print(f"  Length             : {self.length} m")
        print(f"  Grid points        : {self.num_points}")
        print(f"  Grid spacing (dx)  : {self.spacing:.4f} m")
        print(f"  Time step (dt)     : {self.time_step} s")
        print(f"  Total time         : {self.num_steps * self.time_step} s")
        print(f"  Mesh ratio r       : {self.r:.4f}")

    # ----------------------------------------------------------
    # STEP 1: Build the right-hand side
    # ----------------------------------------------------------

    def build_right_side(self):
        """
        In Crank-Nicolson we need to compute the right side
        of our equation first -- this uses the temperatures
        we already know from the current time step.

        For each interior point:
        rhs[i] = r*T[i-1] + (2 - 2r)*T[i] + r*T[i+1]
        """
        rhs = np.zeros(self.num_points)

        for i in range(1, self.num_points - 1):
            rhs[i] = (
                self.r * self.temperature[i - 1]
                + (2 - 2 * self.r) * self.temperature[i]
                + self.r * self.temperature[i + 1]
            )

        return rhs

    # ----------------------------------------------------------
    # STEP 2: Solve for the next temperature using Gauss-Seidel
    # ----------------------------------------------------------

    def solve_next_step(self, rhs):
        """
        Now we find the temperatures at the NEXT time step.
        We do this by iterating -- guessing, then improving
        the guess -- until it stops changing (converges).

        This is the Gauss-Seidel method.
        Think of it like adjusting each point's temperature
        based on its neighbours, again and again, until
        everything agrees with each other.
        """
        new_temperature = self.temperature.copy()
        tolerance       = 1e-6
        max_iterations  = 1000

        for iteration in range(max_iterations):
            old_temperature = new_temperature.copy()

            for i in range(1, self.num_points - 1):
                new_temperature[i] = (
                    rhs[i]
                    + self.r * new_temperature[i - 1]
                    + self.r * old_temperature[i + 1]
                ) / (2 + 2 * self.r)

            # Keep boundaries fixed
            new_temperature[0]  = 100.0
            new_temperature[-1] = 0.0

            # Check if the solution has settled
            change = np.max(np.abs(new_temperature - old_temperature))
            if change < tolerance:
                break

        return new_temperature

    # ----------------------------------------------------------
    # STEP 3: Run the full simulation
    # ----------------------------------------------------------

    def run(self, save_every=None):
        """
        This is the main time loop.
        We step through time, updating temperatures
        at every grid point at every time step.
        """
        if save_every is None:
            save_every = max(1, self.num_steps // 6)

        # Save the starting condition
        self.saved_profiles[0] = self.temperature.copy()
        self.saved_times.append(0)

        print("\nSimulating heat flow through the rod...")

        for step in range(self.num_steps):

            # Build the right side from current temperatures
            rhs = self.build_right_side()

            # Solve for the next time step
            self.temperature = self.solve_next_step(rhs)

            # Save snapshot at regular intervals
            if (step + 1) % save_every == 0:
                current_time = (step + 1) * self.time_step
                self.saved_profiles[current_time] = \
                    self.temperature.copy()
                self.saved_times.append(current_time)

        # Always save the final state
        final_time = self.num_steps * self.time_step
        if final_time not in self.saved_profiles:
            self.saved_profiles[final_time] = \
                self.temperature.copy()
            self.saved_times.append(final_time)

        # Print summary
        mid = self.num_points // 2
        print(f"\n{'':->50}")
        print(f"{'Time (s)':<12} {'Midpoint Temp (C)':<20} {'Status'}")
        print(f"{'':->50}")
        for t, T in sorted(self.saved_profiles.items()):
            status = "initial" if t == 0 else \
                     "final" if t == self.num_steps * \
                     self.time_step else ""
            print(f"{t:<12.0f} {T[mid]:<20.4f} {status}")
        print(f"{'':->50}")
        print("Simulation complete!")

    # ----------------------------------------------------------
    # STEP 4: Plot the results
    # ----------------------------------------------------------

    def plot(self, save_as='fdm_heat_solution.png'):
        """
        Draw the temperature profiles at different times.
        Each curve shows the temperature along the rod
        at one snapshot in time.
        """
        fig, ax = plt.subplots(figsize=(9, 6))

        fig.suptitle(
            'How Heat Spreads Through a 1D Iron Rod\n'
            'Numerical Solution -- Crank-Nicolson Method',
            fontsize=13, fontweight='bold'
        )

        profiles = sorted(self.saved_profiles.items())
        colors   = plt.cm.plasma(
                    np.linspace(0.1, 0.9, len(profiles)))

        for i, (t, T) in enumerate(profiles):
            if t == 0:
                label = 't = 0 s  (room temperature)'
            elif t >= 3600:
                label = f't = {t/3600:.2f} hours'
            else:
                label = f't = {t:.0f} s'

            ax.plot(self.positions, T,
                    color=colors[i], linewidth=2, label=label)

        # Draw where the rod will eventually settle (steady state)
        steady = 100.0 - 100.0 * self.positions / self.length
        ax.plot(self.positions, steady, 'k--',
                linewidth=1.5, label='Final steady state')

        ax.set_xlabel('Position along rod (m)', fontsize=11)
        ax.set_ylabel('Temperature (C)',         fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, self.length])
        ax.set_ylim([-5, 110])

        # Label the two ends
        ax.annotate('Candle end\n100 C',
                    xy=(0, 100), xytext=(0.05, 88),
                    fontsize=9, color='red')
        ax.annotate('Cold end\n0 C',
                    xy=(1, 0), xytext=(0.72, 12),
                    fontsize=9, color='blue')

        plt.tight_layout()
        plt.savefig(save_as, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved as {save_as}")

    # ----------------------------------------------------------
    # STEP 5: Save data to CSV
    # ----------------------------------------------------------

    def save_csv(self, filename='fdm_heat_data.csv'):
        """
        Save all the temperature data to a CSV file.
        We need this later to compare with the
        analytical (exact) solution.

        Columns:
            x     -- position along rod (m)
            t     -- time (s)
            T     -- temperature at that point and time (C)
        """
        rows = []

        for t, T in sorted(self.saved_profiles.items()):
            for position, temp in zip(self.positions, T):
                rows.append({
                    'x (m)'           : round(float(position), 4),
                    't (s)'           : round(float(t), 2),
                    'T_numerical (C)' : round(float(temp), 6)
                })

        data = pd.DataFrame(rows)
        data.to_csv(filename, index=False)

        print(f"Data saved to {filename}")
        print(f"Total rows : {len(data)}")
        print(f"\nSample:")
        print(data.head(6).to_string(index=False))

        return data


# ================================================================
#  RUN THE SIMULATION
# ================================================================

if __name__ == "__main__":

    # Create the rod
    rod = HeatRod(
        length      = 1.0,
        num_points  = 51,
        diffusivity = 2.3e-5,
        time_step   = 100,
        num_steps   = 100
    )

    # Run the simulation
    rod.run(save_every=15)

    # Plot the results
    rod.plot(save_as='fdm_heat_solution.png')

    # Save data for comparison
    rod.save_csv(filename='fdm_heat_data.csv')

    print("\nAll done!")
