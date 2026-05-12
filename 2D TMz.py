#Copyright (C) <2026> <Sean Mullins>
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 1. Physical Constants and Simulation Parameters
# =============================================================================
c = 299792458.0              # Speed of light in vacuum (m/s)
mu0 = 4.0 * np.pi * 1e-7     # Permeability of free space (H/m)
eps0 = 1.0 / (mu0 * c**2)    # Permittivity of free space (F/m)

# Source parameters
f = 200e12                   # Source frequency: 200 THz
lambda_0 = c / f             # Free-space wavelength (~1.5 um)
omega = 2.0 * np.pi * f
tau = 10e-15                 # Gaussian pulse width (10 fs)
t0 = 3.0 * tau               # Pulse delay (30 fs) to ensure a smooth turn-on
source_amplitude = 1.0       # Peak amplitude of the source

# Grid parameters
Nx, Ny = 500, 500            # 500 x 500 cells grid
dl = lambda_0 / 20.0           # Spatial resolution (dx = dy). 20 points per wavelength

# Courant stability condition for 2D FDTD
# dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2))
dt = (dl / (np.sqrt(2.0) * c)) * 0.99  # Time step slightly below Courant limit

# Add in eps_r and update the Ce equation
eps_r = np.ones((Nx, Ny))  # Relative permittivity grid (free space everywhere)
eps_r[350:450, 200:301] = 4.0 # Insert a dielectric block with eps_r=4 in the upper right quadrant

# FDTD update coefficients
Ch = dt / (mu0 * dl)         # Magnetic field multiplier
Ce = dt / (eps0 * eps_r * dl)        # Electric field multiplier

# Total time steps
n_steps = 2000

# Addition of Mur coefficients for absorbing boundary conditions in Q5.
mur_coeff = (c * dt - dl) / (c * dt + dl)  # Coefficient for Mur's first-order ABC


# =============================================================================
# 2. Grid Initialization
# =============================================================================
# Yee Grid Configuration:
# Ez[i, j] is at the cell center: (i+0.5, j+0.5)*dl
# Hx[i, j] is at the cell bottom edge: (i+0.5, j)*dl
# Hy[i, j] is at the cell left edge: (i, j+0.5)*dl
Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny))
Hy = np.zeros((Nx, Ny))

# Center coordinates for the infinite line source
ic, jc = Nx // 2, Ny // 2

# Track current time step
n = 0

# =============================================================================
# 3. FDTD Computation Loop (Pre-calculate all frames)
# =============================================================================
steps_per_frame = 5  # Store 1 frame every 5 steps to save memory
n_frames = n_steps // steps_per_frame

Ez_frames = []
global_max_E = 0.0

Ez_prev = Ez.copy()  # Ensure Ez is initialized as a separate array for frame storage of Mur Coefficient Equation

print("Running FDTD Simulation (calculating time response)...")
for n in range(n_steps):
    # ---------------------------------------------------------------------
    # Magnetic Field Updates
    # ---------------------------------------------------------------------
    Ez_prev[:,:] = Ez # Store the current Ez before updating for Mur's ABC 

    # Hx update (requires Ez above and below)
    Hx[:, 1:] -= Ch * (Ez[:, 1:] - Ez[:, :-1])

    # Hy update (requires Ez to the right and left)
    Hy[1:, :] += Ch * (Ez[1:, :] - Ez[:-1, :])

    # ---------------------------------------------------------------------
    # Electric Field Update
    # ---------------------------------------------------------------------
    # Ez update for the interior core region ONLY.
    # By NOT updating the boundaries (i=0, i=-1, j=0, j=-1), they remain 
    # strictly at 0.0, which naturally enforces the PEC walls perfectly!
    Ez[1:-1, 1:-1] += Ce[1:-1, 1:-1] * (
        (Hy[2:, 1:-1] - Hy[1:-1, 1:-1]) -
        (Hx[1:-1, 2:] - Hx[1:-1, 1:-1])
    )
    # Had to update this equation for Part 3 due to mismatch of size.


    # ---------------------------------------------------------------------
    # Source Injection
    # ---------------------------------------------------------------------
    # Hard source injection (Modulated Gaussian Pulse)
    t = n * dt
    Ez[ic, jc] += source_amplitude * np.exp(-((t - t0) / tau)**2)
    #Ez[ic, jc] = source_amplitude * np.cos(omega * t) * np.exp(-((t - t0) / tau)**2) # Used this for a soft-source

    #Add Mur ABC on right boundary
    Ez[-1, 1:-1] = (
        Ez_prev[-2, 1:-1] +
        mur_coeff * (Ez[-2, 1:-1] - Ez_prev[-1, 1:-1])
    )
    
    # Save frame and track global maximum
    if n % steps_per_frame == 0:
        Ez_frames.append(Ez.copy()) # Copy the array state into memory
        current_max = np.max(np.abs(Ez))
        if current_max > global_max_E:
            global_max_E = current_max
            
    # Progress feedback in the console
    if n % 100 == 0:
        print(f"Computed step {n}/{n_steps}")

# =============================================================================
# 4. Setup Visualization & Animation
# =============================================================================
print("Setting up visualization...")
fig, ax = plt.subplots(figsize=(7, 6))

# Set static vmin and vmax based on the global maximum found during computation
static_vmax = max(global_max_E * 0.1, 1e-9)

# Transpose Ez (Ez.T) so the array maps naturally to x (horizontal) and y (vertical)
im = ax.imshow(Ez_frames[0].T, cmap='bwr', vmin=-static_vmax, vmax=static_vmax, origin='lower')
plt.colorbar(im, ax=ax, label='E_z Field Amplitude (V/m)')
ax.set_title(f'2D TMz FDTD - 200 THz Gaussian Pulse\nTime Step: 0/{n_steps}')
ax.set_xlabel('x (cells)')
ax.set_ylabel('y (cells)')

def update_plot(frame_idx):
    im.set_array(Ez_frames[frame_idx].T)
    time_step = frame_idx * steps_per_frame
    ax.set_title(f'2D TMz FDTD - 200 THz Gaussian Pulse\nTime Step: {time_step}/{n_steps}')
    return [im]

# Create and run the animation
print("Starting animation...")
ani = animation.FuncAnimation(fig, update_plot, frames=len(Ez_frames), 
                              interval=30, blit=False, repeat=False)

# Display the animation (Will block the script until window is closed)
plt.show()
print("Simulation complete.")