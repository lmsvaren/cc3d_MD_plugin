import hoomd
import hoomd.md
import numpy as np
from cc3d.core.PySteppables import *


class adhesionsatSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.sim = None
        self.integrator = None
        self.bead_field = None

        # Simulation parameters
        self.md_dt = 0.002
        self.md_kT = 0.001
        self.spring_k = 1.0
        self.spring_k_cyto = 1.0
        self.spring_r0 = 0.0
        self.bend_k = 0.5
        self.bend_t0 = np.pi
        self.padding = 20.0

        self.num_grid_pts = 0
        self.com_particle_idx = 0

    def cc3d_to_hoomd(self, pos_2d):
        """Shifts CC3D domain [0, dim] to HOOMD centered domain [-Lx/2, Lx/2]."""
        offset = np.array([self.dim.x / 2.0, self.dim.y / 2.0])
        return pos_2d - offset

    def hoomd_to_cc3d(self, pos_2d):
        """Shifts HOOMD centered domain back to CC3D domain."""
        offset = np.array([self.dim.x / 2.0, self.dim.y / 2.0])
        return pos_2d + offset

    def start(self):
        """Initializes HOOMD device, snapshot topology, and bond integrators."""
        self.bead_field = self.create_scalar_field_py("BeadField")

        # Device Selection
        device = hoomd.device.CPU()
        self.sim = hoomd.Simulation(device=device, seed=777)

        # Grid Setup
        step_size = 10
        x_pts = np.arange(0, self.dim.x, step_size)
        y_pts = np.arange(0, self.dim.y, step_size)
        Nx, Ny = len(x_pts), len(y_pts)

        X, Y = np.meshgrid(x_pts, y_pts, indexing="ij")
        grid_points_cc3d = np.column_stack((X.ravel(), Y.ravel()))
        self.num_grid_pts = len(grid_points_cc3d)

        cell = next(iter(self.cell_list), None)
        cell_com_cc3d = (
            np.array([[cell.xCOM, cell.yCOM]])
            if cell
            else np.array([[self.dim.x / 2.0, self.dim.y / 2.0]])
        )

        all_pts_cc3d = np.vstack([grid_points_cc3d, cell_com_cc3d])
        all_pts_hoomd_2d = self.cc3d_to_hoomd(all_pts_cc3d)
        all_pts_hoomd_3d = np.hstack(
            [all_pts_hoomd_2d, np.zeros((len(all_pts_hoomd_2d), 1))]
        )

        self.com_particle_idx = self.num_grid_pts

        # Build Grid Network Topology (Linear Bonds & Angular Triplets)
        def grid_idx(i, j):
            return i * Ny + j

        linear_bonds = []
        angular_bonds = []

        # for i in range(Nx):
            # for j in range(Ny):
                # idx = grid_idx(i, j)

                # # Horizontal linear bonds
                # if i + 1 < Nx:
                    # linear_bonds.append([idx, grid_idx(i + 1, j)])
                # # Vertical linear bonds
                # if j + 1 < Ny:
                    # linear_bonds.append([idx, grid_idx(i, j + 1)])

                # # Horizontal bending angles (i-1, i, i+1)
                # if 0 < i < Nx - 1:
                    # angular_bonds.append(
                        # [grid_idx(i - 1, j), idx, grid_idx(i + 1, j)]
                    # )
                # # Vertical bending angles (j-1, j, j+1)
                # if 0 < j < Ny - 1:
                    # angular_bonds.append(
                        # [grid_idx(i, j - 1), idx, grid_idx(i, j + 1)]
                    # )

        # Cyto springs: Connect initial grid beads touching the cell to cell_com
        cyto_bonds = []
        particle_types_id = np.zeros(self.num_grid_pts, dtype=int)

        for i in range(Nx):
            for j in range(Ny):
                idx = grid_idx(i, j)
                cx, cy = grid_points_cc3d[idx]

                # Identify boundary beads along outer lattice edge
                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1:
                    particle_types_id[idx] = 3  # boundary_bead
                else:
                    # Check cell field at voxel
                    ix, iy = int(round(cx)), int(round(cy))
                    cell_at_pixel = (
                        self.cell_field[ix, iy, 0]
                        if (0 <= ix < self.dim.x and 0 <= iy < self.dim.y)
                        else None
                    )

                    if cell_at_pixel is not None:
                        particle_types_id[idx] = 2  # adhesion_bead
                        cyto_bonds.append([idx, self.com_particle_idx])
                    else:
                        particle_types_id[idx] = 0  # free_bead

        # Snapshot & Box Configuration
        BOX_X = self.dim.x + (2.0 * self.padding)
        BOX_Y = self.dim.y + (2.0 * self.padding)

        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = [BOX_X, BOX_Y, 0, 0, 0, 0]

        snapshot.particles.N = self.num_grid_pts + 1
        snapshot.particles.types = [
            "free_bead",
            "cell_com",
            "adhesion_bead",
            "boundary_bead",
        ]

        total_bonds = len(linear_bonds) + len(cyto_bonds)
        snapshot.bonds.N = total_bonds
        snapshot.bonds.types = ["linear_spring", "no_bond", "cyto_spring"]

        snapshot.angles.N = len(angular_bonds)
        snapshot.angles.types = ["angular_spring", "no_angle"]

        # Populate Particle Data
        for k in range(self.num_grid_pts):
            snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            snapshot.particles.typeid[k] = particle_types_id[k]

        snapshot.particles.position[self.com_particle_idx] = all_pts_hoomd_3d[
            self.com_particle_idx
        ]
        snapshot.particles.typeid[self.com_particle_idx] = 1  # cell_com

        # Populate Bond Groups
        b_idx = 0
        for b in linear_bonds:
            snapshot.bonds.group[b_idx] = b
            snapshot.bonds.typeid[b_idx] = 0  # linear_spring
            b_idx += 1

        for b in cyto_bonds:
            snapshot.bonds.group[b_idx] = b
            snapshot.bonds.typeid[b_idx] = 2  # cyto_spring
            b_idx += 1

        # Populate Angle Groups
        for a_idx, a in enumerate(angular_bonds):
            snapshot.angles.group[a_idx] = a
            snapshot.angles.typeid[a_idx] = 0  # angular_spring

        # Initialize Simulation State
        self.sim.create_state_from_snapshot(snapshot)

        # Integrator & Force Setup
        self.integrator = hoomd.md.Integrator(self.md_dt)

        # Linear bond potential
        beamspring = hoomd.md.bond.Harmonic()
        beamspring.params["linear_spring"] = dict(
            k=self.spring_k, r0=self.spring_r0
        )
        beamspring.params["cyto_spring"] = dict(
            k=self.spring_k_cyto, r0=self.spring_r0
        )
        beamspring.params["no_bond"] = dict(k=0.0, r0=0.0)

        # Angular bond potential
        harmangle = hoomd.md.angle.Harmonic()
        harmangle.params["angular_spring"] = dict(
            k=self.bend_k, t0=self.bend_t0
        )
        harmangle.params["no_angle"] = dict(k=0.0, t0=0.0)

        self.integrator.forces.append(beamspring)
        self.integrator.forces.append(harmangle)

        # Integration Method: Brownian thermostat applied to free and adhesion beads
        active_filter = hoomd.filter.Type(["free_bead", "adhesion_bead"])
        integration_method = hoomd.md.methods.Brownian(
            active_filter, kT=self.md_kT
        )
        self.integrator.methods.append(integration_method)

        self.sim.operations.integrator = self.integrator
        # Reset particle momenta using Maxwell-Boltzmann distribution
        self.sim.state.thermalize_particle_momenta(
            filter=active_filter, kT=self.md_kT
        )

        self.update_bead_field() #visualize
        
        
        cell_a = self.fetch_cell_by_id(1)
        if cell_a:
            # All pixel coordinates in cell_a
            cell_pixels = {
                (pt.pixel.x, pt.pixel.y) for pt in self.get_cell_pixel_list(cell_a)
            }

            # Current snapshot
            snap = self.sim.state.get_snapshot()

            if snap.communicator.rank == 0:
                # Convert HOOMD 2D coordinates back to CC3D
                hoomd_pts = snap.particles.position[: self.num_grid_pts, :2]
                cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)

                # Check each particle position against cell_a pixels
                for particle_idx, (px, py) in enumerate(cc3d_pts):
                    ix, iy = int(round(px)), int(round(py))

                    if (ix, iy) in cell_pixels:
                        # Update particle typeid to 2 (adhesion_bead)
                        snap.particles.typeid[particle_idx] = 2

            # Commit updated state back to HOOMD
            self.sim.state.set_snapshot(snap)



    def step(self, mcs):
        """Updates cell COM, dynamically re-classifies particles inside cell pixels, runs MD, and refreshes the scalar field."""
        # Coordinates of all active cells in the simulation
        cell_pixels = set()
        for cell in self.cell_list:
            for pt in self.get_cell_pixel_list(cell):
                cell_pixels.add((pt.pixel.x, pt.pixel.y))

        # HOOMD snapshot
        snap = self.sim.state.get_snapshot()

        if snap.communicator.rank == 0:
            # Update target Cell COM particle position
            cell = next(iter(self.cell_list), None)
            if cell:
                cell_com_cc3d = np.array([cell.xCOM, cell.yCOM])
                cell_com_hoomd = self.cc3d_to_hoomd(cell_com_cc3d)
                snap.particles.position[self.com_particle_idx] = [
                    cell_com_hoomd[0],
                    cell_com_hoomd[1],
                    0.0,
                ]
                snap.particles.velocity[self.com_particle_idx] = [0.0, 0.0, 0.0]

            # Grid particle position against cell pixel locations
            hoomd_pts = snap.particles.position[: self.num_grid_pts, :2]
            cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)
            
            enter_idx = []
            leave_idx = []
            
            for particle_idx, (px, py) in enumerate(cc3d_pts):
                current_type = snap.particles.typeid[particle_idx]

                # Preserve boundary beads (type 3)
                if current_type == 3:
                    continue

                ix, iy = int(round(px)), int(round(py))

                if (ix, iy) in cell_pixels:
                    # Particle is inside a cell pixel -> set to adhesion_bead (2)
                    snap.particles.typeid[particle_idx] = 2
                    enter_idx.append(particle_idx)
                else:
                    # Particle is outside cell pixels -> set to free_bead (0)
                    if current_type == 2:
                        snap.particles.typeid[particle_idx] = 0
                        leave_idx.append(particle_idx)
                        
                        
                        
            #define A_indices to be the particle ids for new non adhesion particles entering
            new_bonds = []
            for i in enter_idx: # particle id for new adhesion particles
               for j in [self.com_particle_idx]: 
                      new_bonds.append([i, j])
            snap.bonds.N = len(new_bonds) 
            if len(new_bonds) > 0: 
                snap.bonds.group[:] = new_bonds 
                snap.bonds.typeid[:] = 0

        # Particle updates back to HOOMD
        self.sim.state.set_snapshot(snap)
        self.sim.run(50) # Run sim

        # Update bead field
        self.update_bead_field()


    def update_bead_field(self):
        """Visualizes HOOMD grid points on CC3D scalar field."""
        self.bead_field.clear()
        hoomd_pts = self.sim.state.get_snapshot().particles.position[
            : self.num_grid_pts, :2
        ]
        cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)

        for x, y in cc3d_pts:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.dim.x and 0 <= iy < self.dim.y:
                self.bead_field[ix, iy, 0] = 1.0
