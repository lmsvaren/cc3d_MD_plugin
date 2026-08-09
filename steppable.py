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
        self.md_dt = 0.002 #0.003
        self.md_kT = 0.5 #0.001
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
        """Initializes HOOMD device, snapshot, topology, and bond integrators."""
        self.bead_field = self.create_scalar_field_py("BeadField")

        # Device
        device = hoomd.device.CPU()
        self.sim = hoomd.Simulation(device=device, seed=777)

        # Grid & Particle Assembly
        pad = 0
        x_pts = np.arange(pad, self.dim.x - pad, 20)
        y_pts = np.arange(pad, self.dim.y - pad, 20)
        X, Y = np.meshgrid(x_pts, y_pts)
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

        # Snapshot & Box Setup
        BOX_X = self.dim.x + (2.0 * self.padding)
        BOX_Y = self.dim.y + (2.0 * self.padding)

        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = [BOX_X, BOX_Y, 0, 0, 0, 0]

        total_particles = self.num_grid_pts + 1
        snapshot.particles.N = total_particles
        snapshot.particles.types = ["grid_point",  "cell_com", "adhesion_bead", "boundary_bead"] #I would change "grid_point" to "free bead"

        linear_bonds = []
        cyto_bonds = []
        ## We don't just have this number of bonds in the system, will have to fix
        total_bonds= len(linear_bonds) + len(cyto_bonds)
        snapshot.bonds.N = total_bonds
        snapshot.bonds.types = ["linear_spring", "no_bond", "cyto_spring"]
        snapshot.angles.types =[ "angular_spring", "no_angle"]

        '''linear_spring: bonds between free beads
        cyto_spring: bonds between adhesion beads and cell_com
        angular_spring: angles between triplets of beads'''


        # Populate Snapshot Data
        for k in range(self.num_grid_pts):
            snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            snapshot.particles.typeid[k] = 0  # grid_point

        snapshot.particles.position[self.com_particle_idx] = all_pts_hoomd_3d[
            self.com_particle_idx
        ]
        snapshot.particles.typeid[self.com_particle_idx] = 1  # cell_com


        ## This connects all the adhesion_beads to the cell_com with a cyto_spring bond.
        ## Possibly it needs to be initialized as empty before this loop?
        ## Because if at a later time an adhesion bead turns free, it may retain a bond to the cell_com.
        for k in range(self.num_grid_pts):
            if snapshot.particles.typeid[k] == 2: ## if the particle is an adhesion bead
                snapshot.bonds.group[k] = [k, self.com_particle_idx]
                snapshot.bonds.typeid[k] = 2  # cyto_spring
       

       

        # Create Simulation State from Snapshot
        self.sim.create_state_from_snapshot(snapshot)

        # Integrator Setup (Bonds Only â€” No Pair/LJ Potentials)
        self.integrator = hoomd.md.Integrator(self.md_dt)




        # linear bond potential
        beamspring = hoomd.md.bond.Harmonic()

        # bond parameters for the linear spring between linked beads
        beamspring.params["linear_spring"] = dict(
            k=self.spring_k, r0=self.spring_r0
        )


        beamspring.params["no_bond"] = dict(
            k=0, r0=self.spring_r0
        )
       
        # bond parameters for the linear spring between cell_com and grid points

        beamspring.params["cyto_spring"] = dict(
                    k= self.spring_k_cyto , r0=self.spring_r0
                )
           
        ## angular bonds
        harmangle = hoomd.md.angle.Harmonic()
        harmangle.params['angular_spring'] = dict(k=self.bend_k, t0=self.bend_t0)
 

        ### dummy bond
        harmangle.params['no_angle'] = dict(k=0, t0=0)
       
        self.integrator.forces.append(beamspring)
        self.integrator.forces.append(harmangle)


        # Integration Method (Langevin Thermostat for grid points)
        free_particlefilter = hoomd.filter.Type(["grid_point"])
        integration_method = hoomd.md.methods.Brownian(
            free_particlefilter, kT=self.md_kT
        )
        self.integrator.methods.append(integration_method)

        self.sim.operations.integrator = self.integrator
        self.sim.state.thermalize_particle_momenta(
            filter=free_particlefilter, kT=self.md_kT
        )

        self.update_bead_field()

    def step(self, mcs):
        """Updates target COM position, integrates MD steps, and updates lattice visualization."""
        cell = next(iter(self.cell_list), None)
        if cell:
            cell_com_cc3d = np.array([cell.xCOM, cell.yCOM])
            cell_com_hoomd = self.cc3d_to_hoomd(cell_com_cc3d)

            snap = self.sim.state.get_snapshot()
            if snap.communicator.rank == 0:
                snap.particles.position[self.com_particle_idx] = [
                    cell_com_hoomd[0],
                    cell_com_hoomd[1],
                    0.0,
                ]
                snap.particles.velocity[self.com_particle_idx] = [
                    0.0,
                    0.0,
                    0.0,
                ]
            self.sim.state.set_snapshot(snap)

        # Run HOOMD steps
        self.sim.run(50)
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