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
        self.active_bonds = {}  # Dictionary to track active bonds between adhesion beads and cell COM

        # Simulation parameters 
        self.md_dt = 0.0001 
        self.md_kT = 0.001 
        self.spring_k = 100
        self.spring_k_cyto = 100
        self.spring_r0 = 20
        self.bend_k = 100.0
        self.bend_t0 = np.pi
        self.padding = 20.0

        self.num_grid_pts = 0
        self.com_particle_idx = 0
        self.track_beads = True  # Set to True to enable trails/streaks

### Change of coordinates from CC3D to HOOMD and vice versa. CC3D has a domain of [0, dim] while HOOMD has a centered domain of [-Lx/2, Lx/2].

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
        all_pts_hoomd_3d = np.hstack([all_pts_hoomd_2d, np.zeros((len(all_pts_hoomd_2d), 1))])

        self.com_particle_idx = self.num_grid_pts #The last particle in the snapshot is the cell COM particle

        # Snapshot & Box Setup
        BOX_X = self.dim.x + (2.0 * self.padding)
        BOX_Y = self.dim.y + (2.0 * self.padding)

        self.snapshot = hoomd.Snapshot()
        self.snapshot.configuration.box = [BOX_X, BOX_Y, 0, 0, 0, 0]

        self.snapshot.particles.N = self.num_grid_pts + 1 #this is the total number of particles in the simulation (grid points + cell COM)
        self.snapshot.particles.types = ["grid_point",  "cell_com", "adhesion_bead", "boundary_bead"] #I would change "grid_point" to "free bead"


        # Populate Snapshot Data
        for k in range(self.num_grid_pts):
            self.snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            self.snapshot.particles.typeid[k] = 0  # grid_point

        self.snapshot.particles.position[self.com_particle_idx] = all_pts_hoomd_3d[
            self.com_particle_idx
        ]
        self.snapshot.particles.typeid[self.com_particle_idx] = 1  # cell_com

        #define the linear bonds between the beads. Either explicitly define the bonds like in example 1 or use a system like 
        '''
        self.linear_bonds = []
        for k in range(self.num_grid_pts):
            for l in range(k + 1, self.num_grid_pts):
                # either add [k,l] to the list of bonds or don't
                if claim:
                    self.linear_bonds.append([k, l])
                else:
                    pass
        '''
        
        '''#example 1
        self.linear_bonds=[]
        for k in range(50,59):
            self.linear_bonds.append([k,k+1])
        '''
        
        '''
        #example 2
        self.linear_bonds = []

        #Horizontal bonds
        for row in range(10):
            for col in range(3):
                idx = row * 10 + 3*col
                self.linear_bonds.append([idx, idx + 1])

        # Vertical bonds
        for row in range(9):
            for col in range(10):
                idx = row * 10 + col
                self.linear_bonds.append([idx, idx + 10])
        '''
        
        #example 3
        # 1. Define the 5 fibers by their bead IDs
        fiber_chains = [
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], # Row 5 (Horizontal)
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], # Row 3 (Horizontal)
            [0,  11, 22, 33, 44, 55, 66, 77, 88, 99], # Diagonal (/)
        ]

        # 2. Get linear bonds & pin the fiber ends
        self.linear_bonds = []
        boundary_indices = set()

        for chain in fiber_chains:
            boundary_indices.add(chain[0])   # Pin start of fiber
            boundary_indices.add(chain[-1])  # Pin end of fiber
            
            for i in range(len(chain) - 1):
                bond = [chain[i], chain[i+1]]
                if bond not in self.linear_bonds:
                    self.linear_bonds.append(bond)

        
        
        
        self.snapshot.bonds.N = len(self.linear_bonds) + self.num_grid_pts  # linear bonds + grid-to-COM bonds
        self.snapshot.bonds.types = [ "no_bond","linear_spring", "cyto_spring"]
        self.snapshot.angles.types =[ "no_angle", "angular_spring"]

 

        # Create linear bonds
        for i, pair in enumerate(self.linear_bonds):
            self.snapshot.bonds.group[i] = pair
            self.snapshot.bonds.typeid[i] = 1  # linear_spring

        # Create grid-to-COM bonds ONLY for grid points (excludes com_particle_idx)
        for i in range(self.num_grid_pts):
            bond_idx = i + len(self.linear_bonds)
            self.snapshot.bonds.group[bond_idx] = [i, self.com_particle_idx]
            if self.snapshot.particles.typeid[i] == 2:  # adhesion bead
                self.snapshot.bonds.typeid[bond_idx] = 2  # cyto_spring
            else:
                self.snapshot.bonds.typeid[bond_idx] = 0  # no_bond


                # Generate the boundary index set for a 10x10 grid (0-9, 90-99, 10k, 10k+9)
        boundary_indices = set()

        #----
        # Top and Bottom rows
        boundary_indices.update(range(0, 10))        # 0 through 9
        boundary_indices.update(range(90, 100))     # 90 through 99

        # Left and Right edges
        for k in range(10):
            boundary_indices.add(10 * k)             # 0, 10, 20, ... 90
            boundary_indices.add(10 * k + 9)         # 9, 19, 29, ... 99


        # Assign typeids in snapshot initialization
        for k in range(self.num_grid_pts):
            self.snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            
            if k in boundary_indices:
                self.snapshot.particles.typeid[k] = 3  # boundary_bead
            else:
                self.snapshot.particles.typeid[k] = 0  # grid_point
        #----
        
        # Programmatically build angular triplets from linear bonds
        self.angular_triplets = []

        for bond1 in self.linear_bonds:
            j = bond1[1]
            for bond2 in self.linear_bonds:
                i = bond2[0]
                if i == j:
                    self.angular_triplets.append([bond1[0], bond1[1], bond2[1]])
        
        print('LINEAR BONDS ARE',self.linear_bonds,'\n')
        print('ANGULAR TRIPLETS ARE',self.angular_triplets)

        # Allocate and assign snapshot angle arrays
        self.snapshot.angles.N = len(self.angular_triplets)
        self.snapshot.angles.types = ["no_angle", "angular_spring"]

        for idx, triplet in enumerate(self.angular_triplets):
            self.snapshot.angles.group[idx] = triplet
            self.snapshot.angles.typeid[idx] = 1  # angular_spring



        # Create Simulation State from Snapshot
        self.sim.create_state_from_snapshot(self.snapshot)

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
        free_particlefilter = hoomd.filter.Type(["grid_point", "adhesion_bead"])
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
            
            
            for particle_idx, (px, py) in enumerate(cc3d_pts):
                current_type = snap.particles.typeid[particle_idx]
                if current_type == 3 or current_type == 1:
                    continue
                ix, iy = int(round(px)), int(round(py))
                if (ix, iy) in cell_pixels:
                    snap.particles.typeid[particle_idx] = 2
                    self.active_bonds[particle_idx] = [particle_idx, self.com_particle_idx]
                else:
                    if current_type == 2:
                        snap.particles.typeid[particle_idx] = 0
                        self.active_bonds.pop(particle_idx, None)

            # Apply state directly to snapshot bond types
            for i in range(self.num_grid_pts):
                bond_idx = i + len(self.linear_bonds)
                if snap.particles.typeid[i] == 2:  # adhesion bead
                    snap.bonds.typeid[bond_idx] = 2  # cyto_spring
                else:
                    snap.bonds.typeid[bond_idx] = 0  # no_bond
        # Particle updates back to HOOMD
        self.sim.state.set_snapshot(snap)
        self.sim.run(50) # Run sim

        # Update bead field
        self.update_bead_field()

    def update_bead_field(self):
    # Reset field buffer if tracking is OFF (prevents streaks)
        if not getattr(self, 'track_beads', False):
            self.bead_field[:, :, :] = 0.0

        self.bead_field.clear()
        snap = self.sim.state.get_snapshot()
        hoomd_pts = snap.particles.position[: self.num_grid_pts, :2]
        cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)
        typeids = snap.particles.typeid[: self.num_grid_pts]

        for (x, y), tid in zip(cc3d_pts, typeids):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.dim.x and 0 <= iy < self.dim.y:
                # +1 avoids ambiguity with the cleared background value of 0
                self.bead_field[ix, iy, 0] = float(tid) + 1.0

        # Draw COM pixel directly onto BeadField with a distinct scalar value
        cell = next(iter(self.cell_list), None)
        if cell:
            cx = int(np.clip(round(cell.xCOM), 0, self.dim.x - 1))
            cy = int(np.clip(round(cell.yCOM), 0, self.dim.y - 1))
            self.bead_field[cx, cy, 0] = 10.0



