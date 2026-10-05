import hoomd
import hoomd.md
import numpy as np
import json
import os
from cc3d.core.PySteppables import *
import matplotlib.pyplot as plt

class FiberPlotterSteppable(SteppableBasePy):

    def __init__(self, hybrid_steppable, frequency=10):
        SteppableBasePy.__init__(self, frequency)
        self.hybrid_steppable = hybrid_steppable
        
        # Save plots to a dedicated "fiber_plots" folder in current directory
        # self.output_dir = os.path.join(os.getcwd(), "fiber_plots")
        # os.makedirs(self.output_dir, exist_ok=True)

    def step(self, mcs):
        if not self.hybrid_steppable or not self.hybrid_steppable.sim:
            return

        snap = self.hybrid_steppable.sim.state.get_snapshot()
        pts_cc3d = self.hybrid_steppable.hoomd_to_cc3d(snap.particles.position[:, :2])
        typeids = snap.particles.typeid

        fig, ax = plt.subplots(figsize=(7, 7))

        # DRAW CELL OUTLINE & PIXELS
        cell_mask = np.zeros((self.dim.x, self.dim.y), dtype=bool)

        for cell in self.cell_list:
            pixel_list = self.get_cell_pixel_list(cell)
            for pixel_tracker_data in pixel_list:
                pt = pixel_tracker_data.pixel
                if 0 <= pt.x < self.dim.x and 0 <= pt.y < self.dim.y:
                    cell_mask[pt.x, pt.y] = True

        # Draw cell interior as a subtle shaded region
        if np.any(cell_mask):
            # Contour trace around the cell boundary
            ax.contour(
                cell_mask.T, 
                levels=[0.5], 
                colors=['green'], 
                linewidths=1.0, 
                zorder=2
            )
            ax.imshow(
                cell_mask.T, 
                origin='lower', 
                cmap='Greens', 
                alpha=0.35, 
                extent=[0, self.dim.x, 0, self.dim.y], 
                zorder=1
            )

        # DRAW FIBER NETWORK BONDS
        for bond in self.hybrid_steppable.linear_bonds:
            p1, p2 = pts_cc3d[bond[0]], pts_cc3d[bond[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='gray', linewidth=1.0, zorder=3)

        # DRAW MATRIX & ADHESION BEADS
        free_pts = pts_cc3d[typeids == 0]
        adhesion_pts = pts_cc3d[typeids == 2]
        boundary_pts = pts_cc3d[typeids == 3]

        if len(free_pts) > 0:
            ax.scatter(free_pts[:, 0], free_pts[:, 1], c='blue', s=10, label='Free Bead', zorder=4)
        if len(adhesion_pts) > 0:
            ax.scatter(adhesion_pts[:, 0], adhesion_pts[:, 1], c='red', s=25, label='Focal Adhesion', zorder=5)
        if len(boundary_pts) > 0:
            ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='black', s=15, label='Boundary Anchor', zorder=4)

        # Formatting
        ax.set_xlim(0, self.dim.x)
        ax.set_ylim(0, self.dim.y)
        ax.set_aspect('equal')
        ax.set_title(f"Cell & Fiber Network Dynamics (MCS {mcs})")
        ax.legend(loc='upper right', fontsize=8)

        # Save plot
        output_dir = getattr(self, 'output_dir', '.')
        save_path = os.path.join(output_dir, f"fiber_network_mcs_{mcs:05d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class testHybridSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.sim = None
        self.integrator = None
    
        self.bead_field = None
        self.integrin_field = None
        self.active_bonds = {}  # Dictionary to track active bonds between adhesion beads and cell COM

        # Simulation parameters (K_cross = K_polymer, need to add K_cross)
        self.md_dt = 0.0001 #0.0001
        self.md_kT = 0.001 #0.001
        self.spring_k = 10 #100
        self.spring_k_cyto = 10 #100
        self.spring_r0 = 20 #20
        self.bend_k = 10.0 #10.0
        self.bend_t0 = np.pi #np.pi
        self.padding = 20.0 #20

        # Parameters for eq. 4: Catch-slip (FAs are modeled as clusters of catch-slip bonds)
        self.gamma = 1.0 #1.0; Binding rate of integrings
        self.N_tot = 100.0 #100; max number of integrin in an FA allowed 
        self.d_0 = 1.0 #1; Base detachment rate
        self.N_0 = 1.0 #1
        self.f_star = 1.0 #1; force scale
        self.phi_s = 1.0  #1; Slip regime
        self.phi_c = 1.0 #1; Catch regime

        # Dictionary tracking integrin cluster size for active FAs: {particle_idx: N_i}
        self.fa_integrins = {}

        self.num_grid_pts = 0
        self.com_particle_idx = 0
        self.track_beads = False  # Set to True to enable trails/streaks

### Change of coordinates from CC3D to HOOMD and vice versa. CC3D has a domain of [0, dim] while HOOMD has a centered domain of [-Lx/2, Lx/2].

    def cc3d_to_hoomd(self, pos_2d):
        """Shifts CC3D domain [0, dim] to HOOMD centered domain [-Lx/2, Lx/2]."""
        offset = np.array([self.dim.x / 2.0, self.dim.y / 2.0])
        return pos_2d - offset

    def hoomd_to_cc3d(self, pos_2d):
        """Shifts HOOMD centered domain back to CC3D domain."""
        offset = np.array([self.dim.x / 2.0, self.dim.y / 2.0])
        return pos_2d + offset
        

    def catch_slip_rate(self, phi):
        """Calculates dimensionless dissociation rate d(phi) according to eq. 4:
        d(phi) = exp(phi - phi_s) + exp(-phi)
        """
        return np.exp(phi - self.phi_s) + np.exp(self.phi_c-phi)

    def integrate_catch_slip(self, N_i, Phi, dt):
        """
        Integrates focal adhesion integrin turnover (dN/dt) over time step dt
        according to Keijzer et al. Eq. 4 [1]:

            dN/dt = gamma * (N_tot - N_i) - d_0 * d(phi) * N_i

        Parameters:
            N_i (float): Current integrin cluster size (FA size).
            Phi (float): Mechanical force/tension on the focal adhesion.
            dt (float): Integration time step.
        """
        # Compute dimensionless tension per integrin: phi = (f_star * Phi) / N_i
        phi = (self.f_star * Phi) / max(N_i, 1.0)

        # Unbinding rate function d(phi)
        d_phi = self.catch_slip_rate(phi)

        # Full ODE derivative dN/dt = Assembly - Disassembly 
        dN_dt = self.gamma * (self.N_tot - N_i) - self.d_0 * d_phi * N_i

        # Forward Euler update step with boundary clipping [1.0, N_tot]  
        N_next = N_i + dN_dt * dt

        return float(np.clip(N_next, self.N_0, self.N_tot))
    
    def setup_fibers(self, fiber_chains):
        """Parse thorugh user input fibers"""
        pos_map = {}            # Maps rounded (x, y) tuple -> particle index
        all_pts_cc3d = []
        linear_bonds = []
        angular_triplets = []
        boundary_indices = set()

        #Iter through each fiber
        for raw_chain in fiber_chains:
            chain = raw_chain#[::50]
            if len(raw_chain) > 1 and raw_chain[-1] != chain[-1]:
                chain.append(raw_chain[-1])
            chain_indices = []
            # assign particle IDs to each node, deduplicate the nodes
            for pt in chain:
                pt_key = (round(float(pt[0])), round(float(pt[1]))) #2d coords
                # Add pos to mapping if not there already
                if pt_key not in pos_map:
                    idx = len(all_pts_cc3d)
                    pos_map[pt_key] = idx #id to pos
                    all_pts_cc3d.append([pt_key[0], pt_key[1]]) #2d coords
                chain_indices.append(pos_map[pt_key])

            # Endpoints of each fiber are chains
            # CHANGE TO WITHIN A SPECIFIC BORDER
            boundary_indices.add(chain_indices[0])
            boundary_indices.add(chain_indices[-1])

            # Linear bonds
            for i in range(len(chain_indices) - 1):
                bond = sorted([chain_indices[i], chain_indices[i + 1]]) #[i, i+1]
                if bond not in linear_bonds:
                    linear_bonds.append(bond)

            # Angular triplets
            for i in range(len(chain_indices) - 2):
                triplet = [chain_indices[i], chain_indices[i + 1], chain_indices[i + 2]] #[i, i+1, i+2]
                angular_triplets.append(triplet)
        return (
            np.array(all_pts_cc3d),
            linear_bonds,
            angular_triplets,
            boundary_indices
        )

    def start(self):
        """Initializes HOOMD device, snapshot, topology, and bond integrators."""

        self.bead_field = self.create_scalar_field_py("BeadField")
        self.integrin_field = self.create_scalar_field_py("IntegrinField")
        self.fiber_field = self.create_vector_field_py("FiberField")

        # Device and simulation
        device = hoomd.device.CPU()
        self.sim = hoomd.Simulation(device=device, seed=777)

        # DUMMY FIBERS (200,200)
        # user_fibers = [
        #     # Horizontal chain
        #     [(10, 50), (30, 50), (50, 50), (70, 50), (90, 50)],
        #     # Diagonal chain intersecting Fiber 1
        #     [(20, 20), (35, 35), (50, 50), (65, 65), (80, 80)],
        #     # Vertical chain
        #     [(50, 10), (50, 30), (50, 50), (50, 70), (50, 90)]
        # ]
        # Hardcoded dense fiber network spanning [0, 200] x [0, 200]
        
        # GRID OF FIBERS
        user_fibers = [
            # HORIZONTAL FIBERS (y = 10 to 190, step 20)
            [(x, 10) for x in range(10, 191, 20)],
            [(x, 30) for x in range(10, 191, 20)],
            [(x, 50) for x in range(10, 191, 20)],
            [(x, 70) for x in range(10, 191, 20)],
            [(x, 90) for x in range(10, 191, 20)],
            [(x, 110) for x in range(10, 191, 20)],
            [(x, 130) for x in range(10, 191, 20)],
            [(x, 150) for x in range(10, 191, 20)],
            [(x, 170) for x in range(10, 191, 20)],
            [(x, 190) for x in range(10, 191, 20)],

            # VERTICAL FIBERS (x = 10 to 190, step 20)
            [(10, y) for y in range(10, 191, 20)],
            [(30, y) for y in range(10, 191, 20)],
            [(50, y) for y in range(10, 191, 20)],
            [(70, y) for y in range(10, 191, 20)],
            [(90, y) for y in range(10, 191, 20)],
            [(110, y) for y in range(10, 191, 20)],
            [(130, y) for y in range(10, 191, 20)],
            [(150, y) for y in range(10, 191, 20)],
            [(170, y) for y in range(10, 191, 20)],
            [(190, y) for y in range(10, 191, 20)],]

        #     # DIAGONAL CROSS-LINKING FIBERS (Interwoven across domain)
        #     [(x, x) for x in range(10, 191, 20)],                      # Main diagonal /
        #     [(x, 200 - x) for x in range(10, 191, 20)],                # Anti diagonal \
        #     [(x, x + 40) for x in range(10, 151, 20)],                 # Upper diagonal /
        #     [(x + 40, x) for x in range(10, 151, 20)],                 # Lower diagonal /
        # ]
        
        # # OUR SPLINES
        # with open('/Users/svaren/Downloads/testing_sim/shg_splines_42.json', 'r', encoding='utf-8') as file:
            # data = json.load(file)
        # user_fibers = data['splines']
        
        # # ONE VERTICAL FIBER
        # spline_coords = [(375, y) for y in range(0, 501, 5)]
        # user_fibers = [spline_coords]
        
        # user_fibers = [[(round(float(pt[0])), round(float(pt[1]))) for pt in chain] for chain in user_fibers]

        # Grid & Particle Assembly
        # grid_spacing = 20
        # pad = 0 #edge padding
        # x_pts = np.arange(pad, self.dim.x - pad, grid_spacing) 
        # y_pts = np.arange(pad, self.dim.y - pad, grid_spacing)
        # Nx, Ny = len(x_pts), len(y_pts)

        # X, Y = np.meshgrid(x_pts, y_pts)
        # grid_points_cc3d = np.column_stack((X.ravel(), Y.ravel()))

        # Get bonds/triplets info from fibers
        grid_points_cc3d, self.linear_bonds, self.angular_triplets, boundary_indices = self.setup_fibers(user_fibers)

        self.num_grid_pts = len(grid_points_cc3d)

        # Cell COM 
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

        # Snapshot & Box Setup for HOOMD
        BOX_X = self.dim.x + (2.0 * self.padding)
        BOX_Y = self.dim.y + (2.0 * self.padding)

        self.snapshot = hoomd.Snapshot()
        self.snapshot.configuration.box = [BOX_X, BOX_Y, 0, 0, 0, 0]

        self.snapshot.particles.N = self.num_grid_pts + 1 #this is the total number of particles in the simulation (grid points + cell COM)
        self.snapshot.particles.types = ["grid_point",  "cell_com", "adhesion_bead", "boundary_bead"] #I would change "grid_point" to "free bead"

        # Boundary indices (previously below the follwoing section)
        # boundary_indices = set()
        # for r in range(Ny):
        #     for c in range(Nx):
        #         idx = r * Nx + c
        #         if r == 0 or r == Ny-1 or c == 0 or c == Nx-1:
        #             boundary_indices.add(idx)

    
        # Populate Snapshot Data
        for k in range(self.num_grid_pts):
            self.snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            if k in boundary_indices: #this was changed
                self.snapshot.particles.typeid[k] = 3  # boundary bead
            else:
                self.snapshot.particles.typeid[k] = 0 # grid point

        self.snapshot.particles.position[self.com_particle_idx] = all_pts_hoomd_3d[self.com_particle_idx]
        self.snapshot.particles.typeid[self.com_particle_idx] = 1  # cell_com

        #define the linear bonds between the beads. Either explicitly define the bonds like in example 1 or use a system like 
        
        '''#example 1
        self.linear_bonds=[]
        for k in range(50,59):
            self.linear_bonds.append([k,k+1])
        '''
        
        
        # #example 2: Linear Bonds
        # self.linear_bonds = []

        # #Horizontal bonds
        # for row in range(Ny):
        #     for col in range(Nx-1):
        #         idx = row * Nx + col
        #         self.linear_bonds.append([idx, idx + 1])

        # # Vertical bonds
        # for row in range(Ny-1):
        #     for col in range(Nx):
        #         idx = row * Nx + col
        #         self.linear_bonds.append([idx, idx + Nx])
        
        
        '''#example 3
        fiber_chains = [
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59], # Row 5 (Horizontal)
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39], # Row 3 (Horizontal)
            [0,  11, 22, 33, 44, 55, 66, 77, 88, 99], # Diagonal (/)
        ]

        self.linear_bonds = []
        boundary_indices = set()

        for chain in fiber_chains:
            boundary_indices.add(chain[0])   # Pin start of fiber
            boundary_indices.add(chain[-1])  # Pin end of fiber
            
            for i in range(len(chain) - 1):
                bond = [chain[i], chain[i+1]]
                if bond not in self.linear_bonds:
                    self.linear_bonds.append(bond)
        '''
        
        self.snapshot.bonds.N = len(self.linear_bonds) + self.num_grid_pts  # total bonds = linear bonds + grid-to-COM bonds;
        self.snapshot.bonds.types = [ "no_bond","linear_spring", "cyto_spring"]
        # self.snapshot.angles.types =[ "no_angle", "angular_spring"]

        # Create linear bonds in snapshot
        for i, pair in enumerate(self.linear_bonds):
            self.snapshot.bonds.group[i] = pair
            self.snapshot.bonds.typeid[i] = 1  # linear_spring

        # Create grid-to-COM bonds ONLY for grid points (excludes com_particle_idx)
        for i in range(self.num_grid_pts):
            bond_idx = i + len(self.linear_bonds)
            self.snapshot.bonds.group[bond_idx] = [i, self.com_particle_idx]
            # if self.snapshot.particles.typeid[i] == 2:  # adhesion bead
            #     self.snapshot.bonds.typeid[bond_idx] = 2  # cyto_spring
            # else:
            self.snapshot.bonds.typeid[bond_idx] = 0  # no_bond, default


        # MOVED ABOVE
        #         # Generate the boundary index set for a 10x10 grid (0-9, 90-99, 10k, 10k+9)
        # boundary_indices = set()

        # #----
        # # Top and Bottom rows
        # boundary_indices.update(range(0, 10))        # 0 through 9
        # boundary_indices.update(range(90, 100))     # 90 through 99

        # # Left and Right edges
        # for k in range(10):
        #     boundary_indices.add(10 * k)             # 0, 10, 20, ... 90
        #     boundary_indices.add(10 * k + 9)         # 9, 19, 29, ... 99


        # # Assign typeids in snapshot initialization
        # for k in range(self.num_grid_pts):
        #     self.snapshot.particles.position[k] = all_pts_hoomd_3d[k]
            
        #     if k in boundary_indices:
        #         self.snapshot.particles.typeid[k] = 3  # boundary_bead
        #     else:
        #         self.snapshot.particles.typeid[k] = 0  # grid_point
        # #----
        
        # Programmatically build angular triplets from linear bonds
        #There is probably a more efficient way to do this, but I think it's fine as we only have to do it once.
        # self.angular_triplets = []

        # for bond1 in self.linear_bonds:
        #     j = bond1[1]
        #     for bond2 in self.linear_bonds:
        #         i = bond2[0]
        #         if i == j:
        #             self.angular_triplets.append([bond1[0], bond1[1], bond2[1]])
        
        
        # print('LINEAR BONDS ARE',self.linear_bonds,'\n')
        # print('ANGULAR TRIPLETS ARE',self.angular_triplets)

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
        self.update_integrin_field()

    # ADDED 29 sept, 2026
    def update_fa_integrins(self, dt):
        """Extracts tension force from cyto springs and updates focal adhesion integrin count."""
        snap = self.sim.state.get_snapshot() #sim snapshot
        com_pos = snap.particles.position[self.com_particle_idx][:2] #COM position

        # Iterate through all the fa_integrins (dict)
        for p_idx in list(self.fa_integrins.keys()):
            pos = snap.particles.position[p_idx][:2] #specific integrin pos
            dist = np.linalg.norm(pos-com_pos) #dist to com
            tension = self.spring_k_cyto * max(0.0, dist - self.spring_r0) #F = k * max(0, r - r0)

            # Update focal adhesion integrin size
            n_curr = self.fa_integrins[p_idx]
            self.fa_integrins[p_idx] = self.integrate_catch_slip(n_curr, tension, dt)

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
                    snap.particles.typeid[particle_idx] = 2 #adhesion bead
                    # self.active_bonds[particle_idx] = [particle_idx, self.com_particle_idx]
                    if particle_idx not in self.fa_integrins:
                        self.fa_integrins[particle_idx] = self.N_0 #initialize new site
                else:
                    if current_type == 2:
                        snap.particles.typeid[particle_idx] = 0 # grid point
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

        md_steps_per_mcs = 50
        self.sim.run(md_steps_per_mcs) # Run sim
        
        # Integrate Focal Adhesion ODEs (dN/dt) 
        # Catch-slip ODE integration
        self.update_fa_integrins(dt=self.md_dt * md_steps_per_mcs)

        # Update CC3D scalar fields for AdhesiveSatPlugin
        self.update_integrin_field() 
        self.update_bead_field()
        self.update_fiber_field()
        
    def update_bead_field(self):
    # Reset field buffer if tracking is OFF (prevents streaks)
        if not getattr(self, 'track_beads', False):
            # self.bead_field[:, :, :] = 0.0
            self.bead_field.clear()

        # self.bead_field.clear()
        snap = self.sim.state.get_snapshot()
        hoomd_pts = snap.particles.position[: self.num_grid_pts, :2]
        cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)
        typeids = snap.particles.typeid[: self.num_grid_pts]

        # # At this point we might want to use N_i values to color the beads in the BeadField maybe. 
        # for (x, y), tid in zip(cc3d_pts, typeids):
        #     ix, iy = int(round(x)), int(round(y))
        #     if 0 <= ix < self.dim.x and 0 <= iy < self.dim.y:
        #         # +1 avoids ambiguity with the cleared background value of 0
        #         self.bead_field[ix, iy, 0] = float(tid) + 1.0

        # # Draw COM pixel directly onto BeadField with a distinct scalar value
        # cell = next(iter(self.cell_list), None)
        # if cell:
        #     cx = int(np.clip(round(cell.xCOM), 0, self.dim.x - 1))
        #     cy = int(np.clip(round(cell.yCOM), 0, self.dim.y - 1))
        #     self.bead_field[cx, cy, 0] = 10.0

        for (x, y), tid in zip(cc3d_pts, typeids):
            ix = int(np.clip(round(x), 0, self.dim.x - 1))
            iy = int(np.clip(round(y), 0, self.dim.y - 1))
            self.bead_field[ix, iy, 0] = float(tid) + 1.0

        cell = next(iter(self.cell_list), None)
        if cell:
            cx = int(np.clip(round(cell.xCOM), 0, self.dim.x - 1))
            cy = int(np.clip(round(cell.yCOM), 0, self.dim.y - 1))
            self.bead_field[cx, cy, 0] = 10.0
        

    def update_integrin_field(self):
        """Clear and update the CC3D IntegrinField scalar field with current
        N_i values from self.fa_integrins so that the C++ AdhesiveSatPlugin
        can read them during CPM copy attempts.
        """
        if not hasattr(self, "integrin_field") or self.integrin_field is None:
            return

        # Clear previous scalar field values
        self.integrin_field.clear()
        snap = self.sim.state.get_snapshot()

        # Map HOOMD particle positions to CC3D grid pixels and assign N_i
        for p_idx, N_i in self.fa_integrins.items():
            if p_idx >= snap.particles.N:
                continue

            pos_hoomd = snap.particles.position[p_idx][:2]
            pos_cc3d = self.hoomd_to_cc3d(pos_hoomd)

            ix = int(np.clip(round(pos_cc3d[0]), 0, self.dim.x - 1))
            iy = int(np.clip(round(pos_cc3d[1]), 0, self.dim.y - 1))

            # Write current integrin cluster size N_i to the CC3D lattice
            self.integrin_field[ix, iy, 0] = float(N_i)    

    def update_fiber_field(self):
        """Draws directional line segments for all linear bonds in the CC3D lattice."""
        self.fiber_field.clear()
        snap = self.sim.state.get_snapshot()
        
        positions_hoomd = snap.particles.position[:, :2]
        positions_cc3d = self.hoomd_to_cc3d(positions_hoomd)

        for bond in self.linear_bonds:
            idx1, idx2 = bond[0], bond[1]
            pos1 = positions_cc3d[idx1]
            pos2 = positions_cc3d[idx2]

            # Compute vector pointing along the fiber segment
            vec = pos2 - pos1
            
            # Map start position to lattice pixel
            ix = int(np.clip(round(pos1[0]), 0, self.dim.x - 1))
            iy = int(np.clip(round(pos1[1]), 0, self.dim.y - 1))

            # Set vector components (vx, vy, vz) in CC3D vector field
            self.fiber_field[ix, iy, 0] = (float(vec[0]), float(vec[1]), 0.0)

'''Initialize FAs in step(): Set self.fa\_integrins[p\_idx] = self.N\_0 when a bead becomes an adhesion site (typeid == 2), and pop it on unbinding. 
2. Parameter Calibration 
3. Dynamic Boundaries: Currentlly we define boundary indeces explicitly and it is not gebneral nor convenient.
'''