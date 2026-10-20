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
            ax.scatter(free_pts[:, 0], free_pts[:, 1], c='blue', s=5, label='Free Bead', zorder=4)
        if len(adhesion_pts) > 0:
            ax.scatter(adhesion_pts[:, 0], adhesion_pts[:, 1], c='red', s=2, label='Focal Adhesion', zorder=5)
        if len(boundary_pts) > 0:
            ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='black', s=5, label='Boundary Anchor', zorder=4)

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
        # self.active_bonds = {}     # Dictionary to track active bonds between adhesion beads and cell COM
        self.padding = 20.0          # 20; Padding for bead field
        self.site_size = 0.25e-6     # micrometer (um)

        # Simulation parameters (K_cross = K_polymer, need to add K_cross)
        self.md_dt = 0.1             # 0.0001; MD time iterations
        self.md_kT = 0.001           # 0.001; kT for Brownian motion and thermalize_particle_momenta
        self.spring_k = 0.09         # 100; linear spring constant
        self.crosslink_k = 0.09
        self.crosslink_r0 = 3

        # # Additional cross link values from Tsingos et al
        # crosslink_quant_step = 0.3 # minimum crosslinker distance and quantization step
        # crosslink_max_r = 3        # maximum crosslinker distance   
        # crosslink_bin_size = 1/3

        self.spring_k_cyto = 3.10e-4 #10; spring constant for cytoskeleton
        self.spring_r0_cyto = 0

        self.spring_r0 = 5           # 6.25; spring resting length
        self.bend_k = 3.88e-15       # 10.0; angular spring constant
        self.bend_t0 = np.pi         # np.pi; equilibrium angle (theta_0 in radians)
        self.gamma_drag = 10
        
        # Parameters for eq. 4: Catch-slip (FAs are modeled as clusters of catch-slip bonds)
        self.gamma_ode = 2.88        # 1.0; Binding rate of integrings
        self.N_tot = 390.0           # 100; max number of integrin in an FA allowed 
        self.d_0 = 2.88e-2           # 1; Base detachment rate
        self.N_0 = 1.0               # 1; Minimum boundary for n integrins (NEED??)
        self.f_star = 1.29e10        # 1; force scale
        self.phi_s = 4.02            # 1; Slip regime
        self.phi_c = 7.76            # 1; Catch regime

        self.md_steps_per_mcs = 200 # number of MD steps to run in HOOMD (Tsingos et al did 200)

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
        

### Eq 4: catch slip 
    def catch_slip_rate(self, phi):
        """Calculates dimensionless dissociation rate d(phi) according to eq. 4:

            d(phi) = exp(phi - phi_s) + exp(phi_c-phi)

        """
        return np.exp(phi - self.phi_s) + np.exp(self.phi_c - phi)

    def integrate_catch_slip(self, N_i, Phi, dt):
        """
        Integrates focal adhesion integrin turnover (dN/dt) over time step dt
        according to Keijzer et al. Eq. 4:

            dN/dt = gamma * (N_tot - N_i) - d_0 * d(phi) * N_i

        Parameters:
            N_i (float): Current integrin cluster size (FA size).
            Phi (float): Mechanical force/tension on the focal adhesion.
            dt (float): Integration time step.
            N_0 (1): to prevent scaling of phi or dN/dt
        """
        # Compute dimensionless tension per integrin: 
        # phi = (f_star * Phi) / N_i
        phi = (self.f_star * Phi) / max(N_i, self.N_0) # Should be N_i unless N_i=0

        # Unbinding rate function d(phi)
        # d(phi) = exp(phi - phi_s) + exp(phi_c-phi)
        d_phi = self.catch_slip_rate(phi)

        # Full ODE derivative dN/dt = Assembly - Disassembly 
        # dN/dt = gamma * (N_tot - N_i) - d_0 * d(phi) * N_i
        dN_dt = self.gamma_ode * (self.N_tot - N_i) - self.d_0 * d_phi * N_i

        # Forward Euler update step with boundary clipping [1.0, N_tot] 
        # CHANGE SOLVER??? O(h)
        N_next = N_i + dN_dt * dt

        return float(np.clip(N_next, self.N_0, self.N_tot)) #np.clip(a, a_min, a_max,)
    
    def setup_fibers(self, fiber_chains, edge_width=5):
        """Parse thorugh user input fibers"""
        pos_map = {}            # Maps rounded (x, y) tuple -> particle index
        all_pts_cc3d = []
        linear_bonds = []
        angular_triplets = []
        chain_members = []
        end_beads = []
        crosslink_bonds = set()

        # Iter through each fiber
        for chain_id, raw_chain in enumerate(fiber_chains):
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

            # Endpoints of each fiber are chain linked 
            chain_members.append(set(chain_indices))
            end_beads.append((chain_indices[0], chain_id))
            end_beads.append((chain_indices[-1], chain_id))

            # Linear bonds
            for i in range(len(chain_indices) - 1):
                bond = sorted([chain_indices[i], chain_indices[i + 1]]) #[i, i+1]
                if bond not in linear_bonds:
                    linear_bonds.append(bond) # MAKE MORE EFFICIENT??

            # Angular triplets
            for i in range(len(chain_indices) - 2):
                triplet = [chain_indices[i], chain_indices[i + 1], chain_indices[i + 2]] #[i, i+1, i+2]
                angular_triplets.append(triplet)

        pts = np.array(all_pts_cc3d)

        # Cross link bonds
        for end_idx, chain_id in end_beads:
            d2 = np.sum((pts - pts[end_idx]) ** 2, axis=1).astype(float) # Euclidean distance from bead and pt
            d2[list(chain_members[chain_id])] = np.inf # exclude self
            nearest = int(np.argmin(d2)) # get the nearest
            if np.isfinite(d2[nearest]):
                crosslink_bonds.add((min(end_idx, nearest), max(end_idx, nearest)))

        # Added to freeze the edge beads within 5 pixels of the edge
        near_x = (pts[:, 0] <= edge_width) | (pts[:, 0] >= self.dim.x - edge_width)
        near_y = (pts[:, 1] <= edge_width) | (pts[:, 1] >= self.dim.y - edge_width)
        frozen_indices = set(np.flatnonzero(near_x | near_y).tolist())
            
        return (
            np.array(all_pts_cc3d),
            linear_bonds,
            angular_triplets,
            [list(b) for b in crosslink_bonds], # set --> list
            frozen_indices
        )

    def start(self):
        """Initializes HOOMD device, snapshot, topology, and bond integrators."""

        # Initiate the required fields
        self.bead_field = self.create_scalar_field_py("BeadField")
        self.integrin_field = self.create_scalar_field_py("IntegrinField")
        self.fiber_field = self.create_vector_field_py("FiberField")

        # Device and simulation
        device = hoomd.device.CPU() #Eventually make is so either GPU or CPU
        self.sim = hoomd.Simulation(device=device, seed=777)

        # Set up snapshot
        self.snapshot = hoomd.Snapshot()

        # Snapshot & Box Setup for HOOMD
        BOX_X = self.dim.x + (2.0 * self.padding) # dim + 2*padding (for each side)
        BOX_Y = self.dim.y + (2.0 * self.padding)

        self.snapshot.configuration.box = [BOX_X, BOX_Y, 0, # [Lx, Ly, Lz,
                                           0, 0, 0] # xy, xz, yz] tilt factors
        # Tsingos et al 2023 has a loading_margin


        #############################################################################
        # GRID OF FIBERS
        n = 200
        step = 5
        user_fibers = (
            # HORIZONTAL FIBERS (y = 10 to 190, step 5)
            [[(x, y) for x in range(0, n, step)] for y in range(0, n, step)] +
            
            # VERTICAL FIBERS (x = 10 to 190, step 5)
            [[(x, y) for y in range(0, n, step)] for x in range(0, n, step)]
        )
        
        # # OUR SPLINES
        # with open('/Users/svaren/Downloads/shg_splines_42.json', 'r', encoding='utf-8') as file:
            # data = json.load(file)
        # user_fibers = data['splines']
        
        # # ONE VERTICAL FIBER
        # spline_coords = [(375, y) for y in range(0, 501, 5)]
        # user_fibers = [spline_coords]
        

        #############################################################################
        # Get bonds/triplets info from fibers
        grid_points_cc3d, self.linear_bonds, self.angular_triplets, crosslink_bonds, frozen_indices = self.setup_fibers(user_fibers) #crosslink_indices

        self.num_grid_pts = len(grid_points_cc3d)

        # Cell COM 
        cell = next(iter(self.cell_list), None) # Return the first cell in cell list

        # Find the cell center of mass (COM)
        cell_com_cc3d = (
            np.array([[cell.xCOM, cell.yCOM]])
            if cell
            else np.array([[self.dim.x / 2.0, self.dim.y / 2.0]])
        )

        all_pts_cc3d = np.vstack([grid_points_cc3d, cell_com_cc3d])
        all_pts_hoomd_2d = self.cc3d_to_hoomd(all_pts_cc3d)
        all_pts_hoomd_3d = np.hstack([all_pts_hoomd_2d, np.zeros((len(all_pts_hoomd_2d), 1))])

        self.com_particle_idx = self.num_grid_pts #The last particle in the snapshot is the cell COM particle

        # N particles will change if more than one cells
        self.snapshot.particles.N = self.num_grid_pts + 1 #this is the total number of particles in the simulation (grid points + cell COM)
        self.snapshot.particles.types = ["free_bead",  "cell_com", "adhesion_bead", "boundary_bead"] # 0-3 idx

        # Populate Snapshot Data with beads
        for k in range(self.num_grid_pts):

            self.snapshot.particles.position[k] = all_pts_hoomd_3d[k]

            if k in frozen_indices:
                self.snapshot.particles.typeid[k] = 3 # Cross link bead
            else:
                self.snapshot.particles.typeid[k] = 0 # Free bead

        # Cell COM
        self.snapshot.particles.position[self.com_particle_idx] = all_pts_hoomd_3d[self.com_particle_idx]
        self.snapshot.particles.typeid[self.com_particle_idx] = 1  # cell_com

        # Populate Snapshot Data with bonds
        self.snapshot.bonds.N = len(self.linear_bonds) + self.num_grid_pts  # total bonds = linear bonds + grid-to-COM bonds;
        self.snapshot.bonds.types = [ "no_bond","linear_spring", "cyto_spring", "crosslink"]

        # Create linear bonds in snapshot
        for i, pair in enumerate(self.linear_bonds):
            self.snapshot.bonds.group[i] = pair
            self.snapshot.bonds.typeid[i] = 1  # linear_spring

        # # Create cross link bonds
        # for i, pair in enumerate(crosslink_bonds):
        #     self.snapshot.bonds.group[i] = pair
        #     self.snapshot.bonds.typeid[i] = 3

        # Create bead-to-COM bonds ONLY for free beads (excludes com_particle_idx)
        for i in range(self.num_grid_pts):
            bond_idx = i + len(self.linear_bonds)
            self.snapshot.bonds.group[bond_idx] = [i, self.com_particle_idx]
            self.snapshot.bonds.typeid[bond_idx] = 0  # no_bond, default

        # print('LINEAR BONDS ARE',self.linear_bonds,'\n')
        # print('ANGULAR TRIPLETS ARE',self.angular_triplets)

        # Allocate and assign snapshot angle arrays
        self.snapshot.angles.N = len(self.angular_triplets)
        self.snapshot.angles.types = ["no_angle", "angular_spring"]

        for idx, triplet in enumerate(self.angular_triplets):
            self.snapshot.angles.group[idx] = triplet
            self.snapshot.angles.typeid[idx] = 1  # angular_spring


        print("\nHOOMD Snapshot")
        print("Particle types:      ", self.snapshot.particles.types)
        print("Linear bond types:   ", self.snapshot.bonds.types)
        print("Angular bond types:  ", self.snapshot.angles.types)
        print()


        # Create Simulation State from Snapshot
        self.sim.create_state_from_snapshot(self.snapshot)

        # Integrator Setup
        self.integrator = hoomd.md.Integrator(self.md_dt)
        
        # linear bond potential
        beamspring = hoomd.md.bond.Harmonic()

        # bond parameters for the linear spring between linked beads
        beamspring.params["linear_spring"] = dict(
            k=self.spring_k, r0=self.spring_r0
        )

        # # Cross link bonds
        # NEED TO EDIT 
        beamspring.params["crosslink"] = dict(
            k=self.crosslink_k, r0=self.crosslink_r0
        )

        # No bond
        beamspring.params["no_bond"] = dict(
            k=0, r0=self.spring_r0
        )

        # bond parameters for the linear spring between cell_com and grid points
        beamspring.params["cyto_spring"] = dict(
                    k= self.spring_k_cyto , r0=self.spring_r0_cyto
                )

        ## angular bonds
        harmangle = hoomd.md.angle.Harmonic()

        harmangle.params['angular_spring'] = dict(k=self.bend_k, t0=self.bend_t0) 



        ### dummy bond 
        harmangle.params['no_angle'] = dict(k=0, t0=0) 
        
        self.integrator.forces.append(beamspring)
        self.integrator.forces.append(harmangle)

        # Integration Method (Langevin Thermostat for grid points)
        free_particlefilter = hoomd.filter.Type(["free_bead", "adhesion_bead"])
        # Over damped Langevin
        integration_method = hoomd.md.methods.Brownian(
            free_particlefilter, kT=self.md_kT
        )

        # Add gamma value to free bead
        integration_method.gamma['free_bead'] = self.gamma_drag # gamma['adhesion_bead'] defaults to 1

        self.integrator.methods.append(integration_method)
        self.sim.operations.integrator = self.integrator

        # self.sim.state.thermalize_particle_momenta(
        #     filter=free_particlefilter, kT=self.md_kT
        # )

        # Relax field for 1000 steps


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
            dist = np.linalg.norm(pos-com_pos) #dist to comx

            # Phi
            tension = self.spring_k_cyto * dist * self.site_size # CHECK IF THIS IS RIGHT, site_size for unit check

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
                
                
        # HOOMD current snapshot
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

            # Convert coordinates
            cc3d_pts = self.hoomd_to_cc3d(hoomd_pts)
            
            # Iterate through particles
            for particle_idx, (px, py) in enumerate(cc3d_pts):
                current_type = snap.particles.typeid[particle_idx]

                # Continue if free_bead or boundary_bead (Leaves linear and cyto spring)
                if current_type == 3 or current_type == 1: 
                    continue

                # Get coordinates
                ix, iy = int(round(px)), int(round(py))

                # For the particles within a cell pixel
                if (ix, iy) in cell_pixels:
                    # Make all particles FAs
                    snap.particles.typeid[particle_idx] = 2 # adhesion bead (FA)
                    
                    # If the particle is not fa_integrins already, add it as new cite
                    if particle_idx not in self.fa_integrins:
                        self.fa_integrins[particle_idx] = self.N_0 #initialize new site
                else:
                    if current_type == 2:
                        snap.particles.typeid[particle_idx] = 0 # Make free bead
                        self.fa_integrins.pop(particle_idx, None) # Remove from fa_integrins dict

            # Apply state directly to snapshot bond types
            for i in range(self.num_grid_pts):
                bond_idx = i + len(self.linear_bonds)

                if snap.particles.typeid[i] == 2:  # adhesion bead
                    snap.bonds.typeid[bond_idx] = 2  # cyto_spring
                else:
                    snap.bonds.typeid[bond_idx] = 0  # no_bond

        # Particle updates back to HOOMD
        self.sim.state.set_snapshot(snap)

        self.sim.run(self.md_steps_per_mcs) # Run sim
        
        # Integrate Focal Adhesion ODEs (dN/dt) 
        # Catch-slip ODE integration
        self.update_fa_integrins(dt=self.md_dt * self.md_steps_per_mcs)

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
        N_i_totals = {}
        for p_idx, N_i in self.fa_integrins.items():
            if p_idx >= snap.particles.N:
                continue

            pos_hoomd = snap.particles.position[p_idx][:2]
            pos_cc3d = self.hoomd_to_cc3d(pos_hoomd)

            ix = int(np.clip(round(pos_cc3d[0]), 0, self.dim.x - 1))
            iy = int(np.clip(round(pos_cc3d[1]), 0, self.dim.y - 1))
            
            N_i_totals[(ix, iy)] = N_i_totals.get((ix,iy), 0.0) + float(N_i) # Add number of integrins

            # Write current integrin cluster size N_i to the CC3D lattice
        for (ix, iy), total in N_i_totals.items():
            self.integrin_field[ix, iy, 0] = total 

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