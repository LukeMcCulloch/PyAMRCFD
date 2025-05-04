#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 10:27:33 2025

@author: lukemcculloch


Solvers_AMR_Integration.py

This file shows the modifications needed to integrate AMR into your Solvers.py file.
It contains only the modified methods - you should merge these changes into your existing Solvers.py.

Author: Luke McCulloch
"""

import weakref
import numpy as np
import matplotlib.pyplot as plt

from System2D import Grid, Cell, Face, Node
from flux import roe3D, roe2D
from BoundaryConditions import BC_states
from Parameters import Parameters
from Utilities import default_input
from DataHandler import DataHandler
from Debugging import dbInterfaceFlux, dbRoeFlux
from Stencil import StencilLSQ
from AdaptiveMeshRefinement import AMR

class Solvers(object):
    """
    2D Euler/NS equations = 4 equations:
    
    (1)continuity
    (2)x-momentum
    (3)y-momentum
    (4)energy
    
    This is a modified version with AMR support
    """
    
    def __init__(self, mesh):
        # Original initialization
        self.nq = 4 # Euler system size
        self.solver_initialized = False
        self.do_mms = False
        self.mesh = mesh
        self.dim = mesh.dim
        self.dpn = self.dim + 2  # rho, u, v, p
        
        # Parse input parameters
        self.iparams = self.mesh.dhandle.inputParameters
        self.Parameters = Parameters(self.iparams)
        self.aoa = self.Parameters.aoa
        self.M_inf = self.Parameters.M_inf
        self.CFL = self.Parameters.CFL
        self.use_limiter = self.Parameters.use_limiter
        self.second_order = self.Parameters.second_order
        self.eig_limiting_factor = self.Parameters.eig_limiting_factor
        self.inviscid_flux = self.Parameters.inviscid_flux
        self.compute_te_mms = self.Parameters.compute_te_mms
        
        # AMR parameters
        self.do_amr = self.Parameters.do_amr
        self.refine_threshold = self.Parameters.refine_threshold
        self.coarsen_threshold = self.Parameters.coarsen_threshold
        self.amr_frequency = getattr(self.Parameters, 'amr_frequency', 10)
        
        # Initialize solution data
        self.u = np.zeros((mesh.nCells, self.nq), float)  # conservative variables at cells# /nodes
        self.w = np.zeros((mesh.nCells, self.nq), float)  # primitive variables at cells# /nodes
        self.gradw = np.zeros((mesh.nCells, self.nq, self.dim), float)  # gradients of w at cells# /nodes
        
        #source (forcing) data, if any
        self.f = np.zeros((mesh.nCells, self.nq), float)  # source (forcing) data
        self.u0 = np.zeros((mesh.nCells, self.nq), float) # work array
        
        # Solution convergence
        self.res = np.zeros((mesh.nCells, self.nq), float)  # residual vector
        self.res_norm = np.zeros((self.nq, 1), float)
        self.res_norm_shock = np.zeros((self.nq, 3), float)
        
        # Local convergence storage saved for speed
        self.gradw1 = np.zeros((self.nq, self.dim), float)
        self.gradw2 = np.zeros((self.nq, self.dim), float)
        
        # Update (pseudo) time step data
        self.dtau = np.zeros((mesh.nCells), float)
        
        # Accessor integers for clarity
        self.ir = 0  # density
        self.iu = 1  # x-velocity
        self.iv = 2  # y-velocity
        self.ip = 3  # pressure
        
        # Fluid properties
        self.gamma = 1.4  # Ratio of specific heats for air
        self.rho_inf = 1.0
        self.u_inf = 1.0
        self.v_inf = 0.0
        self.p_inf = 1.0 / self.gamma
        
        # Flux
        self.uL3d = np.zeros(5, float)       # conservative variables in 3D
        self.uR3d = np.zeros(5, float)       # conservative variables in 3D
        self.n12_3d = np.zeros(3, float)     # face normal in 3D
        self.num_flux3d = np.zeros(5, float) # numerical flux in 3D
        self.wsn = np.zeros((self.mesh.nCells), float)  # max wave speed array
        
        # Cell-centered limiter data
        self.limiter_beps = 1.0e-14
        self.phi = np.zeros((mesh.nCells), float)
        
        # Least squared gradient
        self.cclsq = np.asarray( [StencilLSQ(cell, mesh) for cell in mesh.cells] )
        #e.g.
        #self.cclsq[0].nghbr_lsq #bulk list of all cells in the 'extended cell halo'
        
        # Precompute least squared gradient coefficients
        self.compute_lsq_coefficients()
        self.test_lsq_coefficients()
        
        # Residual data
        self.num_flux = np.zeros(4, float)
        self.ub = np.zeros(4, float)
        self.wave_speed = 0.
        
        # Local copies of data
        self.unit_face_normal = np.zeros((2), float)
        
        # Exact solution data
        self.w_initial = np.zeros(4, float)
        
        # Boundary conditions
        self.bc_type = {}
        global_counter = 0
        for bound in mesh.bound:
            btype = bound.bc_type
            nfaces = bound.nbfaces
            for f in range(nfaces):
                self.bc_type[global_counter] = btype
                global_counter += 1
        
        # Initialize AMR if enabled
        if self.do_amr:
            self.amr = AMR(mesh, solver=self, max_level=3)
            self.amr.set_thresholds(self.refine_threshold, self.coarsen_threshold)
            print("AMR initialized with:")
            print(f"  Refinement threshold: {self.refine_threshold}")
            print(f"  Coarsening threshold: {self.coarsen_threshold}")
            print(f"  AMR frequency: {self.amr_frequency}")
    
    def perform_amr_step(self, field_name="density"):
        """Perform one adaptive mesh refinement step"""
        if not self.do_amr or not hasattr(self, 'amr'):
            print("AMR is not enabled or initialized")
            return False
        
        # Perform AMR
        result = self.amr.adapt_mesh(field_name)
        
        # Print statistics
        print(f"\nAMR Step {self.amr.amr_step}:")
        print(f"Refined {len(result['refined'])} cells")
        print(f"Coarsened {len(result['coarsened'])} cells")
        print(f"Total cells: {result['num_cells']}")
        print(f"Active cells: {result['num_active_cells']}")
        print(f"Maximum refinement level: {result['max_level']}")
        
        # Recompute LSQ coefficients after mesh adaptation
        self.compute_lsq_coefficients()
        
        # Optional: Output the adapted mesh for visualization
        self.write_solution_to_vtk(f'adapted_mesh_step_{self.amr.amr_step}.vtk')
        
        return True
    
    def explicit_steady_solver(self, 
                               tfinal=1.0, 
                               dt=.01, 
                               tol=1.e-5, 
                               max_iteration=500):
        """
        Explicit Steady Solver: Ut + Fx + Gy = S, Ut -> 0.
        
        Modified to incorporate AMR
        """
        print('call explicit_steady_solver')
        time = 0.0
        
        self.t_final = tfinal
        self.max_iteration = max_iteration
        
        print()
        print("---------------------------------------")
        print(" Pseudo time-Stepping")
        print('max_iteration = ', max_iteration)
        print()
        
        i_iteration = 0
        pseudo_time_loop = True
        
        while (pseudo_time_loop):
            # Compute the residual
            self.compute_residual(roe3D, 'vk_limiter')
            
            # Compute the residual norm for checking convergence
            self.compute_residual_norm()
            
            # Initial (no solution update yet)
            if (i_iteration == 0):
                # Save the initial max res norm
                res_norm_initial = np.copy(self.res_norm)
                
                print(" Iteration   max(res)    max(res)/max(res)_initial ")
                print(i_iteration, np.max(self.res_norm[:]), 1.0)
            else:
                # After the first solution update
                print(i_iteration, np.max(self.res_norm[:]),
                      np.max(self.res_norm[:] / res_norm_initial[:]))
            
            # Exit if the res norm is reduced below the tolerance
            if (np.max(self.res_norm[:] / res_norm_initial[:]) 