#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:07:32 2025

@author: lukemcculloch


main_with_amr.py - Main program to run PyCFD with Adaptive Mesh Refinement

This script demonstrates how to run a simulation with the PyCFD solver
using adaptive mesh refinement. It loads a problem configuration, initializes
the solver, and runs a simulation with periodic mesh adaptation.

Author: Luke McCulloch
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import PyCFD modules
from System2D import Grid
from Solvers import Solvers
from DataHandler import DataHandler
from Parameters import Parameters
from PlotGrids import PlotGrid


def run_with_amr(problem_type='vortex'):
    """
    Run a simulation with adaptive mesh refinement
    
    Parameters:
    -----------
    problem_type : str
        Type of problem to solve ('vortex', 'airfoil', 'cylinder', or 'shock-diffraction')
    """
    # Path to the case directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Select the appropriate case based on problem_type
    if problem_type == 'vortex':
        case_dir = os.path.join(base_dir, 'cases', 'case_unsteady_vortex')
        project_name = 'vortex'
        grid_type = 'tri'
        solver_type = 'explicit_unsteady_solver'
        tfinal = 0.5
        dt = 0.01
    elif problem_type == 'airfoil':
        case_dir = os.path.join(base_dir, 'cases', 'case_steady_airfoil')
        project_name = 'airfoil'
        grid_type = 'quad'
        solver_type = 'explicit_steady_solver'
        tfinal = 1.0
        dt = 0.01
    elif problem_type == 'cylinder':
        case_dir = os.path.join(base_dir, 'cases', 'case_steady_cylinder')
        project_name = 'cylinder'
        grid_type = 'tri'
        solver_type = 'explicit_steady_solver'
        tfinal = 1.0
        dt = 0.01
    elif problem_type == 'shock-diffraction':
        case_dir = os.path.join(base_dir, 'cases', 'case_shock_diffraction')
        project_name = 'shock'
        grid_type = 'quad'
        solver_type = 'explicit_unsteady_solver_efficient_shockdiffraction'
        tfinal = 0.7
        dt = 0.01
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    print(f"Running {problem_type} problem with AMR")
    print(f"Case directory: {case_dir}")
    
    # Create data handler
    dhandler = DataHandler(project_name=project_name, path_to_inputs_folder=case_dir)
    
    # Load the grid
    grid = Grid(generated=False,
                dhandle=dhandler,
                type_=grid_type,
                winding='ccw')
    
    # Initialize solver
    solver = Solvers(mesh=grid)
    
    # Print solver parameters
    solver.print_nml_data()
    
    # Initialize solver for the specific problem
    solver.solver_boot(flowtype=problem_type)
    
    # Save initial solution
    solver.write_solution_to_vtk('initial_solution.vtk')
    
    # Plot initial grid if AMR is enabled
    if solver.do_amr and hasattr(solver, 'amr'):
        solver.amr.plot_mesh(filename='initial_mesh.png')
    
    # Run the solver with AMR
    solver.solver_solve(tfinal=tfinal, dt=dt, solver_type=solver_type)
    
    # Save final solution
    solver.write_solution_to_vtk('final_solution.vtk')
    
    # Plot final grid if AMR is enabled
    if solver.do_amr and hasattr(solver, 'amr'):
        solver.amr.plot_mesh(filename='final_mesh.png')
    
    print(f"Simulation completed for {problem_type}")
    return solver


def main():
    """Main function to run simulations"""
    # Parse command line arguments for problem type
    import argparse
    parser = argparse.ArgumentParser(description='Run PyCFD with Adaptive Mesh Refinement')
    parser.add_argument('--problem', type=str, default='vortex',
                        choices=['vortex', 'airfoil', 'cylinder', 'shock-diffraction'],
                        help='Problem type to solve')
    args = parser.parse_args()
    
    # Run the simulation
    solver = run_with_amr(problem_type=args.problem)
    
    # Return solver for further analysis
    return solver


if __name__ == "__main__":
    solver = main()