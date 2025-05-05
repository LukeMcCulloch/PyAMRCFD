#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:49:32 2025

@author: lukemcculloch


testAMR.py - Test and demonstration of adaptive mesh refinement for PyCFD

This script demonstrates how to use the AMR class to perform adaptive mesh 
refinement on an unstructured grid. It sets up a test case with a strong
gradient (like a shock or vortex) and shows the mesh adaptation process.

Author: Luke McCulloch
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path so we can import PyCFD modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from System2D import Grid
from Solvers import Solvers as EulerSolver
from AdaptiveMeshRefinement import AMR
from Parameters import Parameters
from DataHandler import DataHandler

from Solvers import TestInviscidVortex, TestSteadyCylinder, TestSteadyAirfoil, TestTEgrid, TestShockDiffractiongrid


def test_amr_vortex():
    """Test adaptive mesh refinement with a vortex initial condition"""
    print("=== Testing AMR with Vortex ===")
    
    test = TestInviscidVortex()
    grid = test.grid
    
    # # Create a simple grid
    # grid = Grid(type_='quad', m=20, n=20, 
    #             xb=-10.0, xe=10.0, yb=-10.0, ye=10.0, 
    #             winding='ccw')
    
    # Initialize solver
    solver = EulerSolver(mesh=test.grid)
    solver.solver_boot(flowtype='vortex')
    
    # Create AMR manager
    #amr = AMR(grid, solver = solver)
    amr = AMR(grid, solver = solver, max_level=3)
    amr.set_thresholds(refine_threshold=0.12, coarsen_threshold=0.05)
    
    
    # Plot initial mesh 
    solver.write_solution_to_vtk('initial_vortex.vtk')
    
    #Run solver 
    solver.solver_solve(
        tfinal=0.025, dt=0.01, 
        solver_type='explicit_unsteady_solver')
    
    # Plot solution
    solver.write_solution_to_vtk('final_unadapted_vortex.vtk')
    amr.plot_mesh(filename='final_vortex_solution_unadapted.png')
    
    # Perform AMR steps
    for step in range(3):
        print(f"\nPerforming AMR step {step+1}")
        result = amr.adapt_mesh(field_name="density")
        
        print(f"Refined {result['refined']} cells")
        print(f"Coarsened {result['coarsened']} cells")
        print(f"Total cells: {result['num_cells']}")
        print(f"Active cells: {result['num_active_cells']}")
        print(f"Maximum refinement level: {result['max_level']}")
        
        # Plot adapted mesh
        amr.plot_mesh(filename=f'adapted_mesh_step_{step+1}.png')
        
        
        # Run solver on adapted mesh (optional)
        solver.solver_solve(
            tfinal=0.05, dt=0.01, 
            solver_type='explicit_unsteady_solver')
        
        # Output solution for visualization
        #solver.write_solution_to_vtk(f'adapted_vortex_step_{step+1}.vtk')
        solver.write_amr_solution_to_vtk(f'adapted_vortex_step_{step+1}.vtk')
    
    # Output final mesh statistics
    stats = amr.get_mesh_statistics()
    print("\nFinal Mesh Statistics:")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Active cells: {stats['active_cells']}")
    print(f"Maximum refinement level: {stats['max_level']}")
    print("Cells per level:")
    for level, count in stats['level_counts'].items():
        print(f"  Level {level}: {count} cells")
    print(f"Min cell volume: {stats['min_volume']:.6f}")
    print(f"Max cell volume: {stats['max_volume']:.6f}")
    print(f"Mean cell volume: {stats['mean_volume']:.6f}")


def test_amr_shock():
    """Test adaptive mesh refinement with a shock diffraction problem"""
    print("=== Testing AMR with Shock Diffraction ===")
    
    test = TestShockDiffractiongrid()
    grid = test.grid
    
    
    # Create a test grid for shock diffraction (or load from file)
    # grid = Grid(type_='quad', m=40, n=40, 
    #             xb=0.0, xe=4.0, yb=0.0, ye=3.0, 
    #             winding='ccw')
    
    # Initialize solver
    #solver = EulerSolver(mesh=grid)
    solver = EulerSolver(mesh=test.grid)
    solver.solver_boot(flowtype='shock-diffraction')
    
    # Create AMR manager
    amr = AMR(grid, solver, max_level=2)
    amr.set_thresholds(refine_threshold=0.15, coarsen_threshold=0.05)
    
    # Plot initial mesh and solution
    solver.write_solution_to_vtk('initial_shock.vtk')
    amr.plot_mesh(filename='initial_shock_mesh.png')
    
    # Run solver to develop the shock
    print("\nRunning initial solver to develop shock...")
    solver.solver_solve(tfinal=0.025, dt=0.001, solver_type='explicit_unsteady_solver_efficient_shockdiffraction')
    
    # Output solution
    solver.write_solution_to_vtk('shock_before_amr.vtk')
    
    # Perform AMR steps
    for step in range(3):
        print(f"\nPerforming AMR step {step+1}")
        result = amr.adapt_mesh(field_name="density")
                
        print(f"Refined {result['refined']} cells")
        print(f"Coarsened {result['coarsened']} cells")
        print(f"Total cells: {result['num_cells']}")
        print(f"Active cells: {result['num_active_cells']}")
        print(f"Maximum refinement level: {result['max_level']}")
        
        # Plot adapted mesh
        amr.plot_mesh(filename=f'adapted_shock_mesh_step_{step+1}.png')
        
        # Run solver on adapted mesh
        print(f"\nRunning solver on adapted mesh (step {step+1})...")
        solver.solver_solve(tfinal=0.0025, dt=0.0005, solver_type='explicit_unsteady_solver_efficient_shockdiffraction')
        
        # Output solution for visualization
        solver.write_solution_to_vtk(f'adapted_shock_step_{step+1}.vtk')


    # Output final mesh statistics
    stats = amr.get_mesh_statistics()
    print("\nFinal Mesh Statistics:")
    print(f"Total cells: {stats['total_cells']}")
    print(f"Active cells: {stats['active_cells']}")
    print(f"Maximum refinement level: {stats['max_level']}")
    print("Cells per level:")
    for level, count in stats['level_counts'].items():
        print(f"  Level {level}: {count} cells")
    print(f"Min cell volume: {stats['min_volume']:.6f}")
    print(f"Max cell volume: {stats['max_volume']:.6f}")
    print(f"Mean cell volume: {stats['mean_volume']:.6f}")
    
    
def main():
    """Main function to run AMR tests"""
    # Test AMR with a vortex
    #test_amr_vortex()
    
    # Test AMR with shock diffraction
    # Uncomment to test with shock diffraction
    test_amr_shock()
    
    print("\nAll AMR tests completed.")


if __name__ == "__main__":
    main()