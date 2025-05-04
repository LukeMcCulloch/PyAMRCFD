#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:42:40 2025

@author: lukemcculloch

AdaptiveMeshRefinement.py

Implementation of adaptive mesh refinement (AMR) for PyCFD.
This module extends the Grid, Cell, Face, and Node classes to support
hierarchical refinement of an unstructured grid.

Author: Luke McCulloch
"""

import weakref
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from System2D import Grid, Cell, Face, Node
from Utilities import normalize, normalized, normalize2D, normalized2D, triangle_area

from Stencil import StencilLSQ

class AMR:
    """
    Main class to handle Adaptive Mesh Refinement operations.
    This works as a wrapper around the Grid class, providing additional
    functionality for adaptive refinement.
    """
    
    def __init__(self, grid, solver=None, max_level=3):
        """
        Initialize AMR manager
        
        Parameters:
        -----------
        grid : Grid
            The original grid to be refined
        solver : Solvers (optional)
            Reference to the solver, needed for solution-based refinement
        max_level : int
            Maximum refinement level
        """
        self.grid = grid
        self._solver = weakref.ref(solver) if solver else solver
        self.max_level = max_level
        
        # Track refinement levels for each cell
        self.cell_levels = {cell.cid: 0 for cell in grid.cells}
        
        # Maps to track parent-child relationships
        self.cell_to_children = {}  # Maps parent cell ID to list of child cell IDs
        self.cell_to_parent = {}    # Maps child cell ID to parent cell ID
        
        # Maps to track cells marked for refinement or coarsening
        self.cells_to_refine = set()
        self.cells_to_coarsen = set()
        
        # Thresholds for refinement and coarsening
        self.refine_threshold = 0.2
        self.coarsen_threshold = 0.1
        
        # List of active cells (cells that are not refined further)
        self.active_cells = list(grid.cells)
        
        # Buffer cells to ensure smooth transition between refinement levels
        self.buffer_zone_size = 1
        
        # AMR counter for debug purposes
        self.amr_step = 0
        
    @property
    def solver(self):
        if not self._solver:
            return self._solver
        _solver = self._solver()
        if _solver:
            return _solver
        else:
            raise LookupError("Solver was destroyed")
    
    def set_thresholds(self, refine_threshold, coarsen_threshold):
        """Set the thresholds for refinement and coarsening"""
        self.refine_threshold = refine_threshold
        self.coarsen_threshold = coarsen_threshold
        
    
    def add_buffer_zone(self):
        """
        Add buffer cells around cells flagged for refinement
        to ensure a smooth transition between refinement levels
        """
        original_flagged = set(self.cells_to_refine)
        
        # For each buffer layer
        for layer in range(self.buffer_zone_size):
            new_flagged = set()
            
            # For each already flagged cell
            for cell_id in original_flagged:
                cell = self.get_cell_by_id(cell_id)
                
                # Get face neighbors
                for face in cell.faces:
                    # Skip boundary faces
                    if face.isBoundary:
                        continue
                    
                    # Find neighbor cell
                    if face.adjacentface:
                        neighbor_cell = face.adjacentface.parentcell
                        
                        # Check if neighbor should be flagged
                        if (neighbor_cell.cid not in self.cells_to_refine and 
                            self.cell_levels.get(neighbor_cell.cid, 0) < self.max_level):
                            new_flagged.add(neighbor_cell.cid)
            
            # Add new cells to refinement list
            self.cells_to_refine.update(new_flagged)
            
            # Update for next layer
            original_flagged = new_flagged
            if not original_flagged:
                break
    
    def flag_cells_for_refinement(self, field_name="density"):
        """
        Flag cells for refinement based on gradient magnitude
        
        Parameters:
        -----------
        field_name : str
            Name of the field to compute gradients
        """
        self.cells_to_refine.clear()
        
        # Flag cells based on gradient criteria
        for cell in self.active_cells:
            if self.cell_levels.get(cell.cid, 0) >= self.max_level:
                continue
                
            # Skip cells that are already flagged
            if cell.cid in self.cells_to_refine:
                continue
                
            # Compute gradient magnitude from the solver's gradient data
            grad_mag = self.get_cell_gradient_magnitude(cell, field_name)
            
            # Flag cell for refinement if gradient exceeds threshold
            if grad_mag > self.refine_threshold:
                self.cells_to_refine.add(cell.cid)
        
        # Add buffer zone around flagged cells
        self.add_buffer_zone()
        
        return len(self.cells_to_refine)
    
    def flag_cells_for_coarsening(self, field_name="density"):
        """
        Flag cells for coarsening based on gradient magnitude
        
        Parameters:
        -----------
        field_name : str
            Name of the field to compute gradients
        """
        self.cells_to_coarsen.clear()
        
        # Group cells by parent
        cells_by_parent = {}
        
        for cell in self.active_cells:
            parent_id = self.cell_to_parent.get(cell.cid)
            if parent_id is not None:
                if parent_id not in cells_by_parent:
                    cells_by_parent[parent_id] = []
                cells_by_parent[parent_id].append(cell)
        
        # Flag for coarsening only if all siblings have low gradients
        for parent_id, children in cells_by_parent.items():
            all_can_coarsen = True
            
            # All children must have low gradients
            for child in children:
                grad_mag = self.get_cell_gradient_magnitude(child, field_name)
                if grad_mag > self.coarsen_threshold:
                    all_can_coarsen = False
                    break
            
            # All children must have no further refinement
            if all_can_coarsen:
                for child in children:
                    if child.cid in self.cell_to_children:
                        all_can_coarsen = False
                        break
            
            # All children must not be in the buffer zone of a cell to be refined
            if all_can_coarsen:
                for child in children:
                    # Check all neighbors
                    for face in child.faces:
                        if not face.isBoundary and face.adjacentface:
                            neighbor = face.adjacentface.parentcell
                            if neighbor.cid in self.cells_to_refine:
                                all_can_coarsen = False
                                break
                    
                    if not all_can_coarsen:
                        break
            
            # If all criteria are met, flag all children for coarsening
            if all_can_coarsen:
                for child in children:
                    self.cells_to_coarsen.add(child.cid)
        
        return len(self.cells_to_coarsen)
    
    def refine_cell_triangular(self, cell):
        """
        Refine a triangular cell into four triangles
        
        Parameters:
        -----------
        cell : Cell
            The triangular cell to refine
        
        Returns:
        --------
        list of Cell
            The four new triangular cells
        """
        # Verify cell is triangular
        if len(cell.nodes) != 3:
            raise ValueError(f"Cell {cell.cid} is not triangular")
        
        # Get nodes of the parent cell
        n1, n2, n3 = cell.nodes
        
        # Create midpoint nodes
        grid = self.grid
        mid_n1_n2 = self.create_node((n1.x0 + n2.x0) / 2.0, (n1.x1 + n2.x1) / 2.0)
        mid_n2_n3 = self.create_node((n2.x0 + n3.x0) / 2.0, (n2.x1 + n3.x1) / 2.0)
        mid_n3_n1 = self.create_node((n3.x0 + n1.x0) / 2.0, (n3.x1 + n1.x1) / 2.0)
        
        # Calculate refinement level for new cells
        level = self.cell_levels.get(cell.cid, 0) + 1
        
        # Create four new triangular cells
        child_cells = []
        
        # Child 1: n1, mid_n1_n2, mid_n3_n1
        child1 = self.create_cell([n1, mid_n1_n2, mid_n3_n1], level)
        
        # Child 2: n2, mid_n2_n3, mid_n1_n2
        child2 = self.create_cell([n2, mid_n2_n3, mid_n1_n2], level)
        
        # Child 3: n3, mid_n3_n1, mid_n2_n3
        child3 = self.create_cell([n3, mid_n3_n1, mid_n2_n3], level)
        
        # Child 4: mid_n1_n2, mid_n2_n3, mid_n3_n1
        child4 = self.create_cell([mid_n1_n2, mid_n2_n3, mid_n3_n1], level)
        
        child_cells = [child1, child2, child3, child4]
        
        # Update parent-child relationships
        self.cell_to_children[cell.cid] = [c.cid for c in child_cells]
        for child in child_cells:
            self.cell_to_parent[child.cid] = cell.cid
            
        # Update active cells list
        self.active_cells.remove(cell)
        self.active_cells.extend(child_cells)
        
        # Update connectivity
        self.update_connectivity_after_refinement(cell, child_cells)
        
        return child_cells
    
    def refine_cell_quad(self, cell):
        """
        Refine a quadrilateral cell into four quadrilaterals
        
        Parameters:
        -----------
        cell : Cell
            The quadrilateral cell to refine
        
        Returns:
        --------
        list of Cell
            The four new quadrilateral cells
        """
        # Verify cell is quadrilateral
        if len(cell.nodes) != 4:
            raise ValueError(f"Cell {cell.cid} is not quadrilateral")
        
        # Get nodes of the parent cell
        n1, n2, n3, n4 = cell.nodes
        
        # Create midpoint nodes for edges
        grid = self.grid
        mid_n1_n2 = self.create_node((n1.x0 + n2.x0) / 2.0, (n1.x1 + n2.x1) / 2.0)
        mid_n2_n3 = self.create_node((n2.x0 + n3.x0) / 2.0, (n2.x1 + n3.x1) / 2.0)
        mid_n3_n4 = self.create_node((n3.x0 + n4.x0) / 2.0, (n3.x1 + n4.x1) / 2.0)
        mid_n4_n1 = self.create_node((n4.x0 + n1.x0) / 2.0, (n4.x1 + n1.x1) / 2.0)
        
        # Create center node
        center_x = (n1.x0 + n2.x0 + n3.x0 + n4.x0) / 4.0
        center_y = (n1.x1 + n2.x1 + n3.x1 + n4.x1) / 4.0
        center = self.create_node(center_x, center_y)
        
        # Calculate refinement level for new cells
        level = self.cell_levels.get(cell.cid, 0) + 1
        
        # Create four new quadrilateral cells
        child_cells = []
        
        # Child 1: n1, mid_n1_n2, center, mid_n4_n1
        child1 = self.create_cell([n1, mid_n1_n2, center, mid_n4_n1], level)
        
        # Child 2: n2, mid_n2_n3, center, mid_n1_n2
        child2 = self.create_cell([n2, mid_n2_n3, center, mid_n1_n2], level)
        
        # Child 3: n3, mid_n3_n4, center, mid_n2_n3
        child3 = self.create_cell([n3, mid_n3_n4, center, mid_n2_n3], level)
        
        # Child 4: n4, mid_n4_n1, center, mid_n3_n4
        child4 = self.create_cell([n4, mid_n4_n1, center, mid_n3_n4], level)
        
        child_cells = [child1, child2, child3, child4]
        
        # Update parent-child relationships
        self.cell_to_children[cell.cid] = [c.cid for c in child_cells]
        for child in child_cells:
            self.cell_to_parent[child.cid] = cell.cid
            
        # Update active cells list
        self.active_cells.remove(cell)
        self.active_cells.extend(child_cells)
        
        # Update connectivity
        self.update_connectivity_after_refinement(cell, child_cells)
        
        return child_cells
    
    
    def transfer_solution_to_children(self, parent, child_cells):
        """
        Copy the parent’s primitive & conserved state into each new child.
        Assumes solver.w, solver.u and solver.gradw have already been resized
        so that solver.w[child.cid] is a valid row.
        
        Note: you must call _resize_solver_arrays before this!
        """
        solver = self.solver
        # 1) snapshot every parent field
        w_p     = solver.w[parent.cid].copy()
        u_p     = solver.u[parent.cid].copy()
        gradw_p = solver.gradw[parent.cid].copy()
        res_p   = solver.res[parent.cid].copy()
        dtau_p  = solver.dtau[parent.cid].copy()
        phi_p   = solver.phi[parent.cid].copy()
        wsn_p   = solver.wsn[parent.cid].copy()
        u0_p    = solver.u0[parent.cid].copy()
        
        
        # 2) scatter it into each child slot
        for child in child_cells:
            cid = child.cid
            solver.w[cid]      = w_p
            solver.u[cid]      = u_p
            solver.gradw[cid]  = gradw_p
            solver.res[cid]    = res_p
            solver.dtau[cid]   = dtau_p
            solver.phi[cid]    = phi_p
            solver.wsn[cid]    = wsn_p
            solver.u0[cid]     = u0_p
            
    
    def transfer_solution_to_parent(self, parent, children):
        """
        Transfer solution from children to parent during coarsening
        
        Volume‐weighted average the children back into the parent’s slot.
        Assumes solver.w and solver.u have already been resized so that
        solver.w[parent.cid] still points at the correct row.
        
        Parameters:
        -----------
        parent : Cell
            The parent cell
        children : list of Cell
            The child cells
        """
        solver = self.solver
        p = parent.cid
        total_vol = sum(c.volume for c in children)
        
        # Initialize parent state
        parent_state = np.zeros(self.solver.nq, float)
        #parent_state = {}  # use a dict to make it size agnostic
        
        # For each variable, compute volume-weighted average
        variables = solver.w[0].keys() if hasattr(solver.w[0], 'keys') else range(len(solver.w[0]))
        
        
        # 1) primitives & conservatives (as before)
        for var in variables:
            total_value = 0.0
            #total_volume = 0.0
            
            for child in children:
                total_value += solver.w[child.cid , var] * child.volume
                #total_volume += child.volume
            
            # make it mass conserving
            if total_vol > 0:
                parent_state[var] = total_value / total_vol
        
        # Set parent state
        solver.w[parent.cid, :] = parent_state[:]
        
        # Update conservative variables
        solver.u[parent.cid, :] = solver.w2u(solver.w[parent.cid, :])
        
        

        # 2) gradients: do the same volume‐weighted average
        for q in range(solver.nq):
            for d in range(solver.dim):
                solver.gradw[p, q, d] = sum(
                    solver.gradw[c.cid, q, d] * c.volume for c in children
                ) / total_vol
                
                
        # 3) residuals: you can zero them or average;
        #    averaging typically makes sense if you’re restarting mid‐step:
        solver.res[p] = sum(solver.res[c.cid] * c.volume for c in children) / total_vol
        

        # 4) limiters & local timestep (often you reset these)
        solver.dtau[p] = 0.0
        solver.phi[p]  = 1.0  # or average if you’d rather
    
        # 5) wave‐speed accumulator: average or take the max
        solver.wsn[p] = max(solver.wsn[c.cid] for c in children) #max is safest
        # 5) wave-speed do the volume‐weighted average 
        #solver.wsn[p] = sum(solver.wsn[c.cid] * c.volume for c in children) / total_vol
                
        

    
    def create_node(self, x, y):
        """Create a new node in the grid"""
        grid = self.grid
        nid = len(grid.nodes)
        node = Node(np.asarray([x, y]), nid)
        grid.nodes = np.append(grid.nodes, [node])#TLM todo: check this out closely
        grid.nodeList.append(node)
        grid.nNodes += 1
        return node
    
    def create_cell(self, nodes, level):
        """Create a new cell in the grid"""
        grid = self.grid
        cid = len(grid.cells)
        nface = len(grid.faceList)
        
        # Create the cell
        cell = Cell(nodes, cid, nface, facelist=grid.faceList)
        
        # Update grid
        grid.cells = np.append(grid.cells, [cell])
        grid.cellList.append(cell)
        grid.nCells += 1
        
        # Set the refinement level
        self.cell_levels[cid] = level
        
        return cell
    
    def update_connectivity_after_refinement(self, parent_cell, child_cells):
        """
        Update connectivity information after cell refinement
        
        Parameters:
        -----------
        parent_cell : Cell
            The parent cell that was refined
        child_cells : list of Cell
            The new child cells
        """
        grid = self.grid
        
        # Map new faces to their parent faces
        face_to_children = {}
        for child in child_cells:
            for face in child.faces:
                # If this face is on the boundary of the parent cell
                for parent_face in parent_cell.faces:
                    #print(parent_face, parent_face.fid)
                    if self.is_face_subset(face, parent_face):
                        #if parent_face.fid not in face_to_children:
                        #    face_to_children[parent_face.fid] = []
                        if parent_face not in face_to_children:
                            face_to_children[parent_face] = []
                        #face_to_children[parent_face.fid].append(face)
                        face_to_children[parent_face].append(face)
                        #face_to_children[parent_face.fid].append([parent_face,face])
        
        # Connect to adjacent cells at refinement boundaries
        #for parent_face_id, child_faces in face_to_children.items():
        for parent_face, child_faces in face_to_children.items():
            #print(parent_face_id, child_faces )
            #print(self.get_face_by_id(parent_face_id), child_faces )
            #parent_face = self.get_face_by_id(parent_face_id)
            #parent_face = child_faces[0]
            #child_face = child_faces[1]
            #print(parent_face, child_face)
            #child_faces = 
            
            # If this is an internal face
            if not parent_face.isBoundary and parent_face.adjacentface:
                adjacent_cell = parent_face.adjacentface.parentcell
                
                # If the adjacent cell is not refined, need to handle refinement boundary
                if adjacent_cell.cid not in self.cell_to_children:
                    # For each child face, connect to the adjacent cell
                    for child_face in child_faces:
                        child_face.adjacentface = parent_face.adjacentface
                        parent_face.adjacentface.parentcell.faces.append(child_face)
                        
                # If adjacent cell is also refined, connect corresponding child faces
                else:
                    self.connect_refined_faces(parent_face, child_faces)
        
        # Update the FaceCellMap
        for child in child_cells:
            for face in child.faces:
                grid.FaceCellMap[face.fid] = child
    
    def connect_refined_faces(self, parent_face, child_faces):
        """
        Connect child faces when both sides of a parent face have been refined
        
        Parameters:
        -----------
        parent_face : Face
            The parent face that was refined
        child_faces : list of Face
            The new child faces
        """
        # Get the adjacent face
        adj_face = parent_face.adjacentface
        
        # Get the refined children on both sides
        left_cells = self.cell_to_children.get(parent_face.parentcell.cid, [])
        right_cells = self.cell_to_children.get(adj_face.parentcell.cid, [])
        
        # Get all child faces from adjacent cell
        adjacent_child_faces = []
        for cell_id in right_cells:
            cell = self.get_cell_by_id(cell_id)
            for face in cell.faces:
                if self.is_face_subset(face, adj_face):
                    adjacent_child_faces.append(face)
        
        # Match child faces on both sides of the interface
        for face1 in child_faces:
            for face2 in adjacent_child_faces:
                if self.are_faces_coincident(face1, face2):
                    face1.adjacentface = face2
                    face2.adjacentface = face1
                    face1.isBoundary = False
                    face2.isBoundary = False
                    break
    
    def is_face_subset(self, face, parent_face):
        """Check if face is a subset of parent_face"""
        # Simple case: face is the same as parent_face
        if face.fid == parent_face.fid:
            return True
        
        # A face is a subset if its nodes are coincident with or in between
        # the nodes of the parent face
        n1, n2 = face.nodes
        pn1, pn2 = parent_face.nodes
        
        # Check if the line segments overlap
        # For simplicity, we'll check if both nodes of the child face
        # are on the line defined by the parent face
        return self.is_point_on_line_segment(n1.vector, pn1.vector, pn2.vector) and \
               self.is_point_on_line_segment(n2.vector, pn1.vector, pn2.vector)
    
    def is_point_on_line_segment(self, p, a, b, tol=1e-8):
        """Check if point p is on line segment from a to b"""
        # Check if p is on the line defined by a and b
        ab = np.array([b[0] - a[0], b[1] - a[1]])
        ap = np.array([p[0] - a[0], p[1] - a[1]])
        
        # Cross product should be close to zero if p is on the line
        cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
        
        if cross > tol:
            return False
        
        # Check if p is within the bounds of the segment
        dot = np.dot(ap, ab)
        if dot < 0:
            return False
        
        len_ab_squared = np.dot(ab, ab)
        if dot > len_ab_squared:
            return False
        
        return True
    
    def are_faces_coincident(self, face1, face2, tol=1e-8):
        """Check if two faces share the same space"""
        # Faces are coincident if they share the same nodes or if one is a subset of the other
        n1a, n1b = face1.nodes
        n2a, n2b = face2.nodes
        
        # Compare node positions directly
        endpoints_match = (
            (np.allclose([n1a.x0, n1a.x1], [n2a.x0, n2a.x1], rtol=tol) and
             np.allclose([n1b.x0, n1b.x1], [n2b.x0, n2b.x1], rtol=tol)) or
            (np.allclose([n1a.x0, n1a.x1], [n2b.x0, n2b.x1], rtol=tol) and
             np.allclose([n1b.x0, n1b.x1], [n2a.x0, n2a.x1], rtol=tol))
        )
        
        if endpoints_match:
            return True
        
        # Check if one face is a subset of the other
        return self.is_face_subset(face1, face2) or self.is_face_subset(face2, face1)
    
    # def coarsen_cells(self):
    #     """
    #     Coarsen cells that have been flagged for coarsening
        
    #     Returns:
    #     --------
    #     list of Cell
    #         The reactivated parent cells
    #     """
    #     reactivated_parents = []
        
    #     # Group cells by parent
    #     cells_by_parent = {}
    #     for cell_id in self.cells_to_coarsen:
    #         parent_id = self.cell_to_parent.get(cell_id)
    #         if parent_id is None:
    #             continue
                
    #         if parent_id not in cells_by_parent:
    #             cells_by_parent[parent_id] = []
    #         cells_by_parent[parent_id].append(cell_id)
        
    #     # Process each parent
    #     for parent_id, child_ids in cells_by_parent.items():
    #         parent_cell = self.get_cell_by_id(parent_id)
    #         children = [self.get_cell_by_id(cid) for cid in child_ids]
            
    #         # Only coarsen if all children are flagged
    #         expected_children = self.cell_to_children.get(parent_id, [])
    #         if set(child_ids) != set(expected_children):
    #             continue
            
    #         # Transfer solution data from children to parent
    #         if self.solver:
    #             self.transfer_solution_to_parent(parent_cell, children)
            
    #         # Reactivate parent cell
    #         parent_cell.faces = []  # Clear existing faces, they'll be regenerated
    #         parent_cell.faces = [self.get_face_by_id(fid) for fid in [f.fid for f in parent_cell.faces]]
            
    #         # Update active cells list
    #         self.active_cells.append(parent_cell)
    #         for child in children:
    #             self.active_cells.remove(child)
            
    #         # Remove parent-child relationships
    #         for child_id in child_ids:
    #             if child_id in self.cell_to_parent:
    #                 del self.cell_to_parent[child_id]
            
    #         if parent_id in self.cell_to_children:
    #             del self.cell_to_children[parent_id]
            
    #         # Restore connectivity for parent cell
    #         self.update_connectivity_after_coarsening(parent_cell, children)
            
    #         reactivated_parents.append(parent_cell)
        
    #     return reactivated_parents
    
    def coarsen_cells(self):
        """
        Coarsen cells that have been flagged for coarsening.
    
        Returns:
        --------
        List[Tuple[parent_cell, List[child_cell]]]
            For each parent that can be coarsened, the tuple of (parent_cell, its children).
        """
        parent_child_pairs = []
    
        # 1) Group flagged children by their parent ID
        cells_by_parent = {}
        for child_cid in self.cells_to_coarsen:
            parent_id = self.cell_to_parent.get(child_cid)
            if parent_id is None:
                continue
            cells_by_parent.setdefault(parent_id, []).append(child_cid)
    
        # 2) For each parent, only coarsen if *all* its children are flagged
        for parent_id, child_ids in cells_by_parent.items():
            expected = set(self.cell_to_children.get(parent_id, []))
            if set(child_ids) != expected:
                # skip partial sets
                continue
    
            parent = self.get_cell_by_id(parent_id)
            children = [self.get_cell_by_id(cid) for cid in child_ids]
    
            # 3) Update active_cells & parent/child mappings
            #    (we’ll rebuild faces & neighbors later in adapt_mesh)
            if parent not in self.active_cells:
                self.active_cells.append(parent)
            for ch in children:
                if ch in self.active_cells:
                    self.active_cells.remove(ch)
                # remove the parent link
                self.cell_to_parent.pop(ch.cid, None)
    
            # drop the children list from the mapping
            self.cell_to_children.pop(parent_id, None)
    
            # 4) Fix up connectivity for parent (faces, neighbors, etc.)
            self.update_connectivity_after_coarsening(parent, children)
    
            # 5) Record this pair for adapt_mesh to do the solution transfer
            parent_child_pairs.append((parent, children))
    
        return parent_child_pairs
        
    
    def update_connectivity_after_coarsening(self, parent, children):
        """
        Update connectivity information after cell coarsening
        
        Parameters:
        -----------
        parent : Cell
            The parent cell
        children : list of Cell
            The child cells that are being coarsened
        """
        grid = self.grid
        
        # Recreate parent faces
        for face in parent.faces:
            # Get adjacent cell (if any)
            adjacent_cell = None
            if not face.isBoundary and face.adjacentface:
                adjacent_cell = face.adjacentface.parentcell
            
            # Update the face's parent cell reference
            face.parentcell = parent
            
            # Update adjacency
            if adjacent_cell:
                # Find the appropriate face in the adjacent cell
                for adj_face in adjacent_cell.faces:
                    if self.are_faces_coincident(face, adj_face):
                        face.adjacentface = adj_face
                        adj_face.adjacentface = face
                        break
            
            # Update FaceCellMap
            grid.FaceCellMap[face.fid] = parent
        
        # Remove child cells and their faces
        for child in children:
            for face in child.faces:
                if face.fid in grid.FaceCellMap:
                    del grid.FaceCellMap[face.fid]
    
    def get_face_by_id(self, face_id):
        """Get a face by its ID"""
        for face in self.grid.faceList:
            if face.fid == face_id:
                return face
        return None
    
    def get_cell_by_id(self, cell_id):
        """Get a cell by its ID"""
        for cell in self.grid.cells:
            if cell.cid == cell_id:
                return cell
        return None
    
    def adapt_mesh(self, field_name="density"):
        """
        Main AMR procedure to refine and coarsen the mesh based on solution gradients
        
        Parameters:
        -----------
        field_name : str
            Name of the field to compute gradients
        
        Returns:
        --------
        dict
            Information about the mesh adaptation (refined cells, coarsened cells)
        """
        solver = self.solver
        old_n = solver.mesh.nCells
        
        # reset our queues
        self._pending_child_transfers = []
        self._pending_parent_transfers = []
        
        # 1) Flag cells for refinement
        num_to_refine_dummy = self.flag_cells_for_refinement(field_name)
        
        # 2) Flag & coarsen
        # Flag cells for coarsening
        num_to_coarsen_dummy = self.flag_cells_for_coarsening(field_name)
        
        # after flag_cells_for_refinement & flag_cells_for_coarsening
        # drop any cell from coarsen that is also in refine
        self.cells_to_coarsen = [cid for cid in self.cells_to_coarsen
                                   if cid not in self.cells_to_refine]
        
        # Refine flagged cells
        for cid in list(self.cells_to_refine):
            cell = self.get_cell_by_id(cid)
            
            # Skip if cell is not active
            if cell not in self.active_cells:
                continue
            
            # Refine based on cell type
            if len(cell.nodes) == 3:
                children = self.refine_cell_triangular(cell)
            elif len(cell.nodes) == 4:
                children = self.refine_cell_quad(cell)
            else:
                print(f"Warning: Cannot refine cell {cell.cid} with {len(cell.nodes)} nodes")
                continue
            
            # defer the data move until after resizing
            self._pending_child_transfers.append((cell, children))
    
    
        # NOTE: coarsen_cells() must return a list of (parent_cell, [child1, child2, ...])
        parent_child_pairs = self.coarsen_cells()
        # simply stash that list — no extra loop needed
        self._pending_parent_transfers = parent_child_pairs
    
        # 3) rebuild topology & IDs
        self.grid.make_FaceCellMap()
        self.grid.make_neighbors()
        self.update_indices()
    
        # 4) resize solver arrays to match new nCells
        new_n = len(self.grid.cells)
        solver.nCells = new_n
        if new_n != old_n:
            self._resize_solver_arrays(old_n, new_n, solver)
    
        # 5) now apply **all** the transfers
        for parent, children in self._pending_child_transfers:
            self.transfer_solution_to_children(parent, children)
    
        for parent, children in self._pending_parent_transfers:
            self.transfer_solution_to_parent(parent, children)
    
        # 6) rebuild LSQ stencils
        solver.cclsq = np.asarray([StencilLSQ(c, self.grid)
                                   for c in self.grid.cells])
        solver.compute_lsq_coefficients()
    
        # 7) cleanup & report
        num_refined = len(self._pending_child_transfers)
        num_coarsened = len(self._pending_parent_transfers)
        self._pending_child_transfers.clear()
        self._pending_parent_transfers.clear()
        self.amr_step += 1
    
        return {
            'refined': num_refined,
            'coarsened': num_coarsened,
            'num_cells': new_n,
            'num_active_cells': len(self.active_cells),
            'max_level': max(self.cell_levels.values()) if self.cell_levels else 0
        }
    
    
    def adapt_meshOLD(self, field_name="density"):
        """
        Main AMR procedure to refine and coarsen the mesh based on solution gradients
        
        Parameters:
        -----------
        field_name : str
            Name of the field to compute gradients
        
        Returns:
        --------
        dict
            Information about the mesh adaptation (refined cells, coarsened cells)
        """
        solver = self.solver
        old_n = solver.mesh.nCells
        
        self._pending_child_transfers = []
        self._pending_parent_transfers = []
        
        # 1) Flag cells for refinement
        num_to_refine = self.flag_cells_for_refinement(field_name)
        
        # 2) coarsen (similar pattern)
        # Flag cells for coarsening
        num_to_coarsen = self.flag_cells_for_coarsening(field_name)
        
        # Refine flagged cells
        refined_cells = []
        for cell_id in self.cells_to_refine:
            cell = self.get_cell_by_id(cell_id)
            
            # Skip if cell is not active
            if cell not in self.active_cells:
                continue
            
            # Refine based on cell type
            if len(cell.nodes) == 3:  # Triangle
                children = self.refine_cell_triangular(cell)
            elif len(cell.nodes) == 4:  # Quad
                children = self.refine_cell_quad(cell)
            else:
                print(f"Warning: Cannot refine cell {cell.cid} with {len(cell.nodes)} nodes")
                continue
            
            refined_cells.extend(children)
            # stash them for after resizing
            self._pending_child_transfers.append((cell, children))
            
            #self.transfer_solution_to_children(parent = cell, child_cells = children)
            
        
        # 2) flag & coarsen
        
        
        # # 2) refine flagged cells
        # for cid in list(self.cells_to_refine):
        #     cell = self.get_cell_by_id(cid)
        #     children = (cell.nodes==3
        #                 and self.refine_cell_triangular(cell)
        #                 or self.refine_cell_quad(cell))
        #     # 2a) transfer solution
        #     self.transfer_solution_to_children(parent = cell, child_cells = children)
        
        # self.update_indices()
        
        # # Update grid connectivity (done after coarsening)
        # self.grid.make_FaceCellMap()
        # self.grid.make_neighbors()
        
        
        
        # Coarsen flagged cells (TLM todo, fixme!)
        #coarsened_cells = []
        # NOTE: coarsen_cells() must return a list of (parent_cell, [child1, child2, ...])
        parent_child_pairs = self.coarsen_cells()
        # simply stash that list — no extra loop needed
        self._pending_parent_transfers = parent_child_pairs
        
        #coarsened_cells = self.coarsen_cells()
        #parent_child_pairs = coarsened_cells
        #self._pending_parent_transfers = parent_child_pairs
        
        # for cell_id in self.cells_to_coarsen:
        #     cell = self.get_cell_by_id(cell_id)
            
        #     self._pending_parent_transfers.append((cell, children))
        
        
        # 3) rebuild grid connectivity & IDs
        # Update grid connectivity
        self.grid.make_FaceCellMap()
        self.grid.make_neighbors()
        # Update grid indices
        self.update_indices()
        
        # after update_indices() and after you set solver.nCells = len(self.grid.cells)
        
        # 4) resize every per‐cell solver array
        # 6) resize solver arrays & rebuild stencils
        # rebuild/re-compute your least‑squares stencils over the new connectivity
        # rebuild and recompute LSQ stencils over the new connectivity
        new_n = len(self.grid.cells)
        solver.mesh = self.grid       # if needed
        self.solver.mesh = self.grid       # if needed
        
        solver.nCells = new_n
        self.solver.nCells = len(self.grid.cells)
        new_n = self.solver.nCells       # e.g. now 32769 or higher
        old_n = self.solver.res.shape[0] # still 32768
        
        if new_n != old_n:
            self._resize_solver_arrays(old_n, new_n, solver)
        # # 1) rebuild residual array
        # #    assume res has shape (nCells, nflux)
        # nflux = self.solver.res.shape[1]
        # new_res = np.zeros((new_n, nflux), dtype=self.solver.res.dtype)
        # # copy old values (if you care; otherwise you can skip this and just zero‑start)
        # new_res[:old_n, :] = self.solver.res
        # self.solver.res = new_res
    
        # # 2) rebuild any other cell‑based arrays the same way:
        # #    e.g. dtau (shape (nCells,)), phi (nCells,),  
        # #         w (nCells, nq), u (nCells, nq), gradw (nCells, nq, dim), etc.
        # # For example:
        # self.solver.dtau = np.zeros(new_n, dtype=self.solver.dtau.dtype)
        # self.solver.phi  = np.zeros(new_n, dtype=self.solver.phi.dtype)
           
        
        # primitives & conservatives:
        
        # # rebuild the stencil objects array from scratch
        # # (rebuild or resize solver.w, u, gradw, res, dtau,
        # self.solver.cclsq = np.asarray([StencilLSQ(cell, self.grid) 
        #                            for cell in self.grid.cells])
        # # Update LSQ stencils
        # self.solver.compute_lsq_coefficients()
        
        ## Update LSQ stencils if solver is available
        # if self.solver and hasattr(self.solver, 'compute_lsq_coefficients'):
        #     self.solver.compute_lsq_coefficients()
        
        
        # 5) now that solver arrays are the right size,
        #    do all of the actual data moves
        for parent, children in self._pending_child_transfers:
            self.transfer_solution_to_children(parent, children)
        for parent, children in self._pending_parent_transfers:
            self.transfer_solution_to_parent(parent, children)
    
        # 6) rebuild LSQ stencils on the new mesh
        solver.cclsq = np.asarray([StencilLSQ(c, self.grid)
                                   for c in self.grid.cells])
        solver.compute_lsq_coefficients()
    
        # 7) cleanup & report
        num_refined = len(self._pending_child_transfers)
        num_coarsened = len(self._pending_parent_transfers)
        # clear our queues
        self._pending_child_transfers.clear()
        self._pending_parent_transfers.clear()

        
        # Increment AMR step
        self.amr_step += 1
        
        # Return information about the adaptation
        return {
            'refined': num_refined,#refined_cells,
            'coarsened': num_coarsened,#coarsened_cells,
            'num_cells': len(self.grid.cells),
            'num_active_cells': len(self.active_cells),
            'max_level': max(self.cell_levels.values()) if self.cell_levels else 0
        }
    
    


    def _resize_solver_arrays(self, old_n, new_n, solver):
        """Zero‐initialize (and preserve old data) for every per‐cell array."""
        # 0) figure out how many rows we can actually copy
        # use solver.res as a canonical “old‐shape” source
        old_shape = solver.res.shape[0]
        #n_copy   = min(old_shape, new_n)
        n_copy = min(old_n, new_n)
        
        # 1) residuals
        # nflux = solver.res.shape[1]
        # new_res = np.zeros((new_n, nflux), dtype=solver.res.dtype)
        # new_res[:old_n, :] = solver.res #unsafe?
        # solver.res = new_res
        
        nflux     = solver.res.shape[1]
        new_res   = np.zeros((new_n, nflux), dtype=solver.res.dtype)
        new_res[:n_copy, :] = solver.res[:n_copy, :]
        solver.res = new_res

        # 2) timestep & limiter fields
        #old_dtau     = solver.dtau
        new_dtau     = np.zeros(new_n, dtype=solver.dtau.dtype)
        #solver.dtau = np.zeros(new_n, dtype=solver.dtau.dtype)
        #solver.phi  = np.zeros(new_n, dtype=solver.phi.dtype)
        new_dtau[:n_copy] = solver.dtau[:n_copy]
        solver.dtau = new_dtau

        old_phi      = solver.phi
        new_phi      = np.zeros(new_n, dtype=old_phi.dtype)
        new_phi[:n_copy]  = old_phi[:n_copy]
        solver.phi  = new_phi
        
        
        # 3) primitives variables
        old_w = solver.w
        nq    = old_w.shape[1]
        new_w = np.zeros((new_n, nq), dtype=old_w.dtype)
        print('old_n = {}'.format(old_n))
        print('len(solver.w) = {}'.format(len(new_w)))
        new_w[:n_copy, :] = old_w[:n_copy, :]
        solver.w = new_w
        
        # nq = solver.w.shape[1]
        # new_w = np.zeros((new_n, nq), dtype=solver.w.dtype)
        # print('old_n = {}'.format(old_n))
        # print('len(solver.w) = {}'.format(len(solver.w)))
        # new_w[:old_n, :] = solver.w[:,:]
        # solver.w = new_w

        # 4) conservatives u
        old_u = solver.u
        new_u = np.zeros((new_n, nq), dtype=old_u.dtype)
        new_u[:n_copy, :] = old_u[:n_copy, :]
        solver.u = new_u
        
        # u0
        old_u0 = solver.u0
        new_u0 = np.zeros((new_n, nq), dtype=old_u0.dtype)
        new_u0[:n_copy, :] = old_u0[:n_copy, :]
        solver.u0 = new_u0
        

        # new_u = np.zeros((new_n, nq), dtype=solver.u.dtype)
        # new_u[:old_n, :] = solver.u
        # solver.u = new_u

        # # 4) gradients (nCells, nq, dim)
        # nq, dim = solver.gradw.shape[1], solver.gradw.shape[2]
        # new_g = np.zeros((new_n, nq, dim), dtype=solver.gradw.dtype)
        # new_g[:old_n, :, :] = solver.gradw
        # solver.gradw = new_g
        

        # 5) gradients gradw (nCells, nq, dim)
        old_g = solver.gradw
        dim   = old_g.shape[2]
        new_g = np.zeros((new_n, nq, dim), dtype=old_g.dtype)
        new_g[:n_copy, :, :] = old_g[:n_copy, :, :]
        solver.gradw = new_g
        
        

        # 6) wave‐speed accumulator wsn (shape (nCells,))
        old_wsn = solver.wsn
        new_wsn = np.zeros(new_n, dtype=old_wsn.dtype)
        new_wsn[:n_copy] = old_wsn[:n_copy]
        solver.wsn = new_wsn
    
    def update_indices(self):
        """Update indices for all grid elements"""
        grid = self.grid
        
        # Update node indices
        for i, node in enumerate(grid.nodes):
            node.nid = i
        
        # Update face indices
        for i, face in enumerate(grid.faceList):
            face.fid = i
        
        # Update cell indices
        for i, cell in enumerate(grid.cells):
            cell.cid = i
    
    def get_cell_gradient_magnitude(self, cell, field_name):
        """
        Get the gradient magnitude of a field in a cell
        
        Parameters:
        -----------
        cell : Cell
            The cell
        field_name : str
            Name of the field
        
        Returns:
        --------
        float
            Gradient magnitude
        """
        solver = self.solver
        
        if solver is None:
            return 0.0
            
        # Try to get gradient from solver's gradient data
        if hasattr(solver, 'gradw'):
            if field_name == "density":
                field_idx = solver.ir
            elif field_name == "x-velocity":
                field_idx = solver.iu
            elif field_name == "y-velocity":
                field_idx = solver.iv
            elif field_name == "pressure":
                field_idx = solver.ip
            else:
                field_idx = 0
                
            # Get gradients
            try:
                grad_x = solver.gradw[cell.cid, field_idx, 0]
                grad_y = solver.gradw[cell.cid, field_idx, 1]
                return np.sqrt(grad_x**2 + grad_y**2)
            except (IndexError, KeyError):
                pass
        
        # If gradient isn't available from solver, compute a simple estimate
        #return 0.0
        return self.compute_simple_gradient_magnitude(cell, field_name)
    
    def compute_simple_gradient_magnitude(self, cell, field_name):
        """Compute a simple gradient magnitude estimate if solver gradients aren't available"""
        solver = self.solver
        
        if solver is None:
            return 0.0
            
        # Determine field index
        if field_name == "density":
            field_idx = solver.ir
        elif field_name == "x-velocity":
            field_idx = solver.iu
        elif field_name == "y-velocity":
            field_idx = solver.iv
        elif field_name == "pressure":
            field_idx = solver.ip
        else:
            field_idx = 0
        
        # Get cell value
        cell_value = solver.w[cell.cid, field_idx]
        
        # Get neighbor values and compute simple gradient magnitude
        grad_sum = 0.0
        count = 0
        
        for face in cell.faces:
            if not face.isBoundary and face.adjacentface:
                neighbor = face.adjacentface.parentcell
                neighbor_value = solver.w[neighbor.cid, field_idx]
                
                # Compute distance between cell centroids
                dx = neighbor.centroid[0] - cell.centroid[0]
                dy = neighbor.centroid[1] - cell.centroid[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    # Estimate directional derivative
                    deriv = (neighbor_value - cell_value) / distance
                    grad_sum += deriv**2
                    count += 1
        
        if count > 0:
            return np.sqrt(grad_sum / count)
        else:
            return 0.0
    
    def plot_mesh(self, filename=None, show_level=True):
        """
        Plot the mesh with different colors for different refinement levels
        
        Parameters:
        -----------
        filename : str, optional
            If provided, save the plot to this file
        show_level : bool
            Whether to color cells by refinement level
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Set up colormap for refinement levels
        cmap = plt.cm.viridis
        max_level = max(self.cell_levels.values()) if self.cell_levels else 0
        
        # Plot cells
        patches = []
        colors = []
        
        for cell in self.active_cells:
            vertices = np.array([(node.x0, node.x1) for node in cell.nodes])
            polygon = Polygon(vertices, True)
            patches.append(polygon)
            
            if show_level:
                level = self.cell_levels.get(cell.cid, 0)
                colors.append(level / max(1, max_level))
            else:
                colors.append(0.5)  # Default color
        
        # Create patch collection
        p = PatchCollection(patches, cmap=cmap, alpha=0.7)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        
        # Add colorbar if showing levels
        if show_level and max_level > 0:
            cbar = plt.colorbar(p)
            cbar.set_label('Refinement Level')
        
        # Set plot properties
        ax.set_aspect('equal')
        ax.autoscale()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Adaptive Mesh (Step {self.amr_step})')
        
        # Save or show the plot
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()
    
    def get_mesh_statistics(self):
        """Get statistics about the mesh"""
        total_cells = len(self.grid.cells)
        active_cells = len(self.active_cells)
        
        level_counts = {}
        for cell_id, level in self.cell_levels.items():
            if level not in level_counts:
                level_counts[level] = 0
            level_counts[level] += 1
        
        # Calculate min, max, mean cell volumes
        volumes = [cell.volume for cell in self.active_cells]
        min_vol = min(volumes) if volumes else 0
        max_vol = max(volumes) if volumes else 0
        mean_vol = sum(volumes) / len(volumes) if volumes else 0
        
        return {
            'total_cells': total_cells,
            'active_cells': active_cells,
            'level_counts': level_counts,
            'min_volume': min_vol,
            'max_volume': max_vol,
            'mean_volume': mean_vol,
            'max_level': max(self.cell_levels.values()) if self.cell_levels else 0
            }
    
if __name__ == '__main__':
    print('testeing AdaptiveMeshRefinement module')