#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:30:36 2025

@author: lukemcculloch
"""

import os
import sys
import weakref
try:
    from memory_profiler import profile
    MEM_PROFILE = True
except:
    print( 'please install memory_profiler')
    MEM_PROFILE = False
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import * #quiver...

class StencilLSQ(object):
    """
    #------------------------------------------
    #>> Cell-centered LSQ stencil data
    #------------------------------------------
    """
    def __init__(self, cell, mesh):
        self.cell = cell #reference to cell
        #self.mesh = mesh #reference to mesh
        self._mesh = weakref.ref(mesh) if mesh else mesh
        #
        self.nnghbrs_lsq = None     #number of lsq neighbors
        self.nghbr_lsq = []         #list of lsq neighbors
        self.cx = []                #LSQ coefficient for x-derivative
        self.cy = []                #LSQ coefficient for y-derivative
        #
        #self.node   = np.zeros((self.nNodes),float) #node to cell list
        self.construct_vertex_stencil()
        
    @property
    def mesh(self):
        if not self._mesh:
            return self._mesh
        _mesh = self._mesh()
        if _mesh:
            return _mesh
        else:
            raise LookupError("mesh was destroyed")
            
    # def __del__(self):
    #     pass
    #     #print("delete LSQ",self.cell.cid)
    #     #print("delete", "LSQstencil")
        
        
    def construct_vertex_stencil(self):
        for node in self.cell.nodes:
            for cell in node.parent_cells:
                if cell is not self.cell:
                    self.nghbr_lsq.append(cell)
        
        self.nghbr_lsq = set(self.nghbr_lsq)
        self.nghbr_lsq = list(self.nghbr_lsq)
        self.nnghbrs_lsq = len(self.nghbr_lsq)
        
        # Allocate the LSQ coeffient arrays for the cell i:
        self.cx = np.zeros((self.nnghbrs_lsq),float)
        self.cy = np.zeros((self.nnghbrs_lsq),float)
        return
    
    
    def plot_lsq_reconstruction(self, canvas = None,
                                alpha = .1, saveit = False):
        if canvas is None:
            fig, ax = plt.subplots()
            ax.axis('equal')
        else:
            ax = canvas
            
        fig.suptitle('LSQ reconstruction stencil', fontsize=10)
            
        ax = self.cell.plot_cell(canvas = ax,
                                 fillcolor='green')
        for cell in self.nghbr_lsq:
            ax = cell.plot_cell(canvas = ax)
            
        patch = mpatches.Patch(color='green', label='primary cell')
        plt.legend(handles=[patch])
        
        if saveit:
            mytitle = '../pics/stencil_'+str(self.cell.cid)
            
            self.save_image(filename=mytitle, ftype = '.png')
        return
    
    
    def save_image(self, filename = None, ftype = '.pdf', closeit=True):
        """ save pdf file.
        No file extension needed.
        """
        if filename == None:
            filename = default_input('please enter a name for the picture', 'lsq_reconstruction')
        plt.savefig(filename+ftype, bbox_inches = 'tight')
        if closeit:
            plt.close()
        return
    
