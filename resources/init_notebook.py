# Standard library imports
import os
import sys
import warnings

# Third-party imports
import numpy as np
import pandas as pd
import shapely
from shapely.errors import ShapelyDeprecationWarning
import astropy.units as u 
import astropy.constants as c

# Qiskit Metal imports
from qiskit_metal import designs, Dict
from qiskit_metal.qlibrary.tlines.mixed_path import RouteMixed
from qiskit_metal.analyses.quantization import LOManalysis
from qiskit_metal.analyses.quantization import EPRanalysis

# Local imports
# Add parent directory to Python path
notebook_dir = os.getcwd()
two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(notebook_dir)))
sys.path.append(two_levels_up)

one_level_up = os.path.dirname(os.path.dirname(notebook_dir))
sys.path.append(one_level_up)


import analysis.Transmon_property as trans_p
import analysis.Transmon_specifications as jj
from components.junction.dolan_junction import DolanJunction as junction
from components.qubit.rounded_single_pad import Round_TransmonPocket_Single as transmon
from components.misc.cutout import Cutout 
from components.tm.LaunchpadWirebondCustom import LaunchpadWirebondCustom
from components.tm.CoupledLineTee import CoupledLineTee

# Configure warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Design initialization
design = designs.DesignPlanar({}, True)
design.overwrite_enabled = True

# Chip dimensions
design.chips.main.size['size_x'] = '7 mm'
design.chips.main.size['size_y'] = '7 mm'

# CPW parameters
design.variables['cpw_wdith'] = '15 um'
design.variables['cpw_gap'] = '8.3 um'
design.variables['trace_width'] = '15 um'
design.variables['trace_gap'] = '8.3 um'
cpw_pin_width = 15*u.um
cpw_gap = 8.3*u.um

# Buffer parameters
design.variables['pad_buffer_radius'] = '30 um'
design.variables['buffer_resolution'] = '3'
design.variables['connection_pad_buffer_radius'] = '2 um'

# Renderers
renderer_hfss = design.renderers.hfss
renderer_q3d = design.renderers.q3d

# Layer definitions
qubit_layer = 5
junction_layer = 20
ab_layer = 31
ab_square_layer = 30
junction_area_layer = 60

# Component options
qb_options = dict(
    pad_pocket_distance_top = '39.1um',
    jj_length = '40um',
    jj_pocket_extent = '20.9um',
    connection_pads = dict(
        a = dict(
            loc_W = 0, 
            loc_H = 1, 
            pad_gap = '14.9um',
            pad_height = '15.9um',
            pad_width = '80um',
            pad_cpw_extent = '10um',
            pocket_rise = '0um',
            cpw_extend = '0um',
            pocket_extent = '0um'
        )
    )
)

# Coupled line tee options
TQ_options = dict(
    prime_width = design.variables['cpw_width'],
    prime_gap = design.variables['cpw_gap'],
    second_width = design.variables['cpw_width'],
    second_gap = design.variables['cpw_gap'],
    downlength = '250um',
    coupling_space = '5um',
    open_termination = True,
    hfss_wire_bonds = False,
    q3d_wire_bonds = False,
    layer = qubit_layer
)

# CPW options
CPW_options = Dict(
    trace_width = design.variables['trace_width'],
    trace_gap = design.variables['trace_gap'],
    total_length = '5 mm',
    hfss_wire_bonds = False,
    q3d_wire_bonds = False,
    layer = qubit_layer,
    fillet = '30 um',
    lead = dict(start_straight='5um', end_straight='5um'),
    pin_inputs = Dict(
        start_pin = Dict(component='Q1', pin='a'),
        end_pin = Dict(component='TQ1', pin='second_end')
    )
)

# Transmission line options
trans_options = Dict(
    trace_width = design.variables['trace_width'],
    trace_gap = design.variables['trace_gap'],
    lead = dict(start_straight='5um', end_straight='5um'),
    fillet = '30um',
    layer = qubit_layer,
    total_length = '0.5mm',
    hfss_wire_bonds = True,
    q3d_wirebonds = True,
    pin_inputs = Dict(
        start_pin = Dict(component='TQ1', pin='prime_start'),
        end_pin = Dict(component='TQ2', pin='prime_end')
    )
)

# Pocket options
pocket_options = dict(
    pos_x = '0mm', 
    pos_y = '0mm', 
    orientation = '0',
    frequency = 5.2,
    guess_path = r'/Users/wendy/Desktop/Wendy-qiskit-code/data/educated_guess_0403.csv',
    coupling_path = '',
    sim = True,
    coord = '(0,0)',
    qubit_layer = qubit_layer,
    junction_layer = junction_layer, 
    junction_area_layer = junction_area_layer,
    ab_layer = ab_layer,
    ab_square_layer = ab_square_layer,
    ab_distance = '70um'
) 