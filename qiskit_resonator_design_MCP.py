#!/usr/bin/env python3

"""
Superconducting Resonator Design Class with MCP Tools

This module provides a comprehensive class for designing superconducting resonators
using Qiskit Metal, along with MCP (Model Context Protocol) tools that allow
Large Language Models to customize resonator dimensions and parameters.

The class handles the complete workflow from design to export, including:
- Creating CPW (Coplanar Waveguide) structures with wirebond pads
- Designing couplers with programmable dimensions
- Creating meandered resonators with customizable geometry
- Exporting designs to GDS format for fabrication
- Generating visual screenshots of the design (when GUI is enabled)

All input parameters use micrometers (um) as units, which are automatically
converted to Qiskit Metal's native millimeter units internally.

The GUI can be disabled by setting show_gui=False to run in headless mode,
which is useful for batch processing or environments without display support.
"""

import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
import numpy as np
from IPython.display import Image, display
import manipulate_GDS
import io
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("resonator_design")

# Global designer instance
designer_instance = None

def initialize_designer(show_gui=False):
    """Initialize the global designer instance if not already created."""
    global designer_instance
    if designer_instance is None:
        designer_instance = SuperconductingResonatorDesigner(show_gui=show_gui)
    return designer_instance


class SuperconductingResonatorDesigner:
    """
    A comprehensive class for designing superconducting resonators using Qiskit Metal.

    This class provides a complete workflow for creating, modifying, and exporting
    superconducting resonator designs. It includes coplanar waveguide (CPW) transmission
    lines, couplers, meandered resonators, and wirebond launch pads.

    All dimensional parameters are specified in micrometers (um) and automatically
    converted to Qiskit Metal's native millimeter units.

    Attributes:
        design (DesignPlanar): The main Qiskit Metal design object
        gui (MetalGUI): The graphical user interface for visualization (if enabled)
        show_gui (bool): Flag to control GUI initialization and operations
        cpw_width (float): Width of the coplanar waveguide center conductor (um)
        cpw_gap (float): Gap between center conductor and ground plane (um)
        resonator_length (float): Total electrical length of the resonator (um)
        resonator_height (float): Physical height/footprint of the resonator (um)
        coupler_gap (float): Coupling gap between feedline and resonator (um)
        coupler_length (float): Length of the coupling region (um)
    """

    def __init__(self, cpw_width_um=10, cpw_gap_um=10, show_gui=True):
        """
        Initialize the resonator designer with default CPW parameters.

        Args:
            cpw_width_um (float): Width of CPW center conductor in micrometers
            cpw_gap_um (float): Gap between conductor and ground in micrometers
            show_gui (bool): Whether to initialize and enable GUI operations
        """
        # Store GUI flag
        self.show_gui = show_gui

        # Initialize Qiskit Metal design
        self.design = designs.DesignPlanar()
        self.design.overwrite_enabled = True

        # Initialize GUI only if requested
        if self.show_gui:
            self.gui = MetalGUI(self.design)
        else:
            self.gui = None

        # Store parameters in micrometers for user interface
        self.cpw_width_um = cpw_width_um
        self.cpw_gap_um = cpw_gap_um

        # Convert to millimeters for Qiskit Metal (internal units)
        self.cpw_width = cpw_width_um * 1e-3  # Convert um to mm
        self.cpw_gap = cpw_gap_um * 1e-3      # Convert um to mm

        # Set design variables for CPW geometry
        self.design.variables['cpw_width'] = self.cpw_width
        self.design.variables['cpw_gap'] = self.cpw_gap

        # Initialize resonator parameters (will be set by design_resonator)
        self.resonator_length_um = 10000  # Default 10mm resonator
        self.resonator_height_um = 1000   # Default 1mm height

        # Initialize coupler parameters (will be set by design_coupler)
        self.coupler_gap_um = 20    # Default 20um coupling gap
        self.coupler_length_um = 100 # Default 100um coupling length

        # Store component references for later modification
        self.coupler = None
        self.meander = None
        self.launch_left = None
        self.launch_right = None
        self.feedline = None
        self.end_short_meander = None

    def _um_to_mm(self, value_um):
        """
        Convert micrometers to millimeters for Qiskit Metal compatibility.

        Args:
            value_um (float): Value in micrometers

        Returns:
            float: Value in millimeters
        """
        return value_um * 1e-3

    def is_gui_enabled(self):
        """
        Check if GUI is enabled and available.

        Returns:
            bool: True if GUI is enabled and available, False otherwise
        """
        return self.show_gui and self.gui is not None

    def enable_gui(self):
        """
        Enable GUI if it was previously disabled.

        Returns:
            bool: True if GUI was successfully enabled, False if already enabled
        """
        if not self.show_gui:
            self.show_gui = True
            if self.gui is None:
                self.gui = MetalGUI(self.design)
            return True
        return False

    def disable_gui(self):
        """
        Disable GUI operations.

        Returns:
            bool: True if GUI was successfully disabled, False if already disabled
        """
        if self.show_gui:
            self.show_gui = False
            return True
        return False

    def design_coupler(self, coupler_gap_um, coupler_length_um):
        """
        Design or modify the coupler that connects the feedline to the resonator.

        The coupler is implemented as a CoupledLineTee structure that provides
        controlled electromagnetic coupling between the main feedline and the
        resonator. The coupling strength depends on the gap and length parameters.

        Args:
            coupler_gap_um (float): Gap between feedline and resonator in micrometers.
                                   Smaller gaps provide stronger coupling.
            coupler_length_um (float): Length of the coupling region in micrometers.
                                     Longer lengths provide stronger coupling.
        """
        # Store parameters in micrometers
        self.coupler_gap_um = coupler_gap_um
        self.coupler_length_um = coupler_length_um

        # Convert to millimeters for Qiskit Metal
        coupler_gap_mm = self._um_to_mm(coupler_gap_um)
        coupler_length_mm = self._um_to_mm(coupler_length_um)

        # Configure coupler options for CoupledLineTee
        # The coupling_space parameter is the actual gap minus the CPW gaps
        coupler_options = {
            'prime_width': self.cpw_width,      # Main feedline width
            'prime_gap': self.cpw_gap,          # Main feedline gap
            'second_width': self.cpw_width,     # Resonator connection width
            'second_gap': self.cpw_gap,         # Resonator connection gap
            'coupling_space': coupler_gap_mm - 2*self.cpw_gap,  # Actual coupling gap
            'coupling_length': coupler_length_mm,  # Length of coupling region
            'down_length': self._um_to_mm(50),  # Length extending downward
            'fillet': self._um_to_mm(10),       # Radius for rounded corners
            'orientation': -180                  # Orientation in degrees
        }

        # Create or update the coupler component
        if self.coupler is not None:
            # Remove existing coupler if it exists
            self.design.delete_component('coupler')

        self.coupler = CoupledLineTee(self.design, 'coupler', options=coupler_options)

        print(f"Coupler designed with gap: {coupler_gap_um} um, length: {coupler_length_um} um")

    def design_resonator(self, resonator_length_um, resonator_height_um):
        """
        Design or modify the meandered resonator structure.

        The resonator is implemented as a meandered transmission line that provides
        a long electrical path in a compact physical footprint. The total length
        determines the resonant frequency, while the height affects the physical
        layout and coupling characteristics.

        Args:
            resonator_length_um (float): Total electrical length of the resonator
                                       in micrometers. This determines the fundamental
                                       resonant frequency: f = c/(2*n_eff*L)
                                       where c is speed of light, n_eff is effective
                                       refractive index, and L is the length.
            resonator_height_um (float): Physical height of the meandered structure
                                       in micrometers. Larger heights result in
                                       wider meanders with fewer turns.
        """
        # Store parameters in micrometers
        self.resonator_length_um = resonator_length_um
        self.resonator_height_um = resonator_height_um

        # Convert to millimeters for Qiskit Metal
        resonator_length_mm = self._um_to_mm(resonator_length_um)
        resonator_height_mm = self._um_to_mm(resonator_height_um)

        # Ensure coupler exists before creating resonator
        if self.coupler is None:
            # Create default coupler if none exists
            self.design_coupler(self.coupler_gap_um, self.coupler_length_um)

        # Get the position of the coupler's second end port
        coupler_port_end_x = self.coupler.pins['second_end']['middle'][0]

        # Create short-to-ground termination at the end of the resonator
        if self.end_short_meander is not None:
            self.design.delete_component('end_short_meander')

        self.end_short_meander = ShortToGround(
            self.design,
            'end_short_meander',
            options=dict(
                pos_x=coupler_port_end_x,
                pos_y=resonator_height_mm,
                orientation=90  # Point upward
            )
        )

        # Configure meander options for the resonator
        meander_options = Dict(
            pin_inputs=Dict(
                start_pin=Dict(
                    component='coupler',    # Connect to coupler
                    pin='second_end'        # At the second_end port
                ),
                end_pin=Dict(
                    component='end_short_meander',  # Connect to short
                    pin='short'                     # At the short pin
                )
            ),
            total_length=resonator_length_mm,    # Total electrical length
            fillet=self._um_to_mm(40),          # Radius for rounded bends
            start_straight=self._um_to_mm(100), # Straight section at start
            end_straight=self._um_to_mm(100),   # Straight section at end
            meander=Dict(spacing=self._um_to_mm(100))  # Spacing between meander lines
        )

        # Create or update the meander component
        if self.meander is not None:
            self.design.delete_component('meander')

        self.meander = RouteMeander(self.design, 'meander', meander_options)

        print(f"Resonator designed with length: {resonator_length_um} um, height: {resonator_height_um} um")

        # Update the feedline and launch pads to accommodate new geometry
        self._update_feedline_and_pads()

    def _update_feedline_and_pads(self):
        """
        Update the feedline and wirebond launch pads based on the current resonator geometry.

        This internal method recalculates the positions of the launch pads and
        feedline to ensure they properly encompass the resonator structure.
        It's called automatically when the resonator geometry changes.
        """
        # Get the bounding box of the meander to determine feedline extent
        bounds = self.meander.qgeometry_bounds()

        # Calculate positions for launch pads with some margin
        left_feedline_bound = bounds[0] * 1.5   # 50% margin on left
        right_feedline_bound = bounds[2] * 1.5  # 50% margin on right

        # Create or update left launch pad
        if self.launch_left is not None:
            self.design.delete_component('launch_left')

        self.launch_left = LaunchpadWirebond(
            self.design,
            'launch_left',
            options=dict(
                pad_width=self._um_to_mm(100),      # 100um pad width
                pad_height=self._um_to_mm(100),     # 100um pad height
                pos_x=left_feedline_bound,          # Position at left bound
                pos_y=0,                            # Centered vertically
                orientation=0                       # No rotation
            )
        )

        # Create or update right launch pad
        if self.launch_right is not None:
            self.design.delete_component('launch_right')

        self.launch_right = LaunchpadWirebond(
            self.design,
            'launch_right',
            options=dict(
                pad_width=self._um_to_mm(100),      # 100um pad width
                pad_height=self._um_to_mm(100),     # 100um pad height
                pos_x=right_feedline_bound,         # Position at right bound
                pos_y=0,                            # Centered vertically
                orientation=180                     # 180 degree rotation
            )
        )

        # Create feedline connecting the launch pads
        if self.feedline is not None:
            self.design.delete_component('feedline')

        pin_opt = Dict(
            pin_inputs=Dict(
                start_pin=Dict(
                    component='launch_left',
                    pin='tie'
                ),
                end_pin=Dict(
                    component='launch_right',
                    pin='tie'
                )
            )
        )

        self.feedline = RouteStraight(self.design, 'feedline', pin_opt)

    def get_parameters(self):
        """
        Retrieve all current design parameters.

        Returns a dictionary containing all the key parameters of the resonator
        design in micrometers for easy interpretation.

        Returns:
            dict: Dictionary containing:
                - cpw_width: Width of CPW center conductor (um)
                - cpw_gap: Gap between conductor and ground (um)
                - resonator_length: Total electrical length of resonator (um)
                - resonator_height: Physical height of resonator layout (um)
                - coupler_gap: Gap in the coupling region (um)
                - coupler_length: Length of the coupling region (um)
        """
        parameters = {
            'cpw_width': self.cpw_width_um,
            'cpw_gap': self.cpw_gap_um,
            'resonator_length': self.resonator_length_um,
            'resonator_height': self.resonator_height_um,
            'coupler_gap': self.coupler_gap_um,
            'coupler_length': self.coupler_length_um
        }

        print("Current design parameters:")
        for key, value in parameters.items():
            print(f"  {key}: {value} um")

        return parameters

    def get_picture(self, filename='resonator_design.png', width=500):
        """
        Generate and display a screenshot of the current resonator design.

        This method rebuilds the GUI display, autoscales to fit all components,
        takes a screenshot, and saves it as an image file. It also displays
        the image inline if running in a Jupyter notebook environment.

        Args:
            filename (str): Name of the file to save the screenshot
            width (int): Width of the displayed image in pixels

        Returns:
            str: Path to the saved image file
        """
        # Check if GUI is enabled
        if not self.show_gui or self.gui is None:
            print("GUI is disabled - cannot generate screenshot")
            return None

        # Rebuild and autoscale the GUI to ensure current design is displayed
        if self.gui is not None:
            self.gui.rebuild()
            self.gui.autoscale()

            # Take screenshot and save to file
            self.gui.screenshot()
            self.gui.figure.savefig(filename)
        else:
            print("GUI is not initialized - cannot generate screenshot")
            return None

        # Display the image inline (works in Jupyter notebooks)
        try:
            display_options = dict(width=width)
            display(Image(filename, **display_options))
        except Exception as e:
            print(f"Could not display image inline: {e}")
            print(f"Image saved as: {filename}")

        print(f"Design screenshot saved as: {filename}")
        return filename

    def export_resonator(self, gds_filename='resonator.gds', sonnet_filename='sonnet.gds'):
        """
        Export the resonator design to GDS format for fabrication and simulation.

        This method performs the complete export workflow:
        1. Exports the raw design to GDS format
        2. Calculates appropriate bounding box for the design
        3. Performs boolean operations to clean up the geometry
        4. Creates a simulation-ready GDS file

        The exported files can be used for:
        - Electron beam lithography mask generation
        - Electromagnetic simulation (e.g., in Sonnet)
        - Design verification and analysis

        Args:
            gds_filename (str): Name of the raw GDS export file
            sonnet_filename (str): Name of the processed GDS file for simulation

        Returns:
            tuple: (gds_filename, sonnet_filename) - paths to the exported files
        """
        print(f"Exporting resonator design to GDS format...")

        # Ensure design is up to date
        if self.show_gui and self.gui is not None:
            self.gui.rebuild()

        # Export to GDS format
        self.design.renderers.gds.options['gds_unit'] = 0.001
        self.design.renderers.gds.export_to_gds(gds_filename)
        print(f"Raw GDS exported as: {gds_filename}")

        # Calculate bounding box for the design based on meander geometry
        if self.meander is not None:
            bounds = self.meander.qgeometry_bounds()

            # Define simulation box with margins
            box_bot = self._um_to_mm(-100)      # Bottom margin
            box_top = bounds[3] * 1.2           # Top with 20% margin
            box_left = bounds[0] * 1.2          # Left with 20% margin
            box_right = bounds[2] * 1.2         # Right with 20% margin

            # Perform boolean operations to clean up geometry for simulation
            try:
                manipulate_GDS.slice_and_boolean(
                    inpath=gds_filename,
                    savepath=sonnet_filename,
                    focus_box=np.array([[box_left, box_bot], [box_right, box_top]]),
                    booleans={"layer 1": [(1,0),(1,0)], "layer 2": [(1,10),(1,11)], "layer out": [(100,0),(100,0)], "operation":['or','or']},
                    layers_to_save=[100]
                )
                print(f"Processed GDS for simulation saved as: {sonnet_filename}")
            except Exception as e:
                print(f"Warning: Could not perform GDS post-processing: {e}")
                print(f"Raw GDS file is still available: {gds_filename}")
                sonnet_filename = "FAILED"

        else:
            print("Warning: No meander found, exporting raw GDS only")
            sonnet_filename = "FAILED"

        print("Export completed successfully!")
        return gds_filename, sonnet_filename


# MCP Tool Functions
# These functions provide the Model Context Protocol interface for LLM interaction

@mcp.tool()
def design_resonator_tool(resonator_length_um, resonator_height_um, show_gui=True):
    """
    MCP Tool: Design or modify the resonator dimensions.

    This tool allows an LLM to set the electrical length and physical height
    of the resonator structure. The length primarily determines the resonant
    frequency, while the height affects the physical layout and footprint.

    Args:
        resonator_length_um (float): Total electrical length in micrometers
        resonator_height_um (float): Physical height of layout in micrometers
        show_gui (bool): Whether to enable GUI operations (default: True)

    Returns:
        dict: Status and updated parameters
    """
    global designer_instance
    designer = initialize_designer(show_gui=show_gui)

    try:
        designer.design_resonator(resonator_length_um, resonator_height_um)
        return {
            "status": "success",
            "message": f"Resonator designed with length {resonator_length_um} um and height {resonator_height_um} um",
            "parameters": designer.get_parameters(),
            "gui_enabled": designer.is_gui_enabled()
        }
    except Exception as e:
        return {"error": f"Failed to design resonator: {str(e)}"}


@mcp.tool()
def design_coupler_tool(coupler_gap_um, coupler_length_um, show_gui=True):
    """
    MCP Tool: Design or modify the coupler dimensions.

    This tool allows an LLM to set the coupling gap and length, which control
    the electromagnetic coupling strength between the feedline and resonator.

    Args:
        coupler_gap_um (float): Coupling gap in micrometers
        coupler_length_um (float): Coupling length in micrometers
        show_gui (bool): Whether to enable GUI operations (default: True)

    Returns:
        dict: Status and updated parameters
    """
    global designer_instance
    designer = initialize_designer(show_gui=show_gui)

    try:
        designer.design_coupler(coupler_gap_um, coupler_length_um)
        return {
            "status": "success",
            "message": f"Coupler designed with gap {coupler_gap_um} um and length {coupler_length_um} um",
            "parameters": designer.get_parameters(),
            "gui_enabled": designer.is_gui_enabled()
        }
    except Exception as e:
        return {"error": f"Failed to design coupler: {str(e)}"}


@mcp.tool()
def get_parameters_tool(show_gui=True):
    """
    MCP Tool: Get all current design parameters.

    This tool allows an LLM to retrieve all the current dimensional parameters
    of the resonator design for inspection or further modification.

    Args:
        show_gui (bool): Whether to enable GUI operations (default: True)

    Returns:
        dict: All current design parameters in micrometers
    """
    global designer_instance
    designer = initialize_designer(show_gui=show_gui)

    try:
        parameters = designer.get_parameters()
        return {
            "status": "success",
            "parameters": parameters,
            "gui_enabled": designer.is_gui_enabled()
        }
    except Exception as e:
        return {"error": f"Failed to get parameters: {str(e)}"}


@mcp.tool()
def export_resonator_tool(gds_filename='resonator.gds', sonnet_filename='sonnet.gds', show_gui=True):
    """
    MCP Tool: Export the resonator design to GDS format.

    This tool allows an LLM to export the current design to GDS files suitable
    for fabrication and electromagnetic simulation.

    Args:
        gds_filename (str): Name for the raw GDS export file
        sonnet_filename (str): Name for the simulation-ready GDS file
        show_gui (bool): Whether to enable GUI operations (default: True)

    Returns:
        dict: Status and file paths
    """
    global designer_instance
    designer = initialize_designer(show_gui=show_gui)

    try:
        gds_path, sonnet_path = designer.export_resonator(gds_filename, sonnet_filename)
        return {
            "status": "success",
            "message": "Resonator exported successfully",
            "files": {
                "gds_file": gds_path,
                "sonnet_file": sonnet_path
            },
            "gui_enabled": designer.is_gui_enabled()
        }
    except Exception as e:
        return {"error": f"Failed to export resonator: {str(e)}"}


@mcp.tool()
def get_picture_tool(show_gui=True):
    """
    MCP Tool: Generate and save a screenshot of the current design.

    This tool allows an LLM to generate visual feedback of the current resonator
    design by creating a screenshot image. Requires GUI to be enabled.

    Args:
        show_gui (bool): Whether to enable GUI operations (default: True)

    Returns:
        Image object containing the a render of the resonator design
    """
    global designer_instance
    designer = initialize_designer(show_gui=show_gui)

    try:
        # Check if GUI is enabled
        if not designer.is_gui_enabled():
            raise ValueError("GUI is disabled - cannot generate screenshot. Set show_gui=True to enable GUI operations.")

        # Rebuild and autoscale the GUI to ensure current design is displayed
        if designer.gui is not None:
            designer.gui.rebuild()
            designer.gui.autoscale()

            # Take screenshot and save to memory buffer instead of file
            designer.gui.screenshot()

            # Create a BytesIO buffer to capture the image data
            buf = io.BytesIO()
            designer.gui.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_data = buf.read()
        else:
            raise ValueError("GUI is not initialized - cannot generate screenshot.")

        # Return Image object for MCP
        return Image(data=img_data, format="png")

    except Exception as e:
        raise ValueError(f"Failed to generate screenshot: {str(e)}")


#@mcp.tool()
#def gui_status_tool():
#    """
#    MCP Tool: Check the current GUI status.
#
#    Returns:
#        dict: GUI status information
#    """
#    global designer_instance
#    if designer_instance is None:
#        return {
#            "status": "success",
#            "gui_enabled": False,
#            "message": "Designer not initialized"
#        }
#
#    return {
#        "status": "success",
#        "gui_enabled": designer_instance.is_gui_enabled(),
#        "message": f"GUI is {'enabled' if designer_instance.is_gui_enabled() else 'disabled'}"
#    }
#
#
#@mcp.tool()
#def enable_gui_tool():
#    """
#    MCP Tool: Enable GUI operations.
#
#    Returns:
#        dict: Status of GUI enable operation
#    """
#    global designer_instance
#    designer = initialize_designer(show_gui=True)
#
#    try:
#        was_enabled = designer.enable_gui()
#        return {
#            "status": "success",
#            "gui_enabled": designer.is_gui_enabled(),
#            "message": "GUI enabled successfully" if was_enabled else "GUI was already enabled"
#        }
#    except Exception as e:
#        return {"error": f"Failed to enable GUI: {str(e)}"}
#
#
#@mcp.tool()
#def disable_gui_tool():
#    """
#    MCP Tool: Disable GUI operations.
#
#    Returns:
#        dict: Status of GUI disable operation
#    """
#    global designer_instance
#    if designer_instance is None:
#        return {
#            "status": "success",
#            "gui_enabled": False,
#            "message": "Designer not initialized - GUI already disabled"
#        }
#
#    try:
#        was_disabled = designer_instance.disable_gui()
#        return {
#            "status": "success",
#            "gui_enabled": designer_instance.is_gui_enabled(),
#            "message": "GUI disabled successfully" if was_disabled else "GUI was already disabled"
#        }
#    except Exception as e:
#        return {"error": f"Failed to disable GUI: {str(e)}"}

if __name__ == "__main__":
    mcp.run('stdio')
