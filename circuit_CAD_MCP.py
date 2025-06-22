#!/usr/bin/env python3

"""
PHIDL Superconducting Resonator Design Classes with MCP Tools

This module provides comprehensive classes for designing superconducting resonators
and couplers using PHIDL, along with MCP (Model Context Protocol) tools that allow
Large Language Models to customize device dimensions and parameters.

The module includes two main classes:
1. ResonatorChipDesign - Complete resonator chip with feedlines, couplers, and meandered resonator
2. CouplerChipDesign - Standalone coupler structure for S-parameter analysis

Both classes handle the complete workflow from design to export, including:
- Creating CPW (Coplanar Waveguide) structures with wirebond pads
- Designing couplers with programmable dimensions
- Creating meandered resonators with customizable geometry
- Exporting designs to GDS format for fabrication
- Generating visual images of the design using quickplot

All input parameters use micrometers (um) as units, consistent with PHIDL conventions.
"""

from phidl import Device, CrossSection, Path
import phidl.path as pp
import phidl.geometry as pg
import numpy as np
from phidl import quickplot as qp
from copy import deepcopy
import matplotlib.pyplot as plt
import manipulate_GDS
from quickplot_nosave import quickplot_noshow
import io
from mcp.server.fastmcp import FastMCP, Image
import logging

mcp = FastMCP("CAD_tools")
logging.basicConfig(level=logging.INFO, filename = "circuit_CAD_MCP.log")
logger = logging.getLogger(__name__)

# Global designer instances
resonator_designer_instance = None
coupler_designer_instance = None

def initialize_resonator_designer():
    """Initialize the global resonator designer instance if not already created."""
    global resonator_designer_instance
    if resonator_designer_instance is None:
        resonator_designer_instance = ResonatorChipDesign()
    return resonator_designer_instance

def initialize_coupler_designer():
    """Initialize the global coupler designer instance if not already created."""
    global coupler_designer_instance
    if coupler_designer_instance is None:
        coupler_designer_instance = CouplerChipDesign()
    return coupler_designer_instance

class ResonatorChipDesign:
    """
    A comprehensive class for designing complete superconducting resonator chips using PHIDL.

    This class provides a complete workflow for creating, modifying, and exporting
    superconducting resonator chip designs. It includes coplanar waveguide (CPW)
    transmission lines, couplers, meandered resonators, wirebond launch pads, and
    proper ground plane structures for fabrication.

    The resonator chip consists of:
    - Left and right wirebond pads with CPW tapers for external connections
    - A straight CPW feedline connecting the wirebond pads
    - A coupler structure that electromagnetically couples to the resonator
    - A meandered resonator that provides a long electrical path in compact footprint
    - Proper ground plane cutouts and chip boundaries for fabrication

    All dimensional parameters are specified in micrometers (um), consistent with
    PHIDL conventions and typical superconducting device scales.

    Attributes:
        cpw_width (float): Width of the CPW center conductor (um) - fixed at 10 um
        cpw_gap (float): Gap between center conductor and ground plane (um) - fixed at 10 um
        wirebond_pad_width (float): Width of wirebond pads (um) - fixed at 100 um
        wirebond_pad_height (float): Height of wirebond pads (um) - fixed at 100 um
        wirebond_pad_taper_length (float): Length of taper from pad to CPW (um) - fixed at 50 um
        wirebond_gap (float): Gap around wirebond pads (um) - fixed at 30 um
        resonator_meander_space (float): Spacing between meander lines (um) - fixed at 100 um
        resonator_fillet (float): Radius for rounded corners (um) - fixed at 10 um
        coupler_length_normal (float): Length of normal sections in coupler (um) - fixed at 100 um
        cutout_layer (int): PHIDL layer number for ground plane cutouts - fixed at 11
        cad_layer (int): PHIDL layer number for metal structures - fixed at 12

        # Settable parameters (controlled via generate_cad method):
        resonator_length (float): Total electrical length of resonator (um) - affects resonant frequency
        resonator_width (float): Physical width/footprint of meandered resonator (um)
        coupler_gap (float): Gap between feedline and resonator in coupler (um) - affects coupling strength
        coupler_length (float): Length of coupling region (um) - affects coupling strength

        # Generated objects:
        device (Device): The complete chip design PHIDL device
        drawn_cad (Device): The metal structures without chip boundaries
    """

    def __init__(self):
        """
        Initialize the resonator chip designer with default parameters.

        Sets up all fixed parameters based on the original notebook values,
        and initializes settable parameters to reasonable defaults.
        """
        # Fixed CPW parameters (cannot be changed)
        self.cpw_width = 10  # um
        self.cpw_gap = 10    # um

        # Fixed wirebond parameters
        self.wirebond_pad_width = 100   # um
        self.wirebond_pad_height = 100  # um
        self.wirebond_pad_taper_length = 50  # um
        self.wirebond_gap = 30  # um

        # Fixed resonator geometry parameters
        self.resonator_meander_space = 100  # um - spacing between meander lines
        self.resonator_fillet = 10  # um - radius for rounded corners

        # Fixed coupler parameters
        self.coupler_length_normal = 100  # um - length of normal coupler sections

        # Fixed layer definitions
        self.cutout_layer = 11  # Ground plane cutout layer
        self.cad_layer = 12     # Metal structure layer

        # Settable parameters with defaults from notebook
        self.resonator_length = 8100  # um - total electrical length, determines frequency
        self.resonator_width = 500    # um - physical footprint width of meander
        self.coupler_gap = 20         # um - coupling gap, smaller = stronger coupling
        self.coupler_length = 100     # um - coupling length, longer = stronger coupling

        # Generated design objects (initially None)
        self.device = None      # Complete chip with boundaries
        self.drawn_cad = None   # Just the metal structures

    def _create_wirebond_device(self):
        """
        Create a wirebond pad device with CPW taper.

        This internal method creates the wirebond launch pad structure that
        transitions from a large rectangular pad suitable for wire bonding
        to the narrow CPW transmission line geometry.

        Returns:
            Device: PHIDL device containing wirebond pad and taper
        """
        wirebond = Device("wirebond")

        # Create the main wirebond pad
        wirebond_pad_path = pp.straight(self.wirebond_pad_width)
        wirebond_pad_xc = CrossSection()
        wirebond_pad_xc.add(width=self.wirebond_pad_height, layer=self.cad_layer,
                           name='metal', ports=('in', 'out'))
        wirebond_pad_xc.add(width=self.wirebond_pad_height + 2*self.wirebond_gap,
                           name='cutout', layer=self.cutout_layer)
        wirebond_pad = wirebond_pad_path.extrude(wirebond_pad_xc)

        # Create CPW cross-section
        cpw_xc = CrossSection()
        cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                  name='metal', ports=('in', 'out'))
        cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                  name='cutout', layer=self.cutout_layer)

        # Create smooth transition from wirebond pad to CPW
        wirebond_cpw_trans_xc = pp.transition(wirebond_pad_xc, cpw_xc, width_type='sine')
        wirebond_cpw_trans_path = pp.straight(self.wirebond_pad_taper_length)
        wirebond_cpw_trans = wirebond_cpw_trans_path.extrude(wirebond_cpw_trans_xc, simplify=0.2)

        # Assemble wirebond device
        wirebond << wirebond_pad
        wirebond << wirebond_cpw_trans.move((self.wirebond_pad_width, 0))

        # Add spacer for proper ground plane cutout
        wirebond_spacer = pg.rectangle(
            size=(self.wirebond_gap, self.wirebond_pad_height + 2*self.wirebond_gap),
            layer=self.cutout_layer
        ).move((-self.wirebond_gap, -(self.wirebond_pad_height + 2*self.wirebond_gap)/2))
        wirebond << wirebond_spacer

        # Position wirebond for proper connection
        wirebond.move((-self.wirebond_pad_width - self.wirebond_pad_taper_length, 0))

        return wirebond

    def _create_coupler_device(self):
        """
        Create the coupler device that electromagnetically couples feedline to resonator.

        The coupler uses a coupled-line structure where the main feedline runs
        parallel to a secondary line that connects to the resonator. The coupling
        strength is controlled by the gap and length parameters.

        Returns:
            tuple: (coupler_device, coupler_endpoint) where endpoint is the connection
                   point for the resonator as (x, y) coordinates
        """
        # Create CPW cross-section for coupler
        cpw_xc = CrossSection()
        cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                  name='metal', ports=('in', 'out'))
        cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                  name='cutout', layer=self.cutout_layer)

        # Create primary (feedline) path - straight horizontal line
        coupler_primary_path = pp.straight(self.coupler_length)
        coupler_primary = coupler_primary_path.extrude(cpw_xc)

        # Create secondary (resonator connection) path - L-shaped
        coupler_secondary_path_pts = [
            (0, self.coupler_gap),
            (self.coupler_length, self.coupler_gap),
            (self.coupler_length, self.coupler_gap + self.coupler_length_normal)
        ]
        coupler_secondary_path = pp.smooth(coupler_secondary_path_pts,
                                         radius=self.resonator_fillet,
                                         corner_fun=pp.euler)
        coupler_secondary = coupler_secondary_path.extrude(cpw_xc)

        # Calculate endpoint for resonator connection
        coupler_endpoint = (self.coupler_length, self.coupler_gap + self.coupler_length_normal)

        # Create additional ground plane cutouts for proper isolation
        coupler_secondary_gap = pg.rectangle(
            size=(self.cpw_gap, self.cpw_width + 2*self.cpw_gap),
            layer=self.cutout_layer
        ).move((-self.cpw_gap, self.coupler_gap - (self.cpw_width + 2*self.cpw_gap)/2))

        coupler_secondary_gap2 = pg.rectangle(
            size=(self.cpw_width + 2*self.cpw_gap, self.cpw_gap),
            layer=self.cutout_layer
        ).move((coupler_endpoint[0] - (self.cpw_width + 2*self.cpw_gap)/2, coupler_endpoint[1]))

        # Assemble complete coupler device
        coupler = Device("coupler")
        coupler << coupler_primary
        coupler << coupler_secondary
        coupler << coupler_secondary_gap
        coupler << coupler_secondary_gap2

        return coupler, coupler_endpoint

    def _create_resonator_meander(self, coupler_endpoint):
        """
        Create the meandered resonator structure.

        This method dynamically generates a meandered path that provides the
        specified total electrical length within the given physical width.
        The meander alternates between left and right directions to efficiently
        use the available space.

        Args:
            coupler_endpoint (tuple): (x, y) coordinates where resonator connects to coupler

        Returns:
            tuple: (resonator_device, remaining_length) where remaining_length is any
                   unused length if the meander couldn't accommodate the full specified length
        """
        # Create CPW cross-section for resonator
        cpw_xc = CrossSection()
        cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                  name='metal', ports=('in', 'out'))
        cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                  name='cutout', layer=self.cutout_layer)

        # Initialize meander generation parameters
        resonator_length_current = 0  # Track how much length we've used
        next_meander_direction = 'left'  # Start by going left

        # Define meander boundaries
        meander_x_right = coupler_endpoint[0]
        meander_x_left = coupler_endpoint[0] - self.resonator_width

        # Starting point for meander (slightly offset from coupler endpoint)
        meander_start_x = coupler_endpoint[0] + self.cpw_width/2
        meander_start_y = coupler_endpoint[1] - self.cpw_width/2
        meander_points = [(meander_start_x, meander_start_y)]

        meander_y = meander_start_y

        # Generate meander points by alternating left and right
        while resonator_length_current + self.resonator_width + self.resonator_meander_space < self.resonator_length:
            if next_meander_direction == 'left':
                # Go to left edge
                meander_points.append((meander_x_left, meander_y))
                next_meander_direction = 'right'
                # Move up for next segment
                meander_y += self.resonator_meander_space
                meander_points.append((meander_x_left, meander_y))
            else:
                # Go to right edge
                meander_points.append((meander_x_right, meander_y))
                next_meander_direction = 'left'
                # Move up for next segment
                meander_y += self.resonator_meander_space
                meander_points.append((meander_x_right, meander_y))

            # Update length used
            resonator_length_current += self.resonator_width + self.resonator_meander_space

        # Handle remaining length if any
        remaining_resonator_length = self.resonator_length - resonator_length_current
        if remaining_resonator_length > 2*self.resonator_fillet:  # Only if meaningful length
            if next_meander_direction == 'left':
                meander_points.append((meander_x_right - remaining_resonator_length, meander_y))
            else:
                meander_points.append((meander_x_left + remaining_resonator_length, meander_y))
            remaining_resonator_length = 0  # Used up all length

        # Create smooth path with rounded corners
        resonator_path = pp.smooth(meander_points, radius=self.resonator_fillet,
                                 corner_fun=pp.euler)
        resonator_meander = resonator_path.extrude(cpw_xc, simplify=0.2)

        return resonator_meander, remaining_resonator_length

    def generate_cad(self, resonator_length, resonator_width, coupler_gap, coupler_length,
                     bare_chip_filename="bare_chip.gds",
                     full_chip_filename="full_chip.gds",
                     sonnet_chip_filename="sonnet_chip.gds",
                     image_filename="resonator_chip.png"):
        """
        Generate the complete resonator chip CAD design with specified parameters.

        This method performs the complete design workflow:
        1. Updates design parameters
        2. Creates all individual components (wirebonds, coupler, resonator)
        3. Calculates chip boundaries and feedline dimensions
        4. Assembles the complete chip
        5. Exports to multiple GDS formats for different purposes
        6. Generates and saves a visual image of the design

        The method creates several GDS files:
        - bare_chip.gds: Raw chip design with all layers (customizable filename)
        - full_chip.gds: Complete chip with proper boolean operations for fabrication (customizable filename)
        - sonnet_chip.gds: Simulation-ready version with appropriate boundaries (customizable filename)

        Args:
            resonator_length (float): Total electrical length of resonator in micrometers.
                                    This is the primary parameter that determines the resonant
                                    frequency: f ≈ c/(2*n_eff*L) where c is speed of light,
                                    n_eff is effective refractive index (~6-7 for superconducting
                                    CPW on silicon), and L is this length.
            resonator_width (float): Physical width/footprint of the meandered resonator
                                   in micrometers. Larger widths create wider meanders with
                                   fewer turns. Typical values: 200-1000 um.
            coupler_gap (float): Gap between the feedline and resonator in the coupling
                               region in micrometers. Smaller gaps create stronger coupling.
                               Typical values: 5-50 um.
            coupler_length (float): Length of the coupling region in micrometers.
                                  Longer coupling regions create stronger coupling.
                                  Typical values: 50-200 um.
            bare_chip_filename (str): Filename for the raw GDS export (default: "bare_chip.gds")
            full_chip_filename (str): Filename for the fabrication-ready GDS (default: "full_chip.gds")
            sonnet_chip_filename (str): Filename for the simulation GDS (default: "sonnet_chip.gds")
            image_filename (str): Filename for the design image (default: "resonator_chip.png")

        Returns:
            dict: Status information including:
                - success: True if generation completed successfully
                - files_created: List of GDS files that were created
                - remaining_resonator_length: Any unused resonator length (should be ~0)
                - chip_dimensions: Physical dimensions of the generated chip
                - message: Summary of what was generated
        """
        print(f"Generating resonator chip with parameters:")
        print(f"  Resonator length: {resonator_length} um")
        print(f"  Resonator width: {resonator_width} um")
        print(f"  Coupler gap: {coupler_gap} um")
        print(f"  Coupler length: {coupler_length} um")
        print(f"  Output files: {bare_chip_filename}, {full_chip_filename}, {sonnet_chip_filename}, {image_filename}")

        # Update instance parameters
        self.resonator_length = resonator_length
        self.resonator_width = resonator_width
        self.coupler_gap = coupler_gap + 2*self.cpw_gap
        self.coupler_length = coupler_length

        try:
            # Create wirebond device template
            wirebond = self._create_wirebond_device()

            # Create coupler
            coupler, coupler_endpoint = self._create_coupler_device()

            # Create meandered resonator
            resonator_meander, remaining_length = self._create_resonator_meander(coupler_endpoint)

            # Calculate chip layout dimensions based on resonator bounding box
            resonator_bbox = resonator_meander.bbox
            print(f"Resonator bounding box: {resonator_bbox}")

            # Calculate feedline dimensions (500 um gap on either side of resonator)
            feedline_xmin = resonator_bbox[0][0] - 500
            feedline_xmax = resonator_bbox[1][0] + 500

            # Calculate chip boundary dimensions
            chip_boundary_xmin = feedline_xmin - 1000
            chip_boundary_xmax = feedline_xmax + 1000
            chip_boundary_ymin = -500
            chip_boundary_ymax = resonator_bbox[1][1] + 500

            # Calculate Sonnet simulation boundaries (smaller for efficiency)
            sonnet_boundary_xmin = feedline_xmin + 250
            sonnet_boundary_xmax = feedline_xmax - 250
            sonnet_boundary_ymin = -250
            sonnet_boundary_ymax = resonator_bbox[1][1] + 250

            # Create CPW cross-section for feedline
            cpw_xc = CrossSection()
            cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                      name='metal', ports=('in', 'out'))
            cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                      name='cutout', layer=self.cutout_layer)

            # Create main feedline
            feedline_path = pp.straight(feedline_xmax - feedline_xmin)
            feedline = feedline_path.extrude(cpw_xc).move((feedline_xmin, 0))

            # Assemble all CAD components
            drawn_cad = Device("drawn_cad")
            drawn_cad << feedline
            drawn_cad << resonator_meander
            drawn_cad << coupler
            drawn_cad << deepcopy(wirebond).move((feedline_xmin, 0))
            drawn_cad << deepcopy(wirebond).rotate(180).move((feedline_xmax, 0))

            # Create complete chip with ground plane
            chip = Device("chip")
            chip << pg.rectangle(
                size=(chip_boundary_xmax - chip_boundary_xmin,
                     chip_boundary_ymax - chip_boundary_ymin),
                layer=10
            ).move((chip_boundary_xmin, chip_boundary_ymin))
            chip << drawn_cad

            # Store the generated devices in class attributes
            self.device = chip
            self.drawn_cad = drawn_cad

            # Export to GDS files
            print("Exporting GDS files...")

            # Export raw chip
            chip.write_gds(bare_chip_filename, precision=1e-6)

            # Create full chip with proper boolean operations for fabrication
            manipulate_GDS.slice_and_boolean(
                bare_chip_filename, full_chip_filename,
                np.array([[chip_boundary_xmin, chip_boundary_ymin],
                         [chip_boundary_xmax, chip_boundary_ymax]]),
                {"layer 1": [10, 1], "layer 2": [self.cutout_layer, self.cad_layer],
                 "layer out": [1, 1], "operation": ["not", "or"]},
                [1]
            )

            # Create Sonnet simulation version
            manipulate_GDS.slice_and_boolean(
                bare_chip_filename, sonnet_chip_filename,
                np.array([[sonnet_boundary_xmin, sonnet_boundary_ymin],
                         [sonnet_boundary_xmax, sonnet_boundary_ymax]]),
                {"layer 1": [10, 1], "layer 2": [self.cutout_layer, self.cad_layer],
                 "layer out": [1, 1], "operation": ["not", "or"]},
                [1]
            )

            # Generate and save image using quickplot_noshow
            print("Generating design image...")
            quickplot_noshow(drawn_cad)
            plt.savefig(image_filename, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory

            files_created = [bare_chip_filename, full_chip_filename, sonnet_chip_filename, image_filename]

            chip_dimensions = {
                "width_um": chip_boundary_xmax - chip_boundary_xmin,
                "height_um": chip_boundary_ymax - chip_boundary_ymin,
                "feedline_length_um": feedline_xmax - feedline_xmin
            }

            print("Resonator chip generation completed successfully!")
            print(f"Files created: {files_created}")
            if remaining_length > 0:
                print(f"Warning: {remaining_length} um of resonator length could not be accommodated")

            return {
                "success": True,
                "files_created": files_created,
                "remaining_resonator_length": remaining_length,
                "chip_dimensions": chip_dimensions,
                "message": f"Generated resonator chip with {resonator_length} um resonator"
            }

        except Exception as e:
            print(f"Error generating resonator chip: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate resonator chip"
            }

    def get_params(self):
        """
        Get all current design parameters for the resonator chip.

        Returns both fixed parameters (that cannot be changed) and settable
        parameters (that can be modified via generate_cad). This provides
        complete visibility into the design state.

        Returns:
            dict: Complete parameter set including:
                - Fixed CPW parameters (cpw_width, cpw_gap)
                - Fixed wirebond parameters
                - Fixed resonator geometry parameters
                - Current settable parameters (resonator_length, resonator_width, etc.)
                - Layer definitions
        """
        params = {
            # Fixed CPW parameters
            "cpw_width": self.cpw_width,
            "cpw_gap": self.cpw_gap,

            # Fixed wirebond parameters
            "wirebond_pad_width": self.wirebond_pad_width,
            "wirebond_pad_height": self.wirebond_pad_height,
            "wirebond_pad_taper_length": self.wirebond_pad_taper_length,
            "wirebond_gap": self.wirebond_gap,

            # Fixed resonator geometry parameters
            "resonator_meander_space": self.resonator_meander_space,
            "resonator_fillet": self.resonator_fillet,

            # Fixed coupler parameters
            "coupler_length_normal": self.coupler_length_normal,

            # Layer definitions
            "cutout_layer": self.cutout_layer,
            "cad_layer": self.cad_layer,

            # Settable parameters
            "resonator_length": self.resonator_length,
            "resonator_width": self.resonator_width,
            "coupler_gap": self.coupler_gap,
            "coupler_length": self.coupler_length,

            # Status
            "design_generated": self.device is not None
        }

        print("Current ResonatorChipDesign parameters:")
        print("Fixed parameters:")
        for key in ["cpw_width", "cpw_gap", "wirebond_pad_width", "wirebond_pad_height"]:
            print(f"  {key}: {params[key]}")
        print("Settable parameters:")
        for key in ["resonator_length", "resonator_width", "coupler_gap", "coupler_length"]:
            print(f"  {key}: {params[key]}")

        return params

    def get_image(self, save_to_file=False, filename="resonator_image.png"):
        """
        Get an image of the current resonator chip design.

        Uses the stored CAD object (drawn_cad) to generate a visual representation
        of the current design. If no design has been generated yet, returns an
        error message.

        Args:
            save_to_file (bool): Whether to save the image to a file (default: False)
            filename (str): Filename to save image to if save_to_file=True (default: "resonator_image.png")

        Returns:
            bytes: PNG image data as bytes, or raises ValueError if no design exists
        """
        if self.drawn_cad is None:
            raise ValueError("No design has been generated yet. Call generate_cad() first.")

        try:
            # Generate image using quickplot_noshow
            quickplot_noshow(self.drawn_cad)

            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_data = buf.read()

            # Optionally save to file
            if save_to_file:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Image saved to {filename}")

            plt.close()  # Close to free memory

            return img_data

        except Exception as e:
            raise ValueError(f"Failed to generate image: {str(e)}")

    def save_params_to_file(self, filename="resonator_params.txt"):
        """
        Save current parameters to a text file for documentation.

        Args:
            filename (str): Filename to save parameters to (default: "resonator_params.txt")
        """
        params = self.get_params()

        try:
            with open(filename, 'w') as f:
                f.write("Resonator Chip Design Parameters\n")
                f.write("================================\n\n")

                f.write("Fixed CPW Parameters:\n")
                for key in ["cpw_width", "cpw_gap"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nFixed Wirebond Parameters:\n")
                for key in ["wirebond_pad_width", "wirebond_pad_height", "wirebond_pad_taper_length", "wirebond_gap"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nFixed Resonator Geometry Parameters:\n")
                for key in ["resonator_meander_space", "resonator_fillet"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nFixed Coupler Parameters:\n")
                for key in ["coupler_length_normal"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nLayer Definitions:\n")
                for key in ["cutout_layer", "cad_layer"]:
                    f.write(f"  {key}: {params[key]}\n")

                f.write("\nSettable Parameters:\n")
                for key in ["resonator_length", "resonator_width", "coupler_gap", "coupler_length"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write(f"\nDesign Status:\n")
                f.write(f"  design_generated: {params['design_generated']}\n")

            print(f"Parameters saved to {filename}")

        except Exception as e:
            print(f"Error saving parameters: {str(e)}")

class CouplerChipDesign:
    """
    A class for designing standalone superconducting coupler structures using PHIDL.

    This class creates isolated coupler structures suitable for S-parameter analysis
    and electromagnetic simulation. The coupler chip includes just the coupling
    structure with appropriate feedlines and ground plane boundaries, without the
    full resonator.

    This is useful for:
    - Characterizing coupling strength independently
    - S31 parameter simulations (transmission through coupler)
    - Optimizing coupler design before integration with resonator

    The coupler chip consists of:
    - A straight CPW feedline
    - The coupler structure with programmable gap and length
    - Proper ground plane cutouts and chip boundaries for simulation

    All dimensional parameters are specified in micrometers (um).

    Attributes:
        cpw_width (float): Width of the CPW center conductor (um) - fixed at 10 um
        cpw_gap (float): Gap between center conductor and ground plane (um) - fixed at 10 um
        resonator_fillet (float): Radius for rounded corners (um) - fixed at 10 um
        coupler_length_normal (float): Length of normal sections in coupler (um) - fixed at 100 um
        cutout_layer (int): PHIDL layer number for ground plane cutouts - fixed at 11
        cad_layer (int): PHIDL layer number for metal structures - fixed at 12

        # Settable parameters:
        coupler_gap (float): Gap in the coupling region (um) - affects coupling strength
        coupler_length (float): Length of the coupling region (um) - affects coupling strength

        # Generated objects:
        device (Device): The complete coupler chip design
        coupler_sonnet_sim (Device): The coupler structure without chip boundaries
    """

    def __init__(self):
        """
        Initialize the coupler chip designer with default parameters.
        """
        # Fixed CPW parameters (same as resonator chip)
        self.cpw_width = 10  # um
        self.cpw_gap = 10    # um

        # Fixed geometry parameters
        self.resonator_fillet = 10  # um - radius for rounded corners
        self.coupler_length_normal = 100  # um - length of normal coupler sections

        # Fixed layer definitions
        self.cutout_layer = 11  # Ground plane cutout layer
        self.cad_layer = 12     # Metal structure layer

        # Settable parameters with defaults
        self.coupler_gap = 20    # um - coupling gap
        self.coupler_length = 100  # um - coupling length

        # Generated design objects (initially None)
        self.device = None           # Complete chip with boundaries
        self.coupler_sonnet_sim = None  # Just the coupler structure

    def _create_coupler_device(self):
        """
        Create the standalone coupler device for S-parameter analysis.

        Creates the same coupler structure as used in the resonator chip,
        but optimized for standalone simulation and characterization.

        Returns:
            Device: PHIDL device containing the coupler structure
        """
        # Create CPW cross-section
        cpw_xc = CrossSection()
        cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                  name='metal', ports=('in', 'out'))
        cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                  name='cutout', layer=self.cutout_layer)

        # Create primary (feedline) path - straight horizontal line
        coupler_primary_path = pp.straight(self.coupler_length)
        coupler_primary = coupler_primary_path.extrude(cpw_xc)

        # Create secondary (test port) path - L-shaped
        coupler_secondary_path_pts = [
            (0, self.coupler_gap),
            (self.coupler_length, self.coupler_gap),
            (self.coupler_length, self.coupler_gap + self.coupler_length_normal)
        ]
        coupler_secondary_path = pp.smooth(coupler_secondary_path_pts,
                                         radius=self.resonator_fillet,
                                         corner_fun=pp.euler)
        coupler_secondary = coupler_secondary_path.extrude(cpw_xc)

        # Calculate endpoint coordinates
        coupler_endpoint = (self.coupler_length, self.coupler_gap + self.coupler_length_normal)

        # Create additional ground plane cutouts for proper isolation
        coupler_secondary_gap = pg.rectangle(
            size=(self.cpw_gap, self.cpw_width + 2*self.cpw_gap),
            layer=self.cutout_layer
        ).move((-self.cpw_gap, self.coupler_gap - (self.cpw_width + 2*self.cpw_gap)/2))

        coupler_secondary_gap2 = pg.rectangle(
            size=(self.cpw_width + 2*self.cpw_gap, self.cpw_gap),
            layer=self.cutout_layer
        ).move((coupler_endpoint[0] - (self.cpw_width + 2*self.cpw_gap)/2, coupler_endpoint[1]))

        # Assemble complete coupler device
        coupler = Device("coupler")
        coupler << coupler_primary
        coupler << coupler_secondary
        coupler << coupler_secondary_gap
        coupler << coupler_secondary_gap2

        return coupler

    def generate_cad(self, coupler_gap, coupler_length,
                     coupler_chip_filename="coupler_chip.gds",
                     sonnet_coupler_filename="sonnet_coupler.gds",
                     image_filename="coupler_chip.png"):
        """
        Generate the complete coupler chip CAD design with specified parameters.

        This method creates a standalone coupler structure suitable for S-parameter
        measurements and electromagnetic simulation. The design includes the coupler
        with extended feedlines and proper ground plane boundaries.

        The method creates GDS files:
        - coupler_chip.gds: Raw coupler chip design (customizable filename)
        - sonnet_coupler.gds: Simulation-ready version with boolean operations (customizable filename)

        Args:
            coupler_gap (float): Gap between the two transmission lines in the
                               coupling region in micrometers. This is the primary
                               parameter controlling coupling strength - smaller gaps
                               create stronger coupling. Typical values: 5-50 um.
            coupler_length (float): Length of the parallel coupling region in
                                  micrometers. Longer coupling regions create stronger
                                  coupling. Typical values: 50-200 um.
            coupler_chip_filename (str): Filename for the raw coupler chip GDS (default: "coupler_chip.gds")
            sonnet_coupler_filename (str): Filename for the simulation-ready GDS (default: "sonnet_coupler.gds")
            image_filename (str): Filename for the design image (default: "coupler_chip.png")

        Returns:
            dict: Status information including:
                - success: True if generation completed successfully
                - files_created: List of GDS files that were created
                - coupler_dimensions: Physical dimensions of the generated coupler
                - message: Summary of what was generated
        """
        print(f"Generating coupler chip with parameters:")
        print(f"  Coupler gap: {coupler_gap} um")
        print(f"  Coupler length: {coupler_length} um")
        print(f"  Output files: {coupler_chip_filename}, {sonnet_coupler_filename}, {image_filename}")

        # Update instance parameters
        self.coupler_gap = coupler_gap + 2*self.cpw_gap
        self.coupler_length = coupler_length

        try:
            # Create the coupler structure
            coupler = self._create_coupler_device()

            # Create CPW cross-section for feedline
            cpw_xc = CrossSection()
            cpw_xc.add(width=self.cpw_width, layer=self.cad_layer,
                      name='metal', ports=('in', 'out'))
            cpw_xc.add(width=self.cpw_width + 2*self.cpw_gap,
                      name='cutout', layer=self.cutout_layer)

            # Create extended feedline for S-parameter measurements
            # Add 200 um on each side of the coupler for proper port definition
            feedline_extension = 200
            total_feedline_length = self.coupler_length + 2 * feedline_extension
            coupler_feedline = pp.straight(total_feedline_length).extrude(cpw_xc).move(
                (-(total_feedline_length)/2 + coupler.center[0], 0)
            )

            # Create the coupler simulation device (without chip boundaries)
            coupler_sonnet_sim = Device("coupler_sonnet_sim")
            coupler_sonnet_sim << coupler
            coupler_sonnet_sim << coupler_feedline

            # Create ground plane for the chip
            # Extend ground plane 100 um beyond the highest point and below the CPW gap
            coupler_chip = Device("coupler_chip")
            coupler_chip_groundplane = pg.rectangle(
                size=(coupler_sonnet_sim.xsize, coupler_sonnet_sim.ymax - self.cpw_gap + 100),
                layer=10
            )
            coupler_chip_groundplane.move(
                (-coupler_sonnet_sim.xsize/2 + coupler_sonnet_sim.center[0], -100)
            )

            # Assemble complete chip
            coupler_chip << coupler_sonnet_sim
            coupler_chip << coupler_chip_groundplane

            coupler_chip.move((-self.coupler_length, 0)) # to center the coupler at x=0


            # Store the generated devices in class attributes
            self.device = coupler_chip
            self.coupler_sonnet_sim = coupler_sonnet_sim

            # Export to GDS files
            print("Exporting GDS files...")

            # Export raw coupler chip
            coupler_chip.write_gds(coupler_chip_filename)

            # Create Sonnet simulation version with proper boolean operations
            manipulate_GDS.slice_and_boolean(
                coupler_chip_filename, sonnet_coupler_filename,
                coupler_chip.bbox,
                {"layer 1": [10, 1], "layer 2": [self.cutout_layer, self.cad_layer],
                 "layer out": [1, 1], "operation": ["not", "or"]},
                [1]
            )

            # Generate and save image using quickplot_noshow
            print("Generating design image...")
            quickplot_noshow(coupler_sonnet_sim)
            plt.savefig(image_filename, dpi=150, bbox_inches='tight')
            plt.close()  # Close to free memory

            files_created = [coupler_chip_filename, sonnet_coupler_filename, image_filename]

            coupler_dimensions = {
                "width_um": coupler_chip.xsize,
                "height_um": coupler_chip.ysize,
                "feedline_length_um": total_feedline_length
            }

            print("Coupler chip generation completed successfully!")
            print(f"Files created: {files_created}")

            return {
                "success": True,
                "files_created": files_created,
                "coupler_dimensions": coupler_dimensions,
                "message": f"Generated coupler chip with {coupler_gap} um gap and {coupler_length} um length"
            }

        except Exception as e:
            print(f"Error generating coupler chip: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate coupler chip"
            }

    def get_params(self):
        """
        Get all current design parameters for the coupler chip.

        Returns both fixed parameters (that cannot be changed) and settable
        parameters (that can be modified via generate_cad).

        Returns:
            dict: Complete parameter set including:
                - Fixed CPW parameters (cpw_width, cpw_gap)
                - Fixed geometry parameters
                - Current settable parameters (coupler_gap, coupler_length)
                - Layer definitions
        """
        params = {
            # Fixed CPW parameters
            "cpw_width": self.cpw_width,
            "cpw_gap": self.cpw_gap,

            # Fixed geometry parameters
            "resonator_fillet": self.resonator_fillet,
            "coupler_length_normal": self.coupler_length_normal,

            # Layer definitions
            "cutout_layer": self.cutout_layer,
            "cad_layer": self.cad_layer,

            # Settable parameters
            "coupler_gap": self.coupler_gap,
            "coupler_length": self.coupler_length,

            # Status
            "design_generated": self.device is not None
        }

        print("Current CouplerChipDesign parameters:")
        print("Fixed parameters:")
        for key in ["cpw_width", "cpw_gap", "resonator_fillet", "coupler_length_normal"]:
            print(f"  {key}: {params[key]}")
        print("Settable parameters:")
        for key in ["coupler_gap", "coupler_length"]:
            print(f"  {key}: {params[key]}")

        return params

    def get_image(self, save_to_file=False, filename="coupler_image.png"):
        """
        Get an image of the current coupler chip design.

        Uses the stored CAD object (coupler_sonnet_sim) to generate a visual
        representation of the current design. If no design has been generated
        yet, returns an error message.

        Args:
            save_to_file (bool): Whether to save the image to a file (default: False)
            filename (str): Filename to save image to if save_to_file=True (default: "coupler_image.png")

        Returns:
            bytes: PNG image data as bytes, or raises ValueError if no design exists
        """
        if self.coupler_sonnet_sim is None:
            raise ValueError("No design has been generated yet. Call generate_cad() first.")

        try:
            # Generate image using quickplot_noshow
            quickplot_noshow(self.coupler_sonnet_sim)

            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_data = buf.read()

            # Optionally save to file
            if save_to_file:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Image saved to {filename}")

            plt.close()  # Close to free memory

            return img_data

        except Exception as e:
            raise ValueError(f"Failed to generate image: {str(e)}")

    def save_params_to_file(self, filename="coupler_params.txt"):
        """
        Save current parameters to a text file for documentation.

        Args:
            filename (str): Filename to save parameters to (default: "coupler_params.txt")
        """
        params = self.get_params()

        try:
            with open(filename, 'w') as f:
                f.write("Coupler Chip Design Parameters\n")
                f.write("==============================\n\n")

                f.write("Fixed CPW Parameters:\n")
                for key in ["cpw_width", "cpw_gap"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nFixed Geometry Parameters:\n")
                for key in ["resonator_fillet", "coupler_length_normal"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write("\nLayer Definitions:\n")
                for key in ["cutout_layer", "cad_layer"]:
                    f.write(f"  {key}: {params[key]}\n")

                f.write("\nSettable Parameters:\n")
                for key in ["coupler_gap", "coupler_length"]:
                    f.write(f"  {key}: {params[key]} um\n")

                f.write(f"\nDesign Status:\n")
                f.write(f"  design_generated: {params['design_generated']}\n")

            print(f"Parameters saved to {filename}")

        except Exception as e:
            print(f"Error saving parameters: {str(e)}")

# MCP Tool Functions
# These functions provide the Model Context Protocol interface for LLM interaction
#!/usr/bin/env python3

"""
MCP Tools for PHIDL Superconducting Resonator Design

This module provides MCP (Model Context Protocol) tools that allow
Large Language Models to interact with the ResonatorChipDesign class.
The tools enable customization of resonator dimensions, parameters,
and output filenames.

Tools provided:
1. resonator_generate_cad_tool - Generate complete resonator chip with custom filenames
2. resonator_get_params_tool - Get all current design parameters
3. resonator_get_image_tool - Get visual image of the current design
"""

@mcp.tool()
def resonator_generate_cad_tool(resonator_length: float, resonator_width: float, coupler_gap: float, coupler_length: float,
                               bare_chip_filename="bare_chip.gds",
                               full_chip_filename="full_chip.gds",
                               sonnet_chip_filename="sonnet_chip.gds",
                               image_filename="resonator_chip.png"):
    """
    MCP Tool: Generate a complete superconducting resonator chip design with custom filenames.

    This tool creates a full resonator chip including wirebond pads, feedlines,
    coupler structure, and meandered resonator. The design is optimized for
    superconducting quantum circuits and includes proper ground plane structures
    for fabrication.

    The tool generates multiple output files with customizable names:
    - Raw GDS file with all layers (bare_chip_filename)
    - Fabrication-ready GDS with boolean operations (full_chip_filename)
    - Simulation-optimized GDS file (sonnet_chip_filename)
    - PNG image of the design layout (image_filename)

    Args:
        resonator_length (float): Total electrical length of the resonator in micrometers.
                                This is the primary parameter controlling resonant frequency.
                                Typical range: 2000-15000 um for 2-10 GHz resonators.

        resonator_width (float): Physical width/footprint of the meandered resonator structure
                               in micrometers. This controls how much space the meander occupies.
                               Larger widths create wider meanders with fewer vertical segments.
                               Typical range: 200-1000 um. Should be >> coupler_length for good isolation.

        coupler_gap (float): Gap between the feedline and resonator in the coupling region
                           in micrometers. This is the primary parameter controlling coupling strength.
                           Smaller gaps create stronger coupling (higher external Q factor).
                           Typical range: 30-50 um. Start with 60 um for moderate coupling.

        coupler_length (float): Length of the parallel coupling region in micrometers.
                              Longer coupling regions create stronger coupling. Works together
                              with coupler_gap to set coupling strength. Typical range: 50-200 um.
                              Use longer lengths for weaker gap coupling or vice versa.

        bare_chip_filename (str): Filename for the raw GDS export containing all layers
                                 (default: "bare_chip.gds"). This file contains the complete
                                 design hierarchy and is useful for design verification.

        full_chip_filename (str): Filename for the fabrication-ready GDS file
                                 (default: "full_chip.gds"). This file has proper boolean
                                 operations applied and is ready for mask-making.

        sonnet_chip_filename (str): Filename for the simulation-optimized GDS file
                                   (default: "sonnet_chip.gds"). This file has appropriate
                                   boundaries for electromagnetic simulation.

        image_filename (str): Filename for the design layout image
                             (default: "resonator_chip.png"). High-resolution PNG image
                             showing the complete chip layout.

    Returns:
        dict: Comprehensive status information including:
            - success (bool): Whether the generation completed successfully
            - files_created (list): Names of all files that were created
            - remaining_resonator_length (float): Any resonator length that couldn't be accommodated
            - chip_dimensions (dict): Physical dimensions of the generated chip
            - message (str): Human-readable summary of the operation
    """
    designer = initialize_resonator_designer()

    logger.info("Starting resonator chip generation")

    try:
        result = designer.generate_cad(
            resonator_length=float(resonator_length),
            resonator_width=float(resonator_width),
            coupler_gap=float(coupler_gap),
            coupler_length=float(coupler_length),
            bare_chip_filename=bare_chip_filename,
            full_chip_filename=full_chip_filename,
            sonnet_chip_filename=sonnet_chip_filename,
            image_filename=image_filename
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate resonator chip: {str(e)}",
            "message": "Resonator chip generation failed",
            "files_created": []
        }


@mcp.tool()
def resonator_get_params_tool():
    """
    MCP Tool: Get all current parameters of the resonator chip design.

    This tool returns a comprehensive view of all design parameters, including
    both fixed parameters (that cannot be changed, like CPW dimensions) and
    settable parameters (that can be modified via the generate_cad tool).

    Fixed parameters include:
    - CPW geometry (width, gap)
    - Wirebond pad dimensions and tapers
    - Resonator meander spacing and corner radius
    - Layer definitions for fabrication

    Settable parameters include:
    - Resonator electrical length (affects frequency)
    - Resonator physical width (affects footprint)
    - Coupler gap and length (affect coupling strength)

    This is useful for:
    - Inspecting the current design state
    - Understanding what parameters are available for modification
    - Checking default values before making changes
    - Debugging design issues
    - Documenting design specifications

    Returns:
        dict: Complete parameter set including:
            - success (bool): Whether parameter retrieval was successful
            - parameters (dict): All design parameters with descriptions:
                * Fixed CPW parameters: cpw_width, cpw_gap (micrometers)
                * Fixed wirebond parameters: pad dimensions, taper length, gaps
                * Fixed resonator geometry: meander spacing, fillet radius
                * Fixed coupler parameters: normal section lengths
                * Layer definitions: cutout_layer, cad_layer numbers
                * Settable parameters: resonator_length, resonator_width, coupler_gap, coupler_length
                * design_generated: Boolean indicating if a design exists
            - message (str): Human-readable summary
    """
    designer = initialize_resonator_designer()

    try:
        params = designer.get_params()
        return {
            "success": True,
            "parameters": params,
            "message": "Retrieved resonator chip parameters successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get parameters: {str(e)}",
            "message": "Parameter retrieval failed",
            "parameters": {}
        }


@mcp.tool()
def resonator_get_image_tool():
    """
    MCP Tool: Get a visual image of the current resonator chip design.

    This tool generates a high-quality image of the current resonator chip layout
    using PHIDL's visualization capabilities. The image shows the complete chip
    including wirebond pads, feedlines, coupler, and meandered resonator structure
    with proper colors and scaling.

    The generated image includes:
    - Wirebond pads (left and right) with tapered transitions
    - Straight CPW feedline connecting the pads
    - Coupler structure showing the coupling region
    - Meandered resonator with visible turns and spacing
    - Proper ground plane cutouts (shown as negative space)

    The image is useful for:
    - Visual verification of the design layout and geometry
    - Checking meander structure, spacing, and turn radius
    - Verifying coupler positioning and gap dimensions
    - Creating documentation for papers or presentations
    - Debugging layout issues or unexpected geometry
    - Estimating coupling field distributions qualitatively

    Note: This tool requires that a design has been generated first using the
    resonator_generate_cad_tool. If no design exists, it will return an error.
    The image is generated in memory and returned as an Image object suitable
    for display in MCP-compatible interfaces.

    Returns:
        Image: High-resolution PNG image (typically 150 DPI) of the resonator
               chip layout with proper aspect ratio and clear details, or
               raises ValueError if no design has been generated yet.

    Raises:
        ValueError: If no design has been generated yet or if image generation fails
    """
    designer = initialize_resonator_designer()

    try:
        # Get image data as bytes
        img_data = designer.get_image(save_to_file=False)

        # Return Image object for MCP
        return Image(data=img_data, format="png")

    except Exception as e:
        raise ValueError(f"Failed to get resonator image: {str(e)}")


@mcp.tool()
def resonator_save_params_tool(filename="resonator_params.txt"):
    """
    MCP Tool: Save current resonator design parameters to a text file.

    This tool creates a human-readable documentation file containing all
    current design parameters, both fixed and settable. The file is formatted
    for easy reading and can be used for design documentation, version control,
    or parameter tracking across different design iterations.

    The saved file includes:
    - Header with design type and date
    - Fixed CPW parameters (width, gap)
    - Fixed wirebond parameters (pad dimensions, taper length)
    - Fixed resonator geometry (meander spacing, corner radius)
    - Fixed coupler parameters (normal section length)
    - Layer definitions for GDS export
    - Current settable parameters (length, width, coupling)
    - Design generation status

    This is useful for:
    - Creating design documentation and records
    - Tracking parameter changes across design iterations
    - Sharing design specifications with collaborators
    - Reproducing designs from saved parameters
    - Design version control and history tracking

    Args:
        filename (str): Name of the text file to create (default: "resonator_params.txt").
                       The file will be created in the current working directory.
                       Include the .txt extension for text files, or use other
                       extensions as needed (.log, .config, etc.).

    Returns:
        dict: Status information including:
            - success (bool): Whether the file was saved successfully
            - filename (str): Name of the file that was created
            - message (str): Human-readable summary of the operation
            - parameters_saved (int): Number of parameters written to file
    """
    designer = initialize_resonator_designer()

    try:
        # Save parameters to file
        designer.save_params_to_file(filename)

        # Get parameter count for reporting
        params = designer.get_params()
        param_count = len(params)

        return {
            "success": True,
            "filename": filename,
            "message": f"Successfully saved {param_count} parameters to {filename}",
            "parameters_saved": param_count
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save parameters: {str(e)}",
            "message": f"Could not save parameters to {filename}",
            "filename": filename,
            "parameters_saved": 0
        }


@mcp.tool()
def resonator_save_image_tool(filename="resonator_design.png"):
    """
    MCP Tool: Save the current resonator design image to a file.

    This tool generates and saves a high-quality image of the current resonator
    chip layout to a specified file. Unlike the get_image tool which returns
    the image data for display, this tool saves the image to disk for documentation,
    presentations, or archival purposes.

    The saved image features:
    - High resolution (150 DPI) suitable for publications
    - Proper aspect ratio maintaining design proportions
    - Clear visualization of all chip components
    - Tight bounding box removing excess whitespace
    - Standard PNG format for broad compatibility

    This is useful for:
    - Creating figures for research papers or reports
    - Generating design documentation with visual references
    - Archiving design layouts for future reference
    - Creating presentation slides with chip images
    - Sharing design visuals with collaborators

    Args:
        filename (str): Name of the image file to create (default: "resonator_design.png").
                       Include appropriate extension (.png, .jpg, .pdf, .svg).
                       PNG is recommended for crisp technical drawings.

    Returns:
        dict: Status information including:
            - success (bool): Whether the image was saved successfully
            - filename (str): Name of the file that was created
            - message (str): Human-readable summary of the operation
            - image_size (tuple): Width and height of saved image in pixels (if available)
    """
    designer = initialize_resonator_designer()

    try:
        # Generate and save image to file
        img_data = designer.get_image(save_to_file=True, filename=filename)

        return {
            "success": True,
            "filename": filename,
            "message": f"Successfully saved resonator design image to {filename}",
            "image_size": "150 DPI, auto-sized"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save image: {str(e)}",
            "message": f"Could not save image to {filename}",
            "filename": filename,
            "image_size": None
        }

@mcp.tool()
def coupler_generate_cad_tool(coupler_gap: float, coupler_length: float,
                             coupler_chip_filename="coupler_chip.gds",
                             sonnet_coupler_filename="sonnet_coupler.gds",
                             image_filename="coupler_chip.png"):
    """
    MCP Tool: Generate a standalone superconducting coupler chip design with custom filenames.

    This tool creates an isolated coupler structure suitable for S-parameter
    measurements and electromagnetic simulation. The design includes the coupler
    with extended feedlines and proper ground plane boundaries, but without the
    full resonator structure.

    This standalone coupler is useful for:
    - Characterizing coupling strength independently of the resonator
    - S31 parameter simulations (transmission through the coupler)
    - Optimizing coupler design before integration with full resonator chip
    - Understanding coupling physics and field distributions
    - Validating electromagnetic simulation models

    The tool generates output files with customizable names:
    - Raw coupler chip GDS file (coupler_chip_filename)
    - Simulation-ready GDS with boolean operations (sonnet_coupler_filename)
    - PNG image of the coupler layout (image_filename)

    Args:
        coupler_gap (float): Gap between the two transmission lines in the coupling
                           region in micrometers. This is the primary parameter controlling
                           coupling strength. Smaller gaps create stronger electromagnetic
                           coupling due to increased capacitive and inductive coupling.
                           Physics: Coupling strength ∝ 1/gap for small gaps.
                           Typical range: 20-100 um. Start with 50 um for moderate coupling.

        coupler_length (float): Length of the parallel coupling region in micrometers.
                              Longer coupling regions create stronger coupling by increasing
                              the interaction length between the two transmission lines.
                              Physics: Coupling coefficient increases with length until
                              saturation (typically >λ/4). Typical range: 50-200 um.
                              Use in combination with gap to achieve desired coupling strength.

        coupler_chip_filename (str): Filename for the raw coupler chip GDS file
                                   (default: "coupler_chip.gds"). This file contains
                                   the complete coupler design hierarchy including
                                   the coupler structure, extended feedlines, and
                                   ground plane. Useful for design verification.

        sonnet_coupler_filename (str): Filename for the simulation-ready GDS file
                                     (default: "sonnet_coupler.gds"). This file has
                                     proper boolean operations applied and optimized
                                     boundaries for electromagnetic simulation in
                                     tools like Sonnet or HFSS.

        image_filename (str): Filename for the coupler layout image
                            (default: "coupler_chip.png"). High-resolution PNG image
                            showing the coupler structure, feedlines, and ground plane
                            boundaries for documentation and visualization.

    Returns:
        dict: Comprehensive status information including:
            - success (bool): Whether the generation completed successfully
            - files_created (list): Names of all files that were created
            - coupler_dimensions (dict): Physical dimensions of the generated coupler
                * width_um: Total chip width in micrometers
                * height_um: Total chip height in micrometers
                * feedline_length_um: Length of the feedline including extensions
            - message (str): Human-readable summary of the operation
    """
    designer = initialize_coupler_designer()

    try:
        result = designer.generate_cad(
            coupler_gap=float(coupler_gap),
            coupler_length=float(coupler_length),
            coupler_chip_filename=coupler_chip_filename,
            sonnet_coupler_filename=sonnet_coupler_filename,
            image_filename=image_filename
        )
        return result
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate coupler chip: {str(e)}",
            "message": "Coupler chip generation failed",
            "files_created": []
        }


@mcp.tool()
def coupler_get_params_tool():
    """
    MCP Tool: Get all current parameters of the coupler chip design.

    This tool returns all design parameters for the standalone coupler chip,
    including both fixed parameters (that cannot be changed) and settable
    parameters (that can be modified via the generate_cad tool).

    Fixed parameters include:
    - CPW geometry (center conductor width, gap to ground)
    - Corner radius for smooth bends (resonator_fillet)
    - Normal section length extending from coupling region
    - Layer definitions for GDS export (metal and cutout layers)

    Settable parameters include:
    - Coupler gap (primary coupling strength control)
    - Coupler length (secondary coupling strength control)

    This is useful for:
    - Inspecting the current coupler design state
    - Understanding available parameters for optimization
    - Checking parameter values before making modifications
    - Comparing with resonator chip parameters
    - Documenting design specifications for publications

    Returns:
        dict: Complete parameter set including:
            - success (bool): Whether parameter retrieval was successful
            - parameters (dict): All design parameters:
                * Fixed CPW parameters: cpw_width, cpw_gap (micrometers)
                * Fixed geometry parameters: resonator_fillet, coupler_length_normal
                * Layer definitions: cutout_layer, cad_layer numbers
                * Settable parameters: coupler_gap, coupler_length (micrometers)
                * design_generated: Boolean indicating if a design exists
            - message (str): Human-readable summary
    """
    designer = initialize_coupler_designer()

    try:
        params = designer.get_params()
        return {
            "success": True,
            "parameters": params,
            "message": "Retrieved coupler chip parameters successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to get parameters: {str(e)}",
            "message": "Parameter retrieval failed",
            "parameters": {}
        }


@mcp.tool()
def coupler_get_image_tool():
    """
    MCP Tool: Get a visual image of the current coupler chip design.

    This tool generates a high-quality image of the current standalone coupler
    layout using PHIDL's visualization capabilities. The image shows the coupler
    structure with its feedlines and ground plane boundaries in clear detail.

    The generated image includes:
    - Primary feedline (horizontal CPW transmission line)
    - Secondary coupling line (L-shaped CPW connecting to coupling region)
    - Coupling region showing the gap between transmission lines
    - Extended feedlines for proper S-parameter port definition
    - Ground plane cutouts (shown as negative space)
    - Ground plane boundaries for simulation setup

    Visual elements typically show:
    - Metal structures (conductor): Usually rendered in gold/yellow colors
    - Ground plane cutouts: Darker regions where metal is removed
    - Coupling gap: Visible separation between the two transmission lines
    - Corner radius: Smooth transitions at bends (resonator_fillet parameter)

    The image is useful for:
    - Visual verification of the coupler geometry and dimensions
    - Checking coupling gap and length dimensions visually
    - Verifying feedline connections and ground plane cutouts
    - Creating documentation for S-parameter measurements
    - Understanding field concentration regions in the coupler
    - Preparing figures for research papers or presentations
    - Debugging layout issues or unexpected geometry

    Note: This tool requires that a coupler design has been generated first using
    the coupler_generate_cad_tool. If no design exists, it will return an error.
    The image is generated in memory and returned as an Image object suitable
    for display in MCP-compatible interfaces.

    Returns:
        Image: High-resolution PNG image (150 DPI) of the coupler chip layout
               with proper aspect ratio and clear structural details, or
               raises ValueError if no design has been generated yet.

    Raises:
        ValueError: If no design has been generated yet or if image generation fails
    """
    designer = initialize_coupler_designer()

    try:
        # Get image data as bytes
        img_data = designer.get_image(save_to_file=False)

        # Return Image object for MCP
        return Image(data=img_data, format="png")

    except Exception as e:
        raise ValueError(f"Failed to get coupler image: {str(e)}")


@mcp.tool()
def coupler_save_params_tool(filename="coupler_params.txt"):
    """
    MCP Tool: Save current coupler design parameters to a text file.

    This tool creates a human-readable documentation file containing all
    current design parameters, both fixed and settable. The file is formatted
    for easy reading and can be used for design documentation, version control,
    or parameter tracking across different coupler design iterations.

    The saved file includes:
    - Header identifying this as a coupler design
    - Fixed CPW parameters (center conductor width, gap to ground)
    - Fixed geometry parameters (corner radius, normal section length)
    - Layer definitions for GDS fabrication files
    - Current settable parameters (gap and length)
    - Design generation status

    This is particularly useful for:
    - Creating design documentation and parameter records
    - Tracking parameter changes across coupler optimization iterations
    - Sharing coupler specifications with simulation engineers
    - Reproducing specific coupler designs from saved parameters
    - Design version control and optimization history tracking
    - Comparing parameters between different coupler variants

    Args:
        filename (str): Name of the text file to create (default: "coupler_params.txt").
                       The file will be created in the current working directory.
                       Include the .txt extension for text files, or use other
                       extensions as needed (.log, .config, .dat, etc.).

    Returns:
        dict: Status information including:
            - success (bool): Whether the file was saved successfully
            - filename (str): Name of the file that was created
            - message (str): Human-readable summary of the operation
            - parameters_saved (int): Number of parameters written to file
    """
    designer = initialize_coupler_designer()

    try:
        # Save parameters to file
        designer.save_params_to_file(filename)

        # Get parameter count for reporting
        params = designer.get_params()
        param_count = len(params)

        return {
            "success": True,
            "filename": filename,
            "message": f"Successfully saved {param_count} coupler parameters to {filename}",
            "parameters_saved": param_count
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save parameters: {str(e)}",
            "message": f"Could not save coupler parameters to {filename}",
            "filename": filename,
            "parameters_saved": 0
        }


@mcp.tool()
def coupler_save_image_tool(filename="coupler_design.png"):
    """
    MCP Tool: Save the current coupler design image to a file.

    This tool generates and saves a high-quality image of the current coupler
    chip layout to a specified file. Unlike the get_image tool which returns
    the image data for display, this tool saves the image to disk for documentation,
    S-parameter measurement setup, or archival purposes.

    The saved image features:
    - High resolution (150 DPI) suitable for technical publications
    - Proper aspect ratio maintaining correct geometric proportions
    - Clear visualization of coupling region and gap dimensions
    - Tight bounding box removing excess whitespace
    - Standard PNG format for broad compatibility
    - Professional quality suitable for research documentation

    This is particularly useful for:
    - Creating figures for research papers on coupling analysis
    - Generating documentation for S-parameter measurement procedures
    - Archiving coupler layouts for future reference and comparison
    - Creating presentation slides with coupler geometry illustrations
    - Sharing design visuals with measurement technicians
    - Documenting coupling strength optimization studies

    Args:
        filename (str): Name of the image file to create (default: "coupler_design.png").
                       Include appropriate extension (.png, .jpg, .pdf, .svg).
                       PNG is recommended for crisp technical drawings with
                       fine geometric details.

    Returns:
        dict: Status information including:
            - success (bool): Whether the image was saved successfully
            - filename (str): Name of the file that was created
            - message (str): Human-readable summary of the operation
            - image_size (str): Description of image resolution and sizing
    """
    designer = initialize_coupler_designer()

    try:
        # Generate and save image to file
        img_data = designer.get_image(save_to_file=True, filename=filename)

        return {
            "success": True,
            "filename": filename,
            "message": f"Successfully saved coupler design image to {filename}",
            "image_size": "150 DPI, auto-sized"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save image: {str(e)}",
            "message": f"Could not save coupler image to {filename}",
            "filename": filename,
            "image_size": None
        }

if __name__ == "__main__":
    mcp.run('stdio')
