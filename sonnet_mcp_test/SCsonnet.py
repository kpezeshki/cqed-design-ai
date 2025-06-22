import numpy as np
import pysonnet as ps

class SCSonnetProject(ps.GeometryProject):
    """
    A specialized Sonnet project for superconducting microstrip designs.
    Expects a config dict with:
      - box_x: float, box dimension in x (mils or after unit set)
      - box_y: float, box dimension in y
      - ls: float, sheet inductance for the superconductor (Ohms/sq)
      - metal_name: str, name of the superconductor material
      - layer: int (optional), GDS layer index (default 0)
      - datatype: int (optional), GDS datatype index (default 0)
    """
    def __init__(self, config: dict):
        # Extract required parameters
        box_x = config.get('box_x')
        box_y = config.get('box_y')
        ls = config.get('ls')
        metal_name = config.get('metal_name')
        # Optional parameters
        layer = config.get('layer', 0)
        datatype = config.get('datatype', 0)

        # Initialize base GeometryProject
        super().__init__()

        # Set units to microns
        self.set_units(length='um')

        # Setup simulation box (dimensions in microns)
        # Use some reasonable default for z-size and boundaries
        z_box = config.get('z_box', 2000)
        xy_margin = config.get('xy_margin', 2000)
        self.setup_box(box_x, box_y, xy_margin, xy_margin)

        # Define superconducting metal
        # ls is the sheet inductance
        self.define_metal("general", metal_name, ls=ls)

        # Cover the top of the box with free space
        self.set_box_cover("free space", top=True)

        # Add dielectric layers: air under and silicon
        self.add_dielectric("air", layer, thickness=config.get('air_thickness', 1000))
        self.add_dielectric(
            "silicon", layer + 1,
            thickness=config.get('si_thickness', 100),
            epsilon=config.get('si_epsilon', 11.7),
            dielectric_loss=config.get('si_loss', 0.0),
            conductivity=config.get('si_conductivity', 0.0)
        )

        # Define the technology layer for metal polygons
        self.define_technology_layer(
            "metal", "Superconductor", layer,
            metal_name, fill_type="staircase",
            x_min=config.get('tech_x_min', 1),
            x_max=config.get('tech_x_max', 100),
            y_min=config.get('tech_y_min', 1),
            y_max=config.get('tech_y_max', 100)
        )

        # Enable current density, Q accuracy, and resonance detection
        self.set_options(
            current_density=True,
            q_accuracy=True,
            resonance_detection=True
        )