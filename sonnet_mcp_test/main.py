from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
#from cpw_lib import *
import gdstk
import logging
import numpy as np
import pysonnet as ps
import SCsonnet
import yaml
import threading

#from translation import fig_to_base64_png
import copy


# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()
sonnet_file_path = r"C:/Users/slab/Desktop/Yueheng_Sonnet/sonnet_mcp_test/cpw_mcp_test.son"
path_to_sonnet = r"C:/Program Files/Sonnet Software/18.56/"
sc_sonnet_cfg_path = r"C:/Users/slab/Desktop/Yueheng_Sonnet/sonnet_mcp_test/sc_sonnet_config.yaml"
gds_file_path = r"C:/Users/slab/Desktop/Yueheng_Sonnet/sonnet_mcp_test/cpw_demo.gds"
with open(sc_sonnet_cfg_path, 'r') as file:
    sc_soonet_cfg = yaml.safe_load(file)

# Here's where you define your tools (functions the AI can use)
@mcp.tool()
def sc_sonnet_init(key_args: dict) -> TextContent:
    """
    A specialized Sonnet project for superconducting microstrip designs.
    Args:
        key_args (dict): Dictionary containing configuration parameters for the Sonnet project.
        param (str): The parameter to update.
            - box_x: float, box dimension in x (mils or after unit set)
            - box_y: float, box dimension in y
            - ls: float, sheet inductance for the superconductor (Ohms/sq)
            - metal_name: str, name of the superconductor material
            - layer: int (optional), GDS layer index (default 0)
            - datatype: int (optional), GDS datatype index (default 0)
                value (float): The new value for the parameter
            - gds_file_path (str): Path to the GDS file containing the design.
            - frequency_sweep_type (str): Type of frequency sweep, e.g., "abs" means adaptive sweep."linear" means linear sweep.
            - start_frequency (float): Start frequency for the sweep in GHz.
            - stop_frequency (float): Stop frequency for the sweep in GHz.

      Returns: 
        sc_sonnet_project (SCsonnet.SCSonnetProject): The initialized Sonnet project instance.
        TextContent: Confirmation message with the project initialization status.
    """
    # load params
    try:
        params = copy.copy(sc_soonet_cfg)  # Load default configuration
        for key, value in key_args.items():
            if key not in params:
                pass
            params[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error loading configuration: {e}")
    
    # Initialize the Sonnet project with the provided configuration
    try:
        sc_sonnet_project = SCsonnet.SCSonnetProject(config=params)
    except Exception as e:
        return TextContent(type="text", text=f"Error initializing SC Sonnet settings: {e}")

    try:
        lib = gdstk.read_gds(params["gds_file_path"])
        cell = lib.cells[0]  # Assuming the first cell is the one you want to use
        sc_sonnet_project.add_gdstk_cell(polygon_type='metal',cell=cell,tech_layer='Superconductor')
        #return TextContent(type="text", text=f"GDS cell added to Sonnet project successfully with parameters: {params}")
    except Exception as e:
        return TextContent(type="text", text=f"Error adding GDS cell to Sonnet project: {e}")
        # Initialize the Sonnet project with the provided configuration
    
    # Add frequency sweep settings to the Sonnet project
    try:
        if params["frequency_sweep_type"] == "abs":
            sc_sonnet_project.add_frequency_sweep(params["frequency_sweep_type"],f1=params["start_frequency"],f2=params["stop_frequency"])
            sc_sonnet_project.set_analysis("frequency sweep")
        elif params["frequency_sweep_type"] == "linear":
            sc_sonnet_project.add_frequency_sweep(params["frequency_sweep_type"],f1=params["start_frequency"],f2=params["stop_frequency"], num_points=params["num_points"])
            sc_sonnet_project.set_analysis("frequency sweep")
    except Exception as e:
        return TextContent(type="text", text=f"Error adding frequency sweep: {e}")
    
    # Add ports to the Sonnet project
    try:
        for i, port in enumerate(params["ports"]):
            sc_sonnet_project.add_port("standard",i+1,port["x"],port["y"], resistance =port["resistance"])
    except Exception as e:
        return TextContent(type="text", text=f"Error adding ports to Sonnet project: {e}")
    
    proj_dict['sc_proj'] = sc_sonnet_project

    # Add a touchstone file depending on the number of ports
    try:
        port_count = len(params["ports"])
        sc_sonnet_project.add_syz_parameter_file(
            file_type = 'touchstone',
            file_name = f"sc_sonnet_output.s{port_count}p",
        )
    except Exception as e:
        return TextContent(type="text", text=f"Error adding SYZ parameter file: {e}")
    
    # Create the .son Sonnet file
    try:
        sc_sonnet_project.make_sonnet_file(params["sonnet_file_path"])
        msg = TextContent(type="text", text=f"Superconducting Sonnet project initialized successfully with parameters: {params}")
        return sc_sonnet_project, msg
    except Exception as e:
        return TextContent(type="text", text=f"Error writing Sonnet file: {e}")

#sc_sonnet_project.run
      


@mcp.tool()
def update_sc_sonnet_config(cfg_path: str, param: str, value: float) -> TextContent:
    """
    Update a specific parameter in the superconducting Sonnet project configuration.
    
    Args:
        cfg_path (str): Path to the YAML configuration file.
        param (str): The parameter to update.
            - box_x: float, box dimension in x (mils or after unit set)
            - box_y: float, box dimension in y
            - ls: float, sheet inductance for the superconductor (Ohms/sq)
            - metal_name: str, name of the superconductor material
            - layer: int (optional), GDS layer index (default 0)
            - datatype: int (optional), GDS datatype index (default 0)
                value (float): The new value for the parameter.
        
    Returns:
        TextContent: Confirmation message with the updated parameter.
    """
    try:
        with open(cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)
        
        if param in cfg:
            cfg[param] = value
            with open(cfg_path, 'w') as file:
                yaml.safe_dump(cfg, file)
            return TextContent(type="text", text=f"Successfully updated {param} to {value}")
        else:
            return TextContent(type="text", text=f"Parameter {param} not found in configuration.")
    except Exception as e:
        return TextContent(type="text", text=f"Error updating configuration: {e}")

@mcp.tool()
def run_sonnet_simulation() -> TextContent:
    """
    Run the Sonnet simulation for the superconducting Sonnet project.
    """
    #dummy test to check the Sonnet Software path
    try:
        proj_dict['sc_proj'].locate_sonnet(path_to_sonnet)
        msg = TextContent(type="text", text="Sonnet software located successfully.")
    except Exception as e:
        return TextContent(type="text", text=f"Error locating Sonnet software: {e}")

    # Submit the Sonnet job
    try:
        msg.text += " Frequency sweep job submitted successfully."
        thread = threading.Thread(target=submit_sonnet_job, 
                                  args=(proj_dict['sc_proj'],),daemon=True)
        thread.start()
    except Exception as e:
        return TextContent(type="text", text=f"Error submitting Sonnet simulation: {e}")
    return msg
 
proj_dict  = {}

def submit_sonnet_job(sc_proj):
    try:
        sc_proj.run("frequency sweep")
    except Exception as e:
        return TextContent(type="text", text=f"Error running Sonnet simulation: {e}")

def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()