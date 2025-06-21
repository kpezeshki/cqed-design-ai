from mcp.server.fastmcp import FastMCP, Image
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging

# Import the package
import slab_qick_calib

# Import specific modules
from slab_qick_calib import calib
from slab_qick_calib import exp_handling
from slab_qick_calib import experiments
from slab_qick_calib import gen

# Import specific functions or classes
from slab_qick_calib.calib import qubit_tuning
from slab_qick_calib.experiments.single_qubit import resonator_spectroscopy
from slab_qick_calib.gen import qick_experiment

from translation import fig_to_base64_png
from expt_params import rspec_params_coarse
import copy

import io
import sys
from contextlib import redirect_stdout


# Set up logging (this just prints messages to your terminal for debugging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the MCP server object
mcp = FastMCP()

expt_path = r'C:\Users\slab\Desktop\MCP_server_test\_Data\QB-CB7'
cfg_file='QB-CB7.yml'
ip = '10.108.30.23'
max_t1 = 300

import os
import sys 
sys.path.append('../')
import slab_qick_calib.config as config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qick import QickConfig
import sys 
sys.path.append('../')
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.calib import qubit_tuning, tuneup
from slab_qick_calib import handy

# from tqdm.notebook import tqdm


cfg_path = r'C:\Users\slab\Desktop\MCP_server_test\config\QB-CB7.yml'
auto_cfg = config.load(cfg_path)


# Connect to instruments

im = InstrumentManager(ns_address=ip)
# print(im)
soc = QickConfig(im[auto_cfg['aliases']['soc']].get_cfg())
# print(soc)

cfg_dict = {'soc': soc, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}
# Here's where you define your tools (functions the AI can use)
@mcp.tool()
def qick_config() -> TextContent:
    """
    Return:
        The QickConfig string printout. The string contains the specific qick channel parameters. 
        The most important is the 
        fs printout, which is the frequency of the qick clock in MHz. 
        Can totally just display the raw output. 
        The user might
        ask about specific ADC or DAC channels, which is the correspondance between 
        the x_xxx (all numbers) and the realted channel number. """ 
    return TextContent(type="text", text=str(soc))

@mcp.tool()
def update_readout_config(param:str, value:float, qi:int) -> TextContent:
    """
    Update one config parameter of readout to one given value. For the readout, the parameters are:
    active_reset, chi, fidelity, final_delay, frequency, gain, kappa, lamb, max_gain, phase, qe, qi, 
    readout_length, reps, reps_base, reset_e, reset_g, sigma, soft_avgs, soft_avgs_base, threshold, tm, trig_offset.
    
    ONLY UPDATE THE CONFIG FILE IF YOU RUN THE EXPERIMENT BEFORE AND HAVE OPTIMIZED THE PARAMETERS AND THE FIT IS GOOD.
    
    """
    try:
        auto_cfg = config.update_readout(cfg_path, param, value, qi)
        return TextContent(type="text", text=f"Successfully updated {param} to {value}")
    except Exception as e:
        return TextContent(type="text", text=f"Error: {e}")

@mcp.tool()
def update_qubit_config(param:str, value:float, qi:int) -> TextContent:
    """
    Update one config parameter of qubit to one given value. For the qubit, the parameters are:
    T1, T2e, T2r, f_ef, f_ge, kappa, low_gain, pop, pulses, pi_ef, pi_ge, spec_gain, temp, tuned_up.
    
    ONLY UPDATE THE CONFIG FILE IF YOU RUN THE EXPERIMENT BEFORE AND HAVE OPTIMIZED THE PARAMETERS AND THE FIT IS GOOD.
    """
    try:
        auto_cfg = config.update_qubit(cfg_path, param, value, qi)
        return TextContent(type="text", text=f"Successfully updated {param} to {value}")
    except Exception as e:
        return TextContent(type="text", text=f"Error: {e}")
    
@mcp.tool()
def time_of_flight():
    """Checks the connectivity of the circuit by running a time of flight experiment. Successful connection
    should show a clear step in the plot (blue and orange lines). If there are no step, either the connection port is wrong or the 
    relax delay is incorrectly set. Also show the plot when running this function. If the plot shows oscillation that agerages to zero, 
    without a clear step, then it is not connected. """
    try:
        try:
            tof_exp = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
            
            figure = tof_exp.display(save_fig=False, return_fig=True)
        except Exception as e:
            return TextContent(type="text", text=f"Error: {e}")
            # image_base64 = "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"

            
        return ImageContent(data=figure, mimeType="image/png", type="image")
        
    except Exception as e:
        return TextContent(type="text", text=f"Error: {e}")

# The return format should be one of the types defined in mcp.types. The commonly used ones include TextContent, ImageContent, BlobResourceContents.
# In the case of a string, you can also directly use `return str(a + b)` which is equivalent to `return TextContent(type="text", text=str(a + b))`

@mcp.tool()
def resonator_spectroscopy(key_args: dict):
    """Runs a resonator spectroscopy experiment to find the resonant frequency of the readout resonator.
    In the beginning of the expriment, unless specified, we don't know what the resonant frequency is.
    So we need to sweep the frequency and find the resonant frequency.
    The frequency sweep is done by changing the frequency of the readout pulse.
    
    Move on to the fine resonator spectroscopy when your resonator kappa is 1/2 of the entire span and the fit looks good. 
    Kappa is how wide the resonator is in frequency.     
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            gain (float): Readout pulse amplitude, can only be at maximum 1.0, minimum 0.001
            reps (int): Number of repetitions
            span (float): Frequency span in MHz
            expts (int): Number of frequency points
            start (float): Start frequency in MHz
            style (str): 'coarse' for wide frequency scan
            soft_avgs (int): Number of software averages
        
    Returns:
        list: List containing:
            - ImageContent: Plot showing resonator response vs frequency with fit
            - TextContent: List of identified peak frequencies if peaks are found
    """
    try:
        param = copy.copy(rspec_params_coarse)
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
    try:
        # Create and run the resonator spectroscopy experiment
        rspecc = meas.ResSpec(
            cfg_dict=cfg_dict, 
            qi=param['qi'], 
            style=param['style'], 
            params={
                'start': param['start'],
                'span': param['span'],
                'soft_avgs': param['soft_avgs'],
                'reps': param['reps'],
                'gain': param['gain'],
                'expts': param['expts']
            }
        )
        
        # Get the figure from the experiment
        data = rspecc.analyze()
        figure = rspecc.display(save_fig=False, return_fig=True, fit = param['fit'], plot_res = False)
        
        # Get peak frequencies if available
        peak_text = ""
        if 'coarse_peaks' in data:
            peaks = data['coarse_peaks']
            peak_text = f"Found {len(peaks)} peaks at frequencies:\n"
            for i, peak in enumerate(peaks):
                peak_text += f"Peak {i+1}: {peak:.2f} MHz\n"
        
        # Return both the figure and peak information
        return [
            ImageContent(data=figure, mimeType="image/png", type="image"),
            TextContent(type="text", text=peak_text)
        ]
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")


# @mcp.tool()
# def resonator_fine_spectroscopy(key_args: dict):
#     """Runs a fine tuned resonator spectroscopy experiment to further tune the resonant frequency of the readout resonator.
#     This experiment is used to fine tune the resonant frequency of the readout resonator.
#     The frequency sweep is done by changing the frequency of the readout pulse.
    
#     WARNING: ONLY UPDATE THE CONFIG FILE IF YOU RUN THE EXPERIMENT BEFORE AND HAVE OPTIMIZED THE PARAMETERS AND THE FIT IS GOOD.
    
#     Args:
#         key_args (dict): Dictionary containing experiment parameters:
#             qi (int): Qubit index to measure
#             fit (bool): Whether to fit the data
#             gain (float): Readout pulse amplitude, can only be at maximum 1.0, minimum 0.001
#             reps (int): Number of repetitions
#             span (float): Frequency span in MHz
#             expts (int): Number of frequency points
#             start (float): Start frequency in MHz
#             style (str): 'coarse' for wide frequency scan
#             soft_avgs (int): Number of software averages
#             update (bool): Whether to update the config file
        
#     Returns:
#         list: List containing:
#             - ImageContent: Plot showing resonator response vs frequency with fit
#             - TextContent: List of identified peak frequencies if peaks are found
#     """
#     try:
#         param = copy.copy(rspec_params_coarse)
#         for key, value in key_args.items():
#             if key not in param:
#                 pass
#             param[key] = value
#     except Exception as e:
#         return TextContent(type="text", text=f"Error 1st step: {e}")
#     try:
#         # Create and run the resonator spectroscopy experiment
#         rspecc = meas.ResSpec(
#             cfg_dict=cfg_dict, 
#             qi=param['qi'], 
#             style=param['style'], 
#             params={
#                 'start': param['start'],
#                 'span': param['span'],
#                 'soft_avgs': param['soft_avgs'],
#                 'reps': param['reps'],
#                 'gain': param['gain'],
#                 'expts': param['expts']
#             }
#         )
        
#         # Get the figure from the experiment
#         data = rspecc.analyze()
#         figure = rspecc.display(save_fig=False, return_fig=True, fit = param['fit'], plot_res = False)
        
#         if param['update']:
#             # Capture the output from the update function
           
#             # Redirect stdout to capture the printed text
#             f = io.StringIO()
#             with redirect_stdout(f):
#                 rspecc.update(cfg_dict['cfg_file'])
#             update_output = f.getvalue()
#         else:
#             update_output = "No update"
#         # Return both the figure and peak information
#         return [
#             ImageContent(data=figure, mimeType="image/png", type="image"),
#             TextContent(type="text", text=update_output)
#         ]
        
#     except Exception as e:
#         return TextContent(type="text", text=f"Error in second: {e}")


@mcp.tool()
def resonator_power_spectroscopy(key_args: dict):
    """Runs a resonator power spectroscopy experiment to measure how the resonator response changes with power.
    Because the resonator is coupled to the qubit, the frequency of the resonator will 
    change once hitting a critical value of gain when increasing the gain. That gain value is very important for 
    setting the qubit measurement gain. Future qubit pulse probe or spectroscopy experiments 
    will be performed below this gain value. Also the frequency shift of the resonator is 
    important for noting the qubit-resonator coupling strength. This frequency shift is called
    the Lamb shift.
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            rng (int): Range for the gain sweep, going from max_gain to max_gain/rng
            max_gain (float): Maximum gain value, value need to be smaller than 1.0
            expts_gain (int): Number of gain points
            span (float): Frequency span in MHz
            f_off (float): Frequency offset from resonant frequency in MHz
            min_reps (int): Minimum number of repetitions
            log (bool): Whether to use logarithmic gain spacing
            pulse_e (bool): Whether to apply pi pulse on |g>-|e> transition
        
    Returns:
        list: List containing:
            - ImageContent: Plot showing resonator response vs frequency and power
            - TextContent: Lamb shift measurement if fit is successful
    """
    try:
        # Default parameters for power spectroscopy
        param = {
            'qi': 0,
            'fit': True,
            'rng': 100,
            'max_gain': 1,
            'expts_gain': 20,
            'span': 15,
            'f_off': 4,
            'min_reps': 100,
            'log': True,
            'pulse_e': False
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the resonator power spectroscopy experiment
        rpowspec = meas.ResSpecPower(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            params={
                'rng': param['rng'],
                'max_gain': param['max_gain'],
                'expts_gain': param['expts_gain'],
                'span': param['span'],
                'f_off': param['f_off'],
                'min_reps': param['min_reps'],
                'log': param['log'],
                'pulse_e': param['pulse_e']
            }
        )
        try:
        # Get the figure from the experiment
            data = rpowspec.analyze()
            figure = rpowspec.display(save_fig=False, return_fig=True, fit=param['fit'])
        except Exception as e:
            return TextContent(type="text", text=f"Error in third: {e}")
        # Get Lamb shift measurement if available
        lamb_text = ""
        if 'lamb_shift' in data:
            lamb_text = f"Measured Lamb shift: {data['lamb_shift']:.4f} MHz\n"
            if 'fit_gains' in data:
                high_gain, low_gain = data['fit_gains']
                lamb_text += f"High power peak: {data['fit'][0][2]:.4f} MHz (gain: {high_gain:.4f})\n"
                lamb_text += f"Low power peak: {data['fit'][1][2]:.4f} MHz (gain: {low_gain:.4f})"
        
        # Return both the figure and Lamb shift information
        return [
            ImageContent(data=figure, mimeType="image/png", type="image"),
            TextContent(type="text", text=lamb_text)
        ]
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")

@mcp.tool()
def qubit_spectroscopy(key_args: dict):
    """Runs a qubit spectroscopy experiment to find the qubit transition frequency.
    
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            gain (float): Qubit pulse amplitude, can only be at most 1.0, at least 0.001
            reps (int): Number of repetitions
            span (float): Frequency span in MHz
            expts (int): Number of frequency points
            start (float): Start frequency in MHz
            style (str): 'coarse', 'medium', or 'fine' for different frequency scan ranges
            soft_avgs (int): Number of software averages
            checkEF (bool): Whether to check the |e>-|f> transition
        
    Returns:
        list: List containing:
            - ImageContent: Plot showing qubit response vs frequency with fit
            - TextContent: Fit parameters if fit is successful
    """
    try:
        # Default parameters for qubit spectroscopy
        param = {
            'qi': 0,
            'fit': True,
            'gain': 0.5,
            'reps': 1000,
            'span': 100,
            'expts': 200,
            'start': None,  # Will be calculated based on current qubit frequency
            'style': 'medium',
            'soft_avgs': 1,
            'checkEF': False
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the qubit spectroscopy experiment
        qspec = meas.QubitSpec(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            style=param['style'],
            params={
                'start': param['start'],
                'span': param['span'],
                'soft_avgs': param['soft_avgs'],
                'reps': param['reps'],
                'gain': param['gain'],
                'expts': param['expts'],
                'checkEF': param['checkEF']
            }
        )
        
        # Get the figure from the experiment
        data = qspec.analyze()
        figure = qspec.display(save_fig=False, return_fig=True, fit=param['fit'])
        
        # Get fit parameters if available
        fit_text = ""
        if 'fit' in data:
            f0 = data['fit'][2]  # Center frequency
            kappa = data['fit'][3] * 2  # Linewidth
            fit_text = f"Fit parameters:\n"
            fit_text += f"Center frequency: {f0:.4f} MHz\n"
            fit_text += f"Linewidth: {kappa:.4f} MHz"
        
        # Return both the figure and fit information
        return [
            ImageContent(data=figure, mimeType="image/png", type="image"),
            TextContent(type="text", text=fit_text)
        ]
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")

@mcp.tool()
def qubit_power_spectroscopy(key_args: dict):
    """Runs a qubit power spectroscopy experiment to measure how the qubit response changes with power.
    Experiment can take a long time to run (more than 30min)
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            rng (int): Range for the gain sweep, going from max_gain to max_gain/rng
            max_gain (float): Maximum gain value, value need to be smaller than 1.0
            expts_gain (int): Number of gain points
            span (float): Frequency span in MHz
            f_off (float): Frequency offset from qubit frequency in MHz
            min_reps (int): Minimum number of repetitions
            log (bool): Whether to use logarithmic gain spacing
            checkEF (bool): Whether to check the |e>-|f> transition
        
    Returns:
        list: List containing:
            - ImageContent(s): Plot(s) showing qubit response vs frequency and power
            - TextContent: Power-dependent frequency shift if fit is successful
    """
    try:
        # Default parameters for power spectroscopy
        param = {
            'qi': 0,
            'fit': True,
            'rng': 100,
            'max_gain': 1,
            'expts_gain': 20,
            'span': 100,
            'f_off': 4,
            'min_reps': 100,
            'log': True,
            'checkEF': False
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the qubit power spectroscopy experiment
        qpowspec = meas.QubitSpecPower(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            params={
                'rng': param['rng'],
                'max_gain': param['max_gain'],
                'expts_gain': param['expts_gain'],
                'span': param['span'],
                'f_off': param['f_off'],
                'min_reps': param['min_reps'],
                'log': param['log'],
                'checkEF': param['checkEF']
            }
        )
        
        # Get the figure from the experiment
        data = qpowspec.analyze()
        figure = qpowspec.display(save_fig=False, return_fig=True, fit=param['fit'])
        
        # Get power-dependent frequency shift if available
        shift_text = ""
        if 'fit' in data:
            high_pow_freq = data['fit'][0][2]  # High power frequency
            low_pow_freq = data['fit'][1][2]   # Low power frequency
            freq_shift = high_pow_freq - low_pow_freq
            shift_text = f"Power-dependent frequency shift: {freq_shift:.4f} MHz\n"
            if 'fit_gains' in data:
                high_gain, low_gain = data['fit_gains']
                shift_text += f"High power frequency: {high_pow_freq:.4f} MHz (gain: {high_gain:.4f})\n"
                shift_text += f"Low power frequency: {low_pow_freq:.4f} MHz (gain: {low_gain:.4f})"
        
        # Handle both single figure and list of figures
        result = []
        if isinstance(figure, list):
            # Add each figure as a separate ImageContent
            for fig in figure:
                result.append(ImageContent(data=fig, mimeType="image/png", type="image"))
        else:
            # Single figure returned
            result.append(ImageContent(data=figure, mimeType="image/png", type="image"))
        
        # Add text content
        result.append(TextContent(type="text", text=shift_text))
        
        return result
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")

@mcp.tool()
def qubit_t1_measurement(key_args: dict):
    """Runs a T1 measurement experiment to measure the qubit's energy relaxation time.
    The measurement requires very good calibration of the qubit frequency, resonator frequency, resonator
    gain during measurement, and the qubit pulse gain. 
    
    This experiment measures how long it takes for the qubit to relax from the excited state
    to the ground state. The qubit is first excited with a π pulse, then allowed to relax
    for a variable time before measurement. The resulting decay curve is fit to an exponential
    to extract the T1 time.
    
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            gain (float): Qubit pulse amplitude, can only be at most 1.0, at least 0.001
            reps (int): Number of repetitions
            span (float): Time span in μs
            expts (int): Number of time points
            start (float): Start time in μs
            style (str): 'coarse', 'medium', or 'fine' for different time scan ranges
            soft_avgs (int): Number of software averages
            plot_all (bool): Whether to plot I/Q/Amps or just I
        
    Returns:
        list: List containing:
            - ImageContent(s): Plot(s) showing qubit state vs time with exponential fit
            - TextContent: T1 time from fit if successful
    """
    try:
        # Default parameters for T1 measurement
        param = {
            'qi': 0,
            'fit': True,
            'gain': 0.5,
            'reps': 1000,
            'span': 50,  # 50 μs span for T1 measurement
            'expts': 100,
            'start': 0,
            'style': 'medium',
            'soft_avgs': 1,
            'plot_all': False
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the T1 experiment
        t1_exp = meas.T1Experiment(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            style=param['style'],
            params={
                'start': param['start'],
                'span': param['span'],
                'soft_avgs': param['soft_avgs'],
                'reps': param['reps'],
                'gain': param['gain'],
                'expts': param['expts']
            }
        )
        
        # Get the figure from the experiment
        data = t1_exp.analyze()
        figure = t1_exp.display(save_fig=False, return_fig=True, fit=param['fit'], plot_all=param['plot_all'])
        
        # Get T1 time if fit was successful
        t1_text = ""
        if 'new_t1' in data:
            t1 = data['new_t1']  # T1 time in μs
            t1_text = f"T1 time: {t1:.2f} μs"
        
        # Handle both single figure and list of figures
        result = []
        if isinstance(figure, list):
            # Add each figure as a separate ImageContent
            for fig in figure:
                result.append(ImageContent(data=fig, mimeType="image/png", type="image"))
        else:
            # Single figure returned
            result.append(ImageContent(data=figure, mimeType="image/png", type="image"))
        
        # Add text content
        result.append(TextContent(type="text", text=t1_text))
        
        return result
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")

@mcp.tool()
def qubit_rabi_amplitude(key_args: dict):
    """Runs an amplitude Rabi experiment to find the π-pulse amplitude for qubit control.
    The measurement requires good calibration of the qubit frequency and resonator frequency,
    resonator gain during measurement. This measurement is used to calibrate the qubit pulse gain.
    
    This experiment measures Rabi oscillations by varying the amplitude of a driving pulse
    and measuring the resulting qubit state. The oscillation pattern allows determination
    of the π-pulse amplitude needed for qubit control. The qubit is driven at its resonant
    frequency with a fixed-length pulse of varying amplitude.
    
    Note: This experiment is often run alternating with T2 Ramsey measurements for several
    iterations since both precise frequency (from Ramsey) and precise amplitude (from Rabi) 
    are necessary for optimal qubit control. The two experiments complement each other in
    the calibration process.
    
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data
            gain (float): Maximum qubit pulse amplitude, can only be at most 1.0, at least 0.001
            reps (int): Number of repetitions
            expts (int): Number of amplitude points
            style (str): 'coarse', 'medium', or 'fine' for different amplitude scan ranges
            soft_avgs (int): Number of software averages
            plot_all (bool): Whether to plot I/Q/Amps or just I
            checkEF (bool): Whether to check the |e>-|f> transition
        
    Returns:
        list: List containing:
            - ImageContent(s): Plot(s) showing Rabi oscillations with fit
            - TextContent: π-pulse amplitude from fit if successful
    """
    try:
        # Default parameters for Rabi experiment
        param = {
            'qi': 0,
            'fit': True,
            'gain': 0.5,
            'reps': 1000,
            'expts': 60,
            'style': 'medium',
            'soft_avgs': 1,
            'plot_all': False,
            'checkEF': False,
            'sweep': 'amp'  # Fixed to amplitude sweep
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the Rabi experiment
        rabi_exp = meas.RabiExperiment(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            style=param['style'],
            params={
                'soft_avgs': param['soft_avgs'],
                'reps': param['reps'],
                'gain': param['gain'],
                'expts': param['expts'],
                'checkEF': param['checkEF'],
                'sweep': param['sweep']
            }
        )
        
        # Get the figure from the experiment
        data = rabi_exp.analyze()
        figure = rabi_exp.display(save_fig=False, return_fig=True, fit=param['fit'], plot_all=param['plot_all'])
        
        # Get π-pulse amplitude if fit was successful
        pi_text = ""
        if 'pi_length' in data:
            pi_amp = data['pi_length']  # π-pulse amplitude
            pi_text = f"π-pulse amplitude: {pi_amp:.4f}"
        
        # Handle both single figure and list of figures
        result = []
        if isinstance(figure, list):
            # Add each figure as a separate ImageContent
            for fig in figure:
                result.append(ImageContent(data=fig, mimeType="image/png", type="image"))
        else:
            # Single figure returned
            result.append(ImageContent(data=figure, mimeType="image/png", type="image"))
        
        # Add text content
        result.append(TextContent(type="text", text=pi_text))
        
        return result
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")

@mcp.tool()
def qubit_t2_ramsey(key_args: dict):
    """Runs a T2 Ramsey experiment to fine-tune the qubit frequency and measure dephasing time.
    Experiment can take a long time to run (more than 30min)
    
    This experiment uses two π/2 pulses separated by a variable delay to create coherent 
    oscillations. The frequency of these oscillations reveals the error in the qubit drive
    frequency, enabling precise frequency calibration. The decay envelope gives the T2 
    dephasing time. This measurement often needs to be alternated with amplitude Rabi 
    experiments since both frequency and amplitude are critical for good qubit control.
    
    The Ramsey sequence: π/2 - wait - π/2, where the second π/2 pulse has a phase that
    advances at the Ramsey frequency. Any frequency error shows up as oscillations at
    a different frequency than expected.
    
    Args:
        key_args (dict): Dictionary containing experiment parameters:
            qi (int): Qubit index to measure
            fit (bool): Whether to fit the data to extract T2 and frequency error
            reps (int): Number of repetitions
            expts (int): Number of time points
            span (float): Total time span in μs (default: 3*T2)
            start (float): Start time in μs
            ramsey_freq (float): Ramsey frequency in MHz for phase advancement
            style (str): 'fine' for more averages, 'fast' for fewer points
            soft_avgs (int): Number of software averages
            plot_all (bool): Whether to plot I/Q/Amps or just I
            checkEF (bool): Whether to check the |e>-|f> transition
            fit_twofreq (bool): Whether to fit with two-frequency model
            refit (bool): Whether to refit without slope
        
    Returns:
        list: List containing:
            - ImageContent(s): Plot(s) showing Ramsey oscillations with exponential decay fit
            - TextContent: T2 time and frequency error from fit if successful
    """
    try:
        # Default parameters for T2 Ramsey experiment
        param = {
            'qi': 0,
            'fit': True,
            'reps': 1000,
            'expts': 100,
            'span': None,  # Will be set to 3*T2 if None
            'start': 0.01,
            'ramsey_freq': 'smart',  # Will be set to 1.5/T2
            'style': '',
            'soft_avgs': 1,
            'plot_all': False,
            'checkEF': False,
            'experiment_type': 'ramsey',
            'fit_twofreq': False,
            'refit': False
        }
        
        # Update parameters with user-provided values
        for key, value in key_args.items():
            if key not in param:
                pass
            param[key] = value
    except Exception as e:
        return TextContent(type="text", text=f"Error 1st step: {e}")
        
    try:
        # Create and run the T2 Ramsey experiment
        t2_exp = meas.T2Experiment(
            cfg_dict=cfg_dict, 
            qi=param['qi'],
            style=param['style'],
            params={
                'reps': param['reps'],
                'expts': param['expts'],
                'span': param['span'],
                'start': param['start'],
                'ramsey_freq': param['ramsey_freq'],
                'soft_avgs': param['soft_avgs'],
                'checkEF': param['checkEF'],
                'experiment_type': param['experiment_type']
            }
        )
        
        # Get the figure from the experiment
        data = t2_exp.analyze(fit=param['fit'], fit_twofreq=param['fit_twofreq'], refit=param['refit'])
        figure = t2_exp.display(
            save_fig=False, 
            return_fig=True, 
            fit=param['fit'], 
            plot_all=param['plot_all'],
            fit_twofreq=param['fit_twofreq'],
            refit=param['refit']
        )
        
        # Get T2 time and frequency error if fit was successful
        result_text = ""
        if param['fit'] and 'fit_avgi' in data and data['fit_avgi'] is not None:
            # Extract T2 time (decay parameter, index 3)
            t2_time = data['fit_avgi'][3]
            result_text += f"T2 dephasing time: {t2_time:.2f} μs\n"
            
            # Extract frequency error if available
            if 'f_err' in data:
                freq_error = data['f_err']
                result_text += f"Frequency error: {freq_error:.6f} MHz\n"
                
            if 'new_freq' in data:
                new_freq = data['new_freq']
                result_text += f"Corrected qubit frequency: {new_freq:.6f} MHz\n"
                
            # Extract Ramsey frequency for reference
            ramsey_freq = t2_exp.cfg.expt.ramsey_freq
            result_text += f"Ramsey frequency used: {ramsey_freq:.6f} MHz\n"
            
            # If frequency adjustments are available, show them
            if 't2r_adjust' in data:
                adj = data['t2r_adjust']
                result_text += f"Possible frequency adjustments: {adj[0]:.6f}, {adj[1]:.6f} MHz"
        
        # Handle both single figure and list of figures
        result = []
        if isinstance(figure, list):
            # Add each figure as a separate ImageContent
            for fig in figure:
                result.append(ImageContent(data=fig, mimeType="image/png", type="image"))
        else:
            # Single figure returned
            result.append(ImageContent(data=figure, mimeType="image/png", type="image"))
        
        # Add text content
        result.append(TextContent(type="text", text=result_text))
        
        return result
        
    except Exception as e:
        return TextContent(type="text", text=f"Error in second: {e}")




# This is the main entry point for your server
def main():
    logger.info('Starting your-new-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main()