#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Union, Any
from dataclasses import dataclass
import warnings
import io
import base64
from matplotlib.figure import Figure
from dummy_experiment import DummyResonatorExperiment, ResonanceMode
from mcp.server.fastmcp import FastMCP, Image
from scipy.optimize import curve_fit

mcp = FastMCP("vna")

class VNASimulator:
    """
    A class that simulates a Vector Network Analyzer (VNA) connected to a resonator experiment.

    This class provides methods to perform frequency sweeps, analyze resonances,
    and visualize the results.
    """

    def __init__(self, experiment: DummyResonatorExperiment, num_memories: int = 10):
        """
        Initialize the VNA simulator with a resonator experiment.

        Args:
            experiment: A DummyResonatorExperiment instance
            num_memories: Number of memory slots for storing scan data
        """
        self.experiment = experiment
        self.last_scan_data = None
        self.last_scan_params = None
        self.num_memories = num_memories
        self.memory_scan_data = [None] * num_memories
        self.memory_scan_params = [None] * num_memories
        self.memory_fit_data = [None] * num_memories

    def perform_scan(
        self,
        start_freq: float,
        stop_freq: float,
        num_points: int = 1001,
        power_dbm: float = 0,
        averaging: int = 1,
        return_raw: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a frequency sweep and measure the complex transmission S21.

        Args:
            start_freq: Start frequency in GHz
            stop_freq: Stop frequency in GHz
            num_points: Number of frequency points
            power_dbm: Power in dBm
            averaging: Number of averages to perform
            return_raw: If True, return complex S21 data; if False, return magnitude in dB

        Returns:
            Tuple of (frequencies, s21_data)
        """
        frequencies = np.linspace(start_freq, stop_freq, num_points)
        s21_complex = np.zeros(num_points, dtype=complex)

        # Perform multiple scans and average if requested
        for _ in range(averaging):
            _, scan_data = self.experiment.get_transmission_data(
                start_freq, stop_freq, num_points, power_dbm
            )
            s21_complex += scan_data

        s21_complex /= averaging

        # Store the scan data and parameters
        self.last_scan_data = {
            'frequencies': frequencies,
            's21_complex': s21_complex,
            's21_db': 20 * np.log10(np.abs(s21_complex)),
            's21_phase': np.angle(s21_complex, deg=True)
        }

        self.last_scan_params = {
            'start_freq': start_freq,
            'stop_freq': stop_freq,
            'num_points': num_points,
            'power_dbm': power_dbm,
            'averaging': averaging
        }

        if return_raw:
            return frequencies, s21_complex
        else:
            return frequencies, 20 * np.log10(np.abs(s21_complex))

    def power_sweep(
        self,
        center_freq: float,
        span: float,
        num_points: int = 201,
        powers_dbm: List[float] = None,
        averaging: int = 1
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform a power sweep at multiple power levels around a center frequency.

        Args:
            center_freq: Center frequency in GHz
            span: Frequency span in GHz
            num_points: Number of frequency points per scan
            powers_dbm: List of power levels in dBm
            averaging: Number of averages per scan

        Returns:
            Dictionary mapping power levels to (frequencies, s21_db) tuples
        """
        if powers_dbm is None:
            powers_dbm = [-20, -10, 0, 10, 20]

        start_freq = center_freq - span/2
        stop_freq = center_freq + span/2

        results = {}
        for power in powers_dbm:
            frequencies, s21_db = self.perform_scan(
                start_freq, stop_freq, num_points, power, averaging
            )
            results[power] = (frequencies, s21_db)

        return results

    def plot_scan(
        self,
        fig=None,
        ax=None
    ):
        """
        Plot the results of the last scan.

        Args:
            fig: Matplotlib figure (creates new if None)
            ax: Matplotlib axis (creates new if None)

        Returns:
            Matplotlib figure and axis objects
        """
        if self.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        frequencies = self.last_scan_data['frequencies']
        s21_db = self.last_scan_data['s21_db']

        # Plot the transmission data
        ax.plot(frequencies, s21_db, 'b-', label='S21 Magnitude')

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_title(f"VNA Scan: {self.last_scan_params['start_freq']}-{self.last_scan_params['stop_freq']} GHz, "
                     f"{self.last_scan_params['power_dbm']} dBm")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def write_last_scan_to_mem(self, slot: int):
        """
        Write the last scan data to a memory slot.

        Args:
            slot: Memory slot to write to (0 to num_memories-1)

        Returns:
            True if successful, False otherwise
        """
        if slot < 0 or slot >= self.num_memories:
            return False

        if self.last_scan_data is None:
            return False

        # Make deep copies to ensure independence of memory slots
        self.memory_scan_data[slot] = copy.deepcopy(self.last_scan_data)
        self.memory_scan_params[slot] = copy.deepcopy(self.last_scan_params)

        return True

    def get_scan_plot_mem(self, slot: int):
        """
        Get a plot of the scan data stored in a memory slot.

        Args:
            slot: Memory slot to retrieve from (0 to num_memories-1)

        Returns:
            Matplotlib figure or None if the slot is empty
        """
        if slot < 0 or slot >= self.num_memories:
            return None

        if self.memory_scan_data[slot] is None:
            return None

        # Get the data from memory
        frequencies = self.memory_scan_data[slot]['frequencies']
        s21_db = self.memory_scan_data[slot]['s21_db']

        # Create the plot
        return self.plot_scan(frequencies, s21_db,
                             title=f"Memory Slot {slot}: {self.memory_scan_params[slot]['start_freq']} to {self.memory_scan_params[slot]['stop_freq']} GHz")

    def get_scan_data_mem(self, slot: int):
        """
        Get the scan data stored in a memory slot.

        Args:
            slot: Memory slot to retrieve from (0 to num_memories-1)

        Returns:
            Dictionary with scan data or None if the slot is empty
        """
        if slot < 0 or slot >= self.num_memories:
            return None

        if self.memory_scan_data[slot] is None:
            return None

        return {
            'scan_data': self.memory_scan_data[slot],
            'scan_params': self.memory_scan_params[slot]
        }

    def export_data_mem(self, slot: int, filename: str = None, format: str = 'csv'):
        """
        Export the scan data from a memory slot to a file.

        Args:
            slot: Memory slot to export from (0 to num_memories-1)
            filename: Name of the file to export to (if None, a default name is generated)
            format: Format to export to ('csv' or 'txt')

        Returns:
            Dictionary with export information
        """
        if slot < 0 or slot >= self.num_memories:
            return {'success': False, 'message': f"Invalid memory slot {slot}"}

        if self.memory_scan_data[slot] is None:
            return {'success': False, 'message': f"Memory slot {slot} is empty"}

        # Generate a default filename if none is provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vna_scan_memory_{slot}_{timestamp}"

        # Add the appropriate extension if not present
        if format.lower() == 'csv' and not filename.lower().endswith('.csv'):
            filename += '.csv'
        elif format.lower() == 'txt' and not filename.lower().endswith('.txt'):
            filename += '.txt'

        # Get the data
        frequencies = self.memory_scan_data[slot]['frequencies']
        s21_db = self.memory_scan_data[slot]['s21_db']
        s21_phase = self.memory_scan_data[slot]['s21_phase']
        s21_complex = self.memory_scan_data[slot]['s21_complex']

        # Create a DataFrame
        df = pd.DataFrame({
            'Frequency (GHz)': frequencies,
            'S21 (dB)': s21_db,
            'Phase (rad)': s21_phase,
            'Real': s21_complex.real,
            'Imag': s21_complex.imag
        })

        # Export the data
        try:
            if format.lower() == 'csv':
                df.to_csv(filename, index=False)
            else:  # txt format
                df.to_csv(filename, index=False, sep='\t')

            return {
                'success': True,
                'filename': filename,
                'message': f"Data from memory slot {slot} exported to {filename}"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error exporting data: {str(e)}"
            }

    def fit_lorentzian_resonance_mem(self, slot: int, f_min: float = None, f_max: float = None):
        """
        Fit a Lorentzian resonance to the data in a memory slot.

        Args:
            slot: Memory slot to fit (0 to num_memories-1)
            f_min: Minimum frequency for the fit (if None, uses the full range)
            f_max: Maximum frequency for the fit (if None, uses the full range)

        Returns:
            Dictionary with fit parameters or None if the slot is empty
        """
        if slot < 0 or slot >= self.num_memories:
            return None

        if self.memory_scan_data[slot] is None:
            return None

        # Get the data
        frequencies = self.memory_scan_data[slot]['frequencies']
        s21_db = self.memory_scan_data[slot]['s21_db']

        # Perform the fit
        fit_results = self.fit_lorentzian(frequencies, s21_db, f_min, f_max)

        # Store the fit results in memory
        self.memory_fit_data[slot] = fit_results

        return fit_results

    def export_data(self, filename: str, format: str = 'csv'):
        """
        Export the last scan data to a file.

        Args:
            filename: Output filename
            format: Format type ('csv' or 'npz')
        """
        if self.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        if format.lower() == 'csv':
            # Export as CSV
            data = np.column_stack((
                self.last_scan_data['frequencies'],
                self.last_scan_data['s21_db'],
                self.last_scan_data['s21_phase'],
                self.last_scan_data['s21_complex'].real,
                self.last_scan_data['s21_complex'].imag
            ))

            header = "Frequency (GHz), Magnitude (dB), Phase (deg), Real, Imaginary"
            np.savetxt(filename, data, delimiter=',', header=header)
            print(f"Data exported to {filename}")

        elif format.lower() == 'npz':
            # Export as NumPy compressed file
            np.savez(
                filename,
                frequencies=self.last_scan_data['frequencies'],
                s21_db=self.last_scan_data['s21_db'],
                s21_phase=self.last_scan_data['s21_phase'],
                s21_complex=self.last_scan_data['s21_complex'],
                scan_params=self.last_scan_params
            )
            print(f"Data exported to {filename}")

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'npz'.")

    def get_figure_as_base64(self):
        """
        Convert the current plot to a base64 encoded string.

        Returns:
            Base64 encoded image as string
        """
        if self.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        fig, _ = self.plot_scan()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_str

    def lorentzian_model(self, f, f0, q, amplitude, offset):
        """
        Lorentzian function model for resonance fitting.

        Args:
            f: Frequency points
            f0: Resonance frequency
            q: Quality factor
            amplitude: Amplitude of the resonance
            offset: Vertical offset
            "plot_base64": img_data

        Returns:
            Lorentzian function values at frequency points f
        """
        # For S21 in dB scale, we model the resonance dip
        return offset - amplitude * (1 / (1 + 4 * q**2 * ((f - f0) / f0)**2))

    def fit_lorentzian(self, center_freq=None, span=None):
        """
        Fit a Lorentzian model to the last scan data to extract resonance parameters.

        Args:
            center_freq: Optional center frequency for fitting (GHz)
            span: Optional frequency span for fitting (GHz)

        Returns:
            Dictionary with fit results including resonance frequency, Q factor, etc.
        """
        if self.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        # Get the data
        frequencies = self.last_scan_data['frequencies']
        s21_db = self.last_scan_data['s21_db']

        # Apply frequency window if specified
        if center_freq is not None and span is not None:
            f_min = center_freq - span/2
            f_max = center_freq + span/2
            mask = (frequencies >= f_min) & (frequencies <= f_max)
            frequencies = frequencies[mask]
            s21_db = s21_db[mask]

        # Find initial parameter guesses
        min_idx = np.argmin(s21_db)
        f0_guess = frequencies[min_idx]  # resonance frequency
        offset_guess = np.max(s21_db)    # baseline level
        amplitude_guess = offset_guess - np.min(s21_db)  # depth of resonance

        # Estimate Q from the width of the resonance
        # Find frequencies where the response is at half depth
        half_depth = np.min(s21_db) + amplitude_guess/2
        idx_above = np.where(s21_db > half_depth)[0]
        if len(idx_above) > 0:
            # Find the indices closest to the resonance from both sides
            left_idx = idx_above[idx_above < min_idx]
            right_idx = idx_above[idx_above > min_idx]

            if len(left_idx) > 0 and len(right_idx) > 0:
                left_f = frequencies[left_idx[-1]]
                right_f = frequencies[right_idx[0]]
                fwhm = right_f - left_f  # Full width at half maximum
                q_guess = f0_guess / fwhm
            else:
                q_guess = 100  # Default if we can't estimate
        else:
            q_guess = 100  # Default Q guess

        # Ensure Q is reasonable
        q_guess = min(max(q_guess, 10), 10000)  # Limit Q to reasonable range

        # Initial parameters: [f0, Q, amplitude, offset]
        p0 = [f0_guess, q_guess, amplitude_guess, offset_guess]

        try:
            # Perform curve fitting
            popt, pcov = curve_fit(
                self.lorentzian_model,
                frequencies,
                s21_db,
                p0=p0,
                bounds=([f0_guess*0.99, 5, 0, -100],
                         [f0_guess*1.01, 1e6, 100, 100])
            )

            # Extract fitted parameters
            f0_fit, q_fit, amplitude_fit, offset_fit = popt

            # Calculate errors (standard deviations) from covariance matrix
            perr = np.sqrt(np.diag(pcov))
            f0_err, q_err, amplitude_err, offset_err = perr

            # Calculate the fitted curve
            s21_fit = self.lorentzian_model(frequencies, *popt)

            # Calculate R-squared to assess goodness of fit
            ss_tot = np.sum((s21_db - np.mean(s21_db))**2)
            ss_res = np.sum((s21_db - s21_fit)**2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                "resonance_frequency": float(f0_fit),
                "q_factor": float(q_fit),
                "amplitude_db": float(amplitude_fit),
                "offset_db": float(offset_fit),
                "f0_error": float(f0_err),
                "q_error": float(q_err),
                "r_squared": float(r_squared),
                "fit_successful": True
            }

        except Exception as e:
            return {
                "fit_successful": False,
                "error_message": str(e),
                "f0_guess": float(f0_guess),
                "q_guess": float(q_guess)
            }

    def plot_lorentzian_fit(self, fit_params, fig=None, ax=None):
        """
        Plot the results of Lorentzian fitting.

        Args:
            fit_params: Dictionary of fit parameters from fit_lorentzian
            fig: Matplotlib figure (creates new if None)
            ax: Matplotlib axis (creates new if None)

        Returns:
            Matplotlib figure and axis objects with the plot
        """
        if self.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        if not fit_params.get("fit_successful", False):
            raise ValueError("No successful fit available to plot.")

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        frequencies = self.last_scan_data['frequencies']
        s21_db = self.last_scan_data['s21_db']

        # Plot original data
        ax.plot(frequencies, s21_db, 'b-', label='S21 Magnitude')

        # Generate the fitted curve
        f0 = fit_params["resonance_frequency"]
        q = fit_params["q_factor"]
        amplitude = fit_params["amplitude_db"]
        offset = fit_params["offset_db"]

        # Create a finer frequency array for smooth plotting
        f_min, f_max = np.min(frequencies), np.max(frequencies)
        f_fine = np.linspace(f_min, f_max, 1000)
        s21_fit = self.lorentzian_model(f_fine, f0, q, amplitude, offset)

        # Plot the fit
        ax.plot(f_fine, s21_fit, 'r-', label='Lorentzian Fit')

        # Mark the resonance frequency
        ax.axvline(x=f0, color='g', linestyle='--', alpha=0.5)
        min_y = np.min(s21_db)
        ax.annotate(f'f₀ = {f0:.6f} GHz\nQ = {q:.0f}',
                  xy=(f0, min_y),
                  xytext=(f0 + (f_max-f_min)*0.05, min_y + 1),
                  arrowprops=dict(arrowstyle='->'))

        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_title(f"Lorentzian Fit: f₀ = {f0:.6f} GHz, Q = {q:.0f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig, ax

# Initialize the VNA simulator
experiment = DummyResonatorExperiment()
# Make sure we have the necessary imports
import copy
from datetime import datetime
import pandas as pd
from typing import List, Optional

# Create VNA simulator with 10 memory slots
vna = VNASimulator(experiment, num_memories=10)


@mcp.tool()
def perform_scan(start_freq: float, stop_freq: float, num_points: int = 1001,
                power_dbm: float = 0, averaging: int = 1, return_data: bool = False) -> Dict:
    """
    Perform a frequency sweep and measure the complex transmission S21.

    Args:
        start_freq: Start frequency in GHz
        stop_freq: Stop frequency in GHz
        num_points: Number of frequency points
        power_dbm: Power in dBm
        averaging: Number of averages to perform
        return_data: If True, returns the raw scan data (WARNING: Only set to True if you absolutely need the raw data, as it can be very large and slow down the LLM)

    Returns:
        Dictionary with scan results and optionally raw data
    """
    frequencies, s21_db = vna.perform_scan(
        start_freq, stop_freq, num_points, power_dbm, averaging
    )

    result = {
        "start_freq": start_freq,
        "stop_freq": stop_freq,
        "power_dbm": power_dbm,
        "scan_completed": True,
        "message": f"Scan completed from {start_freq} to {stop_freq} GHz"
    }

    # Only include raw data if specifically requested
    if return_data:
        result["frequencies"] = frequencies.tolist()
        result["s21_db"] = s21_db.tolist()

    return result


@mcp.tool()
def get_last_scan_data(return_full: bool = False) -> Dict:
    """
    Get the data from the last scan.

    Args:
        return_full: If True, returns all raw data including frequency points and S21 data
                     (avoid requesting unless necessary as this can produce large amounts of data)

    Returns:
        Dictionary with the data from the last scan
    """
    if vna.last_scan_data is None:
        return {"error": "No scan data available. Perform a scan first."}

    # Always return scan parameters
    result = {
        "scan_params": vna.last_scan_params,
        "scan_available": True,
        "message": f"Scan data available from {vna.last_scan_params['start_freq']} to {vna.last_scan_params['stop_freq']} GHz"
    }

    # Only return the full data if specifically requested
    if return_full:
        result.update({
            "frequencies": vna.last_scan_data['frequencies'].tolist(),
            "s21_db": vna.last_scan_data['s21_db'].tolist(),
            "s21_phase": vna.last_scan_data['s21_phase'].tolist(),
            "s21_complex_real": vna.last_scan_data['s21_complex'].real.tolist(),
            "s21_complex_imag": vna.last_scan_data['s21_complex'].imag.tolist()
        })

    return result


@mcp.tool()
def get_scan_plot() -> Image:
    """
    Get an image of the last scan plot.

    Returns:
        Image object containing the scan plot
    """
    try:
        if vna.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        fig, _ = vna.plot_scan()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = buf.read()
        plt.close(fig)

        return Image(data=img_data, format="png")
    except ValueError as e:
        raise ValueError(f"Error generating plot: {str(e)}")


@mcp.tool()
def export_data(filename: str, format: str = 'csv') -> str:
    """
    Export the last scan data to a file.

    Args:
        filename: Output filename
        format: Format type ('csv' or 'npz')

    Returns:
        Status message
    """
    try:
        vna.export_data(filename, format)
        return f"Data successfully exported to {filename} in {format} format"
    except ValueError as e:
        return f"Error: {str(e)}"


@mcp.tool()
def fit_lorentzian_resonance(center_freq: float = None, span: float = None,
                            return_data: bool = False) -> Dict:
    """
    Fit a Lorentzian model to the last scan data to extract resonance parameters. To use this, you need to have a narrow sweep centered around the resonance.

    Args:
        center_freq: Optional center frequency for focusing the fit (GHz)
        span: Optional span around center frequency for focusing the fit (GHz)
        return_data: If True, returns additional data points for plotting (WARNING: Only set to True if you absolutely need the raw data)

    Returns:
        Dictionary with fit results including resonance frequency, Q factor, fit quality metrics and optionally raw data
    """
    try:
        # Perform the Lorentzian fit
        fit_results = vna.fit_lorentzian(center_freq, span)

        if not fit_results.get("fit_successful", False):
            return {
                "fit_successful": False,
                "error_message": fit_results.get("error_message", "Unknown fitting error"),
                "message": "Failed to fit Lorentzian model to the data. Try adjusting the frequency range or taking a new scan."
            }

        # Create an image of the fit
        fig, _ = vna.plot_lorentzian_fit(fit_results)

        # Save the plot to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        result = {
            "fit_successful": True,
            "resonance_frequency": fit_results["resonance_frequency"],
            "q_factor": fit_results["q_factor"],
            "amplitude_db": fit_results["amplitude_db"],
            "f0_error": fit_results["f0_error"],
            "q_error": fit_results["q_error"],
            "r_squared": fit_results["r_squared"],
            "message": f"Successfully fit Lorentzian model. f₀ = {fit_results['resonance_frequency']:.6f} GHz, Q = {fit_results['q_factor']:.0f}",
        }

        # Only include raw data if specifically requested
        if return_data and vna.last_scan_data is not None:
            # Generate fit curve with many points for smooth plotting
            frequencies = vna.last_scan_data['frequencies']
            f_min, f_max = np.min(frequencies), np.max(frequencies)
            f_fine = np.linspace(f_min, f_max, 500)  # 500 points is enough for plotting
            s21_fit = vna.lorentzian_model(
                f_fine,
                fit_results["resonance_frequency"],
                fit_results["q_factor"],
                fit_results["amplitude_db"],
                fit_results["offset_db"]
            )

            result.update({
                "original_frequencies": frequencies.tolist(),
                "original_s21_db": vna.last_scan_data['s21_db'].tolist(),
                "fit_frequencies": f_fine.tolist(),
                "fit_s21_db": s21_fit.tolist()
            })

        return result

    except ValueError as e:
        return {
            "fit_successful": False,
            "error_message": str(e),
            "message": "Error fitting Lorentzian: No scan data available. Perform a scan first."
        }


@mcp.tool()
def get_lorentzian_fit_plot() -> Image:
    """
    Get an image of the Lorentzian fit plot for the last scan. To use this, you need to have a narrow sweep centered around the resonance.

    Returns:
        Image object containing the Lorentzian fit plot
    """
    try:
        if vna.last_scan_data is None:
            raise ValueError("No scan data available. Perform a scan first.")

        # Perform the Lorentzian fit
        fit_results = vna.fit_lorentzian()

        if not fit_results.get("fit_successful", False):
            raise ValueError(f"Lorentzian fitting failed: {fit_results.get('error_message', 'Unknown error')}")

        fig, _ = vna.plot_lorentzian_fit(fit_results)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = buf.read()
        plt.close(fig)

        return Image(data=img_data, format="png")
    except ValueError as e:
        raise ValueError(f"Error generating Lorentzian fit plot: {str(e)}")


@mcp.tool()
def write_last_scan_to_mem(slot: int) -> Dict:
    """
    Write the last scan data to a memory slot.

    Args:
        slot: Memory slot to write to (0 to num_memories-1)

    Returns:
        Dictionary with status information
    """
    if vna.last_scan_data is None:
        return {
            "success": False,
            "message": "No scan data available. Perform a scan first."
        }

    if slot < 0 or slot >= vna.num_memories:
        return {
            "success": False,
            "message": f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}."
        }

    # Write the data to memory
    success = vna.write_last_scan_to_mem(slot)

    if success:
        return {
            "success": True,
            "slot": slot,
            "message": f"Scan data successfully written to memory slot {slot}"
        }
    else:
        return {
            "success": False,
            "message": "Failed to write scan data to memory"
        }


@mcp.tool()
def get_scan_data_mem(slot: int, return_full: bool = False) -> Dict:
    """
    Get scan data from a memory slot.

    Args:
        slot: Memory slot to retrieve from (0 to num_memories-1)
        return_full: If True, returns all raw data (avoid requesting unless absolutely necessary!)

    Returns:
        Dictionary with the data from the specified memory slot
    """
    if slot < 0 or slot >= vna.num_memories:
        return {"error": f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}."}

    # Get the data from memory
    mem_data = vna.get_scan_data_mem(slot)

    if mem_data is None:
        return {"error": f"Memory slot {slot} is empty. Use write_last_scan_to_mem first."}

    # Always return scan parameters
    result = {
        "scan_params": mem_data['scan_params'],
        "scan_available": True,
        "slot": slot,
        "message": f"Scan data available in slot {slot} from {mem_data['scan_params']['start_freq']} to {mem_data['scan_params']['stop_freq']} GHz"
    }

    # Only return the full data if specifically requested
    if return_full:
        result.update({
            "frequencies": mem_data['scan_data']['frequencies'].tolist(),
            "s21_db": mem_data['scan_data']['s21_db'].tolist(),
            "s21_phase": mem_data['scan_data']['s21_phase'].tolist(),
            "s21_complex_real": mem_data['scan_data']['s21_complex'].real.tolist(),
            "s21_complex_imag": mem_data['scan_data']['s21_complex'].imag.tolist()
        })

    return result


@mcp.tool()
def get_scan_plot_mem(slot: int) -> Image:
    """
    Get an image of the scan plot from a memory slot.

    Args:
        slot: Memory slot to retrieve from (0 to num_memories-1)

    Returns:
        Image object containing the scan plot
    """
    try:
        if slot < 0 or slot >= vna.num_memories:
            raise ValueError(f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}.")

        # Get the plot from memory
        fig = vna.get_scan_plot_mem(slot)

        if fig is None:
            raise ValueError(f"Memory slot {slot} is empty. Use write_last_scan_to_mem first.")

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = buf.read()
        plt.close(fig)

        return Image(data=img_data, format="png")
    except ValueError as e:
        raise ValueError(f"Error generating plot: {str(e)}")


@mcp.tool()
def export_data_mem(slot: int, filename: str = None, format: str = 'csv') -> Dict:
    """
    Export scan data from a memory slot to a file.

    Args:
        slot: Memory slot to export from (0 to num_memories-1)
        filename: Name of the file to export to (if None, a default name is generated)
        format: Format to export to ('csv' or 'txt')

    Returns:
        Dictionary with export status
    """
    if slot < 0 or slot >= vna.num_memories:
        return {
            "success": False,
            "message": f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}."
        }

    # Export the data from memory
    result = vna.export_data_mem(slot, filename, format)

    return result


@mcp.tool()
def fit_lorentzian_resonance_mem(slot: int, f_min: float = None, f_max: float = None,
                                 return_data: bool = False) -> Dict:
    """
    Fit a Lorentzian model to scan data in a memory slot to extract resonance parameters.

    Args:
        slot: Memory slot to fit (0 to num_memories-1)
        f_min: Optional minimum frequency for focusing the fit (GHz)
        f_max: Optional maximum frequency for focusing the fit (GHz)
        return_data: If True, returns additional data points for plotting

    Returns:
        Dictionary with fit results including resonance frequency, Q factor, and fit quality metrics
    """
    try:
        if slot < 0 or slot >= vna.num_memories:
            return {
                "fit_successful": False,
                "error_message": f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}.",
                "message": f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}."
            }

        # Perform the Lorentzian fit on memory data
        fit_results = vna.fit_lorentzian_resonance_mem(slot, f_min, f_max)

        if fit_results is None:
            return {
                "fit_successful": False,
                "error_message": f"Memory slot {slot} is empty. Use write_last_scan_to_mem first.",
                "message": f"Memory slot {slot} is empty. Use write_last_scan_to_mem first."
            }

        if not fit_results.get("fit_successful", False):
            return {
                "fit_successful": False,
                "error_message": fit_results.get("error_message", "Unknown fitting error"),
                "message": "Failed to fit Lorentzian model to the data. Try adjusting the frequency range."
            }

        result = {
            "fit_successful": True,
            "slot": slot,
            "resonance_frequency": fit_results["resonance_frequency"],
            "q_factor": fit_results["q_factor"],
            "amplitude_db": fit_results["amplitude_db"],
            "f0_error": fit_results["f0_error"],
            "q_error": fit_results["q_error"],
            "r_squared": fit_results["r_squared"],
            "message": f"Successfully fit Lorentzian model to slot {slot}. f₀ = {fit_results['resonance_frequency']:.6f} GHz, Q = {fit_results['q_factor']:.0f}",
        }

        # Only include raw data if specifically requested
        if return_data:
            mem_data = vna.get_scan_data_mem(slot)
            if mem_data is not None:
                frequencies = mem_data['scan_data']['frequencies']
                f_min_plot, f_max_plot = np.min(frequencies), np.max(frequencies)
                f_fine = np.linspace(f_min_plot, f_max_plot, 500)  # 500 points for plotting
                s21_fit = vna.lorentzian_model(
                    f_fine,
                    fit_results["resonance_frequency"],
                    fit_results["q_factor"],
                    fit_results["amplitude_db"],
                    fit_results["offset_db"]
                )

                result.update({
                    "original_frequencies": frequencies.tolist(),
                    "original_s21_db": mem_data['scan_data']['s21_db'].tolist(),
                    "fit_frequencies": f_fine.tolist(),
                    "fit_s21_db": s21_fit.tolist()
                })

        return result

    except Exception as e:
        return {
            "fit_successful": False,
            "error_message": str(e),
            "message": f"Error fitting Lorentzian to memory slot {slot}: {str(e)}"
        }


@mcp.tool()
def perform_multiple_scans(start_freq_list: List[float], stop_freq_list: List[float],
                          num_points_list: Optional[List[int]] = None,
                          power_dbm_list: Optional[List[float]] = None,
                          averaging_list: Optional[List[int]] = None,
                          slots_to_write: Optional[List[int]] = None) -> Dict:
    """
    Perform multiple frequency sweeps and store results in memory slots.

    Args:
        start_freq_list: List of start frequencies in GHz
        stop_freq_list: List of stop frequencies in GHz
        num_points_list: List of number of frequency points (defaults to 1001 if not provided)
        power_dbm_list: List of powers in dBm (defaults to 0 if not provided)
        averaging_list: List of averaging values (defaults to 1 if not provided)
        slots_to_write: List of memory slots to write to (defaults to sequential slots starting from 0)

    Returns:
        Dictionary with scan results summary
    """
    # Validate input lists
    num_scans = len(start_freq_list)

    if len(stop_freq_list) != num_scans:
        return {
            "success": False,
            "message": "Length of stop_freq_list must match length of start_freq_list"
        }

    # Set default values for optional lists
    if num_points_list is None:
        num_points_list = [1001] * num_scans
    elif len(num_points_list) != num_scans:
        return {
            "success": False,
            "message": "Length of num_points_list must match length of start_freq_list"
        }

    if power_dbm_list is None:
        power_dbm_list = [0] * num_scans
    elif len(power_dbm_list) != num_scans:
        return {
            "success": False,
            "message": "Length of power_dbm_list must match length of start_freq_list"
        }

    if averaging_list is None:
        averaging_list = [1] * num_scans
    elif len(averaging_list) != num_scans:
        return {
            "success": False,
            "message": "Length of averaging_list must match length of start_freq_list"
        }

    # Set default slots if not provided
    if slots_to_write is None:
        slots_to_write = list(range(num_scans))
    elif len(slots_to_write) != num_scans:
        return {
            "success": False,
            "message": "Length of slots_to_write must match length of start_freq_list"
        }

    # Check if any slots are invalid
    for slot in slots_to_write:
        if slot < 0 or slot >= vna.num_memories:
            return {
                "success": False,
                "message": f"Invalid memory slot {slot}. Valid slots are 0 to {vna.num_memories-1}."
            }

    # Perform the scans
    results = []
    for i in range(num_scans):
        # Perform the scan
        scan_result = perform_scan(
            start_freq_list[i],
            stop_freq_list[i],
            num_points_list[i],
            power_dbm_list[i],
            averaging_list[i]
        )

        # Write to memory
        mem_result = write_last_scan_to_mem(slots_to_write[i])

        results.append({
            "scan_number": i + 1,
            "slot": slots_to_write[i],
            "start_freq": start_freq_list[i],
            "stop_freq": stop_freq_list[i],
            "scan_success": scan_result.get("scan_completed", False),
            "memory_write_success": mem_result.get("success", False)
        })

    return {
        "success": True,
        "num_scans_requested": num_scans,
        "num_scans_completed": sum(1 for r in results if r["scan_success"]),
        "num_memory_writes_successful": sum(1 for r in results if r["memory_write_success"]),
        "scan_details": results,
        "message": f"Completed {sum(1 for r in results if r['scan_success'])} out of {num_scans} scans"
    }


@mcp.tool()
def get_lorentzian_fit_plot_mem(slot: int) -> Image:
    """
    Get an image of the Lorentzian fit plot for a memory slot. To use this, you need to have
    performed a fit on the memory slot using fit_lorentzian_resonance_mem first.

    Args:
        slot: Memory slot to get the plot for (0 to num_memories-1)

    Returns:
        Image object containing the Lorentzian fit plot
    """
    try:
        if slot < 0 or slot >= vna.num_memories:
            raise ValueError(f"Invalid memory slot. Valid slots are 0 to {vna.num_memories-1}.")

        # Check if memory slot has data
        if vna.memory_scan_data[slot] is None:
            raise ValueError(f"Memory slot {slot} is empty. Use write_last_scan_to_mem first.")

        # Check if memory slot has fit data
        if vna.memory_fit_data[slot] is None:
            raise ValueError(f"No fit data available for memory slot {slot}. Use fit_lorentzian_resonance_mem first.")

        fit_results = vna.memory_fit_data[slot]

        if not fit_results.get("fit_successful", False):
            raise ValueError(f"Lorentzian fitting failed: {fit_results.get('error_message', 'Unknown error')}")

        # Get memory scan data
        frequencies = vna.memory_scan_data[slot]['frequencies']
        s21_db = vna.memory_scan_data[slot]['s21_db']

        # Create a figure with the original data and the fit
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the original data
        ax.plot(frequencies, s21_db, 'bo', alpha=0.5, label="Data")

        # Generate fit curve with many points for smooth plotting
        f_min, f_max = np.min(frequencies), np.max(frequencies)
        f_fine = np.linspace(f_min, f_max, 500)  # 500 points is enough for plotting
        s21_fit = vna.lorentzian_model(
            f_fine,
            fit_results["resonance_frequency"],
            fit_results["q_factor"],
            fit_results["amplitude_db"],
            fit_results["offset_db"]
        )

        # Plot the fit
        ax.plot(f_fine, s21_fit, 'r-', linewidth=2, label="Lorentzian Fit")

        # Add vertical line at resonance frequency
        ax.axvline(x=fit_results["resonance_frequency"], color='g', linestyle='--',
                  alpha=0.7, label=f"f₀ = {fit_results['resonance_frequency']:.6f} GHz")

        # Formatting
        f0 = fit_results["resonance_frequency"]
        q = fit_results["q_factor"]
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Transmission (dB)")
        ax.set_title(f"Memory Slot {slot} - Lorentzian Fit: f₀ = {f0:.6f} GHz, Q = {q:.0f}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Convert the figure to an image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_data = buf.read()
        plt.close(fig)

        return Image(data=img_data, format="png")
    except ValueError as e:
        raise ValueError(f"Error generating Lorentzian fit plot for memory slot {slot}: {str(e)}")


if __name__ == "__main__":
    print("Starting VNA MCP Server...")
    print(f"Initialized VNA with experiment: {experiment}")
    mcp.run(transport="stdio")
