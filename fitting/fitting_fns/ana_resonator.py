"""
Superconducting Resonator Analysis Module

This module provides tools for analyzing superconducting resonator data, including:
- Data loading and preprocessing
- Resonator fitting
- Data visualization
- Temperature and power sweep analysis

The module is organized into classes for better structure and maintainability:
- ResonatorData: Handles data loading and preprocessing
- ResonatorFitter: Handles fitting resonator models to data
- ResonatorAnalyzer: Handles analysis of resonator parameters
- ResonatorPlotter: Handles visualization of resonator data and parameters
"""

import os
import re
import time
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Union, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns
from scipy.interpolate import interp1d

import handy as hy
import fit as scfit
import resonator as scres

# import pyCircFit_v3 as cf

# Define a consistent color palette for plots
COLORS = ["#4053d3", "#b51d14", "#ddb310", "#658b38", "#7e1e9c", "#75bbfd", "#cacaca"]


class ResonatorData:
    """
    Class for handling resonator data loading and preprocessing.

    This class provides methods for:
    - Loading resonator data from files
    - Preprocessing data (phase unwrapping, normalization, etc.)
    - Combining data from multiple files
    """

    @staticmethod
    def get_resonators(
        folder: str, base_path: str, pattern: str
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get a list of resonators from files in a folder.

        Args:
            folder: Folder name
            base_path: Base path to the folder
            pattern: Regex pattern to match resonator files

        Returns:
            Tuple containing:
            - Array of resonator IDs
            - List of matching files
        """
        # List of files
        path = os.path.join(base_path, folder)
        file_list0 = os.listdir(path)
        file_list = [file for file in file_list0 if re.match(pattern, file)]

        tokens = []
        # Get a list of resonators
        for file in file_list:
            tokens.append(re.findall(pattern, file))

        values = [int(token) for sublist in tokens for token in sublist]
        resonators_set = set(values)

        frequency = Counter(values)
        print(f"Resonator frequency: {frequency}")

        resonators = np.array(list(resonators_set))
        resonators.sort()
        return resonators, file_list

    @staticmethod
    def get_resonator_power_list(pattern: str, file_list0: List[str]) -> List[str]:
        """
        Get a list of files for a resonator, sorted by power.

        Args:
            pattern: Regex pattern to match resonator files
            file_list0: List of files to search

        Returns:
            List of files sorted by power (descending)
        """
        # Grab all the files for a given resonator, then sort by power
        file_list = [file for file in file_list0 if re.match(pattern, file)]
        tokens = []
        for file in file_list:
            tokens.append(re.findall(pattern, file)[0])
        powers = np.array(tokens, dtype=float)
        inds = np.argsort(powers)

        sorted_file_list = [file_list[i] for i in inds]
        sorted_file_list = sorted_file_list[::-1]  # Reverse to get descending order
        print(f"Files sorted by power: {sorted_file_list}")
        return sorted_file_list

    @staticmethod
    def get_temp_list(
        base_path: str, max_temp: float = 1500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a list of temperature directories, sorted by temperature.

        Args:
            base_path: Base path to search for temperature directories
            max_temp: Maximum temperature to include

        Returns:
            Tuple containing:
            - Array of temperatures
            - Array of directory names
        """
        directories = [
            name
            for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))
        ]
        directories = sorted(directories)
        print(f"Directories: {directories}")

        temps = np.array([float(d[7:]) for d in directories])
        print(f"Temperatures: {temps}")

        # Filter by max temperature and sort
        inds = np.where(temps < max_temp)
        temps = temps[inds]
        directories = np.array(directories)[inds]
        inds = np.argsort(temps)
        temps = temps[inds]
        directories = directories[inds]
        print(f"Filtered directories: {directories}")
        return temps, directories

    @staticmethod
    def check_phase(data: np.ndarray) -> np.ndarray:
        """
        Check if phase data is in radians or degrees and convert to radians if needed.

        Args:
            data: Phase data

        Returns:
            Phase data in radians
        """
        if (np.max(data) - np.min(data)) > np.pi:
            data = data * np.pi / 180
        return data

    @staticmethod
    def grab_data(
        path: str, fname: str, meas_type: str = "vna", slope: float = 0
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load resonator data from a file.

        Args:
            path: Path to the file
            fname: Filename
            meas_type: Measurement type ('vna' or 'soc')
            slope: Phase slope to remove

        Returns:
            Tuple containing:
            - Dictionary of data
            - Dictionary of attributes
        """
        data, attrs = hy.prev_data(path, fname)

        # Reformat data for scres package
        if meas_type == "vna":
            data["phases"] = np.unwrap(data["phases"])
            data["phases"] = ResonatorData.check_phase(data["phases"])
            data["freqs"] = data["fpts"]
            data["amps"] = data["mags"]
            data["phases"] = data["phases"] - slope * data["freqs"]
        elif meas_type == "vna_old":
            data["phases"] = np.unwrap(data["phases"][0])
            data["phases"] = ResonatorData.check_phase(data["phases"])
            data["freqs"] = data["fpts"][0]
            data["amps"] = data["mags"][0]
            data["phases"] = data["phases"] - slope * data["freqs"]
        elif meas_type == "soc":
            data["phases"] = np.unwrap(data["phases"])
            # Apply phase correction based on slope
            if True:  # This condition seems to always be true in the original code
                data["phases"] = data["phases"][0] + slope * data["xpts"][0]
            else:
                data["phases"] = data["phases"][0] - slope * data["xpts"][0]

            data["phases"] = np.unwrap(data["phases"])
            data["amps"] = np.log10(data["amps"][0]) * 20
            data["freqs"] = data["xpts"][0] * 1e6

        return data, attrs

    @staticmethod
    def fit_phase(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Fit the phase to a line and remove the linear component.

        Args:
            data: Dictionary containing frequency and phase data

        Returns:
            Dictionary with corrected phase data
        """
        # Use the upper 20% of the frequency range for fitting
        freq_range = np.max(data["freqs"]) - np.min(data["freqs"])
        inds = np.where(data["freqs"] > np.max(data["freqs"]) - 0.2 * freq_range)

        # Center frequencies around mean for better numerical stability
        mf = np.mean(data["freqs"])
        slope = np.polyfit(data["freqs"][inds] - mf, data["phases"][inds], 1)

        # Remove linear component
        data["phases"] = data["phases"] - slope[0] * (data["freqs"] - mf) - slope[1]
        data["phases"] = np.unwrap(data["phases"])

        return data

    @staticmethod
    def combine_data(
        data1: Dict[str, np.ndarray],
        attrs: Dict[str, Any],
        data2: Dict[str, np.ndarray],
        fix_freq: bool = True,
        meas_type: str = "soc",
    ) -> Dict[str, np.ndarray]:
        """
        Combine data from two measurements.

        Args:
            data1: First data dictionary
            data2: Second data dictionary
            fix_freq: Whether to adjust phase to match at frequency boundaries
            meas_type: Measurement type

        Returns:
            Combined data dictionary
        """
        data = {}

        # Normalize phases around zero
        mp = np.mean(data1["phases"])
        data1["phases"] = data1["phases"] - mp
        data2["phases"] = data2["phases"] - mp

        # Adjust phase of second dataset to match first at boundary if requested
        if fix_freq:
            phase_interp = interp1d(
                data1["freqs"], data1["phases"], fill_value="extrapolate"
            )
            ph = phase_interp(data2["freqs"][0])
            dphase = data2["phases"][0] - ph
            data2["phases"] = data2["phases"] - dphase

        # Concatenate data
        keys_list = ["freqs", "amps", "phases"]
        for key in keys_list:
            data[key] = np.concatenate([data1[key], data2[key]])

        # Sort by frequency
        inds = data["freqs"].argsort()
        for key in keys_list:
            data[key] = data[key][inds]

        # Ensure phase is unwrapped
        data["phases"] = np.unwrap(data["phases"])

        # Copy VNA power if applicable
        if meas_type == "vna":
            if "vna_power" in data1:
                data["vna_power"] = data1["vna_power"]
            else:
                data["vna_power"] = attrs["vna_power"]

        return data

    @staticmethod
    def norm_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize complex data to have maximum amplitude of 1.

        Args:
            data: Dictionary containing amplitude and phase data

        Returns:
            Dictionary with normalized data
        """
        # Convert from dB to linear amplitude
        amp = 10 ** ((data["amps"]) / 20)
        phs = data["phases"]

        # Convert to complex and normalize
        z = amp * np.exp(1j * phs)
        z = z / np.max(z)

        # Update data dictionary
        data["amps"] = np.log10(np.abs(z)) * 20
        data["phases"] = np.angle(z)
        data["x"] = np.real(z)
        data["y"] = np.imag(z)

        return data

    @staticmethod
    def load_resonator(
        fname: str,
        path: str,
        nfiles: int,
        slope: float,
        meas_type: str,
        ends: List[str],
        fix_freq: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load resonator data from one or more files and combine them.

        Args:
            fname: Filename of the first file
            path: Path to the files
            nfiles: Number of files to load and combine
            slope: Phase slope to remove
            meas_type: Measurement type
            ends: List of filename suffixes for different parts of the data
            fix_freq: Whether to adjust phase when combining data

        Returns:
            Tuple containing:
            - Combined data dictionary
            - Attributes dictionary
        """
        # Load first file
        data1, attrs = ResonatorData.grab_data(path, fname, meas_type, slope)

        if nfiles == 1:
            # Single file case
            data = data1
            data["phases"] = data["phases"] - np.mean(data["phases"])
            return data, attrs

        # Multiple files case
        if nfiles > 1:
            # Load and combine second file
            file2 = fname.replace(ends[0], ends[1])
            data2, _ = ResonatorData.grab_data(path, file2, meas_type, slope)
            data = ResonatorData.combine_data(data1, attrs, data2, fix_freq, meas_type)

        if nfiles > 2:
            # Load and combine third file
            file3 = fname.replace(ends[0], ends[2])
            data3, _ = ResonatorData.grab_data(path, file3, meas_type, slope)
            data = ResonatorData.combine_data(data, attrs, data3, fix_freq, meas_type)

        return data, attrs

    @staticmethod
    def resonator_list(
        directories: List[str], base_path: str, nfiles: int, meas_type: str
    ) -> Tuple[np.ndarray, List[List[List[str]]], List[str]]:
        """
        Get a list of resonators and their files across multiple directories.

        Args:
            directories: List of directories to search
            base_path: Base path to the directories
            nfiles: Number of files per resonator
            meas_type: Measurement type

        Returns:
            Tuple containing:
            - Array of resonator IDs
            - List of files for each resonator in each directory
            - List of filename suffixes
        """
        # Set up pattern and suffixes based on measurement type and number of files
        if nfiles == 3:
            if meas_type == "vna":
                pattern_end = "dbm_"
                ends = ["wide1", "narrow", "wide2"]
            else:
                pattern_end = "_"
                ends = ["wideleft", "narrow", "wideright"]
            pattern0 = r"res_(\d+)_-?\d{1,5}" + pattern_end + ends[0]
        elif nfiles == 2:
            pattern0 = r"res_(\d+)_-?\d{1,5}dbm_wide"
            ends = []
        else:
            if meas_type == "vna":
                pattern_end = "dbm.h5"
            else:
                pattern_end = ""
            pattern0 = r"res_(\d+)_-?\d{1,5}" + pattern_end
            ends = [""]

        # Get list of resonators from first directory
        resonators, _ = ResonatorData.get_resonators(
            directories[0], base_path, pattern0
        )
        file_list_full = []

        # For each directory, get files for each resonator
        for i, directory in enumerate(directories):
            file_list_full.append([])
            resonators, file_list0 = ResonatorData.get_resonators(
                directory, base_path, pattern0
            )

            for j, resonator in enumerate(resonators):
                # Create pattern for this resonator
                if nfiles == 3:
                    pattern = (
                        f"res_{resonator:d}_" + "(-?\d{1,5})" + pattern_end + ends[0]
                    )
                elif nfiles == 2:
                    pattern = f"res_{resonator:d}_" + "(-?\d{1,3})dbm_wide"
                else:
                    pattern = (
                        f"res_{resonator:d}_" + "(-?\d{1,5})" + pattern_end + ends[0]
                    )

                # Get files for this resonator sorted by power
                file_list = ResonatorData.get_resonator_power_list(pattern, file_list0)
                file_list_full[i].append(file_list)

        return resonators, file_list_full, ends


class ResonatorFitter:
    """
    Class for fitting resonator models to data.

    This class provides methods for:
    - Fitting resonator models to data
    - Extracting resonator parameters from fits
    """

    @staticmethod
    def fit_resonator(
        data: Dict[str, np.ndarray],
        filename: str,
        output_path: str,
        fit_type: str = "DCM",
        plot: bool = False,
        pre: str = "circle",
        fix_freq: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Fit a resonator model to data.

        Args:
            data: Dictionary containing frequency, amplitude, and phase data
            filename: Filename for output
            output_path: Path for output files
            fit_type: Fit type ('DCM' or 'CPZM')
            plot: Whether to generate plots
            pre: Preprocessing method
            fix_freq: Whether to fix the resonance frequency

        Returns:
            Tuple containing:
            - List of fit parameters [Q, Qc, frequency, phase]
            - List of parameter errors
        """
        # Create resonator object
        my_resonator = scres.Resonator()
        my_resonator.outputpath = output_path
        my_resonator.filename = filename
        my_resonator.from_columns(data["freqs"], data["amps"], data["phases"])
        my_resonator.fix_freq = fix_freq

        # Set fit parameters
        MC_iteration = 4
        MC_rounds = 1e3
        MC_fix = []
        manual_init = None
        fmt = "png" if plot else None

        # Set preprocessing method
        my_resonator.preprocess_method = pre
        my_resonator.filepath = "./"

        # Perform fit
        my_resonator.fit_method(
            fit_type,
            MC_iteration,
            MC_rounds=MC_rounds,
            MC_fix=MC_fix,
            manual_init=manual_init,
            MC_step_const=0.3,
        )

        # Get fit results
        output = my_resonator.fit(fmt)
        return output

    @staticmethod
    def stow_data(
        params: List[List[float]],
        res_params: List[Dict[str, List]],
        j: int,
        power: List[float],
        err: List[List[float]],
    ) -> List[Dict[str, List]]:
        """
        Store fit parameters for a resonator.

        Args:
            params: List of fit parameters for each power
            res_params: List of dictionaries to store parameters
            j: Index of the resonator
            power: List of powers
            err: List of parameter errors

        Returns:
            Updated res_params list
        """
        # Convert to numpy arrays
        power = np.array(power)
        q = np.array([params[k][0] for k in range(len(params))])
        qc = np.array([params[k][1] for k in range(len(params))])
        freq = np.array([params[k][2] for k in range(len(params))])
        phase = np.array([params[k][3] for k in range(len(params))])

        # Calculate derived parameters
        qi_phi = 1 / (1 / q - 1 / qc)
        Qc_comp = qc / np.exp(1j * phase)
        Qi = (q**-1 - np.real(Qc_comp**-1)) ** -1

        # Store parameters
        res_params[j]["pow"].append(power)
        res_params[j]["q"].append(q)
        res_params[j]["qc"].append(qc)
        res_params[j]["freqs"].append(freq)
        res_params[j]["phs"].append(phase)
        res_params[j]["qi_phi"].append(qi_phi)
        res_params[j]["qi"].append(Qi)

        # Store errors
        res_params[j]["phs_err"].append(np.array([err[i][4] for i in range(len(err))]))
        res_params[j]["q_err"].append(np.array([err[i][0] for i in range(len(err))]))
        res_params[j]["qi_err"].append(np.array([err[i][1] for i in range(len(err))]))
        res_params[j]["qc_err"].append(np.array([err[i][2] for i in range(len(err))]))
        res_params[j]["qc_real_err"].append(
            np.array([err[i][3] for i in range(len(err))])
        )
        res_params[j]["f_err"].append(np.array([err[i][5] for i in range(len(err))]))

        # Create summary plot
        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
        ax[0].plot(power, Qi, ".", markersize=6)
        ax[0].set_ylabel("$Q_i$")
        ax[1].plot(power, qc, ".", markersize=6)
        ax[1].set_ylabel("$Q_c$")
        ax[1].set_xlabel("Power (dBm)")
        fig.suptitle(f"$f_0 = $ {freq[0]/1e9:3.3f} GHz")
        fig.tight_layout()

        return res_params

    @staticmethod
    def stow_data_oth(
        params: List[Dict[str, float]],
        res_params: List[Dict[str, List]],
        j: int,
        power: List[float],
    ) -> List[Dict[str, List]]:
        """
        Store fit parameters from pyCircFit for a resonator.

        Args:
            params: List of fit parameter dictionaries for each power
            res_params: List of dictionaries to store parameters
            j: Index of the resonator
            power: List of powers

        Returns:
            Updated res_params list
        """
        # Convert to numpy arrays
        power = np.array(power)
        q = np.array([params[k]["Qtot"] for k in range(len(params))])
        qc = np.array([params[k]["Qc"] for k in range(len(params))])
        Qi = np.array([params[k]["Qi"] for k in range(len(params))])
        freq = np.array([params[k]["fr"] for k in range(len(params))])
        phase = np.array([params[k]["phi"] for k in range(len(params))])

        # Store parameters
        res_params[j]["pow"].append(power)
        res_params[j]["q"].append(q)
        res_params[j]["qc"].append(qc)
        res_params[j]["freqs"].append(freq)
        res_params[j]["phs"].append(phase)
        res_params[j]["qi"].append(Qi)

        # Store errors
        res_params[j]["phs_err"].append(
            np.array([params[i]["phi_stderr"] for i in range(len(params))])
        )
        res_params[j]["q_err"].append(
            np.array([params[i]["Qtot_stderr"] for i in range(len(params))])
        )
        res_params[j]["qi_err"].append(
            np.array([params[i]["Qi_stderr"] for i in range(len(params))])
        )
        res_params[j]["qc_err"].append(
            np.array([params[i]["Qc_stderr"] for i in range(len(params))])
        )
        res_params[j]["f_err"].append(
            np.array([params[i]["fr_stderr"] for i in range(len(params))])
        )

        # Create summary plot
        fig, ax = plt.subplots(2, 1, figsize=(6, 7.5), sharex=True)
        ax[0].plot(power, Qi, ".", markersize=6)
        ax[0].set_ylabel("$Q_i$")
        ax[1].plot(power, qc, ".", markersize=6)
        ax[1].set_ylabel("$Q_c$")
        ax[1].set_xlabel("Power (dBm)")
        fig.suptitle(f"$f_0 = $ {freq[0]/1e9:3.3f} GHz")
        fig.tight_layout()

        return res_params


class ResonatorAnalyzer:
    """
    Class for analyzing resonator parameters.

    This class provides methods for:
    - Analyzing resonator parameters across temperature and power
    - Converting between different parameter representations
    - Reordering resonators
    """

    @staticmethod
    def convert_power(res_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert power from linear to dBm.

        Args:
            res_params: List of resonator parameter dictionaries

        Returns:
            Updated res_params list
        """
        for i in range(len(res_params)):
            res_params[i]["lin_power"] = res_params[i]["pow"]
            res_params[i]["pow"] = np.log10(res_params[i]["pow"]) * 20 - 30
        return res_params

    @staticmethod
    def reorder(
        params: Dict[str, List],
        res_params: List[Dict[str, Any]],
        use_pitch: bool = True,
    ) -> Tuple[Dict[str, List], List[Dict[str, Any]]]:
        """
        Reorder resonators by pitch.

        Args:
            params: Dictionary of parameters
            res_params: List of resonator parameter dictionaries
            use_pitch: Whether to sort by pitch

        Returns:
            Tuple containing:
            - Updated params dictionary
            - Reordered res_params list
        """
        params["pitch"] = params["pitch"][0 : len(res_params)]
        if use_pitch:
            ord = np.argsort(params["pitch"])
            res_params = [res_params[i] for i in ord]
            params["pitch"] = [params["pitch"][i] for i in ord]
            params["target_freq"] = [params["target_freq"][i] for i in ord]
        return params, res_params

    @staticmethod
    def analyze_sweep_gen(
        directories: List[str],
        base_path: str,
        img_path: str,
        nfiles: int = 3,
        name: str = "res",
        plot: bool = False,
        min_power: float = -120,
        meas_type: str = "vna",
        slope: float = 0,
        fix_freq: bool = True,
        fitphase: bool = False,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Analyze resonator data across multiple directories (temperatures) and powers.

        Args:
            directories: List of directories to analyze
            base_path: Base path to the directories
            img_path: Path for output images
            nfiles: Number of files per resonator
            name: Name prefix for output files
            plot: Whether to generate plots
            min_power: Minimum power to analyze
            meas_type: Measurement type
            slope: Phase slope to remove
            fix_freq: Whether to fix the resonance frequency
            fitphase: Whether to fit and remove phase slope

        Returns:
            List of resonator parameter dictionaries
        """
        # Get list of resonators and their files
        resonators, file_list, ends = ResonatorData.resonator_list(
            directories, base_path, nfiles, meas_type
        )

        # Initialize parameter dictionaries
        res_params = [None] * len(resonators)
        for i in range(len(resonators)):
            res_params[i] = {
                "freqs": [],
                "phs": [],
                "q": [],
                "qi": [],
                "qc": [],
                "qi_phi": [],
                "pow": [],
                "qi_err": [],
                "q_err": [],
                "phs_err": [],
                "qc_err": [],
                "qc_real_err": [],
                "f_err": [],
            }

        # Process each directory (temperature)
        for i, directory in enumerate(directories):
            start = time.time()
            output_path = os.path.join(img_path, f"{name}_{directory}/")
            path = os.path.join(base_path, directory)

            # Process each resonator
            for j, resonator in enumerate(resonators):
                params, err, power = [], [], []

                # Process each power
                for k, fname in enumerate(file_list[i][j]):
                    try:
                        # Load and preprocess data
                        data, attrs = ResonatorData.load_resonator(
                            fname, path, nfiles, slope, meas_type, ends, fix_freq=True
                        )
                        if fitphase:
                            data = ResonatorData.fit_phase(data)
                    except Exception:
                        traceback.print_exc()
                        continue

                    # Skip low power measurements
                    if meas_type == "vna":
                        if attrs["vna_power"] < min_power:
                            continue
                        power.append(attrs["vna_power"])
                    elif meas_type == "vna_old":
                        power.append(data["vna_power"][0])
                    else:
                        if attrs["gain"] < min_power:
                            continue
                        power.append(np.log10(attrs["gain"]) * 20 - 30)

                    print(power[-1])
                    try:
                        # Fit resonator
                        output = ResonatorFitter.fit_resonator(
                            data, fname, output_path, plot=plot, fix_freq=True
                        )
                        params.append(output[0])
                        err.append(output[1])
                    except Exception as error:
                        print(f"An exception occurred: {error}")
                        params.append(np.nan * np.ones(4))
                        err.append(np.nan * np.ones(6))

                # Store parameters
                res_params = ResonatorFitter.stow_data(
                    params, res_params, j, power, err
                )
                print(f"Time elapsed: {time.time() - start}")

        # Convert lists to arrays
        for i in range(len(resonators)):
            for key in res_params[i].keys():
                res_params[i][key] = np.array(res_params[i][key])

        return res_params, file_list


class ResonatorPlotter:
    """
    Class for visualizing resonator data and parameters.

    This class provides methods for:
    - Plotting raw resonator data
    - Plotting resonator parameters vs. power and temperature
    - Plotting resonator parameter comparisons
    """

    @staticmethod
    def plot_raw_data(
        data: Dict[str, np.ndarray],
        phs_off: float = 0,
        amp_off: float = 0,
        circ_only: bool = True,
    ) -> None:
        """
        Plot raw resonator data.

        Args:
            data: Dictionary containing frequency, amplitude, and phase data
            phs_off: Phase offset to apply
            amp_off: Amplitude offset to apply
            circ_only: Whether to only plot the circle fit
        """
        phs = data["phases"] + phs_off
        amp = 10 ** ((data["amps"] - amp_off) / 20)

        if not circ_only:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(data["freqs"] / 1e9, data["phases"] + phs_off)
            ax[1].plot(data["freqs"] / 1e9, data["amps"] - amp_off)
            ax[0].set_xlabel("Frequency (GHz)")
            ax[0].set_ylabel("Phase (rad)")
            ax[1].set_xlabel("Frequency (GHz)")
            ax[1].set_ylabel("Amplitude (dB)")
            fig.tight_layout()

        # Plot circle fit
        plt.figure()
        plt.plot(amp / np.max(amp) * np.cos(phs), amp / np.max(amp) * np.sin(phs), ".")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Circle Fit")
        plt.xlabel("Re")
        plt.ylabel("Im")

    @staticmethod
    def plot_power(
        res_params: List[Dict[str, np.ndarray]],
        cfg: Dict[str, Any],
        ind: int,
        base_path: str,
        use_pitch: bool = True,
    ) -> None:
        """
        Plot resonator parameters vs. power.

        Args:
            res_params: List of resonator parameter dictionaries
            cfg: Configuration dictionary
            base_path: Base path for output files
            use_pitch: Whether to use pitch for labels
        """
        sns.set_palette("coolwarm", len(res_params))
        plt.rcParams["lines.markersize"] = 4

        # Create plots for Q_i vs. power
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        for i in range(len(res_params)):
            if use_pitch:
                label = cfg["pitch"][i]
            else:
                label = round(np.min(res_params[i]["freqs"] / 1e9), 4)

            # Filter by power range
            inds = np.where(
                (res_params[i]["pow"][0] >= cfg["min_power"])
                & (res_params[i]["pow"][0] <= cfg["max_power"])
            )

            # Plot Q_i vs. power
            ax[0].semilogy(
                res_params[i]["pow"][0][inds],
                res_params[i]["qi"][0][inds],
                ".-",
                label=label,
            )
            ax[1].semilogy(
                res_params[i]["pow"][0][inds],
                res_params[i]["qi"][0][inds] / np.nanmax(res_params[i]["qi"][0]),
                ".-",
                label=label,
            )

            # Plot frequency and Q_c vs. power
            ax2[0].plot(
                res_params[i]["pow"][0][inds],
                1e6
                * (
                    res_params[i]["freqs"][0][inds]
                    / np.nanmin(res_params[i]["freqs"][0][inds])
                    - 1
                ),
                ".-",
                label=label,
            )
            ax2[1].plot(
                res_params[i]["pow"][0][inds],
                res_params[i]["qc"][0][inds] / np.nanmin(res_params[i]["qc"][0][inds]),
                ".-",
                label=label,
            )

        # Set labels and legends
        ax[1].set_xlabel("Power (dBm)")
        ax[0].set_ylabel("$Q_i$")
        ax[1].set_ylabel("$Q_i/Q_{i,max}$")
        ax[1].legend(title="Gap", fontsize=8)

        ax2[1].set_xlabel("Power (dBm)")
        ax2[0].set_ylabel("$\Delta f/f$ (ppm)")
        ax2[1].set_ylabel("$Q_c/Q_{c,min}$")

        # Adjust layout and save
        fig.tight_layout()
        fig2.tight_layout()

        try:
            fig2.savefig(base_path + cfg["meas"][ind] + "_Qcfreq_pow.png", dpi=300)
            fig.savefig(base_path + cfg["meas"][ind] + "_Qi_pow.png", dpi=300)
        except Exception as e:
            print(f"Error in plotting: {e}")

    @staticmethod
    def plot_temp(
        res_params: List[Dict[str, np.ndarray]],
        cfg: Dict[str, Any],
        use_pitch: bool,
        base_path: str,
        xval: str = "temp",
    ) -> None:
        """
        Plot resonator parameters vs. temperature.

        Args:
            res_params: List of resonator parameter dictionaries
            cfg: Configuration dictionary
            use_pitch: Whether to use pitch for labels
            base_path: Base path for output files
            xval: X-axis value type ('temp' or 'energy')
        """
        plt.rcParams["lines.markersize"] = 6
        plt.rcParams["lines.linewidth"] = 1.5

        # Sort temperatures
        inds = np.argsort(cfg["temps"])

        # Calculate energy ratio if needed
        i = 0  # Use first resonator for energy calculation
        en = 1e-3 * cfg["temps"][inds] * cs.k / cs.h / res_params[i]["freqs"][inds, 0]

        # Set x-axis values based on xval
        if xval == "temp":
            x = cfg["temps"][inds]
            xlab = "Temperature (mK)"
        else:
            x = en
            xlab = "$k_B T / h f_0$"

        # Set label based on pitch or frequency
        if use_pitch:
            label = cfg["pitch"][i]
        else:
            label = round(np.min(res_params[i]["freqs"] / 1e9), 4)

        # Power index
        j = 0

        # Create plot
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        for i in range(len(res_params)):
            # Filter data points
            inds2 = (
                res_params[i]["qi"][inds, j] / np.max(res_params[i]["qi"][inds, j])
                > 0.72
            )
            min_freq = np.nanmin(res_params[i]["freqs"][inds, :])
            x_filtered = cfg["temps"][inds2]

            # Plot Q_i and frequency vs. temperature
            ax[0].plot(
                x_filtered,
                res_params[i]["qi"][inds2, j] / np.max(res_params[i]["qi"][inds2, j]),
                ".-",
            )
            ax[1].plot(
                x_filtered,
                (res_params[i]["freqs"][inds2, j] - min_freq) / min_freq,
                ".-",
            )

        # Set labels
        ax[0].set_ylabel("$Q_i/Q_{i,max}$")
        ax[1].set_xlabel(xlab)
        ax[1].set_ylabel("$\Delta f/f_0$")

        # Save plot
        fig.tight_layout()
        plt.savefig(base_path + "_" + cfg["res_name"] + "_temp_sweep.png", dpi=300)

    @staticmethod
    def plot_power_temp(
        res_params: List[Dict[str, np.ndarray]],
        i: int,
        cfg: Dict[str, Any],
        base_path: str,
        use_cbar: bool = False,
        xval: str = "temp",
    ) -> None:
        """
        Plot resonator parameters vs. power and temperature.

        Args:
            res_params: List of resonator parameter dictionaries
            i: Index of the resonator to plot
            cfg: Configuration dictionary
            base_path: Base path for output files
            use_cbar: Whether to use a colorbar
            xval: X-axis value type ('temp' or 'energy')
        """
        plt.rcParams["lines.markersize"] = 4
        plt.rcParams["lines.linewidth"] = 1

        # Sort temperatures
        inds = np.argsort(cfg["temps"])

        # Calculate energy ratio if needed
        en = 1e-3 * cfg["temps"][inds] * cs.k / cs.h / res_params[i]["freqs"][inds, 0]

        # Set x-axis values based on xval
        if xval == "temp":
            x = cfg["temps"][inds]
            xlab = "Temperature (mK)"
        else:
            x = en
            xlab = "$k_B T / h f_0$"

        # Get minimum frequency
        min_freq = np.nanmin(res_params[i]["freqs"][inds, :])

        # Set color palette
        sns.set_palette("coolwarm", n_colors=res_params[0]["pow"].shape[1])

        # Create plot for temperature sweep
        fig, ax = plt.subplots(4, 1, figsize=(6, 9), sharex=True)

        for j in range(res_params[i]["pow"].shape[1]):
            ax[0].plot(x, res_params[i]["qi"][inds, j], ".-")
            ax[1].plot(
                x,
                res_params[i]["qi"][inds, j] / np.max(res_params[i]["qi"][inds, j]),
                ".-",
            )
            ax[2].plot(x, res_params[i]["qc"][inds, j], ".-")
            ax[3].plot(x, (res_params[i]["freqs"][inds, j] - min_freq) / min_freq, ".-")

        # Add colorbars if requested
        if use_cbar:
            norm = plt.Normalize(np.min(cfg["temps"]), np.max(cfg["temps"]))
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            ax[1].figure.colorbar(sm, ax=ax[1])

            norm = plt.Normalize(
                np.min(res_params[i]["pow"]), np.max(res_params[i]["pow"])
            )
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
            ax[0].figure.colorbar(sm, ax=ax[0])

        # Set labels
        ax[0].set_ylabel("$Q_i$")
        ax[1].set_ylabel("$Q_i/Q_i(0)$")
        ax[2].set_ylabel("$Q_c$")
        ax[3].set_xlabel(xlab)
        ax[3].set_ylabel("$\Delta f/f_0$")

        # Save temperature sweep plot
        fig.tight_layout()
        fig.savefig(
            base_path + cfg["res_name"] + "_Qi_temp_" + str(i) + ".png", dpi=300
        )

        # Create plot for power sweep
        sns.set_palette("coolwarm", n_colors=res_params[0]["pow"].shape[0])
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

        # Plot power dependence for each temperature
        for j in inds:
            ax[0].plot(
                res_params[i]["pow"][j, :],
                res_params[i]["qi"][j, :],
                ".-",
                label=int(cfg["temps"][j]),
            )
            ax[1].plot(
                res_params[i]["pow"][j, :],
                res_params[i]["qi"][j, :] / np.max(res_params[i]["qi"][j, :]),
                ".-",
            )

        # Set labels and legend
        ax[0].set_ylabel("$Q_i$")
        ax[1].set_ylabel("$Q_i/Q_i(0)$")
        ax[1].set_xlabel("Power (dBm)")
        ax[0].legend(fontsize=8)

        # Save power sweep plot
        fig.tight_layout()
        fig.savefig(base_path + cfg["res_name"] + "_Qi_pow_" + str(i) + ".png", dpi=300)

    @staticmethod
    def plot_res_pars(
        params_list: List[Dict[str, Any]],
        labs: List[str],
        base_path: str,
        name: Optional[str] = None,
    ) -> None:
        """
        Plot resonator parameters comparison.

        Args:
            params_list: List of parameter dictionaries
            labs: List of labels
            base_path: Base path for output files
            name: Name prefix for output files
        """
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True)
        ax = ax.flatten()
        fnames = ""
        sns.set_palette(COLORS)

        # Set labels
        ax[0].set_ylabel("Frequency (GHz)")
        ax[1].set_ylabel("Frequency/Designed Freq")
        ax[2].set_ylabel("Phase (rad)")

        # Set filename prefix
        if name is not None:
            fnames = name + "_"

        # Plot data for each parameter set
        for params, label in zip(params_list, labs):
            try:
                if name is None:
                    fnames += params["meas"] + "_"
            except:
                pass

            ax[0].plot(params["pitch"], params["freqs"] / 1e9, ".", label=label)
            ax[1].plot(
                params["pitch"],
                params["freqs"] / 1e9 / params["target_freq"],
                ".-",
                label=label,
            )
            ax[2].plot(params["pitch"], params["phs"], ".-", label=label)

        # Set x-axis labels and add legend
        for a in ax:
            a.set_xlabel("Gap width ($\mu$m)")
        ax[1].legend(fontsize=8)

        # Save plot
        fig.tight_layout()
        fig.savefig(base_path + fnames + "params_res_full.png", dpi=300)

    @staticmethod
    def plot_all(
        directories: List[str],
        base_path: str,
        output_path: str,
        name: str = "res",
        min_power: float = -120,
        max_power: float = -25,
        norm: bool = False,
        nfiles: int = 3,
        meas_type: str = "vna",
        slope: float = 0,
        circ: bool = False,
        half_norm: bool = False,
    ) -> List[List[List[str]]]:
        """
        Plot all resonator data.

        Args:
            directories: List of directories to analyze
            base_path: Base path to the directories
            output_path: Path for output files
            name: Name prefix for output files
            min_power: Minimum power to analyze
            max_power: Maximum power to analyze
            norm: Whether to normalize data
            nfiles: Number of files per resonator
            meas_type: Measurement type
            slope: Phase slope to remove
            circ: Whether to plot circle fits
            half_norm: Whether to normalize only amplitude

        Returns:
            List of files for each resonator in each directory
        """
        # Get list of resonators and their files
        resonators, file_list, ends = ResonatorData.resonator_list(
            directories, base_path, nfiles, meas_type
        )
        nres = len(resonators)

        # Set color palette
        sns.set_palette("coolwarm", n_colors=int(len(file_list[0][0])))

        # Create plots
        fig, ax = plt.subplots(2, 4, figsize=(10, 7))
        fig2, ax2 = plt.subplots(2, 4, figsize=(10, 7))
        ax = ax.flatten()
        ax2 = ax2.flatten()

        # Process each directory
        for i, directory in enumerate(directories):
            path = os.path.join(base_path, directory)

            # Process each resonator
            for j in range(len(resonators)):
                # Process each power
                for k in range(len(file_list[i][j])):
                    fname = file_list[i][j][k]
                    try:
                        # Load and preprocess data
                        data, _ = ResonatorData.load_resonator(
                            fname, path, nfiles, slope, meas_type, ends, fix_freq=True
                        )
                        data = ResonatorData.fit_phase(data)
                    except:
                        traceback.print_exc()
                        continue

                    # Skip powers outside range
                    if meas_type == "vna":
                        if (
                            data["vna_power"][0] > max_power
                            or data["vna_power"][0] < min_power
                        ):
                            continue

                    # Plot data based on normalization options
                    if norm:
                        # Full normalization
                        x = 10 ** (data["amps"] / 20) * np.cos(data["phases"])
                        y = 10 ** (data["amps"] / 20) * np.sin(data["phases"])
                        z = scfit.preprocess_circle(
                            data["freqs"],
                            x + 1j * y,
                            output_path="hi",
                            fix_freq=True,
                            plot_extra=True,
                        )
                        ax[j].plot(
                            (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                            np.log10(np.abs(z)) * 20,
                            linewidth=1,
                        )
                        ax2[j].plot(data["freqs"] / 1e9, np.angle(z), linewidth=1)
                        if circ:
                            plt.figure()
                            plt.plot(np.real(z), np.imag(z), ".")
                    elif half_norm:
                        # Amplitude normalization only
                        ax[j].plot(
                            (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                            data["amps"] / np.max(data["amps"]),
                            linewidth=1,
                        )
                        ax2[j].plot(
                            (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                            data["phases"],
                            linewidth=1,
                        )
                        data = ResonatorData.norm_data(data)
                        if circ:
                            plt.figure()
                            ResonatorPlotter.plot_raw_data(
                                data, phs_off=0, amp_off=0, circ_only=True
                            )
                    else:
                        # No normalization
                        ax[j].plot(
                            (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                            data["amps"],
                            linewidth=1,
                        )
                        ax2[j].plot(
                            (data["freqs"] - np.mean(data["freqs"])) / 1e3,
                            data["phases"],
                            linewidth=1,
                        )
                        if circ:
                            ResonatorPlotter.plot_raw_data(
                                data, phs_off=0, amp_off=0, circ_only=True
                            )

            # Adjust layout and save plots
            fig.tight_layout()
            fig2.tight_layout()

        return file_list
