#!/usr/bin/env python3

import numpy as np
from scipy.constants import c, epsilon_0
from scipy.special import ellipk
from mcp.server.fastmcp import FastMCP
from typing import Dict

mcp = FastMCP("analytical_approximations")

class analytical_approximations_CPW:
    """
    MCP Server (1) for Analytical Calculations of Superconducting Resonators.

    This server provides analytical models to estimate the resonant frequency (f₀)
    and coupling quality factor (Qc) for a capacitively coupled quarter-wave
    coplanar waveguide (CPW) resonator.

    It is intended to provide a fast, approximate starting point for a design,
    which can then be refined by more accurate EM simulations.
    """

    def __init__(self, epsilon_r: float = 11.9, Z0: float = 50.0, Z_res: float = 50.0):
        """
        Initializes the server with the physical properties of the system.

        Args:
            epsilon_r (float): Relative permittivity of the substrate (e.g., 11.9 for Si).
            Z0 (float): Characteristic impedance of the feedline in Ohms.
            Z_res (float): Characteristic impedance of the resonator line in Ohms.
        """
        if epsilon_r <= 1.0:
            raise ValueError("Relative permittivity (epsilon_r) must be greater than 1.")

        self.epsilon_r = epsilon_r
        # Effective permittivity for a CPW is often approximated as (1 + epsilon_r) / 2
        # This assumes the fields are roughly equally distributed in air and the substrate.
        self.epsilon_eff = (1 + self.epsilon_r) / 2

        self.Z0 = Z0
        self.Z_res = Z_res

        # Physical constants
        self.c = c  # Speed of light in vacuum (m/s)
        self.epsilon_0 = epsilon_0  # Permittivity of free space (F/m)

        print("MCP Server 1 (Analytics) initialized:")
        print(f"  - Substrate epsilon_r: {self.epsilon_r}")
        print(f"  - Effective epsilon_eff: {self.epsilon_eff:.2f}")
        print(f"  - Feedline Z0: {self.Z0} Ohm")
        print(f"  - Resonator Z_res: {self.Z_res} Ohm")
        print("-" * 20)


    def _calculate_capacitance_per_unit_length(self, coupler_width: float, coupler_spacing: float) -> float:
        """
        Calculates the mutual capacitance per unit length between two coplanar strips.

        This uses an accurate formula for the odd-mode capacitance of coupled
        coplanar strips on a dielectric half-space, which is directly related
        to the mutual capacitance.
        Formula from K. C. Gupta, "Microstrip Lines and Slotlines".

        Args:
            coupler_width (float): The width of the resonator's coupling arm (m).
            coupler_spacing (float): The gap between the feedline and the coupling arm (m).

        Returns:
            float: The mutual capacitance per unit length (F/m).
        """
        w = coupler_width
        s = coupler_spacing

        # The modulus k for the elliptic integral
        k = s / (s + 2 * w)

        # The complementary modulus k'
        k_prime = np.sqrt(1 - k**2)

        # ellipk(m) computes the complete elliptic integral K(m), where m = k^2.
        K_k = ellipk(k**2)
        K_k_prime = ellipk(k_prime**2)

        # Capacitance per unit length for coupled lines on a dielectric half-space.
        # This formula provides the mutual capacitance C_m.
        C_per_L = self.epsilon_0 * (self.epsilon_r - 1) / 2 * K_k / K_k_prime

        # An alternative and more common formula for mutual capacitance is C_m = (C_odd - C_even)/2.
        # For simplicity and robustness, we use a well-cited formula for C_m directly:
        # C_m = epsilon_0 * epsilon_eff * K(k') / K(k)
        # Let's use this one as it's more standard for the effective medium.
        C_per_L = self.epsilon_0 * self.epsilon_eff * K_k_prime / K_k

        return C_per_L

    def calculate_resonant_frequency(self, resonator_length: float) -> float:
        """
        Calculates the resonant frequency of a quarter-wave resonator.

        Formula: f₀ = v_p / (4 * L), where v_p = c / sqrt(epsilon_eff).

        Args:
            resonator_length (float): The total physical length of the resonator (m).

        Returns:
            float: The estimated resonant frequency (f₀) in Hz.
        """
        if resonator_length <= 0:
            raise ValueError("Resonator length must be positive.")

        # Phase velocity in the transmission line
        v_p = self.c / np.sqrt(self.epsilon_eff)

        # Resonant frequency for a lambda/4 resonator
        f0 = v_p / (4 * resonator_length)

        return f0

    def calculate_resonator_length(self, target_frequency: float) -> float:
        """
        Calculates the required resonator length for a target resonant frequency.

        This is the inverse of calculate_resonant_frequency.
        Formula: L = v_p / (4 * f₀), where v_p = c / sqrt(epsilon_eff).

        Args:
            target_frequency (float): The desired resonant frequency in Hz.

        Returns:
            float: The required resonator length in meters.
        """
        if target_frequency <= 0:
            raise ValueError("Target frequency must be positive.")

        # Phase velocity in the transmission line
        v_p = self.c / np.sqrt(self.epsilon_eff)

        # Required length for a lambda/4 resonator at target frequency
        resonator_length = v_p / (4 * target_frequency)

        return resonator_length

    def calculate_coupling_q(self, f0: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> tuple[float, float]:
        """
        Calculates the coupling quality factor (Qc) for a capacitively coupled resonator.

        Formula: Qc = π / (4 * ω₀² * Z₀ * Z_res * Cc²)
        where Cc is the total coupling capacitance (Cc = C_per_L * coupler_length).
        This formula is valid for an end-coupled λ/4 resonator.

        Args:
            f0 (float): The resonant frequency in Hz.
            coupler_width (float): The width of the resonator's coupling arm (m).
            coupler_spacing (float): The gap between the feedline and the coupling arm (m).
            coupler_length (float): The length of the parallel coupling section (m).

        Returns:
            tuple[float, float]: A tuple containing:
                - Qc (float): The dimensionless coupling quality factor.
                - Cc (float): The total coupling capacitance in Farads.
        """
        if not all(arg > 0 for arg in [f0, coupler_width, coupler_spacing, coupler_length]):
            raise ValueError("All input arguments must be positive.")

        # Angular resonant frequency
        omega0 = 2 * np.pi * f0

        # 1. Calculate capacitance per unit length
        C_per_L = self._calculate_capacitance_per_unit_length(coupler_width, coupler_spacing)

        # 2. Calculate total coupling capacitance
        Cc = C_per_L * coupler_length

        # 3. Calculate Coupling Q
        # This formula is derived from the power balance and equivalent circuit model
        # at resonance for a shunt-coupled resonator.
        Qc_numerator = np.pi
        Qc_denominator = 4 * (omega0**2) * self.Z0 * self.Z_res * (Cc**2)

        if Qc_denominator == 0:
            return float('inf'), Cc

        Qc = Qc_numerator / Qc_denominator

        return Qc, Cc

    def run_analysis(self, resonator_length: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> dict:
        """
        A wrapper function to perform all calculations for a given geometry.
        This is the primary entry point for the LLM.

        Args:
            resonator_length (float): Total resonator length in meters.
            coupler_width (float): Coupling arm width in meters.
            coupler_spacing (float): Gap between feedline and coupler in meters.
            coupler_length (float): Length of the coupling section in meters.

        Returns:
            dict: A dictionary containing the calculated design parameters:
                  - 'f0_GHz' (float): Resonant frequency in GHz.
                  - 'Qc' (float): Coupling quality factor.
                  - 'Cc_fF' (float): Total coupling capacitance in femto-Farads.
                  - 'C_per_L_pF_m' (float): Capacitance per unit length in pF/m.
        """
        # Calculate resonant frequency first, as it's needed for Qc
        f0_Hz = self.calculate_resonant_frequency(resonator_length)

        # Calculate coupling Q and capacitance
        Qc, Cc_F = self.calculate_coupling_q(f0_Hz, coupler_width, coupler_spacing, coupler_length)

        # Get C_per_L for reporting
        C_per_L = self._calculate_capacitance_per_unit_length(coupler_width, coupler_spacing)

        # Package results into a dictionary with convenient units
        results = {
            "f0_GHz": f0_Hz / 1e9,
            "Qc": Qc,
            "Cc_fF": Cc_F * 1e15,
            "C_per_L_pF_m": C_per_L * 1e12
        }

        return results

    def run_length_analysis(self, target_frequency: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> dict:
        """
        A wrapper function to calculate resonator length for a target frequency and then
        perform all calculations for that geometry.

        Args:
            target_frequency (float): Desired resonant frequency in Hz.
            coupler_width (float): Coupling arm width in meters.
            coupler_spacing (float): Gap between feedline and coupler in meters.
            coupler_length (float): Length of the coupling section in meters.

        Returns:
            dict: A dictionary containing the calculated design parameters:
                  - 'resonator_length_m' (float): Required resonator length in meters.
                  - 'resonator_length_mm' (float): Required resonator length in millimeters.
                  - 'f0_GHz' (float): Actual resonant frequency in GHz (should match target).
                  - 'Qc' (float): Coupling quality factor.
                  - 'Cc_fF' (float): Total coupling capacitance in femto-Farads.
                  - 'C_per_L_pF_m' (float): Capacitance per unit length in pF/m.
        """
        # Calculate required resonator length
        resonator_length = self.calculate_resonator_length(target_frequency)

        # Run full analysis with the calculated length
        results = self.run_analysis(resonator_length, coupler_width, coupler_spacing, coupler_length)

        # Add length information to results
        results.update({
            "resonator_length_m": resonator_length,
            "resonator_length_mm": resonator_length * 1e3,
            "target_frequency_Hz": target_frequency,
            "target_frequency_GHz": target_frequency / 1e9
        })

        return results

# Create analytical approximations instance
analytical_approximations = analytical_approximations_CPW()


@mcp.tool()
def calculate_resonant_frequency(resonator_length: float) -> Dict:
    """
    Calculates the resonant frequency of a quarter-wave resonator.

    Formula: f₀ = v_p / (4 * L), where v_p = c / sqrt(epsilon_eff).

    Args:
        resonator_length (float): The total physical length of the resonator (m).

    Returns:
        Dict: Dictionary containing the estimated resonant frequency (f₀) in Hz and GHz.
    """
    try:
        f0_Hz = analytical_approximations.calculate_resonant_frequency(resonator_length)
        return {
            "f0_Hz": f0_Hz,
            "f0_GHz": f0_Hz / 1e9,
            "resonator_length_m": resonator_length,
            "message": f"Calculated resonant frequency: {f0_Hz/1e9:.4f} GHz for length {resonator_length*1e3:.2f} mm"
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def calculate_resonator_length(target_frequency: float) -> Dict:
    """
    Calculates the required resonator length for a target resonant frequency.

    This is the inverse of calculate_resonant_frequency.
    Formula: L = v_p / (4 * f₀), where v_p = c / sqrt(epsilon_eff).

    Args:
        target_frequency (float): The desired resonant frequency in Hz.

    Returns:
        Dict: Dictionary containing the required resonator length in meters and millimeters.
    """
    try:
        length_m = analytical_approximations.calculate_resonator_length(target_frequency)
        return {
            "resonator_length_m": length_m,
            "resonator_length_mm": length_m * 1e3,
            "target_frequency_Hz": target_frequency,
            "target_frequency_GHz": target_frequency / 1e9,
            "message": f"Required resonator length: {length_m*1e3:.2f} mm for target frequency {target_frequency/1e9:.4f} GHz"
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def calculate_coupling_q(f0: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> Dict:
    """
    Calculates the coupling quality factor (Qc) for a capacitively coupled resonator.

    Formula: Qc = π / (4 * ω₀² * Z₀ * Z_res * Cc²)
    where Cc is the total coupling capacitance (Cc = C_per_L * coupler_length).
    This formula is valid for an end-coupled λ/4 resonator.

    Args:
        f0 (float): The resonant frequency in Hz.
        coupler_width (float): The width of the resonator's coupling arm (m).
        coupler_spacing (float): The gap between the feedline and the coupling arm (m).
        coupler_length (float): The length of the parallel coupling section (m).

    Returns:
        Dict: Dictionary containing the dimensionless coupling quality factor (Qc) and total coupling capacitance (Cc) in Farads.
    """
    try:
        Qc, Cc = analytical_approximations.calculate_coupling_q(f0, coupler_width, coupler_spacing, coupler_length)
        return {
            "Qc": Qc,
            "Cc_F": Cc,
            "Cc_fF": Cc * 1e15,
            "f0_Hz": f0,
            "f0_GHz": f0 / 1e9,
            "coupler_width_um": coupler_width * 1e6,
            "coupler_spacing_um": coupler_spacing * 1e6,
            "coupler_length_um": coupler_length * 1e6,
            "message": f"Calculated Qc: {Qc:.2f}, Cc: {Cc*1e15:.4f} fF"
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_analysis(resonator_length: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> Dict:
    """
    A wrapper function to perform all calculations for a given geometry.
    This is the primary entry point for the LLM.

    Args:
        resonator_length (float): Total resonator length in meters.
        coupler_width (float): Coupling arm width in meters.
        coupler_spacing (float): Gap between feedline and coupler in meters.
        coupler_length (float): Length of the coupling section in meters.

    Returns:
        Dict: A dictionary containing the calculated design parameters:
              - 'f0_GHz' (float): Resonant frequency in GHz.
              - 'Qc' (float): Coupling quality factor.
              - 'Cc_fF' (float): Total coupling capacitance in femto-Farads.
              - 'C_per_L_pF_m' (float): Capacitance per unit length in pF/m.
    """
    try:
        results = analytical_approximations.run_analysis(
            resonator_length, coupler_width, coupler_spacing, coupler_length
        )

        # Add input parameters to results for reference
        results.update({
            "input_resonator_length_m": resonator_length,
            "input_resonator_length_mm": resonator_length * 1e3,
            "input_coupler_width_m": coupler_width,
            "input_coupler_width_um": coupler_width * 1e6,
            "input_coupler_spacing_m": coupler_spacing,
            "input_coupler_spacing_um": coupler_spacing * 1e6,
            "input_coupler_length_m": coupler_length,
            "input_coupler_length_um": coupler_length * 1e6,
            "message": f"Analysis complete: f₀={results['f0_GHz']:.4f} GHz, Qc={results['Qc']:.2f}"
        })

        return results
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def run_length_analysis(target_frequency: float, coupler_width: float, coupler_spacing: float, coupler_length: float) -> Dict:
    """
    A wrapper function to calculate resonator length for a target frequency and then
    perform all calculations for that geometry.

    Args:
        target_frequency (float): Desired resonant frequency in Hz.
        coupler_width (float): Coupling arm width in meters.
        coupler_spacing (float): Gap between feedline and coupler in meters.
        coupler_length (float): Length of the coupling section in meters.

    Returns:
        Dict: A dictionary containing the calculated design parameters including required length.
    """
    try:
        results = analytical_approximations.run_length_analysis(
            target_frequency, coupler_width, coupler_spacing, coupler_length
        )

        # Add input parameters to results for reference
        results.update({
            "input_target_frequency_Hz": target_frequency,
            "input_target_frequency_GHz": target_frequency / 1e9,
            "input_coupler_width_m": coupler_width,
            "input_coupler_width_um": coupler_width * 1e6,
            "input_coupler_spacing_m": coupler_spacing,
            "input_coupler_spacing_um": coupler_spacing * 1e6,
            "input_coupler_length_m": coupler_length,
            "input_coupler_length_um": coupler_length * 1e6,
            "message": f"Length analysis complete: L={results['resonator_length_mm']:.2f} mm for f₀={results['target_frequency_GHz']:.4f} GHz, Qc={results['Qc']:.2f}"
        })

        return results
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_system_parameters() -> Dict:
    """
    Returns the current system parameters used in the analytical model.

    Returns:
        Dict: Dictionary containing the system parameters including substrate properties and impedances.
    """
    return {
        "epsilon_r": analytical_approximations.epsilon_r,
        "epsilon_eff": analytical_approximations.epsilon_eff,
        "Z0_feedline": analytical_approximations.Z0,
        "Z_res_resonator": analytical_approximations.Z_res,
        "speed_of_light": analytical_approximations.c,
        "epsilon_0": analytical_approximations.epsilon_0,
        "message": f"System: εᵣ={analytical_approximations.epsilon_r}, εₑff={analytical_approximations.epsilon_eff:.2f}, Z₀={analytical_approximations.Z0}Ω"
    }


if __name__ == "__main__":
    print("Starting Analytical Approximations MCP Server...")
    print(f"Initialized with substrate εᵣ={analytical_approximations.epsilon_r}")
    print(f"Effective permittivity εₑff={analytical_approximations.epsilon_eff:.2f}")
    print(f"Feedline impedance Z₀={analytical_approximations.Z0}Ω")
    print(f"Resonator impedance Zᵣₑₛ={analytical_approximations.Z_res}Ω")
    mcp.run(transport="stdio")
