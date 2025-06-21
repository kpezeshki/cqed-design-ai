#!/usr/bin/env python3

"""
Test script to validate the new resonator length calculation functionality.
This script tests the inverse relationship between calculate_resonant_frequency
and calculate_resonator_length methods.
"""

import sys
import os

# Add the current directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from analytical_approximations_MCP import analytical_approximations_CPW
    import numpy as np

    def test_length_frequency_inverse():
        """Test that length and frequency calculations are proper inverses."""
        print("Testing inverse relationship between length and frequency calculations...")
        print("=" * 60)

        # Create instance with default parameters
        calc = analytical_approximations_CPW()

        # Test cases: various resonator lengths
        test_lengths = [0.001, 0.002, 0.005, 0.010, 0.020]  # 1mm to 20mm

        print("Forward calculation (Length -> Frequency -> Length):")
        print("-" * 50)

        for original_length in test_lengths:
            # Calculate frequency from length
            frequency = calc.calculate_resonant_frequency(original_length)

            # Calculate length back from frequency
            calculated_length = calc.calculate_resonator_length(frequency)

            # Check if we get back the original length
            relative_error = abs(calculated_length - original_length) / original_length * 100

            print(f"Original length: {original_length*1000:.2f} mm")
            print(f"Calculated frequency: {frequency/1e9:.4f} GHz")
            print(f"Calculated length: {calculated_length*1000:.2f} mm")
            print(f"Relative error: {relative_error:.6f}%")
            print()

            # Assert the error is very small (numerical precision)
            assert relative_error < 1e-10, f"Large error: {relative_error}%"

        print("✓ All forward calculations passed!")
        print()

        # Test cases: various target frequencies
        test_frequencies = [1e9, 2e9, 5e9, 10e9, 15e9]  # 1 GHz to 15 GHz

        print("Reverse calculation (Frequency -> Length -> Frequency):")
        print("-" * 50)

        for original_frequency in test_frequencies:
            # Calculate length from frequency
            length = calc.calculate_resonator_length(original_frequency)

            # Calculate frequency back from length
            calculated_frequency = calc.calculate_resonant_frequency(length)

            # Check if we get back the original frequency
            relative_error = abs(calculated_frequency - original_frequency) / original_frequency * 100

            print(f"Original frequency: {original_frequency/1e9:.2f} GHz")
            print(f"Calculated length: {length*1000:.2f} mm")
            print(f"Calculated frequency: {calculated_frequency/1e9:.4f} GHz")
            print(f"Relative error: {relative_error:.6f}%")
            print()

            # Assert the error is very small (numerical precision)
            assert relative_error < 1e-10, f"Large error: {relative_error}%"

        print("✓ All reverse calculations passed!")
        print()

    def test_run_length_analysis():
        """Test the run_length_analysis wrapper function."""
        print("Testing run_length_analysis wrapper function...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test parameters
        target_freq = 5e9  # 5 GHz
        coupler_width = 10e-6  # 10 μm
        coupler_spacing = 5e-6  # 5 μm
        coupler_length = 100e-6  # 100 μm

        # Run the analysis
        results = calc.run_length_analysis(target_freq, coupler_width, coupler_spacing, coupler_length)

        print("Input parameters:")
        print(f"  Target frequency: {target_freq/1e9:.2f} GHz")
        print(f"  Coupler width: {coupler_width*1e6:.1f} μm")
        print(f"  Coupler spacing: {coupler_spacing*1e6:.1f} μm")
        print(f"  Coupler length: {coupler_length*1e6:.1f} μm")
        print()

        print("Results:")
        print(f"  Required length: {results['resonator_length_mm']:.2f} mm")
        print(f"  Actual frequency: {results['f0_GHz']:.4f} GHz")
        print(f"  Coupling Q: {results['Qc']:.2f}")
        print(f"  Coupling capacitance: {results['Cc_fF']:.4f} fF")
        print()

        # Verify the frequency matches our target
        freq_error = abs(results['f0_GHz'] - target_freq/1e9) / (target_freq/1e9) * 100
        print(f"Frequency accuracy: {freq_error:.6f}% error")
        assert freq_error < 1e-10, f"Frequency mismatch: {freq_error}%"

        print("✓ run_length_analysis test passed!")
        print()

    def test_edge_cases():
        """Test edge cases and error handling."""
        print("Testing edge cases and error handling...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test negative frequency
        try:
            calc.calculate_resonator_length(-1e9)
            assert False, "Should have raised ValueError for negative frequency"
        except ValueError as e:
            print(f"✓ Correctly caught negative frequency: {e}")

        # Test zero frequency
        try:
            calc.calculate_resonator_length(0)
            assert False, "Should have raised ValueError for zero frequency"
        except ValueError as e:
            print(f"✓ Correctly caught zero frequency: {e}")

        # Test very high frequency (should give very small length)
        high_freq = 100e9  # 100 GHz
        length = calc.calculate_resonator_length(high_freq)
        print(f"✓ High frequency test: {high_freq/1e9:.0f} GHz -> {length*1e6:.2f} μm")

        # Test very low frequency (should give very large length)
        low_freq = 1e6  # 1 MHz
        length = calc.calculate_resonator_length(low_freq)
        print(f"✓ Low frequency test: {low_freq/1e6:.0f} MHz -> {length:.3f} m")

        print("✓ All edge case tests passed!")
        print()

    def main():
        """Run all tests."""
        print("RESONATOR LENGTH CALCULATION VALIDATION")
        print("=" * 60)

        try:
            test_length_frequency_inverse()
            test_run_length_analysis()
            test_edge_cases()

            print("🎉 ALL TESTS PASSED! 🎉")
            print("The new resonator length calculation functionality is working correctly.")

        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            return 1

        return 0

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("  - numpy")
    print("  - scipy")
    print("  - mcp (if running MCP tools)")
    sys.exit(1)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
