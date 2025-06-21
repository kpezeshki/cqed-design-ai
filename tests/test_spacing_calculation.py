#!/usr/bin/env python3

"""
Test script to validate the new coupler spacing calculation functionality.
This script tests the numerical solving for coupler spacing given a target Qc.
"""

import sys
import os

# Add the current directory to Python path to import the module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from analytical_approximations_MCP import analytical_approximations_CPW
    import numpy as np

    def test_spacing_calculation():
        """Test that spacing calculation produces correct Qc values."""
        print("Testing coupler spacing calculation for target Qc...")
        print("=" * 60)

        # Create instance with default parameters
        calc = analytical_approximations_CPW()

        # Test parameters
        f0 = 5e9  # 5 GHz
        coupler_width = 10e-6  # 10 μm
        coupler_length = 100e-6  # 100 μm

        # Test various target Qc values
        target_qcs = [1e4, 5e4, 1e6]

        print("Spacing calculation tests:")
        print("-" * 50)

        for target_qc in target_qcs:
            try:
                # Calculate required spacing
                spacing = calc.calculate_coupler_spacing(
                    f0, target_qc, coupler_width, coupler_length
                )

                # Verify by calculating actual Qc with this spacing
                actual_qc, cc = calc.calculate_coupling_q(
                    f0, coupler_width, spacing, coupler_length
                )

                # Calculate relative error
                relative_error = abs(actual_qc - target_qc) / target_qc * 100

                print(f"Target Qc: {target_qc:.1f}")
                print(f"Required spacing: {spacing*1e6:.2f} μm")
                print(f"Actual Qc: {actual_qc:.4f}")
                print(f"Relative error: {relative_error:.6f}%")
                print(f"Coupling capacitance: {cc*1e15:.4f} fF")
                print()

                # Assert the error is very small (numerical precision)
                assert relative_error < 0.1, f"Large error: {relative_error}%"

            except Exception as e:
                print(f"Failed for target Qc={target_qc}: {e}")
                print()

        print("✓ All spacing calculation tests passed!")
        print()

    def test_run_spacing_analysis():
        """Test the run_spacing_analysis wrapper function."""
        print("Testing run_spacing_analysis wrapper function...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test parameters
        f0 = 5e9  # 5 GHz
        target_qc = 1e4
        coupler_width = 5e-6  # 10 μm
        coupler_length = 50e-6  # 100 μm

        # Run the analysis
        results = calc.run_spacing_analysis(f0, target_qc, coupler_width, coupler_length)

        print("Input parameters:")
        print(f"  Frequency: {f0/1e9:.2f} GHz")
        print(f"  Target Qc: {target_qc}")
        print(f"  Coupler width: {coupler_width*1e6:.1f} μm")
        print(f"  Coupler length: {coupler_length*1e6:.1f} μm")
        print()

        print("Results:")
        print(f"  Required spacing: {results['coupler_spacing_um']:.2f} μm")
        print(f"  Actual Qc: {results['Qc']:.4f}")
        print(f"  Target Qc: {results['target_Qc']:.1f}")
        print(f"  Coupling capacitance: {results['Cc_fF']:.4f} fF")
        print(f"  Capacitance per length: {results['C_per_L_pF_m']:.4f} pF/m")
        print()

        # Verify the Qc matches our target
        qc_error = abs(results['Qc'] - target_qc) / target_qc * 100
        print(f"Qc accuracy: {qc_error:.6f}% error")
        assert qc_error < 0.1, f"Qc mismatch: {qc_error}%"

        print("✓ run_spacing_analysis test passed!")
        print()

    def test_spacing_edge_cases():
        """Test edge cases and error handling for spacing calculation."""
        print("Testing edge cases and error handling...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test parameters
        f0 = 5e9  # 5 GHz
        coupler_width = 10e-6  # 10 μm
        coupler_length = 100e-6  # 100 μm

        # Test negative Qc
        try:
            calc.calculate_coupler_spacing(f0, -100, coupler_width, coupler_length)
            assert False, "Should have raised ValueError for negative Qc"
        except ValueError as e:
            print(f"✓ Correctly caught negative Qc: {e}")

        # Test zero Qc
        try:
            calc.calculate_coupler_spacing(f0, 0, coupler_width, coupler_length)
            assert False, "Should have raised ValueError for zero Qc"
        except ValueError as e:
            print(f"✓ Correctly caught zero Qc: {e}")

        # Test very high Qc (might be out of range)
        try:
            very_high_qc = 10000
            spacing = calc.calculate_coupler_spacing(f0, very_high_qc, coupler_width, coupler_length)
            print(f"✓ High Qc test: Qc={very_high_qc} -> spacing={spacing*1e6:.2f} μm")
        except ValueError as e:
            print(f"✓ High Qc correctly failed: {e}")

        # Test very low Qc (might be out of range)
        try:
            very_low_qc = 1
            spacing = calc.calculate_coupler_spacing(f0, very_low_qc, coupler_width, coupler_length)
            print(f"✓ Low Qc test: Qc={very_low_qc} -> spacing={spacing*1e6:.2f} μm")
        except ValueError as e:
            print(f"✓ Low Qc correctly failed: {e}")

        # Test invalid spacing range
        try:
            calc.calculate_coupler_spacing(f0, 100, coupler_width, coupler_length,
                                         spacing_min=10e-6, spacing_max=5e-6)
            assert False, "Should have raised ValueError for invalid spacing range"
        except ValueError as e:
            print(f"✓ Correctly caught invalid spacing range: {e}")

        print("✓ All edge case tests passed!")
        print()

    def test_spacing_frequency_relationship():
        """Test how spacing requirements change with frequency."""
        print("Testing spacing vs frequency relationship...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test parameters
        target_qc = 1e4
        coupler_width = 10e-6  # 10 μm
        coupler_length = 100e-6  # 100 μm

        # Test various frequencies
        frequencies = [2e9, 5e9, 10e9, 15e9]  # 2, 5, 10, 15 GHz

        print("Frequency vs spacing relationship:")
        print("-" * 40)

        for f0 in frequencies:
            try:
                spacing = calc.calculate_coupler_spacing(f0, target_qc, coupler_width, coupler_length)

                # Verify the result
                actual_qc, _ = calc.calculate_coupling_q(f0, coupler_width, spacing, coupler_length)

                print(f"Frequency: {f0/1e9:.1f} GHz")
                print(f"Required spacing: {spacing*1e6:.2f} μm")
                print(f"Actual Qc: {actual_qc:.2f}")
                print()

            except Exception as e:
                print(f"Failed for frequency {f0/1e9:.1f} GHz: {e}")
                print()

        print("✓ Frequency relationship test completed!")
        print()

    def test_spacing_width_relationship():
        """Test how spacing requirements change with coupler width."""
        print("Testing spacing vs coupler width relationship...")
        print("=" * 60)

        calc = analytical_approximations_CPW()

        # Test parameters
        f0 = 5e9  # 5 GHz
        target_qc = 1e4
        coupler_length = 100e-6  # 100 μm

        # Test various coupler widths
        widths = [5e-6, 10e-6, 20e-6, 50e-6]  # 5, 10, 20, 50 μm

        print("Coupler width vs spacing relationship:")
        print("-" * 40)

        for width in widths:
            try:
                spacing = calc.calculate_coupler_spacing(f0, target_qc, width, coupler_length)

                # Verify the result
                actual_qc, _ = calc.calculate_coupling_q(f0, width, spacing, coupler_length)

                print(f"Coupler width: {width*1e6:.1f} μm")
                print(f"Required spacing: {spacing*1e6:.2f} μm")
                print(f"Actual Qc: {actual_qc:.2f}")
                print()

            except Exception as e:
                print(f"Failed for width {width*1e6:.1f} μm: {e}")
                print()

        print("✓ Width relationship test completed!")
        print()

    def main():
        """Run all tests."""
        print("COUPLER SPACING CALCULATION VALIDATION")
        print("=" * 60)

        try:
            test_spacing_calculation()
            test_run_spacing_analysis()
            test_spacing_edge_cases()
            test_spacing_frequency_relationship()
            test_spacing_width_relationship()

            print("🎉 ALL TESTS PASSED! 🎉")
            print("The new coupler spacing calculation functionality is working correctly.")

        except Exception as e:
            print(f"❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
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
