import unittest
import numpy as np
import astropy.units as u
from gordian.conversions import (
    convert_wave,
    convert_flux,
    return_maggie_flux_unit,
    return_cgs_flux_unit,
    return_si_flux_unit,
    MAGGIE_UNIT
)


class TestWavelengthConversions(unittest.TestCase):
    """Test wavelength unit conversions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wave_um = np.arange(1, 10)
    
    def test_microns_to_angstroms(self):
        """Test conversion from microns to Angstroms"""
        wave_A = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.AA, return_quantity=True)
        
        # Check type
        self.assertIsInstance(wave_A, u.Quantity)
        self.assertEqual(wave_A.unit, u.AA)
        
        # Check values (1 micron = 10,000 Angstroms)
        expected = self.wave_um * 10000
        np.testing.assert_array_almost_equal(wave_A.value, expected)
    
    def test_microns_to_meters(self):
        """Test conversion from microns to meters"""
        wave_m = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.m, return_quantity=True)
        
        # Check type
        self.assertIsInstance(wave_m, u.Quantity)
        self.assertEqual(wave_m.unit, u.m)
        
        # Check values (1 micron = 1e-6 meters)
        expected = self.wave_um * 1e-6
        np.testing.assert_array_almost_equal(wave_m.value, expected)
    
    def test_return_value_mode(self):
        """Test that return_quantity=False returns plain numpy array"""
        wave_A = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.AA, return_quantity=False)
        
        # Should be numpy array, not Quantity
        self.assertIsInstance(wave_A, np.ndarray)
        self.assertNotIsInstance(wave_A, u.Quantity)


class TestFluxConversions(unittest.TestCase):
    """Test flux unit conversions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maggie_unit = return_maggie_flux_unit()
        self.cgs_unit = return_cgs_flux_unit()
        self.si_unit = return_si_flux_unit()
        self.wave_um = np.arange(1, 10)
        self.wave_A = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.AA, return_quantity=True)
        self.wave_m = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.m, return_quantity=True)
    
    def test_si_to_si_no_conversion(self):
        """Test that SI to SI returns same values"""
        flux_si = np.arange(1, 10)
        flux_out, err_out = convert_flux(flux_in=flux_si, in_unit=self.si_unit, out_unit=self.si_unit)
        
        np.testing.assert_array_almost_equal(flux_out, flux_si)
        self.assertIsNone(err_out)
    
    def test_si_to_cgs(self):
        """Test conversion from SI to CGS units"""
        flux_si = np.arange(1, 10)
        flux_cgs, err_cgs = convert_flux(flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit)
        
        # Check that conversion occurred
        self.assertIsNotNone(flux_cgs)
        self.assertIsNone(err_cgs)
        
        # SI to CGS: W/m^3 to erg/(s·cm^2·Å)
        # 1 W/m^3 = 1e-7 erg/(s·cm^2·Å)
        expected = flux_si * 1e-7
        np.testing.assert_array_almost_equal(flux_cgs, expected)
    
    def test_cgs_to_jy_requires_wavelength(self):
        """Test that CGS to Jy conversion requires wavelength"""
        flux_cgs = np.array([1.0, 2.0, 3.0])
        
        # Should raise error without wavelength
        with self.assertRaises(ValueError):
            convert_flux(flux_in=flux_cgs, in_unit=self.cgs_unit, out_unit=u.Jy, return_quantity=True)
    
    def test_cgs_to_jy_with_wavelength(self):
        """Test CGS to Jy conversion with wavelength"""
        flux_si = np.arange(1, 10)
        flux_cgs, _ = convert_flux(flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit)
        
        flux_jy, err_jy = convert_flux(
            flux_in=flux_cgs, in_unit=self.cgs_unit, out_unit=u.Jy,
            wave=self.wave_A, return_quantity=True
        )
        
        # Check type
        self.assertIsInstance(flux_jy, u.Quantity)
        self.assertEqual(flux_jy.unit, u.Jy)
        self.assertIsNone(err_jy)
    
    def test_return_quantity_mode(self):
        """Test return_quantity parameter"""
        flux_si = np.array([1.0, 2.0, 3.0])
        
        # With return_quantity=True
        flux_qty, err_qty = convert_flux(
            flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit,
            return_quantity=True
        )
        self.assertIsInstance(flux_qty, u.Quantity)
        
        # With return_quantity=False
        flux_val, err_val = convert_flux(
            flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit,
            return_quantity=False
        )
        self.assertIsInstance(flux_val, np.ndarray)
        self.assertNotIsInstance(flux_val, u.Quantity)


class TestMagnitudeConversions(unittest.TestCase):
    """Test magnitude conversions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maggie_unit = return_maggie_flux_unit()
        self.cgs_unit = return_cgs_flux_unit()
        self.si_unit = return_si_flux_unit()
        self.wave_um = np.arange(1, 10)
        self.wave_A = convert_wave(wave_in=self.wave_um, in_unit=u.um, out_unit=u.AA, return_quantity=True)
    
    def test_cgs_to_magnitude_requires_wavelength(self):
        """Test that CGS to magnitude conversion requires wavelength"""
        flux_cgs = np.array([1e-7, 2e-7, 3e-7])
        
        # Should raise error without wavelength
        with self.assertRaises(ValueError):
            convert_flux(flux_in=flux_cgs, in_unit=self.cgs_unit, out_unit=u.mag, return_quantity=True)
    
    def test_cgs_to_magnitude_with_wavelength(self):
        """Test CGS to magnitude conversion with wavelength"""
        flux_si = np.arange(1, 10)
        flux_cgs, _ = convert_flux(flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit)
        
        flux_mag, err_mag = convert_flux(
            flux_in=flux_cgs, in_unit=self.cgs_unit, out_unit=u.mag,
            wave=self.wave_A, return_quantity=True
        )
        
        # Check type
        self.assertIsInstance(flux_mag, u.Quantity)
        self.assertEqual(flux_mag.unit, u.mag)
        self.assertIsNone(err_mag)
    
    def test_jy_to_magnitude_roundtrip(self):
        """Test that Jy → mag → Jy gives consistent results"""
        flux_jy_original = np.array([100.0, 200.0, 300.0])
        
        # Convert Jy → mag
        flux_mag, _ = convert_flux(
            flux_in=flux_jy_original, in_unit=u.Jy, out_unit=u.mag,
            return_quantity=True
        )
        
        # Convert mag → Jy
        flux_jy_recovered, _ = convert_flux(
            flux_in=flux_mag.value, in_unit=u.mag, out_unit=u.Jy,
            return_quantity=True
        )
        
        # Should match original within numerical precision
        np.testing.assert_array_almost_equal(
            flux_jy_recovered.value, flux_jy_original, decimal=10
        )
    
    def test_cgs_to_maggie_with_wavelength(self):
        """Test CGS to maggie conversion"""
        flux_si = np.arange(1, 10)
        flux_cgs, _ = convert_flux(flux_in=flux_si, in_unit=self.si_unit, out_unit=self.cgs_unit)
        
        flux_maggie, err_maggie = convert_flux(
            flux_in=flux_cgs, in_unit=self.cgs_unit, out_unit=self.maggie_unit,
            wave=self.wave_A, return_quantity=True
        )
        
        # Check that maggie can be converted to Jy
        flux_jy = flux_maggie.to(u.Jy)
        self.assertIsInstance(flux_jy, u.Quantity)
        self.assertEqual(flux_jy.unit, u.Jy)


class TestErrorPropagation(unittest.TestCase):
    """Test flux error propagation"""
    
    def test_jy_to_magnitude_with_errors(self):
        """Test error propagation for Jy to magnitude conversion"""
        flux_with_err = np.array([100.0, 200.0, 300.0])
        err_with_err = np.array([10.0, 15.0, 20.0])
        
        flux_mag_out, err_mag_out = convert_flux(
            flux_in=flux_with_err, in_unit=u.Jy, out_unit=u.mag,
            err_in=err_with_err, return_quantity=True
        )
        
        # Check that errors were propagated
        self.assertIsNotNone(err_mag_out)
        self.assertIsInstance(err_mag_out, u.Quantity)
        self.assertEqual(err_mag_out.unit, u.mag)
        
        # Error should be positive
        self.assertTrue(np.all(err_mag_out.value > 0))
    
    def test_magnitude_to_jy_with_errors(self):
        """Test error propagation for magnitude to Jy conversion"""
        flux_mag = np.array([15.0, 16.0, 17.0])
        err_mag = np.array([0.1, 0.15, 0.2])
        
        flux_jy_out, err_jy_out = convert_flux(
            flux_in=flux_mag, in_unit=u.mag, out_unit=u.Jy,
            err_in=err_mag, return_quantity=True
        )
        
        # Check that errors were propagated
        self.assertIsNotNone(err_jy_out)
        self.assertIsInstance(err_jy_out, u.Quantity)
        self.assertEqual(err_jy_out.unit, u.Jy)
        
        # Error should be positive
        self.assertTrue(np.all(err_jy_out.value > 0))
    
    def test_no_error_returns_none(self):
        """Test that when no error is provided, None is returned"""
        flux_only, err_only = convert_flux(
            flux_in=100.0, in_unit=u.Jy, out_unit=u.mag,
            return_quantity=True
        )
        
        self.assertIsNotNone(flux_only)
        self.assertIsNone(err_only)
    
    def test_simple_conversion_with_errors(self):
        """Test error propagation for simple unit conversions"""
        flux_jy = np.array([100.0, 200.0, 300.0])
        err_jy = np.array([10.0, 15.0, 20.0])
        
        flux_ujy, err_ujy = convert_flux(
            flux_in=flux_jy, in_unit=u.Jy, out_unit=u.uJy,
            err_in=err_jy, return_quantity=True
        )
        
        # Check that errors scaled correctly (1 Jy = 1e6 uJy)
        np.testing.assert_array_almost_equal(err_ujy.value, err_jy * 1e6)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cgs_unit = return_cgs_flux_unit()
    
    def test_single_value_conversion(self):
        """Test conversion with single scalar value"""
        flux_jy = 100.0
        
        flux_mag, _ = convert_flux(
            flux_in=flux_jy, in_unit=u.Jy, out_unit=u.mag,
            return_quantity=True
        )
        
        # Should work with scalar
        self.assertIsInstance(flux_mag, u.Quantity)
    
    def test_zero_flux_magnitude(self):
        """Test that zero flux raises appropriate error or warning"""
        # Zero flux in magnitudes is problematic (log of zero)
        flux_jy = np.array([0.0, 100.0, 200.0])
        
        # This should handle zero flux appropriately
        # (either raise error or return inf/-inf)
        try:
            flux_mag, _ = convert_flux(
                flux_in=flux_jy, in_unit=u.Jy, out_unit=u.mag,
                return_quantity=True
            )
            # If it doesn't raise, check for inf
            self.assertTrue(np.isinf(flux_mag.value[0]))
        except (ValueError, RuntimeWarning):
            # Either error or warning is acceptable
            pass
    
    def test_negative_flux_magnitude(self):
        """Test that negative flux in magnitude conversion is handled"""
        flux_jy = np.array([-100.0, 100.0, 200.0])
        
        # Negative flux should cause issues with log
        try:
            flux_mag, _ = convert_flux(
                flux_in=flux_jy, in_unit=u.Jy, out_unit=u.mag,
                return_quantity=True
            )
            # If it doesn't raise, check for NaN
            self.assertTrue(np.isnan(flux_mag.value[0]))
        except (ValueError, RuntimeWarning):
            # Either error or warning is acceptable
            pass


class TestConsistency(unittest.TestCase):
    """Test consistency between different conversion paths"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.maggie_unit = return_maggie_flux_unit()
        self.cgs_unit = return_cgs_flux_unit()
        self.wave_A = convert_wave(
            wave_in=np.array([5000.0]), in_unit=u.AA, out_unit=u.AA, return_quantity=True
        )
    
    def test_multi_step_conversion_consistency(self):
        """Test that multi-step conversions give same result as direct"""
        flux_jy = np.array([1000.0])
        
        # Direct: Jy → mag
        flux_mag_direct, _ = convert_flux(
            flux_in=flux_jy, in_unit=u.Jy, out_unit=u.mag,
            return_quantity=True
        )
        
        # Multi-step: Jy → maggie → mag
        flux_maggie, _ = convert_flux(
            flux_in=flux_jy, in_unit=u.Jy, out_unit=self.maggie_unit,
            return_quantity=True
        )
        flux_mag_multi, _ = convert_flux(
            flux_in=flux_maggie.value, in_unit=self.maggie_unit, out_unit=u.mag,
            return_quantity=True
        )
        
        # Should be approximately equal
        np.testing.assert_array_almost_equal(
            flux_mag_direct.value, flux_mag_multi.value, decimal=10
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)