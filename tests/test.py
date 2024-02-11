# 
# Copyright (C) 2024  Lorenzo Pierfederici

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#

import sys
import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
# if the test is running in interactive mode (-i)
if sys.flags.interactive:
    plt.ion()

# This adds src to the list of directories the interpreter will search
# for the required module. main.py mustn't be moved from src
# directory
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from src.main import VoltageData

class TestVoltageData(unittest.TestCase):
    """Class for unit-testing."""

    def setUp(self, sample_size=10):
        self.sample_size = sample_size
        self.t = np.linspace(0., 2., self.sample_size)
        self.v = np.random.uniform(0.5, 1.5, self.sample_size)
        self.v_err = np.repeat(0.05, self.sample_size)

    def load_from_sample_arrays(self):
        """Loads data from sample arrays."""
        return VoltageData(self.t, self.v), \
            VoltageData(self.t, self.v, self.v_err)
    
    def test_constructor(self):
        """Test the constructor. Note that we access a private member, for
        testing purpose. This is generally not a good practice.
        """
        v_data, v_data_with_errs = self.load_from_sample_arrays()
        # _.data.shape gives (nrows, ncolumns)
        self.assertEqual(v_data._data.shape, (self.sample_size, 2))
        self.assertEqual(v_data_with_errs._data.shape, (self.sample_size, 3))

    def test_num_columns(self):
        """Test the number of columns."""
        v_data, v_data_with_errs = self.load_from_sample_arrays()
        # Two columns for the data without errors, three for data with errors
        self.assertEqual(v_data.num_columns(), 2)
        self.assertEqual(v_data_with_errs.num_columns(), 3)
    
    def _test_attributes(self, *attributes, expected_size=None):
        """Private workhorse function for testing the attribute in a loop,
        to be called by the actual test functions (notice the _ in front of the
        name)."""
        if expected_size is None:
            expected_size = self.sample_size
        for attr in attributes:
            # Test that the attribute is a numpy array
            self.assertTrue(isinstance(attr, np.ndarray))
            # Test the sape: they must be 1D with the right lenght
            self.assertEqual(attr.shape[0], expected_size)

    def test_attributes(self):
        """Test that timestamps, voltages and voltages_err return the right
        thing. In case of data without error test that the correct exception
        is raised."""
        v_data, v_data_with_errs = self.load_from_sample_arrays()
        # Test that timestamps and voltages wrok
        self._test_attributes(v_data.timestamps,
                              v_data.voltages)
        # Test that accessing voltage_errs on a istance of the data without
        # errors trigger a AttributeError. We use a lambda function, since
        # assertRaises() requires a callable as second argument:
        # https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertRaises
        self.assertRaises(AttributeError, lambda v : v.voltage_errs, v_data)
        # Test the three attributes on a istance with errors
        self._test_attributes(v_data_with_errs.timestamps,
                              v_data_with_errs.voltages,
                              v_data_with_errs.voltage_errs)

    def test_random_access(self):
        """Test __getitem__"""
        v_data, v_data_with_errs = self.load_from_sample_arrays()
        # Test the first value of the first column (timestamps)
        self.assertAlmostEqual(v_data[0,0], self.t[0])
        # Test the last value of the errors column
        self.assertAlmostEqual(
           v_data_with_errs[v_data_with_errs.num_rows() - 1, 2], self.v_err[-1])
        # Test slicing
        self.assertEqual(v_data[1:5, :].shape, (4, v_data.num_columns()))
        self.assertEqual(v_data_with_errs[: , 1:].shape,
                         (v_data_with_errs.num_rows(), 2))

    def test_iteration(self):
        """Test __iter__"""
        v_data, v_data_with_errs = self.load_from_sample_arrays()
        # A row must be a numpy array with size 2 or 3
        self._test_attributes(*(row for row in v_data),
                              expected_size=v_data.num_columns())
        self._test_attributes(*(row for row in v_data_with_errs),
                              expected_size=v_data_with_errs.num_columns())

if __name__ == '__main__':
    unittest.main()