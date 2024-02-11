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

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt

class VoltageData:

    """Class for handling voltage-time data.

    Attributes
    ----------
    times : array of float64
            Array of time values.
    
    voltages : array of float64
               Array of voltage values.

    voltage_errors : array of float64, optional
                     Array of voltage uncertainties, defaults to None.
    """

    def __init__(self, times, voltages, voltage_errors=None):
        """Class constructor."""
        t = np.array(times, dtype="float64")
        v = np.array(voltages, dtype="float64")
        self._data = np.column_stack((t, v))
        if voltage_errors is not None:
            v_err = np.array(voltage_errors, dtype="float64")
            self._data = np.column_stack((self._data, v_err))

    @property
    def timestamps(self):
        """Returns times."""
        return self._data[:, 0]

    @property
    def voltages(self):
        """Returns voltages."""
        return self._data[:, 1]

    @property
    def voltage_errs(self):
        """Return the voltage errors as a numpy array."""
        # If there is no error column, numpy will raise a IndexError exception.
        # This is not very explicative for the user: instead we want to raise
        # an AttributeError, which is what you usually get when you call obj.x
        # and x doesn't exist. So we catch the exception and raise another one.
        try:
            return self._data[:, 2]
        except IndexError:
            err_msg = "The optional column 'voltage_errs' is not present."
            raise AttributeError(err_msg)

    @classmethod
    def from_file(cls, file_path):

        """Classmethod for reading values from a .txt file. Time and voltage values mustn't be blank. If a value in a row isn't a float all the values in the same row won't be red.
        
        Attributes
        ----------
        file_path : str
                    Path of the source file.
        
        Returns
        -------
        t, v : lists of float
               Values that initialize the class.
        """

        t = []
        v = []
        dv = []
        with open(file_path, encoding="utf-8") as file:
            for i, line in enumerate(file):
                if line.startswith("#"):
                    continue
                values = line.strip().strip("\n").split("\t")
                try:
                    t_value = float(values[0])
                    v_value = float(values[1])
                    t.append(t_value)
                    v.append(v_value)
                    dv_value = float(values[2])
                    dv.append(dv_value)
                except IndexError:
                    continue
                except ValueError as e:
                    print(f"Line {i} error: {e}\n")
                    continue
            if len(dv) != 0:
                return cls(t, v, dv)
            return cls(t, v)

    def num_rows(self):
        """Number of rows."""
        return self._data.shape[0]

    def num_columns(self):
        """Number of columns (can be 2 or 3)."""
        return self._data.shape[1]

    def __len__(self):
        """Number of data points (or rows in the file, which is the same)."""
        return self.num_rows()

    def __getitem__(self, index):
        """Gets the values in the index-th row."""
        # We use composition and simply call __getitem__ from _data
        return self._data[index]

    def __iter__(self):
        """Returns the values row by row."""
        # We use a generator expression here. The syntax is very readible!
        for i in range(len(self)):
            yield self._data[i, :]

    def __call__(self, t):
        """Returns the interpolated value of voltage at time t."""
        spline = self.spline()
        return spline(t)
    
    def __repr__(self):
        """Prints the full content row by row."""
        # Define the row format. We loop on self.num_columns(), so the number
        # of {} always match the number of fields to print.
        row_fmt = " ".join("{}" for _ in range(self.num_columns()))
        # Join the rows with the newline as separator. Note how we pass the
        # elements of the row to format with the unpacking syntax (*), which is
        # the same used in passing arguments
        return "\n".join(row_fmt.format(*row) for row in self)

    def __str__(self):
        """ Prints the full content row-by-row with a nice formatting."""
        # Define the row format
        row_fmt = "Row {} -> t : {:.1f} s   V : {:.2f} mV"
        # Add the third field only when appropriate
        if self.num_columns() == 3:
            row_fmt += "    dV : {:.2f} mV"
        # Generator expression for substituting the {} in the format string with
        # the actual values (lazy evaluation - no loop on the next line).
        row_fmt_gen = (row_fmt.format(i, *row) for i, row in enumerate(self))
        # Eventually join the string with a newline. Note that this is the only
        # place where the generators are actually evaluated
        return "\n".join(row_fmt_gen)
    
    def spline(self):
        """Returns the spline of the data points, only if len(self)>3 (the degree of the spline).
        """
        if len(self)>3:
            return InterpolatedUnivariateSpline(self.timestamps, self.voltages, k=3)
        print('Insufficient number of points (>3)')
        return None
    
    def plot(self, ax=None, fmt='bo', **plot_options):
       
       """ Draws the data points using matplotlib.pyplot.
       
       Attributes
       ----------
       ax : obj, optional
            Existing figure to add to the plot, defaults to None.
       
       fmt : str, optional
            Color, marker, linestyle formatter, defaluts to 'bo'.

        plot_options : str, optional
            Other plot options.

        Returns
        -------
        ax : obj
            Figure object.
       """

       # The user can provide an existing figure to add the plot, otherwise we
       # create a new one.
       if ax is not None:
           plt.sca(ax) # sca (Set Current Axes) selects the given figure
       else:
           ax = plt.figure('voltage_vs_time')
       plt.plot(self.timestamps, self.voltages, fmt, **plot_options)
       if len(self)>3:
        plt.plot(self.timestamps, self(self.timestamps), label='Cubic spline')
        plt.legend()
       plt.title('Voltage vs time')
       plt.xlabel('Time [s]')
       plt.ylabel('Voltage [mV]')
       plt.grid(True)
       return ax # We return the axes, just in case

if __name__ == '__main__':
    #lab_data = VoltageData.from_file("prova.txt")
    lab_data = VoltageData([1., 2., 3., 4.], [1., 2., 3., 4.])
