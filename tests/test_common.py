# pylint: disable=invalid-name

'''
Pytests for the common utilities included in this package. Includes:

    - conversions.py
    - specs.py
    - utils.py

To run the tests, type the following in the top level repo directory:

    python -m pytest --grib-file [path/to/gribfile]

'''

from string import digits, ascii_letters

from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np

import adb_graphics.conversions as conversions
import adb_graphics.specs as specs
import adb_graphics.utils as utils

def test_conversion():

    ''' Test that conversions return at numpy array for input of np.ndarray,
    list, or int '''

    a = np.ones([3, 2]) * 300
    b = list(a)
    c = a[0, 0]
    assert np.array_equal(conversions.k_to_c(a), a - 273.15)
    assert np.array_equal(conversions.m_to_dm(a), a / 10)
    assert np.array_equal(conversions.pa_to_hpa(a), a / 100)

    assert np.array_equal(conversions.k_to_c(b), a - 273.15)
    assert np.array_equal(conversions.m_to_dm(b), a / 10)
    assert np.array_equal(conversions.pa_to_hpa(b), a / 100)

    assert np.array_equal(conversions.k_to_c(c), a[0, 0] - 273.15)
    assert np.array_equal(conversions.m_to_dm(c), a[0, 0] / 10)
    assert np.array_equal(conversions.pa_to_hpa(c), a[0, 0] / 100)


class MockSpecs(specs.VarSpec):

    ''' Mock class for the VarSpec abstract class '''

    @property
    def clevs(self):
        return list(range(15))


def test_specs():

    ''' Test VarSpec properties. '''

    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    # Ensure correct return type
    assert isinstance(varspec.t_colors, np.ndarray)
    assert isinstance(varspec.ps_colors, np.ndarray)
    assert isinstance(varspec.yml, dict)

    # Ensure the appropriate number of colors is returned
    assert np.shape(varspec.t_colors) == (len(varspec.clevs), 4)
    assert np.shape(varspec.ps_colors) == (105, 4)


def test_utils():

    ''' Test that utils works appropriately. '''

    assert callable(utils.get_func('conversions.k_to_c'))


class TestDefaultSpecs():

    ''' Test contents of default_specs.yml. '''

    config = 'adb_graphics/default_specs.yml'
    varspec = MockSpecs(config)

    cfg = varspec.yml

    @property
    def allowable(self):

        ''' Each entry in the dict names a function that tests a key in
        default_specs.yml. '''

        return {
            'clevs': self.is_a_clev,
            'cmap': self.is_a_cmap,
            'colors': self.is_a_color,
            'subst': self.is_a_key,
            'ticks': self.is_int,
            'transform': self.is_callable,
            'unit': self.is_string,
            }

    @staticmethod
    def is_a_clev(clev):

        ''' Returns true for a clev that is a list, a range, or a callable function. '''

        if isinstance(clev, list):
            return True

        if 'range' in clev.split('[')[0]:
            clean = lambda x: x.strip().split('-')[-1].replace('.', '1')
            items = clev.split(' ', 1)[1].strip('[').strip(']').split(',')
            nums = [clean(i).isnumeric() for i in items]
            return all(nums)

        return callable(utils.get_func(clev))

    @staticmethod
    def is_a_cmap(cmap):

        ''' Returns true for a cmap that is a Colormap object. '''

        return isinstance(cm.get_cmap(cmap), mcolors.Colormap)

    def is_a_color(self, color):

        ''' Returns true if color is contained in the list of recognized colors. '''

        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        return color in colors.keys() or self.is_callable(color)

    def is_a_level(self, key):

        '''
        Returns true if the key fits one of the level descriptor formats.

        Allowable formats include:

            [str_descriptor]     e.g. sfc, max, mup
            [numeric][lev_type]  e.g. 500mb, or 2m
            [stat][numeric]      e.g. mn02, mx25

        '''

        allowed_levels = [
                'esbl',    # ???
                'esblmn',  # ???
                'max',     # maximum in column
                'maxsfc',  # max surface value
                'mdn',     # maximum downward
                'mnsfc',   # min surface value
                'mup',     # maximum upward
                'sfc',     # surface
                'ua',      # upper air
                ]

        allowed_lev_type = [
                'cm',      # centimeters
                'ds',      # difference
                'm',       # meters
                'mb',      # milibars
                ]

        allowed_stat = [
                'in',      # ???
                'm',       # ???
                'maxm',    # ???
                'mn',      # minimum
                'mx',      # maximum
                ]

        # Easy check first -- it is in the allowed_levels list
        if key in allowed_levels:
            return True

        # Check for [numeric][lev_type] pattern
        for lev in allowed_lev_type:
            ks = key.split(lev)

            # If the lev didn't appear in the key, lenght of list is 1.
            # If the lev didn't match exactly, the second element will the remainder of the string
            if len(ks) == 2 and len(ks[1]) == 0:
                numeric = ks[0].isnumeric()
                allowed = ''.join([c for c in key if c in ascii_letters]) in allowed_lev_type

                if numeric and allowed:
                    return True

        # Check for [stat][numeric]
        for stat in allowed_stat:
            ks = key.split(stat)
            if len(ks) == 2 and len(ks[0]) == 0:

                numeric = ks[1].isnumeric()
                allowed = ''.join([c for c in key if c in ascii_letters]) in allowed_stat

                if numeric and allowed:
                    return True

        return False

    def is_a_key(self, key):

        ''' Returns true if key exists as a key in the config file. '''

        return self.cfg.get(key) is not None

    def is_callable(self, func):

        ''' Returns true if func is the name of a callable function. '''

        return hasattr(self.varspec, func) or callable(utils.get_func(func))

    @staticmethod
    def  is_int(i):

        ''' Returns true if i is an integer. '''

        if isinstance(i, int):
            return True
        return i.isnumeric() and len(i.split('.')) == 1

    @staticmethod
    def is_string(s):

        ''' Returns true if s is a string. '''

        return isinstance(s, str)

    def check_keys(self, d):

        ''' Helper function that recursively checks the keys in the dictionary by calling the
        function defined in allowable. '''

        if not isinstance(d, dict):
            return
        for k, v in d.items():
            assert (k in self.allowable.keys()) or self.is_int(k)
            if isinstance(v, dict):
                self.check_keys(v)
            else:
                checker = self.allowable.get(k)
                assert checker(v)

    def test_keys(self):

        ''' Tests each of top-level variables in the config file by calling the helper function. '''

        for spec in self.cfg.values():
            self.check_keys(spec)