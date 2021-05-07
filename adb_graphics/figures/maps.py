# pylint: disable=invalid-name,too-few-public-methods

'''
Module contains classes relevant to plotting maps. The Map class handles all the
functionality related to a Basemap, and adding airports to a blank map. The
DataMap class takes as input a Map object and a DataHandler object (e.g.,
UPPData object) and creates a standard plot with shaded fields, contours, wind
barbs, and descriptive annotation.
'''

from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.offsetbox as mpob
import matplotlib.patches as mpatches
from mpl_toolkits.basemap import Basemap
import numpy as np

import adb_graphics.utils as utils

# TILE_DEFS is a dict with predefined tiles specifying the corners of the grid to be plotted.
#     Order: [lower left lat, upper right lat, lower left lon, upper right lon]

TILE_DEFS = {
    'NC': [36, 51, -109, -85],
    'NE': [36, 48, -91, -62],
    'NW': [35, 52, -126, -102],
    'SC': [24, 41, -107, -86],
    'SE': [22, 37, -93.5, -72],
    'SW': [24.5, 45, -122, -103],
    'AKZoom': [52, 73, -162, -132],
    'ATL': [31.2, 35.8, -87.4, -79.8],
    'CA-NV': [30, 45, -124, -114],
    'CentralCA': [34.5, 40.5, -124, -118],
    'CHI-DET': [39, 44, -92, -83],
    'DCArea': [36.7, 40, -81, -72],
    'EastCO': [36.5, 41.5, -108, -101.8],
    'GreatLakes': [37, 50, -96, -70],
    'HI': [16.6, 24.6, -157.6, -157.5],
    'NYC-BOS': [40, 43, -78.5, -68.5],
    'SEA-POR': [43, 50, -125, -119],
    'SouthCA': [31, 37, -120, -114],
    'SouthFL': [24, 28.5, -84, -77],
    'VortexSE': [30, 37, -92.5, -82],
}


class Map():

    '''
    Class includes utilities needed to create a Basemap object, add airport
    locations, and draw the blank map.

        Required arguments:

          airport_fn    full path to airport file
          ax            figure axis

        Keyword arguments:

          map_proj      dict describing the map projection to use.
                        The only options currently are for lcc settings in
                        _get_basemap()
          corners       list of values lat and lon of lower left (ll) and upper
                        right(ur) corners:
                             ll_lat, ur_lat, ll_lon, ur_lon
          tile          a string corresponding to a pre-defined tile in the
                        TILE_DEFS dictionary
    '''

    def __init__(self, airport_fn, ax, **kwargs):

        self.ax = ax
        self.grid_info = kwargs.get('grid_info', {})
        self.tile = kwargs.get('tile', 'full')
        self.airports = self.load_airports(airport_fn)

        if self.tile in ['full', 'conus', 'AK',]:
            self.corners = self.grid_info.pop('corners')
        else:
            self.corners = self.get_corners()
            self.grid_info.pop('corners')

        # Some of Hawaii's smaller islands don't show up with a larger
        # threshold.
        area_thresh = 1000
        if self.tile == 'HI':
            area_thresh = 100

        self.m = self._get_basemap(area_thresh=area_thresh, **self.grid_info)

    def boundaries(self):

        ''' Draws map boundaries - coasts, states, countries. '''

        try:
            self.m.drawcoastlines(linewidth=0.5)
        except ValueError:
            self.m.drawcounties(color='k',
                                linewidth=0.4,
                                zorder=2,
                                )
        else:
            if self.tile not in ['full', 'conus', 'AK']:
                self.m.drawcounties(antialiased=False,
                                    color='gray',
                                    linewidth=0.1,
                                    zorder=2,
                                    )
        self.m.drawstates()
        self.m.drawcountries()

    def draw(self):

        ''' Draw a map with political boundaries and airports only. '''

        self.boundaries()
        self.draw_airports()

    def draw_airports(self):

        ''' Plot each of the airport locations on the map. '''

        lats = self.airports[:, 0]
        lons = 360 + self.airports[:, 1] # Convert to positive longitude
        x, y = self.m(lons, lats)
        self.m.plot(x, y, 'ko',
                    ax=self.ax,
                    color='w',
                    fillstyle='full',
                    markeredgecolor='k',
                    markeredgewidth=0.5,
                    markersize=4,
                    )

    def _get_basemap(self, **get_basemap_kwargs):

        ''' Wrapper around basemap creation '''

        basemap_args = dict(
            ax=self.ax,
            resolution='i',
            )
        corners = self.corners
        if corners is not None:
            basemap_args.update(dict(
                llcrnrlat=corners[0],
                llcrnrlon=corners[2],
                urcrnrlat=corners[1],
                urcrnrlon=corners[3],
                ))

        basemap_args.update(get_basemap_kwargs)

        return Basemap(**basemap_args)

    def get_corners(self):

        '''
        Gather the corners for a specific tile. Corners are supplied in the
        following format:

        lat and lon of lower left (ll) and upper right(ur) corners:
             ll_lat, ur_lat, ll_lon, ur_lon
        '''

        return TILE_DEFS[self.tile]

    @staticmethod
    def load_airports(fn):

        ''' Load lat, lon pairs from a text file, return a list of lists. '''

        with open(fn, 'r') as f:
            data = f.readlines()
        return np.array([l.strip().split(',') for l in data], dtype=float)


class DataMap():
    #pylint: disable=too-many-arguments

    '''
    Class that combines the input data and the chosen map to plot both together.

    Input:

        field             datahandler data object for data field to shade
        contour_fields    list of datahandler object fields to contour
        hatch_fields      list of datahandler object fields to hatch over shaded
                          fields
        map               maps object

    '''

    def __init__(self, field, map_, contour_fields=None, hatch_fields=None, model_name=None):

        self.field = field
        self.contour_fields = contour_fields
        self.hatch_fields = hatch_fields
        self.map = map_
        self.model_name = model_name

    @staticmethod
    def add_logo(ax):

        ''' Puts the NOAA logo at the bottom left of the matplotlib axes. '''

        logo = mpimg.imread('static/noaa-logo-50x50.png')

        imagebox = mpob.OffsetImage(logo)
        ab = mpob.AnnotationBbox(
            imagebox,
            (0, 0),
            box_alignment=(-0.2, -0.2),
            frameon=False,
            )

        ax.add_artist(ab)


    def _colorbar(self, cc, ax):

        ''' Internal method that plots the color bar for a contourf field.
            If ticks is set to zero, use a user-defined list of clevs from default_specs
            If ticks is less than zero, use abs(ticks) as the step for labeling clevs '''

        if self.field.ticks > 0:
            ticks = np.arange(np.amin(self.field.clevs),
                              np.amax(self.field.clevs+1), self.field.ticks)
        elif self.field.ticks == 0:
            ticks = self.field.clevs
        else:
            ticks = self.field.clevs[0:len(self.field.clevs):-self.field.ticks]
        ticks = np.around(ticks, 4)

        cbar = plt.colorbar(cc,
                            ax=ax,
                            orientation='horizontal',
                            pad=0.02,
                            shrink=1.0,
                            ticks=ticks,
                            )

        if self.field.short_name == 'flru':
            ticks = [label.rjust(30) for label in ['VFR', 'MVFR', 'IFR', 'LIFR']]

        cbar.ax.set_xticklabels(ticks, fontsize=14)

    @utils.timer
    def draw(self, show=False): # pylint: disable=too-many-locals

        ''' Main method for creating the plot. Set show=True to display the
        figure from the command line. '''

        ax = self.map.ax

        # Draw a map and add the shaded field
        self.map.draw()
        cf = self._draw_field(ax=ax,
                              colors=self.field.colors,
                              extend='both',
                              field=self.field,
                              func=self.map.m.contourf,
                              levels=self.field.clevs,
                              )
        self._colorbar(ax=ax, cc=cf)

        not_labeled = [self.field.short_name]
        if self.hatch_fields:
            not_labeled.extend([h.short_name for h in self.hatch_fields])

        # Contour secondary fields, if requested
        if self.contour_fields:
            for contour_field in self.contour_fields:
                levels = contour_field.contour_kwargs.pop('levels',
                                                          contour_field.clevs)

                cc = self._draw_field(ax=ax,
                                      field=contour_field,
                                      func=self.map.m.contour,
                                      levels=levels,
                                      **contour_field.contour_kwargs,
                                      )
                if contour_field.short_name not in not_labeled:
                    try:
                        clab = plt.clabel(cc, levels[::4],
                                          colors='w',
                                          fmt='%1.0f',
                                          fontsize=10,
                                          inline=1,
                                          )
                        # Set the background color for the line labels to black
                        _ = [txt.set_bbox(dict(color='k')) for txt in clab]

                    except ValueError:
                        print(f'Cannot add contour labels to map for {self.field.short_name} \
                                {self.field.level}')

        # Add hatched fields, if requested
        # Levels should be included in the settings dict here since they don't
        # correspond to a full field of contours.
        if self.hatch_fields:
            handles = []
            for field in self.hatch_fields:
                colors = field.contour_kwargs.get('colors', 'k')
                hatches = field.contour_kwargs.get('hatches', '----')
                labels = field.contour_kwargs.get('labels', 'XXXX')
                linewidths = field.contour_kwargs.get('linewidths', 0.1)
                handles.append(mpatches.Patch(edgecolor=colors[-1], facecolor='lightgrey', \
                               label=labels, hatch=hatches[-1]))

                cf = self._draw_field(ax=ax,
                                      extend='both',
                                      field=field,
                                      func=self.map.m.contourf,
                                      **field.contour_kwargs,
                                      )

                # For each level, we set the color of its hatch
                for collection in cf.collections:
                    collection.set_edgecolor(colors)
                    collection.set_facecolor(['None'])
                    collection.set_linewidth(linewidths)

            # Create legend for precip type field
            if self.field.short_name == 'ptyp':
                plt.legend(handles=handles, loc=[0.25, 0.03])

        # Add wind barbs, if requested
        add_wind = self.field.vspec.get('wind', False)
        if add_wind:
            self._wind_barbs(add_wind)

        # Finish with the title
        self._title()

        # Create a pop-up to display the figure, if show=True
        if show:
            plt.tight_layout()
            plt.show()

        self.add_logo(ax)

    def _draw_field(self, ax, field, func, **kwargs):

        '''
        Internal implementation that calls a matplotlib function.

        Input args:
            ax:      Figure axis
            field:   Field to be plotted
            func:    Matplotlib function to be called.

        Keyword args:
            Can be any of the keyword args accepted by original func in
            matplotlib.

        Return:
            The return from the function called.
        '''

        x, y = self._xy_mesh(field)

        return func(x, y, field.values()[::],
                    ax=ax,
                    **kwargs,
                    )

    def _title(self):

        ''' Creates the standard annotation for a plot. '''

        f = self.field
        atime = f.date_to_str(f.anl_dt)
        vtime = f.date_to_str(f.valid_dt)

        # Create a descriptor string for the first hatched field, if one exists
        contoured = []
        not_labeled = [f.short_name]
        if self.hatch_fields:
            cf = self.hatch_fields[0]
            not_labeled.extend([h.short_name for h in self.hatch_fields])
            if not any(list(set(cf.short_name).intersection(['pres']))):
                title = cf.vspec.get('title', cf.field.long_name)
                contoured.append(f'{title} ({cf.units}, hatched)')

        # Add descriptor string for the important contoured fields
        if self.contour_fields:
            for cf in self.contour_fields:
                if cf.short_name not in not_labeled:
                    title = cf.vspec.get('title', cf.field.long_name)
                    contoured.append(f'{title} ({cf.units}, contoured)')

        contoured = ', '.join(contoured)

        # Analysis time (top) and forecast hour (bottom) on the left
        plt.title(f"{self.model_name}: {atime}\nFcst Hr: {f.fhr}",
                  alpha=None,
                  fontsize=14,
                  loc='left',
                  )

        # Add "experimental" label
        if self.model_name not in ['RAP-NCEP', 'HRRR-NCEP']:
            plt.title('Experimental',
                      fontsize=16,
                      loc='right',
                      )

        # Two lines for shaded data (top), and contoured data (bottom)
        title = f.vspec.get('title')

        level, lev_unit = ('', '')
        if title is None:
            title = f.field.long_name
            level, lev_unit = f.numeric_level(index_match=False)
            level = str(level[0]) if isinstance(level, list) else level

        if f.vspec.get('print_units', True):
            units = f'({f.units}, shaded)'
        else:
            units = f''
        plt.title(f"{level} {lev_unit} {title} {units}\n {contoured}",
                  fontsize=16,
                  horizontalalignment='right',
                  position=(1, 1.05),
                  )


        # X label shows forecast valid time.
        plt.xlabel(f"Valid time: {vtime}", fontsize=14, labelpad=75)

    def _wind_barbs(self, level):

        ''' Draws the wind barbs. '''

        u, v = self.field.wind(level)

        tile = self.map.tile

        # Set the stride and size of the barbs to be plotted with a masked array.
        if self.map.m.projection == 'lcc' and tile == 'full':
            stride = 30
            length = 5
        elif tile == 'HI':
            stride = 1
            length = 4
        elif len(tile) == 2 or tile in ['full', 'conus', 'GreatLakes', 'CA-NV']:
            stride = 10
            length = 4
        else:
            stride = 4
            length = 4

        mask = np.ones_like(u)
        mask[::stride, ::stride] = 0

        mu, mv = [np.ma.masked_array(c, mask=mask) for c in [u, v]]
        x, y = self._xy_mesh(self.field)
        self.map.m.barbs(x, y, mu, mv,
                         barbcolor='k',
                         flagcolor='k',
                         length=length,
                         linewidth=0.2,
                         sizes={'spacing': 0.25},
                         )

    @lru_cache()
    def _xy_mesh(self, field):

        ''' Helper function to create mesh for various plot. '''

        lat, lon = field.latlons()
        adjust = 360 if np.any(lon < 0) else 0
        return self.map.m(adjust + lon, lat)
