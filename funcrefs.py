'''
INTRODUCTION

To use these functions, import the file (like packages!) and call the functions in the same manner
NOTE: if importing the file from another folder, you may need to add the directory to the path.
Always check your current directory to know where you are in the file system!

1) if that's the case, you can follow this example (otherwise skip to step 3):

        import os 
        print(os.getcwd())

2) if the file is in a different directory, you can add it to the path like this:
        
        os.sys.path.append('path/to/your/file')

3) then you can import the file like this (exclude the .py part):
   * if you skipped to this step, make sure to import the os package first

        from funcrefs import fnrefs as rfs (or whatever you want to call it)
        from funcrefs import convenience_functions as cf

4) call functions from within this class like this: 

        rfs.create_stack(insert parameters here)
        cf.show_image(insert parameters here)

'''

# Packages Galore! -----------------------------------------------------------------------------------
import os
from re import search
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ccdproc import Combiner
from astropy.nddata import CCDData
import astropy.visualization as vis
import matplotlib.pyplot as plt
from astropy.nddata import block_reduce, Cutout2D
import time
from IPython.display import display, HTML, Image, clear_output
from textwrap import wrap
import astropy.units as u
import warnings

''' Custom Colormap ----------------------------------------------------------------------------- '''
# https://xkcd.com/color/rgb/

from matplotlib.colors import LinearSegmentedColormap
colors = ["black", "xkcd:very dark purple", 
          "xkcd:cornflower", "xkcd:light blue grey", 
          "xkcd:light khaki", "xkcd:dandelion"]
custom_colormap = LinearSegmentedColormap.from_list("custom", colors, N=256)



''' My Functions ----------------------------------------------------------------------------------- '''

class fnrefs:
    
    # [please preserve the indentation in the next line]
    ''' 
FUNCTION 1) ------------------ MAPPING FITS FILES IN A GRIDLIKE PLOT ------------------
    '''
    
    # this is useful for viewing multiple fits files in a grid (see show_iamge for single fits files)
    def map_fits(files, hdul_index, nrows, ncols, cmap, interpolation,
                   figsize=(15, 15), animation=False, interval=None, stretch=None, 
                   vmin=None, vmax=None, contrast=None, bias=None, power=1, percentile=1):
                   # just setting default power & percentile = 1 bc they can't have a value of None
        
        # aspect ratio for plots
        with fits.open(files[0]) as hdul:
            data = hdul[hdul_index].data
        aspect_ratio = data.shape[0] / data.shape[1]
        figsize = (max(figsize) * aspect_ratio, max(figsize))

        if animation == False:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axs = axs.ravel()

        # ------ START OF FOR LOOP ------
        for i, file in enumerate(files):  # iterating through each file and displaying the image
            with fits.open(file) as hdul:
                data = hdul[hdul_index].data
            
            # creating dictionaries for available intervals and stretches in astropy.visualization (vis in this notebook)
            intervals = {'ZScale': vis.ZScaleInterval(),
                        'MinMax': vis.MinMaxInterval(),
                        'Percentile': vis.PercentileInterval(percentile),
                        'AsymPercentile': vis.AsymmetricPercentileInterval(vmin, vmax),
                        'Manual': vis.ManualInterval(vmin=vmin, vmax=vmax)}
            
            stretches = {'Linear': vis.LinearStretch(),
                        'Asinh': vis.AsinhStretch(),
                        'Log': vis.LogStretch(),
                        'Sqrt': vis.SqrtStretch(),
                        'Hist' : vis.HistEqStretch(data),
                        'Power': vis.PowerStretch(power),
                        'Sinh': vis.SinhStretch(),
                        'Contrast': vis.ContrastBiasStretch(contrast=contrast, bias=bias)}

            # converting data to float type and normalizing
            data = np.nan_to_num(data.astype(float))
            vis_vmin, vis_vmax = np.percentile(data, [vmin, vmax])
            norm = vis.ImageNormalize(data, vmin=vis_vmin, vmax=vis_vmax, interval=intervals[interval], stretch=stretches[stretch])

            # plotting as "gif" (may not be useful for this project but i have this from my research)
            if animation == True:
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_title("\n".join(wrap(file, width = 40)), weight='bold', loc='center')
                ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
                plt.tight_layout(pad=0, h_pad=0, w_pad=2)
                plt.show()

                time.sleep(0.1)
                if file != files[-1]:
                    clear_output(wait=True)
            # otherwise will plot as a grid of images, rows and columns specified
            else:
                ax = axs[i]
                ax.set_title("\n".join(wrap(file, width = 40)), weight='bold', loc='center')
                ax.imshow(data, cmap=cmap, norm=norm, interpolation=interpolation)
            # -------- END OF FOR LOOP --------
        
        # have to do this part outside of the for loop as a final call 
        if animation == False:
            plt.tight_layout(pad=0, h_pad=0.2, w_pad=0.2)
            plt.show()



    # [please preserve the indentation in the next line]
    '''
FUNCTION 2) ------------------ GENERIC NORMALIZING FUNCTION FOR ARRAYS ------------------
    '''
    def normalize(arr, t_min=None, t_max=None):
        if t_min is None:
            t_min = np.amin(arr)
        if t_max is None:
            t_max = np.amax(arr)
        
        norm_arr = np.empty(arr.shape, dtype=arr.dtype)
        diff = t_max - t_min
        diff_arr = t_max - t_min

        # np.ndenumerate preserves original array coordinates and values
        for index, i in np.ndenumerate(arr):  
            norm_arr[index] = (((i - t_min)*diff)/diff_arr) + t_min
        # for i in np.nditer(arr):
        #     temp = (((i - t_min)*diff)/diff_arr) + t_min
        #     norm_arr.append(temp)
        return norm_arr
    


    # [please preserve the indentation in the next line]
    '''
FUNCTION 3) ------------------ CREATING A STACK OF FITS FILES (PLEASE READ) ------------------

    ------------------------
    Purpose of This Function
    ------------------------

    To stack multiple fits files into a single array. Default setting is to return the
    combined data as an array, but the user can also save the data to a new fits file.

    More comments on the function are provided at the end, but here is a brief overview:


    ------------------------
    About the Parameters
    ------------------------

    fol_dir: PATH of folder where files are located (type: string)

    writeto: path name of the file to write the stack to (type: string)
    *As default, the combined data is returned as an array if writeto not specified*
    
    overwrite: if True, will overwrite the file if it already exists (bool)
    
    keyword: optional query to search for particular file names (string), where
        the default setting is to include all files in the specified directory

    warn: defaulted to 'ignore' to suppress warnings (string)

    normalize: if True, will normalize the data before stacking

    '''
    def create_stack(fol_dir, writeto=False, overwrite=False, keyword=False, warn='ignore', normalize = False):
        with warnings.catch_warnings(): # this just suppesses warnings to make the output cleaner
            warnings.simplefilter(warn) # Hint: set to 'default' to see warnings

            image_data = [] # empty list to store the data from each file for combining later

            for file in [f for f in os.listdir(fol_dir) if ( (not keyword) or search(keyword, f) )]:
                '''
                loop through files (file) in a list that contains the file names (f) in 
                the passed directory (file_dir) with an optional keyword to search names

                "or" can be interpreted like "otherwise" here, so if a keyword isn't passed,
                it will just return the list of all files in fol_dir. otherwise, if a keyword
                is a non-empty string, it will return the list of files that match the query
                '''
                with fits.open(fol_dir + file) as hdul:
                    data = hdul[0].data
                    if (normalize): # executes when not False (aka some truthy value)
                        data = fnrefs.normalize(data)
                    
                    # converting the pixels to world coordinates since the objects aren't lined up
                    wcs = WCS(hdul[0].header) 
                    world_coords = wcs.pixel_to_world((data.shape[0], data.shape[1]), 0)
                    ccd = CCDData(data, unit=u.adu)
                    
                    image_data.append(ccd)
                
            combined_data = np.asarray( Combiner(image_data).median_combine() )
            combined_data = np.nan_to_num(combined_data, nan=0) # replaces NaN values with 0

            # if writeto is specified, will write the combined data to a new fits file
            if (writeto): # skips execution if writeto = False (aka some falsy value)
                fits.writeto(str(writeto), combined_data, overwrite=overwrite)
            else: return combined_data # this is the default action 
    '''
    
    ------------------------
    A Walkthrough of the Steps
    ------------------------

    1. function will iterate through the files in the directory specified by fol_dir

    2. if keyword is specified, will only include file names that contain the keyword

    3. in each iteration, the function will open the file and extract the data and wcs

    4. the function will then convert the pixel coordinates to world coordinates

    5. the function will then append the ccd data to an array (outside of the loop)

    6. the ccd data will be combined using Combiner from ccdproc and median_combine()

    7. the combined data will be written to a new fits file specified by writeto


    ------------------------
    Additional Comments on This Function
    ------------------------

    - this function is designed to use the PATHING of a folder containing the files
      to be combined, and puts the stacked data wherever the user specifies

    - unlike the CCDProc package, this function takes advantage of the OS package,
      which can offer more flexibility in terms of how files are located and combined

    - this is designed to be flexible, allowing the user to specify a keyword as a
      means to search for names of files in the folder containing a particular string

    - to integrate the CCDProc package, the function uses the Combiner class to stack
      files, rather than the combine() function, which, although nearly identical, seems
      to be more intuitive to use in this context

        - particularly, Combiner is a class (or group) of functions that can be used
          with one another in a more human-readable way (just like this fnrefs class)

        - the combine() function is a single function that requires several parameters
          to be specified, and the only advantage is that you can set a limit on memory 
          usage. however, the scope of this class shouldn't require heavy computing power

    - to combine the data, the function uses the median_combine() method, which is an
      alternative way to stack, rather than using sigma clipping or other methods

        - median_combine() is a member of the Combiner class, and can be used to call
          back on other members, to apply median combination to the data

        - sigma clipping uses a different method to stack the data, which clips off the
          outliers in the data (decided by the user) and stacks what's left

        - sigma clipping can be useful for noisy data, but may also risk the chance of
          losing valuable information, which is why median combination is used here

    - if specified, the user can also choose to normalize the data BEFORE stacking them,
      though the default value is set to False, as this may not always be necessary

      - the normalizing function is another function I have made within the fnrefs class

      - if the user wants to normalize the data AFTER stacking, they can do so by using
        this function after the data has been combined and saved, or passing create_stack
        as a parameter in the normalizing function to keep it to a single line of code
    '''
# END OF SECTION -----------------------------------------------------------------------------------



from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAnnulus, aperture_photometry, ApertureStats
from astropy.stats import sigma_clipped_stats, SigmaClip
from matplotlib.patches import Rectangle
from matplotlib.colors import rgb2hex
import photutils as pu

# class for photometry annulus, should be used AFTER background/sky subtraction
class PhotAnnulus: 
    def __init__(self, data):
        '''
        just a way of declaring variables local to entire class, can be used in multiple
        functions within the class without having to redefine them each time. referred to
        by using self.[variable name]

        otherwise, if you don't use self, the variable will only be local to the function

        __init__ is like a list of variables that are used within the class, and the 
        functions defined later are the actions that can be taken with those variables.
        to use self variables later, add self as the first parameter of a function
        '''
        self.data = data
        self.snippet = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        
        # initial skeleton variables, will be defined later when used
        self.norm = None
        self.mean = None
        self.median = None
        self.std = None

        self.srcs = None
        self.finder = None
        self.positions = None
        self.annuli = None
        self.an_areas = None
        self.an_stats = None
        self.r_in = None
        self.r_out = None
        self.phot_table = None

        self.vmin = None
        self.vmax = None
        self.percentile = 1 # these can't be None values, so defaulting to 1
        self.power = 1
        self.contrast = None
        self.bias = None

        # dictionary for easier use later
        self.intervals = {'ZScale': vis.ZScaleInterval(),
                        'MinMax': vis.MinMaxInterval(),
                        'Percentile': vis.PercentileInterval(self.percentile),
                        'AsymPercentile': vis.AsymmetricPercentileInterval(self.vmin, self.vmax),
                        'Manual': vis.ManualInterval(vmin=self.vmin, vmax=self.vmax)}
            
        self.stretches = {'Linear': vis.LinearStretch(),
                        'Asinh': vis.AsinhStretch(),
                        'Log': vis.LogStretch(),
                        'Sqrt': vis.SqrtStretch(),
                        'Hist' : vis.HistEqStretch(self.data),
                        'Power': vis.PowerStretch(self.power),
                        'Sinh': vis.SinhStretch(),
                        'Contrast': vis.ContrastBiasStretch(contrast=self.contrast, bias=self.bias)}
        
    # -----------------------------------------------------------------------------------
    def normalizer(self, interval='ZScale', stretch='Sqrt', pmin=1, pmax=99.75):
        # setting class-wide variable to the values passed
        self.vmin, self.vmax = np.percentile(self.data, [pmin, pmax])

        self.norm = vis.ImageNormalize(self.data, vmin=self.vmin, vmax=self.vmax, 
                                       interval=self.intervals[interval], 
                                       stretch=self.stretches[stretch])
        # returning for outside access during use, not for within class
        return self.norm

    # -----------------------------------------------------------------------------------
    '''
    get full photometry table here. manually remove bad sources later
    '''
    def sources(self, xdim, ydim, sigma, fwhm=4, threshold=5, r_in=10, r_out=20):
        '''
        Creating Snippet
        '''
        self.x_start, self.x_end = xdim
        self.y_start, self.y_end = ydim
        print(f"Snippet Resolution:  {self.x_end - self.x_start} x {self.y_end - self.y_start} px")

        self.snippet = self.data[self.y_start:self.y_end, self.x_start:self.x_end]
        self.mean, self.median, self.std = sigma_clipped_stats(self.snippet, sigma=sigma)

        '''
        Finding Sources
        '''
        self.finder = DAOStarFinder(fwhm=fwhm, threshold=threshold*self.std)
        self.srcs = self.finder(self.snippet - self.median)

        for col in self.srcs.colnames:  
            if col not in ('id', 'xcentroid', 'ycentroid'):
                self.srcs[col].format = '%.2f'  # for consistent table output

        # print(f"Sources Found: {len(self.srcs)}") # can also print this outside function
        
        '''
        Photometry Part 1
        '''
        self.positions = np.transpose((self.srcs['xcentroid'], self.srcs['ycentroid']))
        self.r_in, self.r_out = r_in, r_out
        self.annuli = CircularAnnulus(self.positions, r_in=r_in, r_out=r_out)
        self.phot_table = aperture_photometry(self.data, self.annuli)

        # returning for outside access during use, not for within class
        return self.snippet, self.srcs

    # -----------------------------------------------------------------------------------
    '''
    manually remove bad sources before passing to this function as phot_table
    this function will outline snippet on original image and display annuli sources

    recommended to use with normalizer function defined above
    '''
    def view_sources(self, srcs, norm, figsize=(16, 9), cmap='magma', mcolor='xkcd:wine', interpolation='hermite', nightmode=True):
        warnings.simplefilter('ignore') # booo warnings

        # updating sources and others for the rest of the class
        self.srcs = srcs
        self.positions = np.transpose((self.srcs['xcentroid'], self.srcs['ycentroid']))
        self.annuli = CircularAnnulus(self.positions, r_in=self.r_in, r_out=self.r_out)
        self.phot_table = aperture_photometry(self.data, self.annuli)
        
        # flashbangs?!?!
        with plt.style.context('dark_background' if nightmode else 'default'):
            plt.figure(figsize=figsize)

            # ignore this part idk why she's not workinggg

            # Compute the contrast color (mcolor) for annotations to stand out
            # colormap = plt.get_cmap(cmap)
            # avg_color = colormap(np.mean([self.vmin, self.vmax]))
            # mcolor = tuple(1 - x for x in avg_color[:3])  # convert to RGB components, ignore the alpha (opacity) channel
            # mcolor = rgb2hex(mcolor)
            
            # plotting image and marking the snippet -----------------------------------------
            plt.subplot(1, 2, 1)
            plt.title(f'Snippet of Image', weight='bold')
            plt.xlabel(f'Snip Resolution: {self.snippet.shape[0]} x {self.snippet.shape[1]} px      Sources found: {len(self.srcs)}')
            
            width = self.x_end - self.x_start
            height = self.y_end - self.y_start
            # extent = [self.x_start, self.x_start + width, self.y_start, self.y_start + height]
            border = Rectangle((self.x_start, self.y_start), width, height, edgecolor=mcolor, facecolor='none', lw=1.5)

            plt.imshow(self.data, cmap=cmap, norm=norm, interpolation=interpolation)
            plt.gca().add_patch(border) # adding outline of snippet

            # plotting annului --------------------------------------------------------------
            plt.subplot(1, 2, 2)
            plt.title(f'Annuli Sources', weight='bold')

            plt.imshow(self.snippet, cmap=cmap, origin='lower', norm=norm, interpolation=interpolation)
            self.annuli.plot(color=mcolor, lw=1.5, alpha=0.5)

            # annotating the annuli with their corresponding source id number
            for i, (x, y) in enumerate(zip(self.srcs['xcentroid'], self.srcs['ycentroid'])):
                plt.text(x, y, f"{self.srcs['id'][i]}", color=mcolor, fontsize=12, ha='center', va='center')


            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]) # hide that shi
            plt.tight_layout()
            plt.show()


    # -----------------------------------------------------------------------------------
    def magnitudes(self, ids, sigma=3.0, maxiters=10):
        
        self.an_areas = pu.aperture.PixelAperture.area_overlap(self.annuli, self.data)
        self.an_stats = ApertureStats(self.data, self.annuli, sigma_clip=SigmaClip(sigma=sigma, maxiters=maxiters))

        # total background within the annulus
        self.phot_table['total_bkg'] = self.an_stats.median * self.an_areas

        # background-subtracted photometry
        self.phot_table['bkgsub'] = self.phot_table['aperture_sum'] - self.phot_table['total_bkg']

        for col in self.phot_table.colnames:  
            if col not in ('id'):
                self.phot_table[col].format = '%.2f'  # for consistent table output

        self.phot_table['id'] = self.srcs['id']

        magnitudes = []
        for source_id in ids:
            source_row = self.srcs[self.srcs['id'] == source_id]
            x, y = np.round(source_row['xcentroid'][0], 2), np.round(source_row['ycentroid'][0], 2)
            phot_row = self.phot_table[(np.round(self.phot_table['xcenter'].value, 2) == x) & (np.round(self.phot_table['ycenter'].value, 2) == y)]

            # print(f"[ID {source_id}] Magnitude: {phot_row['bkgsub'][0]:.2f}")
            magnitudes.append([source_id, phot_row['bkgsub'][0]])

        return magnitudes, self.phot_table



'''Convenience Functions given from Tuttle on canvas --------------------------------------------------'''

class convenience_functions: 

    # this is only useful for viewing a single fits file
    def show_image(image,
                percl=99, percu=None, is_mask=False,
                figsize=(10, 10),
                cmap='viridis', log=False, clip=True,
                show_colorbar=True, show_ticks=True,
                fig=None, ax=None, input_ratio=None):
        """
        Show an image in matplotlib with some basic astronomically-appropriat stretching.

        Parameters
        ----------
        image
            The image to show
        percl : number
            The percentile for the lower edge of the stretch (or both edges if ``percu`` is None)
        percu : number or None
            The percentile for the upper edge of the stretch (or None to use ``percl`` for both)
        figsize : 2-tuple
            The size of the matplotlib figure in inches
        """
        if percu is None:
            percu = percl
            percl = 100 - percl

        if (fig is None and ax is not None) or (fig is not None and ax is None):
            raise ValueError('Must provide both "fig" and "ax" '
                            'if you provide one of them')
        elif fig is None and ax is None:
            if figsize is not None:
                # Rescale the fig size to match the image dimensions, roughly
                image_aspect_ratio = image.shape[0] / image.shape[1]
                figsize = (max(figsize) * image_aspect_ratio, max(figsize))

            fig, ax = plt.subplots(1, 1, figsize=figsize)


        # To preserve details we should *really* downsample correctly and
        # not rely on matplotlib to do it correctly for us (it won't).

        # So, calculate the size of the figure in pixels, block_reduce to
        # roughly that,and display the block reduced image.

        # Thanks, https://stackoverflow.com/questions/29702424/how-to-get-matplotlib-figure-size
        fig_size_pix = fig.get_size_inches() * fig.dpi

        ratio = (image.shape // fig_size_pix).max()

        if ratio < 1:
            ratio = 1

        ratio = input_ratio or ratio

        reduced_data = block_reduce(image, ratio)

        if not is_mask:
            # Divide by the square of the ratio to keep the flux the same in the
            # reduced image. We do *not* want to do this for images which are
            # masks, since their values should be zero or one.
            reduced_data = reduced_data / ratio**2

        # Of course, now that we have downsampled, the axis limits are changed to
        # match the smaller image size. Setting the extent will do the trick to
        # change the axis display back to showing the actual extent of the image.
        extent = [0, image.shape[1], 0, image.shape[0]]

        if log:
            stretch = vis.LogStretch()
        else:
            stretch = vis.LinearStretch()

        norm = vis.ImageNormalize(reduced_data,
                                interval=vis.AsymmetricPercentileInterval(percl, percu),
                                stretch=stretch, clip=clip)

        if is_mask:
            # The image is a mask in which pixels should be zero or one.
            # block_reduce may have changed some of the values, so reset here.
            reduced_data = reduced_data > 0
            # Set the image scale limits appropriately.
            scale_args = dict(vmin=0, vmax=1)
        else:
            scale_args = dict(norm=norm)

        im = ax.imshow(reduced_data, origin='lower',
                    cmap=cmap, extent=extent, aspect='equal', **scale_args)

        if show_colorbar:
            # I haven't a clue why the fraction and pad arguments below work to make
            # the colorbar the same height as the image, but they do....unless the image
            # is wider than it is tall. Sticking with this for now anyway...
            # Thanks: https://stackoverflow.com/a/26720422/3486425
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # In case someone in the future wants to improve this:
            # https://joseph-long.com/writing/colorbars/
            # https://stackoverflow.com/a/33505522/3486425
            # https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html#colorbar-whose-height-or-width-in-sync-with-the-master-axes

        if not show_ticks:
            ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)


    def image_snippet(image, center, width=50, axis=None, fig=None,
                    is_mask=False, pad_black=False, **kwargs):
        """
        Display a subsection of an image about a center.

        Parameters
        ----------

        image : numpy array
            The full image from which a section is to be taken.

        center : list-like
            The location of the center of the cutout.

        width : int, optional
            Width of the cutout, in pixels.

        axis : matplotlib.Axes instance, optional
            Axis on which the image should be displayed.

        fig : matplotlib.Figure, optional
            Figure on which the image should be displayed.

        is_mask : bool, optional
            Set to ``True`` if the image is a mask, i.e. all values are
            either zero or one.

        pad_black : bool, optional
            If ``True``, pad edges of the image with zeros to fill out width
            if the slice is near the edge.
        """
        if pad_black:
            sub_image = Cutout2D(image, center, width, mode='partial', fill_value=0)
        else:
            # Return a smaller subimage if extent goes out side image
            sub_image = Cutout2D(image, center, width, mode='trim')
        
        # assuming class was imported as cf
        convenience_functions.show_image(sub_image.data, cmap='gray', ax=axis, fig=fig,
                show_colorbar=False, show_ticks=False, is_mask=is_mask,
                **kwargs)


    def _mid(sl):
        return (sl.start + sl.stop) // 2


    def display_cosmic_rays(cosmic_rays, images, titles=None,
                            only_display_rays=None):
        """
        Display cutouts of the region around each cosmic ray and the other images
        passed in.

        Parameters
        ----------

        cosmic_rays : photutils.segmentation.SegmentationImage
            The segmented cosmic ray image returned by ``photuils.detect_source``.

        images : list of images
            The list of images to be displayed. Each image becomes a column in
            the generated plot. The first image must be the cosmic ray mask.

        titles : list of str
            Titles to be put above the first row of images.

        only_display_rays : list of int, optional
            The number of the cosmic ray(s) to display. The default value,
            ``None``, means display them all. The number of the cosmic ray is
            its index in ``cosmic_rays``, which is also the number displayed
            on the mask.
        """
        # Check whether the first image is actually a mask.

        if not ((images[0] == 0) | (images[0] == 1)).all():
            raise ValueError('The first image must be a mask with '
                            'values of zero or one')

        if only_display_rays is None:
            n_rows = len(cosmic_rays.slices)
        else:
            n_rows = len(only_display_rays)

        n_columns = len(images)

        width = 12

        # The height below is *CRITICAL*. If the aspect ratio of the figure as
        # a whole does not allow for square plots then one ends up with a bunch
        # of whitespace. The plots here are square by design.
        height = width / n_columns * n_rows
        fig, axes = plt.subplots(n_rows, n_columns, sharex=False, sharey='row',
                                figsize=(width, height))

        # Generate empty titles if none were provided.
        if titles is None:
            titles = [''] * n_columns

        display_row = 0

        for row, s in enumerate(cosmic_rays.slices):
            if only_display_rays is not None:
                if row not in only_display_rays:
                    # We are not supposed to display this one, so skip it.
                    continue

            x = convenience_functions._mid(s[1])
            y = convenience_functions._mid(s[0])

            for column, plot_info in enumerate(zip(images, titles)):
                image = plot_info[0]
                title = plot_info[1]
                is_mask = column == 0
                ax = axes[display_row, column]
                convenience_functions.image_snippet(image, (x, y), width=80, axis=ax, fig=fig,
                            is_mask=is_mask)
                if is_mask:
                    ax.annotate('Cosmic ray {}'.format(row), (0.1, 0.9),
                                xycoords='axes fraction',
                                color='cyan', fontsize=20)

                if display_row == 0:
                    # Only set the title if it isn't empty.
                    if title:
                        ax.set_title(title)

            display_row = display_row + 1

        # This choice results in the images close to each other but with
        # a small gap.
        plt.subplots_adjust(wspace=0.1, hspace=0.05)
