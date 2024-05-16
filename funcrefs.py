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

# My Functions -----------------------------------------------------------------------------------

class fnrefs:
    
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

    # normalizing arrays
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
    

    ''' ------------------ creating a stack of fits files ------------------
    fol_dir: path of folder where files are located (string)

    writeto: path name of the file to write the stack to (string)
    *As default, the combined data is returned as an array if writeto not specified*
    
    overwrite: if True, will overwrite the file if it already exists (bool)
    
    keyword: optional query to search for particular file names (string), where
        the default setting is to include all files in the specified directory

    warn: defaulted to 'ignore' to suppress warnings (string)

    normalize: if True, will normalize the data

    1. function will iterate through the files in the directory specified by fol_dir
    2. if keyword is specified, will only include file names that contain the keyword
    3. in each iteration, the function will open the file and extract the data and wcs
    4. the function will then convert the pixel coordinates to world coordinates
    5. the function will then append the ccd data to an array (outside of the loop)
    6. the ccd data will be combined using Combiner from ccdproc and median_combine()
    7. the combined data will be written to a new fits file specified by writeto
    '''
    def create_stack(fol_dir, writeto=False, overwrite=False, keyword=False, warn='ignore', normalize = False):
        with warnings.catch_warnings():
            warnings.simplefilter(warn) # Hint: set to 'default' to see warnings

            image_data = []

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
                    wcs = WCS(hdul[0].header)
                    
                    world_coords = wcs.pixel_to_world((data.shape[0], data.shape[1]), 0)
                    ccd = CCDData(data, unit=u.adu)
                    
                    image_data.append(ccd)
                
            combined_data = np.asarray( Combiner(image_data).median_combine() )
            combined_data = np.nan_to_num(combined_data, nan=0) # replaces NaN values with 0

            if (normalize): # executes when not False or some truthy value
                combined_data = fnrefs.normalize(combined_data)

            # if writeto is specified, will write the combined data to a new fits file
            if (writeto): # skips execution if writeto = False or some falsy value
                fits.writeto(str(writeto), combined_data, overwrite=overwrite)
            else: return combined_data

    # old version
    # def create_master(files, master_name):
    #     stack = []
    #     for file in files:
    #         with fits.open(file) as hdul:
    #             stack.append(hdul[0].data)
    #     stack = np.dstack(stack)
    #     mean = np.mean(stack, axis=2)
    #     master = fits.PrimaryHDU(mean)
    #     master.writeto(master_name, overwrite=True)



# Convenience Functions given from Tuttle on canvas --------------------------------------------------

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
