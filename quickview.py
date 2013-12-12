# coding: utf-8

""" A quick-view tool for raw HERMES data files """

__author__ = "Andy Casey <andy@astrowizici.st>"

# Standard libraries
import logging
import os
import pickle
import threading
from glob import glob
from time import sleep, time
# Third party libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyfits
import wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.ticker import MaxNLocator

from traits.api import \
    Any, Array, Instance, property_depends_on, Button, DelegatesTo, \
    PrototypedFrom, HasTraits, HasStrictTraits, Event, Password, Property, \
    Enum, Float, File, Str, Dict, Int, Bool, List, on_trait_change, Range, \
    Directory

from traitsui.api import \
    Handler, HSplit, HFlow, View, VSplit, ButtonEditor, EnumEditor, \
    DirectoryEditor, Action, Label, Item, Group, HGroup, InstanceEditor, VGroup, Menu, \
    MenuBar, ImageEditor, FileEditor, TableEditor, TextEditor, Tabbed, \
    TabularEditor, ToolBar, TitleEditor, UItem, spring, VGrid, ListStrEditor

from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

# Module-specific libraries
import calibrate


__all__ = ["QuickView"]



class SelectDirectory(HasTraits):
    """Select Data Directory for GALAH Slow View"""

    directory = Directory

    view = View(
        Item("directory", editor=DirectoryEditor(), style="simple"),
        title="GALAH Slow View - Select Data Folder",
        width=500,
        height=150,
        buttons=["OK", "Cancel"])



class MonitorFolder(threading.Thread):
    """ A parallel thread to monitor a folder for new OBJECT images """

    def __init__(self, folder, expected_blue_images, expected_blue_images_to_ignore,
        callback):
        threading.Thread.__init__(self)

        self.kill_thread = False
        self.is_running = True
        self.folder = folder
        self.expected_blue_images = expected_blue_images
        self.callback = callback

        # Cache non-object images
        self.expected_blue_images_to_ignore = expected_blue_images_to_ignore

    def run(self):
        while True:
            if self.kill_thread:
                self.is_running = False
                break

            # We will identify new images based on the blue channel
            blue_images = sorted(glob(os.path.join(self.folder[0], "ccd_1/??????????.fits")))
            
            all_known_images = []
            [all_known_images.extend(images) for images in [self.expected_blue_images, self.expected_blue_images_to_ignore]]
            
            new_image_filenames = set(blue_images).symmetric_difference(all_known_images)
            
            # Any change in blue images?
            if len(new_image_filenames) > 0:

                # Check to make sure the new images are actually object images
                object_filenames = []
                for filename in new_image_filenames:
                    
                    # Keep trying to open this new file
                    while True:
                        try:
                            with pyfits.open(filename) as image:
                                obstype = image[0].header["OBSTYPE"]

                        except:
                            logging.info("Image {0} not fully written to disk yet. Trying again in one second..".format(filename))
                            sleep(1)
                            continue

                        else:
                            # It's okay to load. Check that it's an OBJECT file.
                            if obstype == "OBJECT":
                                object_filenames.append(filename)

                            else:
                                # Ignore this filename in the future
                                self.expected_blue_images_to_ignore.append(filename)
                            break

                if len(object_filenames) > 0:
                    self.callback(sorted(self.expected_blue_images + object_filenames))
                    self.expected_blue_images.extend(object_filenames)
            sleep(1)
            

class _MPLFigureEditor(Editor):
    """ Editor class for containing a matplotlib figure within a
    TraitsUI GUI window """

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the matplotlib canvas """

        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        
        mpl_control = FigureCanvasWxAgg(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        
        return panel


class MPLFigureEditor(BasicEditorFactory):
    """ Factory class for generating editors that contain matplotlib
    figures and can be placed within a TraitsUI GUI """
    klass = _MPLFigureEditor


class QuickViewHandler(Handler):

    def object_title_str_changed(self, info):
        """ Update the title of the GUI window """
        info.ui.title = info.object.title_str

    def close(self, info, is_ok=True):
        """ Closes the Slow View GUI dialog """
        if hasattr(info.object, "watch_folder_thread"):
            info.object.watch_folder_thread.kill_thread = True
        info.ui.owner.close()


class QuickView(HasTraits):
    """ A GUI dialog for quickly examining HERMES raw data files """

    folder_to_watch = Str
    select_folder_to_watch_button = Button("Select Folder")
    science_image_wildmasks = List(Str)
    current_science_images = List(Str)

    show_science_image_wildmasks = Enum(values="science_image_wildmasks")
    automatically_show_new_images = Bool(True)
    automatically_show_new_images_label = Str("Automatically show new images")
    
    display = Instance(plt.Figure)
    diagnostic_display = Instance(plt.Figure)

    raw_images_displayed = Bool(False)
    down_sample_raw_image_display = Bool(True)

    sliced_image_index = None
    column_slice_button = Button("Column Slice")
    row_slice_button = Button("Row Slice")
    slice_all_images = Bool(False)
    slice_all_images_label = Str("Perform slice on all images")

    # For an edit-able title
    default_title_str = "GALAH Slow View"
    title_str = Str(default_title_str)

    blank = Str(" ")

    view = View(
        VGroup(
            HGroup(
                Item("show_science_image_wildmasks", style="simple", width=600, label="Current object image wildmask", padding=5),
                "5",
                Item("select_folder_to_watch_button", show_label=False, padding=5),
                spring,
                Item("automatically_show_new_images", show_label=False),
                Item("automatically_show_new_images_label", show_label=False, padding=5, style="readonly")),
            HSplit(
                VGroup(
                    Item("display", editor=MPLFigureEditor(), show_label=False),
                    HGroup(
                        Item("column_slice_button", enabled_when="raw_images_displayed", show_label=False),
                        "10",
                        Item("row_slice_button", enabled_when="raw_images_displayed", show_label=False),
                        Item("blank", style="readonly", springy=True, show_label=False),
                        Item("slice_all_images", enabled_when="raw_images_displayed", show_label=False),
                        Item("slice_all_images_label", show_label=False, style="readonly"),
                        label="Image Slicing",
                        show_border=True),
                    ),
                Item("diagnostic_display", editor=MPLFigureEditor(), show_label=False),
            ),
        ),
        width       = 1280,
        height      = 700,
        handler     = QuickViewHandler(),
        resizable   = True,
        title       = default_title_str
        )

    def __init__(self, folder_to_watch=None):
        """ Initialises the Slow View GUI """
        HasTraits.__init__(self)

        self.slice_mode = None
        if folder_to_watch is None:
            self.folder_to_watch = ""
            self.science_image_wildmasks = ["(None)"]
        else: self.folder_to_watch = folder_to_watch


    def _display_default(self):
        """ Initialises the display """

        figure = plt.Figure()
        figure.subplots_adjust(left=0.13, bottom=0.09,
            right=0.95, top=0.95, wspace=0.05, hspace=0.05)

        self._image_displays = []
        self._display_image_slices = []

        rect = figure.patch
        rect.set_facecolor("w")
        return figure


    def _diagnostic_display_default(self):
        """ Initialises the diagnostic displays """

        figure = plt.Figure()
        #figure.subplots_adjust(left=0.10, bottom=0.15,
        #    right=0.95, top=0.90, wspace=0.25)

        axes = figure.add_subplot(211)
        axes.xaxis.set_major_locator(MaxNLocator(6))
        axes.yaxis.set_major_locator(MaxNLocator(6))

        self._diagnostic_display_image_slices = [axes.plot([], [], color)[0] for color in ["blue", "green", "yellow", "red"]]
        axes.set_ylabel("Counts")
        axes.set_xlabel("Pixel")


        axes = figure.add_subplot(223)
        axes.xaxis.set_major_locator(MaxNLocator(6))
        axes.yaxis.set_major_locator(MaxNLocator(6))
        axes.set_xlabel("Row (px)")
        axes.set_ylabel("Estimated S/N")

        self._display_column_slice, = axes.plot([], [], "k")

        axes = figure.add_subplot(224)
        axes.xaxis.set_major_locator(MaxNLocator(6))
        axes.yaxis.set_major_locator(MaxNLocator(6))
        axes.set_xlabel("Estimated Mean Total S/N")
        axes.set_ylabel("Fibres")

        [axis.set_visible(False) for axis in figure.axes]


        initial_axis = figure.add_subplot(111, aspect="equal")
        initial_axis.set_frame_on(False)
        #initial_axis.xaxis.set_visible(False)
        initial_axis.xaxis.set_ticklabels([""] * len(initial_axis.xaxis.get_ticklabels()))
        [item.set_visible(False) for item in initial_axis.xaxis.get_majorticklines()]
        initial_axis.set_xlabel("GALAH Slow View (v0.01)\nAndy Casey (andy@the.astrowizici.st)")
        initial_axis.yaxis.set_visible(False)
        
        galah_image = plt.imread("galah.jpg")
        self._initial_display_axis = initial_axis.imshow(galah_image[::-1, :])

        self._initiated_all_axes = False
        
        rect = figure.patch
        rect.set_facecolor("w")

        return figure


    def _select_folder_to_watch_button_fired(self, value):

        dialog = SelectDirectory()
        dialog.configure_traits(kind="modal")

        print(dialog.__dict__)
        if dialog.directory:
            self.folder_to_watch = dialog.directory

    '''
    def _display_point_clicked(self, event):
        """ Point clicked on raw image display """

        if event.xdata is None: return
        column = int(np.round(event.xdata))

        # Draw a line on the plot
        if hasattr(self, "_display_column_slice_marker"):
            self._display_column_slice_marker.set_xdata([column, column])

        else:
            xlim = self.display.axes[0].get_xlim()
            ylim = self.display.axes[0].get_ylim()
            self._display_column_slice_marker, = self.display.axes[0].plot([column, column], ylim, "#046380", lw=2)
            self.display.axes[0].set_xlim(xlim)
            self.display.axes[0].set_ylim(ylim)

        # Update the line splice plot
        self._update_column_slice_display(column)
        wx.CallAfter(self.display.canvas.draw)
    
    '''
    def convert_filenames_to_wildmasks(self, images):

        wildmasks = []
        for filename in images:
            # Convert to a wildmask
            filename = filename.split("/")

            # Replace the CCD_X
            filename[-2] = filename[-2].replace("ccd_1", "ccd_?")

            # Replace the CCD_X number in the filename: (18nov10018 -> 18onv?0018)
            filename[-1] = list(filename[-1])
            filename[-1][5] = "?"
            filename[-1] = "".join(filename[-1])

            filename = "/".join(filename)
            wildmasks.append(filename)

        return wildmasks


    def filter_object_images(self, filenames):
        """ Returns only filenames that are OBJECT types. """

        object_images = []
        for filename in filenames:
            with pyfits.open(filename) as image:
                if image[0].header["OBSTYPE"] == "OBJECT":
                    object_images.append(filename)

        return object_images


    def _folder_to_watch_changed(self, folder_to_watch):
        """ The GALAH data folder has changed and we should update our
        current science images and wildmasks """
        
        # True images are in:
        # <folder_to_watch>/ccd_?/??feb?0018.fits

        # Search for folders in <folder_to_watch>/ccd_1/??????????.fits
        # Replace as <folder_to_watch>/ccd_?/18nov?0018.fits

        blue_images = self.filter_object_images(glob(os.path.join(folder_to_watch, "ccd_1/??????????.fits")))

        if len(blue_images) == 0:
            self.science_image_wildmasks = ["(No object images found)"]
            self._update_display_plots([])
            self.title_str = self.default_title_str

        else:
            self.science_image_wildmasks = self.convert_filenames_to_wildmasks(blue_images)
            self.title_str = " - ".join([self.default_title_str, folder_to_watch])
            self._show_science_image_wildmasks_changed(self.science_image_wildmasks[-1])

        # Initiate thread
        def callback(object_images):
            """ Callback function for when there are changes to
            the folder we're monitoring """

            wildmasks = self.convert_filenames_to_wildmasks(object_images)
            new_image_present = True if len(wildmasks) > 0 and wildmasks[-1] not in self.science_image_wildmasks else False
            print("new images present", new_image_present)
            self.science_image_wildmasks = wildmasks
            print("show new images", self.automatically_show_new_images)

            # If the first one in new_science_image_wildmasks is
            # not in our old_science_images, make it
            # the currently selected one, too
            if new_image_present and self.automatically_show_new_images:
                
                # Wait to make sure all files can be opened
                latest_image_filenames = glob(self.science_image_wildmasks[-1])
                print("latest images are", latest_image_filenames)

                while True:
                    try:
                        for latest_image_filename in latest_image_filenames:
                            with pyfits.open(latest_image_filename) as fp:
                                print("opened", latest_image_filename)
                            
                    except:
                        print("Image {0} not fully written to disk yet. Trying again in one second..".format(latest_image_filename))
                        sleep(1)
                        continue

                    else:
                        # Looks OK to load
                        print("UPDATING TO SHOW IMAGES", self.science_image_wildmasks[-1])
                        self.show_science_image_wildmasks = self.science_image_wildmasks[-1]
                        #self.current_science_images = latest_images
                        break

            
        if not hasattr(self, "watch_folder_thread"):
            self.watch_folder_thread = MonitorFolder(
                [folder_to_watch],
                blue_images,
                [],
                callback
                )
            self.watch_folder_thread.start()

        else:
            self.watch_folder_thread.expected_blue_images = blue_images
            self.watch_folder_thread.expected_blue_images_to_ignore = []
            self.watch_folder_thread.folder = [folder_to_watch]


    def _science_image_wildmasks_changed(self, science_image_wildmasks):
        """ Object images have been added or removed """
        print("SCIENCE IMAGE WILDMASKS:", science_image_wildmasks)

    def _show_science_image_wildmasks_changed(self, selected_science_image_wildmask):
        """ Updating the selected object image wildmask """
        print("images to show updated")
        self.current_science_images = glob(selected_science_image_wildmask)

        images = map(pyfits.open, self.current_science_images)

        print("Calculating with....", self.current_science_images)
        self._update_total_fibre_snr_plot(images)
        self._update_display_plots(images)

        [image.close() for image in images]


   


    def _update_total_fibre_snr_plot(self, images):
        """ Updates the estimated total fibre S/N for the current images """

        # Sometimes (18nov10013 vs [18nov10014 or 18nov20013]) the fibre table is the first index
        # sometimes it is not.

        fibre_table_extension = 2 if images[0][1].data.shape == (1, ) else 1
        fibre_table = images[0][fibre_table_extension].data

        # Find which fibres are sky fibres
        sky_fibres = np.where(fibre_table["TYPE"] == "S")[0]

        # Find which fibres are program fibres
        program_fibres = np.where(fibre_table["TYPE"] == "P")[0]

        def clear_snr_plot():
            [patch.set_visible(False) for patch in self.diagnostic_display.axes[2].patches]
            self.diagnostic_display.axes[2].patches = []
            if self.diagnostic_display.canvas is not None:
                wx.CallAfter(self.diagnostic_display.canvas.draw)
            

        if len(sky_fibres) == 0:
            clear_snr_plot()
            raise IOError("No sky fibres found in image! No estimate of fibre S/N can be made.")
            
        if len(program_fibres) == 0:
            clear_snr_plot()
            raise IOError("No program fibres found in image! No estimate of fibre S/N can be made.")
     
        extraction_widths = {
            "blue": 3.5,
            "green": 3.5,
            "yellow": 3.5,
            "red": 3.5
        } 

        estimated_channel_snr = {}
        for channel, image in zip(extraction_widths.keys(), images):
            start_time = time()

            # Open a tram-line map
            tram_map_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tram_maps/{0}.map".format(channel))
            if not os.path.exists(tram_map_filename):
                raise IOError("tram map filename not found: {0}".format(tram_map_filename))

            with open(tram_map_filename, "r") as fp:
                tram_map = pickle.load(fp)

            # Perform fibre extraction
            sky_fibre_fluxes = calibrate.extract_fibres(image, tram_map,
                extraction_widths[channel], sky_fibres, mode="median")
            program_fibre_fluxes = calibrate.extract_fibres(image, tram_map,
                extraction_widths[channel], program_fibres, mode="sum")

            # Calculate mean sky fibre fluxes along each column (Wavelength)
            median_sky_fibre_flux = np.mean(sky_fibre_fluxes, axis=0)
            
            # Calculate object pixel fluxes / mean sky fibre fluxes
            #estimated_pixel_snr = program_fibre_fluxes / (program_fibre_fluxes + median_sky_fibre_flux)**0.5
            estimated_pixel_counts = program_fibre_fluxes / median_sky_fibre_flux
            estimated_mean_fibre_counts = np.mean(estimated_pixel_counts, axis=1)

            print("num fibres", len(estimated_mean_fibre_counts))


            # Draw histogram of S/N
            estimated_channel_snr[channel] = estimated_mean_fibre_counts
            print("Completed {0} channel in {1:.2f}".format(channel, time() - start_time))

        # Clear any previous patches
        clear_snr_plot()

        bin_size = 2
        x_limits = sum([[np.min(channel_snr), np.max(channel_snr)] for channel_snr in estimated_channel_snr.values()], [])
        x_limits = [np.floor(np.min(x_limits)) - bin_size, np.ceil(np.max(x_limits)) + bin_size]

        bins = np.arange(x_limits[0], x_limits[1] + bin_size, bin_size)
        all_y_values = []
        for channel, snr_values in estimated_channel_snr.iteritems():
            y_values, returned_bins, patches = self.diagnostic_display.axes[2].hist(
                snr_values, bins=bins, color=channel, alpha=0.5)
            all_y_values.append(y_values)

        self.diagnostic_display.axes[2].set_ylim(0, max(map(max, all_y_values)))
        self.diagnostic_display.axes[2].set_xlim(x_limits)
        
        if self.diagnostic_display.canvas is not None:
            wx.CallAfter(self.diagnostic_display.canvas.draw)


    def _update_display_plots(self, images):
        print("drawing")

        channels = ["blue", "green", "yellow", "red"]
        if not self._initiated_all_axes:
            self._initial_display_axis.set_visible(False)
            del self._initial_display_axis
            self.diagnostic_display.axes[-1].set_axis_off()
            self.diagnostic_display.axes[-1].set_frame_on(False)

            axes = [self.display.add_subplot(2, 2, i) for i in xrange(1, 5)]
            
            [axis.set_visible(True) for axis in self.diagnostic_display.axes]

            # Hide unnecessary axes
            [axes[i].xaxis.set_visible(False) for i in [0, 1]]
            [axes[i].yaxis.set_visible(False) for i in [1, 3]]

            # Put labels on where relevant
            [axes[i].set_xlabel("Column (px)") for i in [2, 3]]
            [axes[i].set_ylabel("Row (px)") for i in [0, 2]]

            for axis, channel in zip(axes, channels):
                for line in axis.spines:
                    axis.spines[line].set_edgecolor(channel)
                    axis.spines[line].set_linewidth(3)

            self._initiated_all_axes = True

        # Draw the images
        for i, (axis, channel, image) in enumerate(zip(self.display.axes, channels, images)):
            
            data = image[0].data

            # Down-sample?
            if self.down_sample_raw_image_display:

                dpi = 400

                # Get axes size in inches
                extent = axis.get_window_extent().transformed(self.display.dpi_scale_trans.inverted())
                display_num_x_pixels, display_num_y_pixels = map(abs, (extent.p0 - extent.p1) * dpi)

                down_sample_rate = int(np.floor(np.min([data.shape[0]/display_num_x_pixels, data.shape[1]/display_num_y_pixels])))
                print("Down sampling rate is {0}".format(down_sample_rate))

                data = data[::down_sample_rate, ::down_sample_rate]

            else: down_sample_rate = 1

            # Clip the data +/- 1.5 sigma just for plotting purposes
            clip, mean, sigma = 1.5, np.mean(data), np.std(data)
            data = np.clip(data, mean - clip*sigma, mean + clip*sigma)

            if len(self._image_displays) < 4:
                self._image_displays.append(axis.imshow(data, cmap="gray", aspect="auto", interpolation="nearest"))
                self._display_image_slices.append(axis.plot([], lw=2, color=channel)[0])
                
            else:

                self._image_displays[i].set_data(data)
                self._image_displays[i].autoscale()

            axis.set_xlim(0, data.shape[0])
            axis.set_ylim(0, data.shape[1])

            axis.set_xticklabels(["{0:.0f}".format(int(num) * down_sample_rate) for num in axis.get_xticks()])
            axis.set_yticklabels(["{0:.0f}".format(int(num) * down_sample_rate) for num in axis.get_yticks()])


            self.raw_images_displayed = True

        if self.display.canvas is not None:
            wx.CallAfter(self.display.canvas.draw)


        return None

    def _update_diagnostic_display_plots(self):
        raise NotImplementedError


    def _column_slice_button_fired(self, value):
        self.slice_mode = "column"
        print("Click the mouse where you would like to perform a column slice")

        if not hasattr(self, "_mouse_click_event"):
            self._mouse_click_event = self.display.canvas.mpl_connect("button_press_event", self._raw_image_mouse_click_fired)


    def _row_slice_button_fired(self, value):
        self.slice_mode = "row"
        print("Click the mouse where you would like to perform a row slice")

        if not hasattr(self, "_mouse_click_event"):
            self._mouse_click_event = self.display.canvas.mpl_connect("button_press_event", self._raw_image_mouse_click_fired)


    def _raw_image_mouse_click_fired(self, event):

        print(event.__dict__)

        # Column slice or row slice?
        if self.slice_mode == "row":
            x_data = self.display.axes[0].get_xlim()
            y_data = [event.ydata, event.ydata]

        else:
            x_data = [event.xdata, event.xdata]
            y_data = self.display.axes[0].get_ylim()

        # Save the axes index for if self.slice_all_images changes!
        self.sliced_image_index = self.display.axes.index(event.inaxes)

        # Draw slice(s) 
        for axis, line in zip(self.display.axes, self._display_image_slices):
            current_xlim, current_ylim = axis.get_xlim(), axis.get_ylim()

            line.set_data(x_data, y_data)

            visible = self.slice_all_images or axis == event.inaxes
            line.set_visible(visible)

            axis.set_xlim(current_xlim)
            axis.set_ylim(current_ylim)

        wx.CallAfter(self.display.canvas.draw)

        # Update slice plot
        y_limits = None
        for axis, image, display_slice in zip(self.display.axes, self._image_displays, self._diagnostic_display_image_slices):

            # Extract the data slice
            x_offset, y_offset = (0, 1) if self.slice_mode == "row" else (1, 0)
            data_slice = np.array(image.get_array())[int(y_data[0]):int(y_data[1]) + y_offset, int(x_data[0]):int(x_data[1]) + x_offset].flatten()
            print(data_slice)
            display_slice.set_data(np.arange(len(data_slice)), data_slice)

            visible = self.slice_all_images or axis == event.inaxes
            display_slice.set_visible(visible)

            if visible and y_limits is None:
                y_limits = np.min(data_slice), np.max(data_slice)

            elif visible:
                y_limits = np.min([y_limits[0], np.min(data_slice)]), np.max([y_limits[1], np.max(data_slice)])

        self.diagnostic_display.axes[0].set_xlabel("Column (px)" if self.slice_mode == "row" else "Row (px)")
        self.diagnostic_display.axes[0].set_xlim(0, len(data_slice))
        self.diagnostic_display.axes[0].set_ylim(y_limits)

        wx.CallAfter(self.display.canvas.draw)
        wx.CallAfter(self.diagnostic_display.canvas.draw)


    def _slice_all_images_changed(self, slice_all_images):

        # Is there a previous slice?
        if self.sliced_image_index is None: return

        # Hide the lines on axes that weren't originally sliced
        for index, (axis, line) in enumerate(zip(self.display.axes, self._display_image_slices)):
            if index != self.sliced_image_index:
                line.set_visible(slice_all_images)

        current_ylim = self.diagnostic_display.axes[0].get_ylim()
        new_ylim = [] + list(current_ylim)
        for index, (axis, line) in enumerate(zip(self.display.axes, self._diagnostic_display_image_slices)):
            if index != self.sliced_image_index:
                line.set_visible(slice_all_images)

            if slice_all_images:
                new_ylim = np.min([new_ylim[0], np.min(line.get_data()[1])]), np.max([new_ylim[1], np.max(line.get_data()[1])])
            elif index == self.sliced_image_index:
                new_ylim = np.min(line.get_data()[1]), np.max(line.get_data()[1])

        self.diagnostic_display.axes[0].set_ylim(new_ylim)

        wx.CallAfter(self.display.canvas.draw)
        wx.CallAfter(self.diagnostic_display.canvas.draw)


    

if __name__ == "__main__":
    quickview = QuickView("/Users/andycasey/thesis/research/hermes/reduction-testing/18nov2013")
    quickview.configure_traits()