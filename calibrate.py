# coding: utf-8

""" Simple data reduction routines for AAO/HERMES multiplex data """

__author__ = "Andy Casey <andy@astrowizici.st>"

# Standard libraries
import logging
import os
from time import time 
import multiprocessing as mpi

# Third party libraries
import numpy as np
import pyfits

__all__ = ["read_tramline_map", "extract_fibres"]

def read_tramline_map(filename, order=2):
    """ Reads a 2dfdr-reduced tramline map and returns an array of polynomial
    coefficients that will reproduce the tramline map for all fibres. """

    with pyfits.open(filename) as image:

        data = image[0].data
        num_fibres, num_pixels = data.shape
        if num_fibres != 400:
            print("Warning: Expected 400 fibres in tramline map but found {0}".format(num_fibres))

        x = np.arange(num_pixels)
        coefficients = np.zeros((num_fibres, order + 1))
        for i in xrange(num_fibres):
            coefficients[i] = np.polyfit(x, data[i], order)

    return coefficients


def extract_fibres(image, tram_map, extraction_width=2, fibre_indices="all", mode="sum"):
    """ Performs a rough extraction of fibres from a raw image. """

    if mode not in ("sum", "median", "mean"):
        raise ValueError("extraction mode must be sum, median or mean")

    else:
        extraction_method = {"sum": sum, "median": np.median, "mean": np.mean}[mode]

    fibre_indices = xrange(400) if fibre_indices == "all" else fibre_indices
    fibre_table_extension = 2 if image[1].data.shape == (1, ) else 1
    
    # Where are the data on the image?
    x_data_slice = [image[0].header[key] for key in ("DETECXS", "DETECXE")]
    y_data_slice = [image[0].header[key] for key in ("DETECYS", "DETECYE")]

    # x_data_slice and y_data_slice are inclusive indices. Python works differently
    # for indexation of arrays, and as such we have to do this:
    x_data_slice[0] -= 1
    y_data_slice[0] -= 1

    # Slice out the real data
    image_data = image[0].data[x_data_slice[0]:x_data_slice[1], y_data_slice[0]:y_data_slice[1]]

    # Prepare the flux array
    num_x_pixels, num_y_pixels = map(np.ptp, [x_data_slice, y_data_slice])
    x_values, y_values = map(np.arange, [num_x_pixels, num_y_pixels])
    extracted_flux = np.zeros((len(fibre_indices), num_x_pixels))

    # [TODO] Thread it?
    # tram_map, x_values, extraction_width, image_data, extraction_method
    for i, fibre_index in enumerate(fibre_indices):

        # Create the tramline polynomials for the fibres to extract
        tram_line_y_values = np.polyval(tram_map[fibre_index], x_values)
        extracted_fibre_flux = np.empty(len(x_values))
        for x, y in zip(x_values, tram_line_y_values):

            # Retrieve all the y-pixels in the extraction region for this x-pixel
            flux_indices = [int(np.ceil(y - extraction_width)), int(np.floor(y + extraction_width)) + 1]

            # Aggregate (eg sum, median, mean) all the pixels along the y-direction
            # NOTE: The indexation here for images differs between 2dfdr and Python: x, y = y, x
            extracted_flux[i, x] = extraction_method(image_data[flux_indices[0]:flux_indices[1], x])

    return extracted_flux

    

