import sys
sys.path.insert(1, '/Users/andrew/Documents/python/pygplates_rev28_python37_MacOS64')
import pygplates
import numpy as np
import glob
import pandas as pd
import pickle
import dill

import slab_tracker_utils as slab
import splits_and_merges as snm
import slab_cross_section_utils as sxs

import xarray as xr
from scipy import spatial

from scipy.interpolate import interp1d


def get_subducted_slabs(rotation_model, slab_times, grids, dip_data):
    '''
    get iso-sub chrons with dips from slab 2.0 geometry
    '''

    output_data = []

    time_list = slab_times

    if handle_splits:
        plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(
                                             reconstruction_model,
                                             time_list,
                                             verbose=True)

        print (plate_disappearance_time_lut)

    # loop over a series of times at which we want to extract trench iso-sub-chrons
    for time in time_list:

        print( 'time %0.2f Ma' % time)
        tmp_plate_snapshot = reconstruction_model.plate_snapshot(time)
        # Get subduction boundary segments
        subduction_boundary_sections = tmp_plate_snapshot.get_boundary_features('subduction')

        # Set up a grid interpolator for this time, to be used
        # for each tessellated line segment
        lut = []
        if grid_filename is not None:
            for ind,i in enumerate(grid_filename):
                count = ind + 1
                if count % 2 != 0:
                    grdfile = '%s%d%s' % (grid_filename[ind],time,grid_filename[ind+1])
                    lut.append(slab.make_grid_interpolator(grdfile))


        #print subduction_boundary_sections

        # Loop over each segment
        for segment_index,subduction_segment in enumerate(subduction_boundary_sections):

            # find the overrding plate id (and only continue if we find it)
            overriding_and_subducting_plates = slab.find_overriding_and_subducting_plates(subduction_segment,time)

            if not overriding_and_subducting_plates:
                continue
            overriding_plate, subducting_plate, subduction_polarity = overriding_and_subducting_plates

            overriding_plate_id = overriding_plate.get_resolved_feature().get_reconstruction_plate_id()
            subducting_plate_id = subducting_plate.get_resolved_feature().get_reconstruction_plate_id()

            #if (opid!=224 or cpid!=909):
            #if (subducting_plate_id!=911 and subducting_plate_id!=909):
            #if subducting_plate_id<900:
            #    continue

            subducting_plate_disappearance_time = -1.
            if handle_splits:
                for plate_disappearance in plate_disappearance_time_lut:
                    if plate_disappearance[0]==subducting_plate_id:
                        subducting_plate_disappearance_time = plate_disappearance[1]

            tessellated_line = subduction_segment.get_resolved_geometry().to_tessellated(line_tessellation_distance)

            variable_results = []
            if lut is not None:
                x = tessellated_line.to_lat_lon_array()[:,1]
                y = tessellated_line.to_lat_lon_array()[:,0]
                for i in lut:
                    variable_results.append(i.ev(np.radians(y+90.),np.radians(x+180.)))
            else:
                # if no age grids, just fill the ages with zero
                variable_results = [0. for point in tessellated_line.to_lat_lon_array()[:,1]]

            # CALL THE MAIN WARPING FUNCTION
            (points,
             point_depths,
             polyline,dips, convergence_rates) = slab.warp_subduction_segment(tessellated_line,
                                                      rotation_model,
                                                      subducting_plate_id,
                                                      overriding_plate_id,
                                                      subduction_polarity,
                                                      time,
                                                      end_time,
                                                      time_step,
                                                      clean_dips,
                                                      ground_pixel_tree,
                                                      subducting_plate_disappearance_time)
            #print(dips)
            output_data.append([time,polyline,point_depths,variable_results, dips, overriding_plate_id, subducting_plate_id, convergence_rates])


    # write out the features for GPlates
    #output_features = pygplates.FeatureCollection(point_features)

    ### write results to file

    #NB not corrected for multiple variables (yet)

    #slab.write_subducted_slabs_to_xyz(output_filename,output_data)
    #close dataset

    return output_data
