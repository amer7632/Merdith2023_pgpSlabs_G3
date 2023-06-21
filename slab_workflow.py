# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:12:04 2023

@author: Andrew Merdith, comments by Kev Wong
"""

from gplately import pygplates
import numpy as np
import pandas as pd
import slab_tracker_utils as slab
import splits_and_merges as snm
from scipy import spatial
from scipy.interpolate import interp1d
from slabdip import SlabDipper

# Define some input files and import them into pygplates.
# In this case, the Muller et al., 2016, AREPS is used...
RotFile_List = ['./updatedMullerModel/Global_EarthByte_230-0Ma_GK07_AREPS.rot']
GPML_List = ['./updatedMullerModel/Global_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries.gpml',
             './updatedMullerModel/Global_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks' +
             '.gpml']

rotation_model = pygplates.RotationModel(RotFile_List)
topology_features = pygplates.FeatureCollection()

for file in GPML_List:
    topology_feature = pygplates.FeatureCollection(file)
    topology_features.add(topology_feature)

# Input parameters for slab tracing.
start_time = 15.
end_time = 0.
time_step = 1.0
dip_angle_degrees = 45.0
line_tessellation_distance = np.radians(1.0)
handle_splits = True
output_filename = 'subduction_3d_geometries_time_%0.2fMa_dip_%0.2fdeg.p' % (end_time,
                                                                            dip_angle_degrees)

# Try to use small circle path for stage rotation to rotate along velocity dip.
# i.e., distance to stage rotation pole matches distance to original stage pole.
# use_small_circle_path = False

# def find_with_list(myList, target):  # I'm pretty sure this function is not needed (KW), and I'm
# pretty sure that what is does is already a python method
#     '''
#     find index of thing in list
#     '''

#     inds = []
#     for i in range(len(myList)):
#         if myList[i] == target:
#             inds += i,
#     return inds


def get_slab_surfaces(slab_dir, FNAME):
    """
    Generate slab surface information from a directory and Syracuse et al., 2010 temperature
    profile file.

    Parameters
    ----------
    slab_dir : string
        String of the directory where the slab file can be found.
    FNAME : string
        Name of the slab temperature file. Syracuse files are tab-delimited files arranged by:
            Column 0: surface ('0') or Moho ('1')
            Column 1: trench-orthoganal distance (km)
            Column 2: slab feature (surface/Moho) depth
            Column 3: temperature (degC).

    Returns
    -------
    surface_output : list of numpy.arrays
        Two-item list comprising array of slab surface temperatures and array of Moho temperatures.
    isotherm_output : list of lists
        Three-item list comprising one-item lists of slab surface Curie temperature depth,
        Moho Curie temperature depth, and slab surface magnetite-out depth.

    """

    # Shift = 0  # Not used (KW)
    # Some lists for storage...
    MagnetiteIsoDepths = []
    SurfaceCurieIsoDepths = []
    MohoCurieIsoDepths = []

    # Open the file with the slab info:
    fid = open(slab_dir+'/'+FNAME, 'r')
    SurfaceArray = []
    MohoArray = []

    # Iterate over all lines in the slab file:
    for line in fid:
        # strip removes the '\n' at the end
        tline = line.strip()
        # print(tline[0])
        # Case where line contains the top surface of slab (in Syracuse files '0' represents the
        # slab top and '7' the Moho surface)
        if '0' in tline[0]:
            tline = list(map(float, tline.split()))
            SurfaceArray.append(tline)
        # Else we're at the moho
        else:
            tline = list(map(float, tline.split()))
            MohoArray.append(tline)
    fid.close()  # This line was originally commented out but is probably necessary
    SurfaceArray = np.asarray(SurfaceArray)
    MohoArray = np.asarray(MohoArray)

    # Interpolate over the Surface array
    SurfaceCurieInterpolator = interp1d(SurfaceArray[:, 3], SurfaceArray[:, 2], kind='linear')

    # Find the Curie isotherm depth of the slab surface at 550 degC
    try:
        SurfaceCurieIsoDepth = SurfaceCurieInterpolator(550.)
    except ValueError:
        SurfaceCurieIsoDepth = np.nan

    # Repeat for the Moho Curie isotherm:
    MohoCurieInterpolator = interp1d(MohoArray[:, 3], MohoArray[:, 2], kind='linear')
    try:
        MohoCurieIsoDepth = MohoCurieInterpolator(550.)
    except ValueError:
        MohoCurieIsoDepth = np.nan

    # One more time for the slab surface magnetite isotherm at 200 degC:
    MagnetiteInterpolator = interp1d(SurfaceArray[:, 3], SurfaceArray[:, 2], kind='linear')
    try:
        MagnetiteIsoDepth = MagnetiteInterpolator(200.)
    except ValueError:
        MagnetiteIsoDepth = np.nan

    # Append everything to the lists and output
    SurfaceCurieIsoDepths.append(SurfaceCurieIsoDepth)
    MagnetiteIsoDepths.append(MagnetiteIsoDepth)
    MohoCurieIsoDepths.append(MohoCurieIsoDepth)
    surface_output = [SurfaceArray, MohoArray]
    isotherm_output = [SurfaceCurieIsoDepths, MohoCurieIsoDepths, MagnetiteIsoDepths]

    return surface_output, isotherm_output

# KW: I have commented out this function since a) it is not needed in this particular code as far
# as I can tell, and b) it refers to a module
# shorthanded 'sxs' which is never called.
# def get_intersecting_values(intersecting_lines_1, cross_section_line):
#     """
#     now that we have our intersecting lines, we can find the containing segment
#     of the line, in order to access the correct points (and then the correct
#     depths, variables etc.)
#     I don't know why this function appears here as opposed to further below when the lines are
#     defined (KW)

#     Parameters
#     ----------
#     intersecting_lines_1 : TYPE
#         DESCRIPTION.
#     cross_section_line : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     intersecting_points_reversed : TYPE
#         DESCRIPTION.
#     intersecting_points : TYPE
#         DESCRIPTION.
#     interpolated_variables : TYPE
#         DESCRIPTION.
#     interpolated_depths : TYPE
#         DESCRIPTION.

#     """

#     intersecting_points = []
#     interpolated_variables = []
#     interpolated_depths = []

#     for ind, line in enumerate(intersecting_lines_1):

#         # for clarity we will enunciate each iso-subchron
#         age_of_subdcution = line[0]
#         iso_subchron = line[1]
#         depth = line[2]

#         # need to convert convergence rate to units
#         variable_data = []
#         # print(variable_data)
#         conv_rates = []
#         for conv_rate in line[7]:
#             if isinstance(conv_rate, float):
#                 conv_rates.append(conv_rate)
#             else:
#                 conv_rates.append(conv_rate.get_magnitude())
#         conv_rates = np.asarray(conv_rates)
#         variable_data.append(line[3])
#         variable_data = np.asarray(variable_data[0])
#         merged_variable_data = np.vstack((variable_data, conv_rates))

#         # get intersect points, and starting indices of the segments that contain intersect
#         # points of our two lines (cross section line, and iso-subchron)
#         closest_point_data = pygplates.GeometryOnSphere.distance(iso_subchron,
#                                                                  cross_section_line,
#                                                                  return_closest_positions=True,
#                                                                  return_closest_indices=True)
#         # for clarity we will enunciate the closest point data
#         # tmp_distance = closest_point_data[0]  # This is just never assigned (KW)
#         # NB these next two should be the same
#         tmp_sub_isochron_intercept = closest_point_data[1]
#         # tmp_cross_section_intercept = closest_point_data[2]  # Never assigned (KW)

#         # the indices refer to the start of the containing segment
#         tmp_sub_isochron_segment_index = closest_point_data[3]
#         # tmp_cross_section_segment_index = closest_point_data[4]  # Never assigned (KW)

#         # set depths and variable indices
#         # print(merged_variable_data)
#         for variable_index, array in enumerate(merged_variable_data):
#             variables = merged_variable_data[variable_index][tmp_sub_isochron_segment_index:
#                                                              tmp_sub_isochron_segment_index+2]
#             interpolated_variable = sxs.get_intercept_values(tmp_sub_isochron_intercept,  # sxs?
#                                                              iso_subchron, variables)
#             interpolated_variables.append(interpolated_variable)

#         depths = depth[tmp_sub_isochron_segment_index:tmp_sub_isochron_segment_index+2]

#         # NB (check)
#         # because we are plotting depths and explicit distances along, we inherently correct for
#         # true apparent dip
#         interpolated_depth = sxs.get_intercept_values(tmp_sub_isochron_intercept, #sxs? (KW))
#                                                       iso_subchron, depths)
#         interpolated_depths.append(interpolated_depth)

#         tmp_values = (age_of_subduction,
#                       tmp_sub_isochron_intercept,
#                       depth[tmp_sub_isochron_segment_index],
#                       [i[tmp_sub_isochron_segment_index] for i in merged_variable_data])

#         intersecting_points.append(tmp_values)

#     # reverse intersecting points so we're starting from the subduction zone
#     intersecting_points_reversed = intersecting_points[::-1]

#     return (intersecting_points_reversed, intersecting_points,
#             interpolated_variables, interpolated_depths)


def cross_section_line_pygplates(lat1, lon1, lat2, lon2, spacing):
    """
    Given two points and spacing between points, return a pygplates tessellated line.

    Parameters
    ----------
    lat1 : float
        Latitude of point 1 in decimal degrees.
    lon1 : float
        Latitude of point 1 in decimal degrees.
    lat2 : float
        Latitude of point 2 in decimal degrees.
    lon2 : float
        Latitude of point 2 in decimal degrees.
    spacing : float
        Spacing of cross section line in degrees.

    Returns
    -------
    cross_section_line : list of pygplates.PointOnSphere
        List of Cartesian points correponding to a tesselated polyline.

    """

    cross_section_points = []
    cross_section_points.append((lat1, lon1))
    cross_section_points.append((lat2, lon2))
    # make line feature
    cross_section_line = pygplates.PolylineOnSphere(cross_section_points)
    # tessellate line
    cross_section_line = cross_section_line.to_tessellated(np.radians(float(spacing)))

    return cross_section_line


def get_subducted_slabs(start_time, end_time, time_step, grid_filename, slab_XR):
    """
    Get iso-sub chrons (isochrons of original trench position) with dips from Slab2 geometry

    Parameters
    ----------
    start_time : integer or float
        Start time of reconstruction.
    end_time : integer or float
        Finishing time of reconstruction.
    time_step : integer or float
        Time steps taken between start_time and end_time.
    grid_filename : netcdf4 file
        DESCRIPTION.
    slab_XR : TYPE  # Some kind of Slab2 data?
        DESCRIPTION.

    Returns
    -------
    output_data : TYPE
        DESCRIPTION.

    """

    # I assume that this is Slab2 data sans all NaN points? KW
    clean_dips = slab_XR.dip[slab_XR.dip.notnull()]

    # I now assume that this extracts an array of slab latitudes and longitudes?
    coords = np.column_stack((clean_dips.latitude.values.ravel(),
                              clean_dips.longitude.values.ravel()))

    # This creates a kd-tree which an be used for quick nearest-neighbour lookup.
    ground_pixel_tree = spatial.KDTree(slab.transform_coordinates(coords))

    # Establishing some other things...
    output_data = []
    time_list = np.arange(start_time, end_time-time_step, -time_step)

    # This is the handle_splits boolean variable called at the start of the file. This really
    # should be a parameter in its own right here to avoid confusion... KW
    # Anyways, if the plate splits, then disappearance times are printed.
    if handle_splits:
        plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(topology_features,
                                                                            rotation_model,
                                                                            time_list,
                                                                            verbose=True)

        print(plate_disappearance_time_lut)

    # Loop over a series of times at which we want to extract trench iso-sub-chrons
    for time in time_list:
        print('time %0.2f Ma' % time)

        # Call function to get subduction boundary segments
        subduction_boundary_sections = slab.getSubductionBoundarySections(topology_features,
                                                                          rotation_model,
                                                                          time)

        # Set up an grid interpolator for this time, to be used for each tessellated line segment
        lut = []
        if grid_filename is not None:
            # Iterate over grid indices and values
            for ind, i in enumerate(grid_filename):
                count = ind + 1
                if count % 2 != 0:
                    grdfile = '%s%d%s' % (grid_filename[ind], time, grid_filename[ind+1])
                    lut.append(slab.make_age_interpolator(grdfile))

        # print(subduction_boundary_sections)

        # Loop over each segment
        for segment_index, subduction_segment in enumerate(subduction_boundary_sections):

            # find the overrding plate id (and only continue if we find it)
            overriding_and_subducting_plates = slab.find_overriding_and_subducting_plates(
                subduction_segment, time)
            if not overriding_and_subducting_plates:
                continue
            (overriding_plate, subducting_plate,
             subduction_polarity) = overriding_and_subducting_plates

            overriding_plate_id = overriding_plate.get_resolved_feature(
                ).get_reconstruction_plate_id()
            subducting_plate_id = subducting_plate.get_resolved_feature(
                ).get_reconstruction_plate_id()

            # Some legacy code here...
            # if (opid != 224 or cpid != 909):
            # if (subducting_plate_id != 911 and subducting_plate_id != 909):
            # if subducting_plate_id < 900:
            #    continue

            # Find the time at which the plate disappears
            subducting_plate_disappearance_time = -1.
            if handle_splits:
                for plate_disappearance in plate_disappearance_time_lut:
                    if plate_disappearance[0] == subducting_plate_id:
                        subducting_plate_disappearance_time = plate_disappearance[1]
            print(subducting_plate_disappearance_time)

            # Tessellate the polyline representing subduction
            tessellated_line = subduction_segment.get_resolved_geometry(
                ).to_tessellated(line_tessellation_distance)

            # dip_at_tesselated_line = get_dip(points)

            variable_results = []
            if lut:

                # x, y represent longitude and latitude on a sphere respectively
                x = tessellated_line.to_lat_lon_array()[:, 1]
                y = tessellated_line.to_lat_lon_array()[:, 0]
                for i in lut:
                    # Not really too sure what is happening over here - maybe something Slab2? (KW)
                    variable_results.append(i.ev(np.radians(y+90.), np.radians(x+180.)))
            else:
                # if no age grids, just fill the ages with zero
                variable_results = [0. for point in tessellated_line.to_lat_lon_array()[:, 1]]

            # CALL THE MAIN WARPING FUNCTION
            (points,
             point_depths,
             polyline, dips, convergence_rates) = slab.warp_subduction_segment(
                 tessellated_line,
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
            # print(dips)
            output_data.append([time, polyline, point_depths, variable_results, dips,
                                overriding_plate_id, subducting_plate_id, convergence_rates])

    # Legacy code and comments here... (KW)
    # write out the features for GPlates
    # output_features = pygplates.FeatureCollection(point_features)
    # write results to file
    # NB not corrected for multiple variables (yet)
    # slab.write_subducted_slabs_to_xyz(output_filename,output_data)
    # close dataset

    return output_data

# This function seems depreciated so I have commented it out (KW)
# def get_subducted_slabs_ORIGINAL(start_time, end_time, time_step, grid_filename, slab_XR):
#     '''
#     get iso-sub chrons with dips from slab 2.0 geometry
#     '''
#     clean_dips = slab_XR.dip[slab_XR.dip.notnull()]

#     coords = np.column_stack((clean_dips.latitude.values.ravel(),
#                               clean_dips.longitude.values.ravel()))

#     ground_pixel_tree = spatial.cKDTree(slab.transform_coordinates(coords))

#     output_data = []

#     time_list = np.arange(start_time, end_time-time_step, -time_step)

#     if handle_splits:
#         plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(topology_features,
#                                                                             rotation_model,
#                                                                             time_list,
#                                                                             verbose=True)

#         print(plate_disappearance_time_lut)

#     # loop over a series of times at which we want to extract trench iso-sub-chrons
#     for time in time_list:

#         print('time %0.2f Ma' % time)

#         # call function to get subduction boundary segments
#         subduction_boundary_sections = slab.getSubductionBoundarySections(topology_features,
#                                                                           rotation_model,
#                                                                           time)

#         # Set up an grid interpolator for this time, to be used
#         # for each tessellated line segment
#         lut = []
#         if grid_filename is not None:
#             for ind, i in enumerate(grid_filename):
#                 count = ind + 1
#                 if count % 2 != 0:
#                     grdfile = '%s%d%s' % (grid_filename[ind], time, grid_filename[ind+1])
#                     lut.append(slab.make_age_interpolator(grdfile))

#         # print(subduction_boundary_sections)

#         # Loop over each segment
#         for segment_index, subduction_segment in enumerate(subduction_boundary_sections):

#             # find the overrding plate id (and only continue if we find it)
#             overriding_and_subducting_plates = slab.find_overriding_and_subducting_plates(
#                 subduction_segment, time)

#             if not overriding_and_subducting_plates:
#                 continue
#             (overriding_plate, subducting_plate,
#              subduction_polarity) = overriding_and_subducting_plates

#             overriding_plate_id = overriding_plate.get_resolved_feature(
#                 ).get_reconstruction_plate_id()
#             subducting_plate_id = subducting_plate.get_resolved_feature(
#                 ).get_reconstruction_plate_id()

#             # if (opid != 224 or cpid != 909):
#             # if (subducting_plate_id != 911 and subducting_plate_id != 909):
#             # if subducting_plate_id < 900:
#             #    continue

#             subducting_plate_disappearance_time = -1.
#             if handle_splits:
#                 for plate_disappearance in plate_disappearance_time_lut:
#                     if plate_disappearance[0] == subducting_plate_id:
#                         subducting_plate_disappearance_time = plate_disappearance[1]

#             tessellated_line = subduction_segment.get_resolved_geometry().to_tessellated(
#                 line_tessellation_distance)

#             # dip_at_tesselated_line = get_dip(points)

#             variable_results = []
#             if lut is not None:
#                 x = tessellated_line.to_lat_lon_array()[:, 1]
#                 y = tessellated_line.to_lat_lon_array()[:, 0]
#                 for i in lut:
#                     variable_results.append(i.ev(np.radians(y+90.), np.radians(x+180.)))
#             else:
#                 # if no age grids, just fill the ages with zero
#                 variable_results = [0. for point in tessellated_line.to_lat_lon_array()[:, 1]]

#             # CALL THE MAIN WARPING FUNCTION
#             (points, point_depths, polyline, dips,
#              convergence_rates) = slab.warp_subduction_segment(
#                  tessellated_line,
#                  rotation_model,
#                  subducting_plate_id,
#                  overriding_plate_id,
#                  subduction_polarity,
#                  time,
#                  end_time,
#                  time_step,
#                  clean_dips,
#                  ground_pixel_tree,
#                  subducting_plate_disappearance_time)
#             # print(dips)
#             output_data.append([time, polyline, point_depths, variable_results, dips,
#                                 overriding_plate_id, subducting_plate_id, convergence_rates])

#     # write out the features for GPlates
#     # output_features = pygplates.FeatureCollection(point_features)

#     # write results to file

#     # NB not corrected for multiple variables (yet)

#     # slab.write_subducted_slabs_to_xyz(output_filename,output_data)
#     # close dataset

#     return output_data


# Original function had a typo which I have corrected
def calculate_pressure(density, depth):
    """
    Calculate pressure at a series of points of set depths (km) with designated densities (g/cm3).
    Assumesgravity is constant (9.8 m/s2).
    The basic equation we use is:
    pressure = density • gravity • depth (with depth being total depth, or thickness of the layer)

    Parameters
    ----------
    density : list or numpy.array
        The collection of densities in g/cm3. Must correspond to each depth.
    depth : list or numpy.array
        The collection of depths (from surface) in km. Must have a corresponding density for each
        depth.

    Returns
    -------
    pressure : numpy.array
        The cumulative pressure at each depth point in MPa.

    """

    # Set gravity
    g = 9.8

    # Check if density is a list, if so convert to array
    if isinstance(density, list):
        density = np.asarray(density)

    # Check if depth is a list, if so convert to array
    if isinstance(depth, list):
        depth = np.asarray(depth)

    # Convert density to kg/m3
    rho = density * 1000

    # Convert depth to m
    Z_bot = depth * 1000

    # Get incremental depths - list of zeros that we reassign with layer thicknesses
    layer_thicknesses = np.zeros_like(depth)
    for ind, i in enumerate(Z_bot):
        # We just use the current depth and subtract the previous one from it
        if ind == 0:
            # For the first point we just need index 0
            layer_thicknesses[ind] = Z_bot[ind]
        else:
            # Subtract current depth from previous one to get the change (i.e. the thickness of
            # each layer)
            layer_thicknesses[ind] = (Z_bot[ind] - Z_bot[ind-1])

    # We sum the layer thicknesses and corresponding densities, and multiply by gravity to get
    # Pressure in Pascals
    pressure = np.sum(layer_thicknesses * rho) * g

    # Convert to Megapascals
    pressure = pressure * 1e-6

    return pressure


def calc_point_rho(depth):
    """
    Calculate the pressure at a given point, assuming that continental crust thickness and layering
    is equivalent to that of Kaban and Mooney, 2001 (see their Eq. 1 and adjacent text).
    May be worth renaming this to calc_point_pressure because 'rho' is density (KW)

    Parameters
    ----------
    depth : float
        Depth in km.

    Returns
    -------
    pressure_GPa : float
        Pressure in GPa.

    """
    # Convert depth to m
    depth = depth * 1e3

    # Values from Kaban and Mooney, 2001
    # Thicknesses in km
    upper_crust_thickness = 11.5
    lower_crust_thickness = np.clip(depth-upper_crust_thickness, 0, 25-11.5)
    upper_mantle_thickness = np.clip(depth-upper_crust_thickness-lower_crust_thickness, 0, 1e99)

    # Densities in kg/m3
    rho_upper_crust = 2700
    rho_lower_crust = 2930
    rho_upper_mantle = 3350

    # Gravity (m/s2) is assigned here but not needed
    # gravity = 9.8

    # This gives pressure in Pascals, divide by 1e9 to get Gigapascal
    # Andrew- double check that this is working as intended in the original file, unsure what the
    # effect of backslash is on BODMAS (KW).
    pressure = ((upper_crust_thickness * rho_upper_crust) +
                (lower_crust_thickness * rho_lower_crust) +
                (upper_mantle_thickness * rho_upper_mantle)) * 9.8
    pressure_GPa = pressure / 1e9

    return pressure_GPa


def get_sub_parameters(shared_boundary_sections):
    """
    This function gets some subduction parameters from a series of shared boundary sections. Both
    'cross_section_start_points' and 'sub_length' should have the same shape.

    Parameters
    ----------
    shared_boundary_sections : list of pygplates.shared_boundary_section objects
        List of shared_boundary_section objects from which to collect subduction parameters.

    Returns
    -------
    cross_section_start_points : numpy.array
        The mid point of each segment (for use to start a cross section)
    sub_length : numpy.array
        The length of each segment in km.
    segments : list of pygplates.GreatCircleArc objects
        List of segments within shared_boundary_sections subduction zones.
    polarity : list of strings
        List of strings ('left' or 'right') describing subduction polarity at time = 0.

    """

    # We want to sample the mid point of each segment of subduction zone for our cross sections.
    # Some lists for storage:
    cross_section_start_points = []
    sub_length = []
    segments = []
    polarity = []

    # Loop through shared boundary sections to get subduction zones
    for shared_boundary_section in shared_boundary_sections:

        # Separate the subduction zones from everything else:
        if (
                shared_boundary_section.get_feature().get_feature_type() ==
                pygplates.FeatureType.gpml_subduction_zone
                ):
            # Then iterate over the shared subduction segments within each boundary section.
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():

                # Need polarity of subduction zones
                tmp_polarity = slab.find_overriding_and_subducting_plates(shared_sub_segment, 0)

                # Skip sections with no polarity (should be fixed in the plate model) - may be
                # worth including a print function here to double check that it is corrected (KW)
                if tmp_polarity is None:
                    continue

                # Loop through segments (defined as straight line between two points in a cross
                # section, i.e., a great circle arc).
                for segment in shared_sub_segment.get_geometry().get_segments():
                    # Get mean lat/lon of segment (i.e., centre point) to use as a point to anchor
                    # the cross section
                    segment_mean_lat = np.mean((segment.get_start_point().to_lat_lon()[0],
                                                segment.get_end_point().to_lat_lon()[0]))
                    segment_mean_lon = np.mean((segment.get_start_point().to_lat_lon()[1],
                                                segment.get_end_point().to_lat_lon()[1]))

                    # Append the data of this segment centre point to the lists defined above.
                    cross_section_start_points.append([segment_mean_lat, segment_mean_lon])
                    segments.append(segment)
                    sub_length.append(segment.get_arc_length()*pygplates.Earth.mean_radius_in_kms)
                    polarity.append(tmp_polarity[2])

    cross_section_start_points = np.asarray(cross_section_start_points)
    sub_length = np.asarray(sub_length)

    return cross_section_start_points, sub_length, segments, polarity


def make_cross_section(forward_distance, back_distance, cross_section_start_points, segments,
                       polarity):
    """
    Given a lat-lon point and two distances, constructs a cross section between the start
    and end points through the lat-lon point

    Parameters
    ----------
    forward_distance : integer or float
        Forwards distance from the cross section in radians. (Is this not immediately converted
        into radians? KW)
    back_distance : integer or float
        Backwards distance from the cross section in radians (this and forward_distance define the
        extent of the cross section).
    cross_section_start_points : numpy.array
        Input latitude/longitude that defines the cross sections.
    segments : list of pygplates.GreatCircleArc objects
        Segments of each subduction zone.
    polarity : list or array of strings
        Polarity of subduction zone segments.

    Returns
    -------
    new_cross_section_start_points : numpy.array
        Array of (new) start points for cross sections.
    cross_section_end_points : numpy.array
        Array of end points for cross sections. Both this and new_cross_section_start_points should
        be the same length as the input data.

    """

    # To get our cross section start and end points, we use angular distance to sample our cross
    # section forwards and backwards from our segment point.
    angular_distance_forwards = np.radians(forward_distance)
    angular_distance_backwards = np.radians(back_distance)

    # Replace mid points from previous function with 'new start points'
    cross_section_end_points = []
    new_cross_section_start_points = []

    # Loop through points
    for index in range(len(cross_section_start_points)):
        # print(index)

        # Legacy code here:
        # Skip small segments, mainly because they usually occur at edges of subduction zones, and
        # can then cause issues with overlap
        # if segments[index].get_arc_length() * pygplates.Earth.mean_radius_in_kms < 25:
        #     continue

        # Mid point of cross section segment
        mid_point = pygplates.PointOnSphere(cross_section_start_points[index])

        # Get normal great circle to segment
        normal = segments[index].get_great_circle_normal().to_normalised()

        # Get the unnormalised vector along the normal from the mid point
        stage_pole_x, stage_pole_y, stage_pole_z = pygplates.Vector3D.cross(
                                    mid_point.to_xyz(), normal).to_xyz()

        # Turn vector into a stage pole? i.e., a point on the great cricle
        stage_pole = pygplates.PointOnSphere(
                            stage_pole_x, stage_pole_y, stage_pole_z, normalise=True)

        # Normal great circle always to the left of the subduction zone, so have to reverse
        # print(polarity[index])
        if polarity[index] == 'Left':
            subducting_normal_reversal = 1
        else:
            # print(index)
            subducting_normal_reversal = -1
        # Get the rotation of the stage pole using a set angle to get cross section end point
        stage_rotation = pygplates.FiniteRotation(stage_pole, angular_distance_forwards *
                                                  subducting_normal_reversal)
        # Get cross section end point
        cross_section_end_point = stage_rotation * mid_point
        cross_section_end_points.append([cross_section_end_point.to_lat_lon_array()[0][0],
                                        cross_section_end_point.to_lat_lon_array()[0][1]])

        # Need to extend the start point back a bit, so just multiply by -1 to get the other
        # direction
        stage_rotation = pygplates.FiniteRotation(stage_pole, angular_distance_backwards *
                                                  subducting_normal_reversal * -1)
        new_cross_section_start_point = stage_rotation * mid_point

        # Append the new start point lat/lons to the list defined earlier
        new_cross_section_start_points.append(
            [new_cross_section_start_point.to_lat_lon_array()[0][0],
             new_cross_section_start_point.to_lat_lon_array()[0][1]])

    cross_section_end_points = np.asarray(cross_section_end_points)
    new_cross_section_start_points = np.asarray(new_cross_section_start_points)

    # Legacy code here:
    # now because slabs are in 0–360..
    # for i in cross_section_end_points[:,1]:
    #     if i > 180:
    #         input_lon = input_lon-360

    return new_cross_section_start_points, cross_section_end_points


def populate_cross_section(output_data, cross_section_end_points, cross_section_start_points,
                           steps):
    """
    Given a lat-lon point and two distances, constructs a cross section between the start and end
    points through the lat-lon point for pygmt and rockhunter.

    Parameters
    ----------
    output_data : numpy.array
        Array describing a subducted slab.
    cross_section_end_points : numpy.array
        Input latitude and longitude points that define the start of the cross sections.
    cross_section_start_points : numpy.array
        Input latitude and longitude points that define the end of the cross sections.
    steps : integer
        Number of steps within the cross section between start and end.

    Returns
    -------
    cross_section_points : pandas.DataFrame
        pandas.DataFrame of longitude and latitude (used for pygmt).
    cross_section_lines : numpy.array
        Array of latitude and longitude points defining a cross-section.
    sorted_intersecting_lines : numpy.array
        Array of lines from a subduction-isochron from a subducted slab that intersect with our
        cross section (this is where our data is stored).

    """

    # These lines here seem a bit unnecessary (KW)
    cross_section_end_point = cross_section_end_points[:]
    cross_section_start_point = cross_section_start_points[:]

    # Define line for the cross section. We need two types of cross section, as unfortunately the
    # section lines for Slab2.0 and pygmt need to be built in different ways.
    # Here we start with the Slab2.0 section:
    cross_section_points = []
    cross_section_lines = []
    intersecting_lines = []
    distance_to_lines = []
    sorted_intersecting_lines = []

    # Iterate over all cross section start and end points:
    for ind, (end_point, start_point) in enumerate(zip(cross_section_end_point,
                                                       cross_section_start_point)):
        start_lat = start_point[0]
        start_lon = start_point[1]
        end_lat = end_point[0]
        end_lon = end_point[1]

        # Get cross_section line using function described above
        cross_section_line = cross_section_line_pygplates(start_lat, start_lon,
                                                          end_lat, end_lon, 0.1)

        # Get the subduction isochrons that intersect the cross section line
        intersecting_line = []
        distance_to_line = []
        # NB this returns the lines in a non-random, but non correct order
        for ind1, polyline in enumerate(output_data):
            # Legacy code:
            # print(ind)
            # if not polyline:
            #     continue

            # Get minimum distance between 'iso-sub-chron' and our cross section
            min_distance_to_feature = pygplates.GeometryOnSphere.distance(
                polyline[1], cross_section_line)

            # If min distance is 0, then they intersect and we want the rest of the data
            if min_distance_to_feature == 0:
                intersecting_line.append(polyline)
                distance_to_line.append(pygplates.GeometryOnSphere.distance(
                    polyline[1], pygplates.PointOnSphere([start_lat, start_lon])))
        # print(distance_to_line)
        # Now we can order (sort) our lines correctly based on distance to start of cross_section
        sorted_lines = [x for _, x in sorted(zip(distance_to_line, intersecting_line))]

        # More storage for intersecting lines:
        new_intersecting_lines = []

        # Already sorted based on distance to present-day subduction zone so the last entry is the
        # '5' Ma. (I assume that this is derived from output_data? KW)
        if sorted_lines:
            # print(len(sorted_lines))
            # Get present day over-riding and downgoing plate
            # tmp_overriding_plate = sorted_lines[0][5]  # Never called (KW)
            tmp_downgoing_plate = sorted_lines[0][6]
            # Check to make sure that all lines have same downgoing plate

            for sorted_line in sorted_lines:
                # print(sorted_line[6])
                # THIS IS IMPORTANT, -2 refers to downgoing plate - think this may be depreciated?
                # ~KW
                if sorted_line[6] == tmp_downgoing_plate:
                    new_intersecting_lines.append(sorted_line)
        # print(ind, len(sorted_lines), len(new_intersecting_lines))
        steps = 200
        lat = np.linspace(start_lat, end_lat, int(steps))
        lon = np.linspace(start_lon, end_lon, int(steps))

        # pygmt track needs lon/lat as separate columns in pandas DataFrame, so establish a
        # dictionary here to turn into a DataFrame
        d = {'lon': lon, 'lat': lat}
        points = pd.DataFrame(data=d)

        # Shuttle everything into list storage...
        cross_section_points.append(points)
        cross_section_lines.append(cross_section_line)
        intersecting_lines.append(intersecting_line)
        sorted_intersecting_lines.append(new_intersecting_lines)
        # sorted_intersecting_lines.append(sorted_lines)
        distance_to_lines.append(distance_to_line)

    return cross_section_points, cross_section_lines, sorted_intersecting_lines


def haversine_formula(lon1, lon2, lat1, lat2):
    """
    Function to get great circle distance between two latitude/longitude points.

    Parameters
    ----------
    lon1 : float
        Longitude of point 1.
    lon2 : float
        Longitude of point 2.
    lat1 : float
        Latitude of point 1.
    lat2 : float
        Latitude of point 2.

    Returns
    -------
    distance : float
        Distance in km.

    """

    #  convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles.
    distance = c * r
    return distance


def get_distances(intersecting_points, intersecting_depths, tracks):
    """
    Returns distance between points on a cross section and in the depth profile.

    Parameters
    ----------
    intersecting_points : numpy.array
        Array of latitude/longitude points.
    intersecting_depths : numpy.array
        Array of depths corresponding to each latitude/longitude point.
    tracks : TYPE  # This parameter is never called in this function (KW)
        DESCRIPTION.

    Returns
    -------
    distance_range : numpy.array
        Distance of each point in the depth profile.
    cum_distances : numpy.array
        Array of cumulative distance from first point to each intersecting line of the cross
        section.
    incremental_distances : numpy.array
        Array of the incremental distance between points in km (i.e., distance between each point)
        for each intersecting line.

    """

    # Storage...
    incremental_distances = []
    cum_distances = []
    distance_range = []

    for index, point in enumerate(intersecting_points):
        # print(index)
        # print(point)

        # Calculate distance as going across cross section. Each cross section has equally placed
        # points. Two lists here for the incremental distances and the cumulative distance.
        tmp_incremental_distance = []
        tmp_cum_distances = []

        for ind, i in enumerate(point):
            # print(i)
            if ind == 0:
                # print(i[1])
                incremental_distance = 0
            else:
                # We need current point, and previous point to get the distance between them
                incremental_distance = pygplates.GeometryOnSphere.distance(
                    i[1], intersecting_points[index][ind-1][1])

            # To convert from radians to km we have to multiply by radius, but as we are at depth,
            # the radius is the distance between Earth's radius and the depth of the point
            radius = pygplates.Earth.mean_radius_in_kms - intersecting_depths[index][::-1][ind]
            tmp_incremental_distance.append(incremental_distance*radius)
            tmp_cum_distances.append(np.sum(tmp_incremental_distance))
            # print(intersecting_depths[index][-1][::-1][ind], radius, distance, distance*radius)

        # Append those incremental and cumulative distances:
        incremental_distances.append(tmp_incremental_distance)
        cum_distances.append(tmp_cum_distances)

        tmp_distance_range = []
        if len(tracks[index]) < 2:
            tmp_distance_range.append(1)
        # print(tracks[index]['lat'])
        # Use haversine formula (great-circle distance between two points on a sphere) to convert
        # to km. Get distance, equally spaced so we can define at the start.
        else:
            lat1 = tracks[index]['lat'].values[0]
            lat2 = tracks[index]['lat'].values[1]
            lon1 = tracks[index]['lon'].values[0]
            lon2 = tracks[index]['lon'].values[1]

            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
            distance = haversine_formula(lon1, lon2, lat1, lat2)
            # print(distance)
            # Get the incremental range:
            for i in range(tracks[index]['depth'].count()):
                tmp_distance_range.append(i*distance)
        distance_range.append(tmp_distance_range)

    return distance_range, cum_distances, incremental_distances


def _find_nearest(array, value):
    """
    Find the index in array whose element is nearest to value.

    Parameters
    ----------
    array : numpy.array
        The array.

    value : float
        The value.

    Returns
    -------
    integer
        The index in array whose element is nearest to value.

    """

    if array.argmax() == array.size - 1 and value > array.max():
        return 0  # array.size
    return (np.abs(array - value)).argmin()


# This is practically the same function as the one before.
# def _find_nearest_temp(array, value):
#     """
#     Find the index in array whose element is nearest to value.

#     Parameters
#     ----------
#     array : numpy.array
#         The array.

#     value : float
#         The value.

#     Returns
#     -------
#     integer
#         The index in array whose element is nearest to value.

#     """
#     if array.argmax() == array.size - 1 and value > array.max():
#         return array.size
#     return (np.abs(array - value)).argmin()


def warp_points(subduction_lats, subduction_lons, rotation_model, subducting_plate_ids,
                subduction_normals, time, end_time, time_step, dips,
                subducting_plate_disappearance_times, use_small_circle_path=False):
    """
    Warp subducting points to determine their position and depth after a specified time interval
    according to a GPlates reconstruction model

    Parameters
    ----------
    subduction_lats : numpy.array or list
        Latitudes of subducting points.
    subduction_lons : numpy.array or list
        Longitudes of subducting points.
    rotation_model : rot file
        GPlates rotation file.
    subducting_plate_ids : list of integers
        List of GPlates model plate IDs.
    subduction_normals : numpy.array or list
        Collection of subduction normal vectors (? KW).
    time : float or integer
        Starting time of warping process.
    end_time : float or integer
        Finishing time of warping process.
    time_step : float or integer
        Time interval between warping calculations.
    dips : numpy.array or list
        Collection of slab dips (? KW).
    subducting_plate_disappearance_times : numpy.array or list
        Collection of subducting plate disappearance times (? KW).
    use_small_circle_path : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    points : numpy.array
        Array of latitude and longitude points at a given time.
    point_depths : list
        List of depths of each point after subduction.
    point_dips : list
        List of dips of each point after subduction.
    point_convergence_rates : list
        List of point convergence rates after subduction (down-slab? KW).
    no_dip_points : numpy.array
        List of points assumping that subduction occurs at zero dip.
    distances : numpy.array
        Distances subducted from trench (down-slab? KW).

    """

    # get 'initial points' that we will be warping
    points = np.empty_like(subduction_lats, dtype=object)
    no_dip_points = np.empty_like(subduction_lats, dtype=object)
    distances = np.empty_like(subduction_lats, dtype=object)
    stage_rotations = np.empty_like(subduction_lats, dtype=object)

    # Iterate over the number of points, find the rotated position of each point assuming no dip
    for ind in range(len(subduction_lats)):
        # print(subduction_lats[ind],subduction_lons[ind])
        points[ind] = pygplates.PointOnSphere(subduction_lats[ind],
                                              subduction_lons[ind])
        no_dip_points[ind] = pygplates.PointOnSphere(subduction_lats[ind],
                                                     subduction_lons[ind])
        stage_rotations[ind] = rotation_model.get_rotation(
            subducting_plate_disappearance_times[ind]+time_step,
            subducting_plate_ids[ind], subducting_plate_disappearance_times[ind])

    # Set up some lists to keep information about each individual point:
    point_depths = [0. for point in points]
    point_dips = [0. for point in points]
    point_convergence_rates = [0. for point in points]
    # point_convergence_rates_magnitude = [0. for point in points]  # Never called

    # Get a copy of points for warping
    warped_points = np.copy(points)

    # This part here is used to keep progress on how the process is going.
    warped_end_time = end_time
    if warped_end_time < 0:
        warped_end_time = 0
    print('times', time, warped_end_time-time_step, -time_step)

    # Loop through times
    for warped_time in np.arange(time, warped_end_time-time_step, -time_step):
        # print(warped_time)
        stage_rotations = np.empty_like(subducting_plate_disappearance_times, dtype=object)

        # Iterate over subducting plate disappearance times to provide rotations if a plate
        # disappears within the warping timeframe
        for ind, subducting_plate_disappearance_time in enumerate(
                subducting_plate_disappearance_times):
            # print(ind, subducting_plate_disappearance_time)

            # If the time of warping occurs after the subducting plate has disappeared, we can use
            # the rotation of the time immediately before it disappears.
            if warped_time <= subducting_plate_disappearance_time:
                print('Using %0.2f to %0.2f Ma stage pole for plate %d' % (
                    subducting_plate_disappearance_time+time_step,
                    subducting_plate_disappearance_time,
                    subducting_plate_ids[ind]))
                stage_rotations[ind] = rotation_model.get_rotation(
                    subducting_plate_disappearance_time+time_step,
                    subducting_plate_ids[ind],
                    subducting_plate_disappearance_time)
                # print('here, 1')

            # Else we use the rotation at the current time.
            else:
                # The stage rotation that describes the motion of the subducting plate, with
                # respect to the fixed plate for the rotation model.
                # print('here, 2')
                stage_rotations[ind] = rotation_model.get_rotation(warped_time-time_step,
                                                                   subducting_plate_ids[ind],
                                                                   warped_time)

        if use_small_circle_path:  # Still not too sure what this is for (KW)
            stage_poles = np.asarray(list(map(
                lambda stage_rotations: stage_rotations.get_euler_pole_and_angle(),
                stage_rotations)))[:, 0]

            stage_pole_angles_radians = np.asarray(list(map(
                lambda stage_rotations: stage_rotations.get_euler_pole_and_angle(),
                stage_rotations)))[:, 1]

        # print(stage_rotations)
        # This part calculates the velocities for each point according to the stage rotations
        # calculated above.
        relative_velocity_vectors = np.asarray([pygplates.calculate_velocities(
            point, stage_rotation, time_step, pygplates.VelocityUnits.kms_per_my)
            for point, stage_rotation in zip(points, stage_rotations)]).reshape(len(points),)

        normals = np.copy(subduction_normals)
        # print(normals, points)
        # Find subduction parallel directions for each individual point using the normal vectors.
        # However, this variable is just never called (KW)
        # parallels = np.asarray([pygplates.Vector3D.cross(
        #     point.to_xyz(), normal).to_normalised()
        #     for point, normal in zip(points, normals)]).reshape(len(points),)

        # Make copies of various things...
        # point_lats = np.copy(subduction_lats)  # This variable is never called
        # point_lons = np.copy(subduction_lons)  # This variable is never called
        dip_angles_degrees = dips
        dip_angles_radians = np.radians(dip_angles_degrees)
        velocities = np.copy(relative_velocity_vectors)

        # Store our new points that have been warped
        warped_points = np.empty_like(points)
        no_dip_warped_points = np.empty_like(no_dip_points)
        warped_points_depth = np.empty_like(points)
        warped_dips = np.empty_like(points)
        warped_convergence_rates = np.empty_like(points)
        warped_distances_from_trench = np.empty_like(points)
        # warped_dip_stage_pole = []  # np.empty_like(points)  # This variable is never called

        # Iterate over the velocities that we have calculated:
        for ind, velocity in enumerate(velocities):
            # print('here')

            # If the velocity is zero then the point hasn't moved. The warped points are the same
            # as the original points.
            if velocity.is_zero_magnitude():
                warped_points[ind] = points[ind]
                no_dip_warped_points[ind] = no_dip_points[ind]
                warped_points_depth[ind] = point_depths[ind]
                warped_distances_from_trench[ind] = distances[ind]
                warped_dips[ind] = point_dips[ind]
                warped_convergence_rates[ind] = point_convergence_rates[ind]
                continue

            # If the velocity is not zero, then we need to figure out where our points have moved
            # to with some vector maths.
            else:
                normal_angle = pygplates.Vector3D.angle_between(velocity, normals[ind])
                velocity_normal = np.cos(normal_angle) * velocity.get_magnitude()
                normal_vector = normals[ind].to_normalised() * velocity_normal
                # Trench-parallel variables below are never called
                # parallel_angle = pygplates.Vector3D.angle_between(velocity, parallels[ind])
                # velocity_parallel = np.cos(parallel_angle) * velocity.get_magnitude()
                # parallel_vector = parallels[ind].to_normalised() * velocity_parallel

                # Adjust velocity based on subduction vertical dip angle.
                # velocity_dip = parallel_vector + np.cos(dip_angles_radians[ind]) * normal_vector
                velocity_dip = np.cos(dip_angles_radians[ind]) * normal_vector

                warped_dips[ind] = np.copy(dip_angles_degrees[ind])
                warped_convergence_rates[ind] = np.copy(velocity.get_magnitude())

                # deltaZ is the amount that this point increases in depth within the time step
                deltaZ = np.abs(np.sin(dip_angles_radians[ind]) * velocity_normal)

                # NB sometimes the normal angle gives a -ve cos, which makes velocity_normal -ve
                # below print statement shows if necessary
                if deltaZ < 0:
                    print(normal_angle, np.cos(normal_angle), velocity_normal,
                          velocity.get_magnitude(), deltaZ)

                # Again no idea what this does (KW)
                if use_small_circle_path:
                    # Rotate original stage pole by the same angle that effectively rotates the
                    # velocity vector to the dip velocity vector.
                    dip_stage_pole_rotate = pygplates.FiniteRotation(
                        points[ind], pygplates.Vector3D.angle_between(velocity_dip, velocity))
                    dip_stage_pole = dip_stage_pole_rotate * stage_poles[ind]
                    # The following should be 90 degrees always (remember to uncomment all the
                    # trench parallel stuff above).
                    # print(np.degrees(np.arccos(pygplates.Vector3D.dot(
                    #     normal_vector, parallel_vector))))

                # Get the unnormalised vector perpendicular to both the point and velocity vector.
                else:
                    (dip_stage_pole_x,
                     dip_stage_pole_y,
                     dip_stage_pole_z) = pygplates.Vector3D.cross(points[ind].to_xyz(),
                                                                  velocity_dip).to_xyz()

                    # PointOnSphere requires a normalised (i.e., unit length) vector (x, y, z).
                    dip_stage_pole = pygplates.PointOnSphere(dip_stage_pole_x,
                                                             dip_stage_pole_y,
                                                             dip_stage_pole_z,
                                                             normalise=True)

                # Not really too sure what this is. I think the dip velocity vector magnitude is
                # multiplied by some constants to get a dip angle in radians? (KW)
                dip_stage_angle_radians = velocity_dip.get_magnitude()*(
                    time_step / pygplates.Earth.mean_radius_in_kms)

                if use_small_circle_path:
                    # Increase rotation angle to adjust for fact that we're moving a shorter
                    # distance with small circle (compared to great circle).
                    dip_stage_angle_radians /= np.abs(np.sin(
                        pygplates.Vector3D.angle_between(dip_stage_pole.to_xyz(),
                                                         points[ind].to_xyz())))

                    # Use same sign as original stage rotation.
                    if stage_pole_angles_radians[ind] < 0:
                        dip_stage_angle_radians = -dip_stage_angle_radians

                # Get the stage rotation that describes the lateral motion of the point taking the
                # dip into account.
                dip_stage_rotation = pygplates.FiniteRotation(dip_stage_pole,
                                                              dip_stage_angle_radians)
                no_dip_stage_rotation = stage_rotations[ind]

                # increment the point longitude, latitude, and depth for the next time step.
                warped_points[ind] = dip_stage_rotation * points[ind]
                no_dip_warped_point = no_dip_stage_rotation * no_dip_points[ind]
                # print(no_dip_warped_point.to_lat_lon())
                # print(subduction_lons[ind], no_dip_warped_point.to_lat_lon()[1],
                #       subduction_lats[ind], no_dip_warped_point.to_lat_lon()[0])
                no_dip_warped_points[ind] = no_dip_warped_point
                # warped_points.append(warped_point)
                warped_points_depth[ind] = point_depths[ind] + deltaZ
                distance = haversine_formula(subduction_lons[ind],
                                             no_dip_warped_point.to_lat_lon()[1],
                                             subduction_lats[ind],
                                             no_dip_warped_point.to_lat_lon()[0])
                warped_distances_from_trench[ind] = np.abs(distance)

            # Up to here.
            # For next warping iteration.

        # Export everything into various numpy arrays.
        points = warped_points
        point_depths = warped_points_depth
        point_dips = warped_dips
        point_convergence_rates = warped_convergence_rates
        no_dip_points = no_dip_warped_points
        distances = warped_distances_from_trench
        # print(distances)

    return points, point_depths, point_dips, point_convergence_rates, no_dip_points, distances


def get_subducted_points(start_time, end_time, time_step, model, grid_filename,
                         tessellation_threshold_radians):
    """
    Function to obtain position of subducted points after defined time steps.

    Parameters
    ----------
    start_time : float or integer
        DESCRIPTION.
    end_time : float or integer
        DESCRIPTION.
    time_step : float or integer
        DESCRIPTION.
    model : gplately.PlateReconstruction object
        pyGPlates plate reconstruction model, comprising rotations, topology features.
    grid_filename : list of netcdf4 files
        List of netcdf4 files describing gridded variables.
    tessellation_threshold_radians : float
        Interval over which polyline and grid tessellation occur.

    Returns
    -------
    output_data : list of lists
        Output data as follows: reconstruction time, points, point depths, variable results (from
        grids), dips, subduction lengths, overriding plate IDs, subducting plate IDs, point
        convergence rates, no dip points, subduction longitudes, subduction latitudes, distances.

    """
    output_data = []
    start_time = start_time
    end_time = end_time
    time_step = time_step
    times = np.arange(start_time, end_time-time_step, -time_step)
    handle_splits = True  # Maybe this and 'filt' could be adjustable variables? (KW)
    filt = False

    # Set up dip calculator
    dipper = SlabDipper()
    dipper.model = model

    # If splits are handled, then disappearance times are adjusted.
    if handle_splits:
        plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(
            topology_features, rotation_model, times, verbose=False)

    # Iterate over the timeframe that the subducting points are calculated.
    for reconstruction_time in times:
        print(reconstruction_time)
        # This variable below is never used:
        # subduction_data = model.tessellate_subduction_zones(
        #     reconstruction_time, tessellation_threshold_radians=tessellation_threshold_radians,
        #     ignore_warnings=True, output_subducting_absolute_velocity_components=True)

        dipper.set_age_grid_filename(grid_filename[1] % reconstruction_time)
        dipper.set_spreading_rate_grid_filename(grid_filename[2] % reconstruction_time)

        dataFrame = dipper.tessellate_slab_dip(
            reconstruction_time, tessellation_threshold_radians=tessellation_threshold_radians)

        # These variables are used:
        subduction_lons = dataFrame['lon'].values
        subduction_lats = dataFrame['lat'].values
        subduction_pids_sub = dataFrame['pid_sub'].values
        subduction_pids_over = dataFrame['pid_over'].values
        subduction_lengths = dataFrame['length'].values
        subduction_convergence = dataFrame['vel'].values
        subduction_normals = dataFrame['normals'].values
        dips = dataFrame['slab_dip'].values

        # These are not:
        # subduction_angles = dataFrame['angle'].values
        # subduction_norms = dataFrame['norm'].values
        # subduction_migration = dataFrame['trench_vel'].values
        # subduction_plate_vels = dataFrame['slab_vel_abs'].values

        # Remove entries that have "negative" subduction. This occurs when the subduction
        # obliquity is greater than 90 degrees
        subduction_convergence = np.clip(subduction_convergence, 0, 1e99)

        # Legacy code
        # dips, vratio, rho_prime, theta_qv, volatile_flux = get_dip(reconstruction_time,
        #                                                            gdownload, subduction_data)
        # print(np.shape(dips), np.shape(subduction_lats))

        subducting_plate_ids = np.copy(subduction_pids_sub)
        overriding_plate_ids = np.copy(subduction_pids_over)

        # cap dips at 80°
        # dips[dips > 40] = 40

        # Not too sure what this part does, but I assume it's something to do with the information
        # stored within the netcdf4 files.
        lut = []
        if grid_filename is not None:
            for ind, i in enumerate(grid_filename):
                # count = ind + 1
                # if count % 2 != 0:
                grdfile = grid_filename[ind] % reconstruction_time
                lut.append(slab.make_age_interpolator(grdfile))

        # Might have to walk through plate IDs individually
        subducting_plate_disappearance_times = np.ones_like(subducting_plate_ids)*-1.
        if handle_splits:
            for plate_disappearance in plate_disappearance_time_lut:
                if np.isin(plate_disappearance[0], subduction_pids_sub):
                    index_of_sub_id = np.where(subduction_pids_sub == plate_disappearance[0])
                    subducting_plate_disappearance_times[
                        index_of_sub_id[0][0]] = plate_disappearance[1]

        # Something to do with the Andes here?
        if filt:
            min_lon = -80
            max_lon = -60
            min_lat = -40
            max_lat = -15

            nu_subduction_lons = subduction_lons[np.logical_and(
                        np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                        np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            nu_subduction_lats = subduction_lats[np.logical_and(
                np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            dips = dips[np.logical_and(np.logical_and(
                min_lon <= subduction_lons, max_lon >= subduction_lons),
                np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            subduction_normals = subduction_normals[np.logical_and(
                np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            subducting_plate_ids = subducting_plate_ids[np.logical_and(
                np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            overriding_plate_ids = overriding_plate_ids[np.logical_and(
                np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))]
            subducting_plate_disappearance_times = (
                subducting_plate_disappearance_times[np.logical_and(
                    np.logical_and(min_lon <= subduction_lons, max_lon >= subduction_lons),
                    np.logical_and(min_lat <= subduction_lats, max_lat >= subduction_lats))])

            subduction_lons = nu_subduction_lons
            subduction_lats = nu_subduction_lats

        # this gets our variables at the lat/lons
        variable_results = []
        if lut is not None:
            x = np.copy(subduction_lons)
            y = np.copy(subduction_lats)

            for i in lut:
                variable_results.append(i.ev(np.radians((y+90.).astype(float)),
                                             np.radians((x+180.).astype(float))))
        else:
            # if no grids, just fill the ages with zero
            variable_results = [0. for point in subduction_lons]

        # call main warping function
        (points,
         point_depths,
         point_dips,
         point_convergence_rates,
         no_dip_points,
         distances) = warp_points(subduction_lats, subduction_lons, rotation_model,
                                  subducting_plate_ids, subduction_normals, reconstruction_time,
                                  end_time, time_step, dips, subducting_plate_disappearance_times)

        output_data.append([reconstruction_time, points, point_depths, variable_results, dips,
                            subduction_lengths, overriding_plate_ids, subducting_plate_ids,
                            point_convergence_rates, no_dip_points, subduction_lons,
                            subduction_lats, distances])
    return output_data
