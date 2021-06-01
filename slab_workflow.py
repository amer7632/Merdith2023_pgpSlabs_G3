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


#######################################################
# Define Input files for Muller 2016 AREPS model
#######################################################
#'''

MODELDIR = '/Users/andrew/Documents/GitHub/EarthBytePlateMotionModel-ARCHIVE/Muller++_2015_AREPS_CORRECTED/'
RotFile_List = ['%sGlobal_EarthByte_230-0Ma_GK07_AREPS.rot' % MODELDIR]
GPML_List = ['%sGlobal_EarthByte_230-0Ma_GK07_AREPS_PlateBoundaries_ASM.gpml' % MODELDIR,\
              '%sGlobal_EarthByte_230-0Ma_GK07_AREPS_Topology_BuildingBlocks.gpml' % MODELDIR]

#####################################
rotation_model = pygplates.RotationModel(RotFile_List)
topology_features = pygplates.FeatureCollection()

for file in GPML_List:
    topology_feature = pygplates.FeatureCollection(file)
    topology_features.add(topology_feature)
#'''


#############################
# INPUT PARAMETERS for slab tracing
#
start_time = 15.
end_time = 0.
time_step = 1.0
dip_angle_degrees = 45.0
line_tessellation_distance = np.radians(1.0)
handle_splits = True
# Try to use small circle path for stage rotation to rotate along velocity dip.
# Ie, distance to stage rotation pole matches distance to original stage pole.
#use_small_circle_path = False

output_filename = 'subduction_3d_geometries_time_%0.2fMa_dip_%0.2fdeg.p' % (end_time,dip_angle_degrees)


#############################


def find_with_list(myList, target):
    '''
    find index of thing in list
    '''

    inds = []
    for i in range(len(myList)):
        if myList[i] == target:
            inds += i,
    return inds

def get_slab_surfaces(slab_dir, FNAME):

    Shift=0

    MagnetiteIsoDepths = []
    SurfaceCurieIsoDepths = []
    MohoCurieIsoDepths = []


    fid=open(slab_dir+'/'+FNAME, 'r');

    SurfaceArray = [];
    MohoArray = [];

    for line in fid:
        #strip removes the '\n' at the end
        tline = line.strip()
        #print(tline[0])
        # Case where line contains the top surface of slab
        if '0' in tline[0]:
            tline = list(map(float, tline.split()))
            SurfaceArray.append(tline)
        #else we're at the moho
        else:
            tline = list(map(float, tline.split()))
            MohoArray.append(tline)
    #fid.close()
    SurfaceArray = np.asarray(SurfaceArray)
    MohoArray = np.asarray(MohoArray)

    SurfaceCurieInterpolator = interp1d(SurfaceArray[:,3],SurfaceArray[:,2],kind='linear')
    try:
        SurfaceCurieIsoDepth = SurfaceCurieInterpolator(550.)
    except:
        SurfaceCurieIsoDepth = np.nan
    MohoCurieInterpolator = interp1d(MohoArray[:,3],MohoArray[:,2],kind='linear')
    try:
        MohoCurieIsoDepth = MohoCurieInterpolator(550.)
    except:
        MohoCurieIsoDepth = np.nan
    MagnetiteInterpolator = interp1d(SurfaceArray[:,3],SurfaceArray[:,2],kind='linear')
    try:
        MagnetiteIsoDepth = MagnetiteInterpolator(200.)
    except:
        MagnetiteIsoDepth = np.nan

    SurfaceCurieIsoDepths.append(SurfaceCurieIsoDepth)
    MagnetiteIsoDepths.append(MagnetiteIsoDepth)
    MohoCurieIsoDepths.append(MohoCurieIsoDepth)

    surface_output = [SurfaceArray, MohoArray]
    isotherm_output = [SurfaceCurieIsoDepths, MohoCurieIsoDepths, MagnetiteIsoDepths]
    return surface_output,isotherm_output

def get_intersecting_values(intersecting_lines_1, cross_section_line):
    '''
    now that we have our intersecting lines, we can find the containing segment
    of the line, in order to access the correct points (and then the correct
    depths, variables etc.)
    '''

    intersecting_points = []
    interpolated_variables = []
    interpolated_depths = []

    for ind, line in enumerate(intersecting_lines_1):

        #for clarity we will enunciate each iso-subchron
        age_of_subdcution = line[0]
        iso_subchron = line[1]
        depth = line[2]

        ##need to convert convergence rate to units
        variable_data = []
        ##print(variable_data)
        conv_rates = []
        for conv_rate in line[7]:
            if isinstance(conv_rate, float):
                conv_rates.append(conv_rate)
            else:
                conv_rates.append(conv_rate.get_magnitude())
        conv_rates = np.asarray(conv_rates)
        variable_data.append(line[3])
        variable_data = np.asarray(variable_data[0])
        merged_variable_data = np.vstack((variable_data, conv_rates))
        #get intersect points, and starting indices of the segments that contain intersect points of our two lines
        #(cross section line, and iso-subchron)
        closest_point_data = \
        pygplates.GeometryOnSphere.distance(iso_subchron,
                                            cross_section_line,
                                            return_closest_positions=True,
                                            return_closest_indices=True)

        #for clarity we will enunciate the closest point data
        tmp_distance = closest_point_data[0]
        #NB these next two should be the same
        tmp_sub_isochron_intercept = closest_point_data[1]
        tmp_cross_section_intercept = closest_point_data[2]

        #the indices refer to the start of the containing segment
        tmp_sub_isochron_segment_index = closest_point_data[3]
        tmp_cross_section_segment_index = closest_point_data[4]

        #set depths and variable indices
        #print(merged_variable_data)
        for variable_index, array in enumerate(merged_variable_data):
            variables = merged_variable_data[variable_index][tmp_sub_isochron_segment_index:tmp_sub_isochron_segment_index+2]
            interpolated_variable = sxs.get_intercept_values(tmp_sub_isochron_intercept, iso_subchron, variables)
            interpolated_variables.append(interpolated_variable)

        depths = depth[tmp_sub_isochron_segment_index:tmp_sub_isochron_segment_index+2]

        #NB (check)
        #because we are plotting depths and explicit distances along, we inherently correct for true/
        #apparent dip
        interpolated_depth = sxs.get_intercept_values(tmp_sub_isochron_intercept, iso_subchron, depths)
        interpolated_depths.append(interpolated_depth)

        tmp_values = age_of_subdcution, \
                     tmp_sub_isochron_intercept, \
                     depth[tmp_sub_isochron_segment_index], \
                     [i[tmp_sub_isochron_segment_index] for i in merged_variable_data]

        intersecting_points.append(tmp_values)

    #reverse intersecting points so we're starting from the subduction zone
    intersecting_points_reversed = intersecting_points[::-1]

    return intersecting_points_reversed, intersecting_points, interpolated_variables, interpolated_depths

def cross_section_line_pygplates(lat1, lon1, lat2, lon2, spacing):
    '''
    Given two points and spacing between points, return a pygplates tessellated
    line.
    lat1/lon1/lat2/lon2: degree co-ordinates
    spacing: float to be converted to radians
    '''

    cross_section_points = []
    cross_section_points.append((lat1,lon1))
    cross_section_points.append((lat2,lon2))
    #make line feature
    cross_section_line = pygplates.PolylineOnSphere(cross_section_points)
    #tessellate line
    cross_section_line = cross_section_line.to_tessellated(np.radians(float(spacing)))

    return cross_section_line

def get_subducted_slabs(start_time, end_time, time_step, grid_filename, slab_XR):
    '''
    get iso-sub chrons with dips from slab 2.0 geometry
    '''
    clean_dips = slab_XR.dip[slab_XR.dip.notnull()]

    coords = np.column_stack((clean_dips.latitude.values.ravel(),
                              clean_dips.longitude.values.ravel()))

    ground_pixel_tree = spatial.cKDTree(slab.transform_coordinates(coords))

    output_data = []

    time_list = np.arange(start_time,end_time-time_step,-time_step)

    if handle_splits:
        plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(topology_features,
                                                                            rotation_model,
                                                                            time_list,
                                                                            verbose=True)

        print (plate_disappearance_time_lut)

    # loop over a series of times at which we want to extract trench iso-sub-chrons
    for time in time_list:

        print( 'time %0.2f Ma' % time)

        # call function to get subduction boundary segments
        subduction_boundary_sections = slab.getSubductionBoundarySections(topology_features,
                                                                          rotation_model,
                                                                          time)

        # Set up an age grid interpolator for this time, to be used
        # for each tessellated line segment
        lut = []
        if grid_filename is not None:
            for ind,i in enumerate(grid_filename):
                count = ind + 1
                if count % 2 != 0:
                    grdfile = '%s%d%s' % (grid_filename[ind],time,grid_filename[ind+1])
                    lut.append(slab.make_age_interpolator(grdfile))


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

def calcualte_pressure(density, depth):

    '''
    Calculate pressure at a series of points of set depths (km) with designated densities (g/cm3). Assumes
    gravity is constant (9.8 m/s**2).

    The basic equation we use is:

    pressure = density • gravity • depth (with depth being total depth, or thickness of the layer)


    Parameters
    -----------
    Density: list or array of densities, in g/cm3.
        The collection of densities. Must correspond to each depth.

    Depth: list or array of depths, in km
        The collection of depths (from surface). Must have a corresponding density for each depth.

    Returns
    -------
    Pressure: array of cumulative pressure, in Megapascals.
        The cumulative pressure at each depth point.
    '''
    #set gravity
    g = 9.8

    #check if density is a list, if so convert to array
    if isinstance(density, list):
        density = np.asarray(density)

    #check if depth is a list, if so convert to array
    if isinstance(depth, list):
        depth = np.asarray(depth)

    #convert density to kg/m3
    rho = density * 1000

    #conver density to m
    Z_bot = depth * 1000

    #get incremental depths
    layer_thicknesses = np.zeros_like(depth)
    for ind,i in enumerate(Z_bot):
        #we just use the current depth and subtract the previous one from it
        if ind == 0:
            #for the first point we just need index 0
            layer_thicknesses[ind] = Z_bot[ind]
        else:
            #subtract current depth from previous one to get the change (i.e. the thickness of each layer)
            layer_thicknesses[ind] = (Z_bot[ind] - Z_bot[ind-1])

    #we sum the layer thicknesses and corresponding densities, and multiply by gravity to get pressure in pascals
    pressure = np.sum(layer_thicknesses * rho) * g
    #in megapascals
    pressure = pressure * 1e-6

    return pressure

def get_sub_parameters(shared_boundary_sections):

    '''
    this function gets some subduction parameters from a series of shared boundary
    sections

    Inputs
    _________
    shared_boundary_sections: list of shared boundary sections of a GPLates
                               full plate mode

    Returns
    ________
    cross_section_start_points: array, the mid point of each segment (for use to start a cross
                                section)
    sub_length: array, the length of each segment in km

    both returned variabels should have the same shape
    '''


    #we want to sample the mid point of each segment of subduction zone for our cross sections
    cross_section_start_points = []
    sub_length = []
    polarity = []
    segments = []
    #loop through shared boundary sections to get subduction zones
    for shared_boundary_section in shared_boundary_sections:
        if shared_boundary_section.get_feature().get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():

                #need polarity of subduction zones
                tmp_polarity = slab.find_overriding_and_subducting_plates(shared_sub_segment, 0)

                #skip sections with no polarity (should be fixed in the plate model)
                if tmp_polarity is None:
                    continue

                #loop through segments (defined as straight line between two points in a cross section)
                for segment in shared_sub_segment.get_geometry().get_segments():

                    segment_mean_lat = np.mean((segment.get_start_point().to_lat_lon()[0],
                                                segment.get_end_point().to_lat_lon()[0]))
                    segment_mean_lon = np.mean((segment.get_start_point().to_lat_lon()[1],
                                                segment.get_end_point().to_lat_lon()[1]))

                    #get mean lat/lon of segment (i.e. centre point) to use as a point to anchor the cross section
                    cross_section_start_points.append([segment_mean_lat, segment_mean_lon])
                    segments.append(segment)
                    sub_length.append(segment.get_arc_length()*pygplates.Earth.mean_radius_in_kms)
                    polarity.append(tmp_polarity[2])

    cross_section_start_points = np.asarray(cross_section_start_points)
    sub_length = np.asarray(sub_length)

    return cross_section_start_points, sub_length, segments, polarity

def make_cross_section(forward_distance, back_distance,cross_section_start_points, segments, polarity):

    '''
    given a lat-lon point and two distances, constructs a cross section between the start
    and end points through the lat-lon point

    Inputs
    _______
    forward_distance and backward_distance: int or float
            distance from the cross section forwards and backwards in radians (defines the extent)

    cross_section_start_points: array
        input lat/lons that define our cross-sections

    segments: array
        segments of each subduction zone

    polarity: array
        polarity of subduction zones

    Returns
    ________
    Cross_section_start_points: array
        array of (new) start points for cross sections
    Cross_section_end_points: array
        array of end points for cross sections

    both should be the same length as input data
    '''


    #now to get our cross section start and end points
    #we use angular distance to sample our cross section forwards and backwards from our segment point
    angular_distance_forwards = np.radians(forward_distance)
    angular_distance_backwards = np.radians(back_distance)

    #replace mid points from previous cell with 'new start points'
    cross_section_end_points = []
    new_cross_section_start_points = []

    #loop through points
    for index in range(len(cross_section_start_points)):
        #print(index)
        #skip small segments, mainly because they usually occur
        #at edges of subduction zones, and can then cause issues with overlap
        #if segments[index].get_arc_length() * pygplates.Earth.mean_radius_in_kms < 25:

            #continue

        #mid point of cross section segment
        mid_point = pygplates.PointOnSphere(cross_section_start_points[index])

        #get normal great circle to segment
        normal = segments[index].get_great_circle_normal().to_normalised()

        # Get the unnormalised vector along the normal from the mid point
        stage_pole_x, stage_pole_y, stage_pole_z = pygplates.Vector3D.cross(
                                    mid_point.to_xyz(), normal).to_xyz()

        #turn vector into a stage pole? i.e. a point on the great cricle
        stage_pole = pygplates.PointOnSphere(
                            stage_pole_x, stage_pole_y, stage_pole_z, normalise=True)

        #normal great circle always to the left of the subduction zone, so have to reverse otherwise
        #print(polarity[index])
        if polarity[index] == 'Left':
            subducting_normal_reversal = 1
        else:
            #print(index)
            subducting_normal_reversal = -1
        #get the rotation of the stage pole using a set angle to get cross section end point
        stage_rotation = pygplates.FiniteRotation(stage_pole, angular_distance_forwards * subducting_normal_reversal)
        #get cross section end point
        cross_section_end_point = stage_rotation * mid_point
        cross_section_end_points.append([cross_section_end_point.to_lat_lon_array()[0][0],
                                        cross_section_end_point.to_lat_lon_array()[0][1]])

        #need to extend the start point back a bit, so just multiply by -1 to get the other direction
        stage_rotation = pygplates.FiniteRotation(stage_pole, angular_distance_backwards * subducting_normal_reversal *-1)
        new_cross_section_start_point = stage_rotation * mid_point

        new_cross_section_start_points.append([new_cross_section_start_point.to_lat_lon_array()[0][0],
                                               new_cross_section_start_point.to_lat_lon_array()[0][1]])


    cross_section_end_points = np.asarray(cross_section_end_points)
    new_cross_section_start_points = np.asarray(new_cross_section_start_points)


    #now because slabs are in 0–360..
    #for i in cross_section_end_points[:,1]:
    #    if i > 180:
    #        input_lon = input_lon-360

    return new_cross_section_start_points, cross_section_end_points

def populate_cross_section(output_data, cross_section_end_points,
                       cross_section_start_points, steps):

    '''
    given a lat-lon point and two distances, constructs a cross section between the start
    and end points through the lat-lon point for pygmt and rockhunter

    Inputs
    _______
    output_data: array
        series of arrays describing a subducted slab

    cross_section_start_points and cross_section_end_points: array
        input lat/lons that define the start/end pooints of our cross-sections

    steps: int
        number of steps in the cross section

    Returns
    ________
    cross_section_points: dataframe
        pandas dataframe of longitude and latitude (USED FOR PYGMT)
    cross_section_lines: array
        array of latitude and longitude points defining a cross-section
    intersecting_lines: array
        array of lines from a sub-isochron from a subducted slab that intersect
        with our cross section (this is where our data is stored)
    '''

    cross_section_end_point = cross_section_end_points[:]
    cross_section_start_point = cross_section_start_points[:]
    #define line for cross section
    #we need two types of cross sections, one for slab 2.0
    #one for pygmt
    #they, unfortunately, have to be built in different ways

    #slab2.0
    cross_section_points = []
    cross_section_lines = []
    intersecting_lines = []
    distance_to_lines = []
    sorted_intersecting_lines = []
    for ind, (end_point, start_point) in enumerate(zip(cross_section_end_point, cross_section_start_point)):
        start_lat = start_point[0]
        start_lon = start_point[1]
        end_lat = end_point[0]
        end_lon = end_point[1]

        #get cross_section line
        cross_section_line = cross_section_line_pygplates(start_lat,
                                                          start_lon,
                                                          end_lat,
                                                          end_lon,
                                                          0.1)

        #get the iso-subchrons that intersect the cross section line

        intersecting_line = []
        distance_to_line = []
        #NB this returns the lines in a non-random, but non correct order
        for ind1, polyline in enumerate(output_data):
            #print(ind)
            #if not polyline:
            #    continue
            #get min distance between 'iso-sub-chron' and our cross section
            min_distance_to_feature = pygplates.GeometryOnSphere.distance(polyline[1], cross_section_line)
            #if min distance is 0, then they intersect and we want the rest of the data

            if min_distance_to_feature == 0:
                intersecting_line.append(polyline)
                distance_to_line.append(pygplates.GeometryOnSphere.distance(polyline[1],
                                                                pygplates.PointOnSphere([start_lat,
                                                                    start_lon])))
        #print(distance_to_line)
        #now we can order (sort) our lines correctly based on distance to start of cross_section
        sorted_lines = [x for _,x in sorted(zip(distance_to_line,intersecting_line))]

        new_intersecting_lines = []
        #already sorted based on distance to present-day subduction zone
        #so the last entry is the '5' Ma
        if sorted_lines:
            #print(len(sorted_lines))
            #get present day over-riding and downgoing plate
            tmp_overriding_plate = sorted_lines[0][5]
            tmp_downgoing_plate = sorted_lines[0][6]
            #check to make sure that all lines have same downgoing plate

            for sorted_line in sorted_lines:
                #print(sorted_line[6])
                ###THIS IS important, -2 refers to downgoing plate
                if sorted_line[6] == tmp_downgoing_plate:
                    new_intersecting_lines.append(sorted_line)
        #print(ind, len(sorted_lines), len(new_intersecting_lines))
        steps = 200
        lat = np.linspace(start_lat,end_lat, int(steps))
        lon = np.linspace(start_lon,end_lon, int(steps))

        #pygmt track needs lon/lat as separate columns in pandas dataframe
        d = { 'lon': lon,'lat': lat}
        points = pd.DataFrame(data=d)

        cross_section_points.append(points)
        cross_section_lines.append(cross_section_line)
        intersecting_lines.append(intersecting_line)
        sorted_intersecting_lines.append(new_intersecting_lines)
        #sorted_intersecting_lines.append(sorted_lines)
        distance_to_lines.append(distance_to_line)
    return cross_section_points, cross_section_lines, sorted_intersecting_lines

def get_distances(intersecting_points, intersecting_depths, tracks):

    '''
    returns distance between points on a cross section and in the depth profile

    Inputs
    --------
    Intersecting_points: array
        array of lat/lon points
    Intersecting_depth: array
        array of depths corresponding to each lat/lon point

    Returns
    ---------
    incremental_distances: array
        array of the incremental distance between points in km (i.e distance between each point) for each
        intersecting line
    cum_distances: array
        array of cumulative distance from first point to each intersecting line of the cross section
    distance_range: array
        distance of each point in the depth profile
    '''

    incremental_distances = []
    cum_distances = []
    distance_range = []

    for index, point in enumerate(intersecting_points):
        #print(index)
        #print(point)
        #calculate distance as going across cross section.
        #each cross section has equally placed points

        tmp_incremental_distance = []
        tmp_cum_distances = []

        for ind, i in enumerate(point):
            #print(i)
            if ind == 0:
                #print(i[1])
                incremental_distance = 0
            else:
                #we need current point, and previous point to get the distance between

                incremental_distance = pygplates.GeometryOnSphere.distance(i[1],
                                                                           intersecting_points[index][ind-1][1])

            #to convert from radians to km we have to multiply by radius,
            #but as we are at depth, the radius is slightly different
            radius = pygplates.Earth.mean_radius_in_kms - intersecting_depths[index][::-1][ind]
            tmp_incremental_distance.append(incremental_distance*radius)
            tmp_cum_distances.append(np.sum(tmp_incremental_distance))

            #print(intersecting_depths[index][-1][::-1][ind], radius, distance, distance*radius)
        incremental_distances.append(tmp_incremental_distance)
        cum_distances.append(tmp_cum_distances)

        tmp_distance_range = []
        if len(tracks[index]) < 2:
            tmp_distance_range.append(1)
        #print(tracks[index]['lat'])
        #use haversine formula to convert to km
        #get distance, equally spaced so we can define at the start
        else:
            lat1 = tracks[index]['lat'].values[0]
            lat2 = tracks[index]['lat'].values[1]
            lon1 = tracks[index]['lon'].values[0]
            lon2 = tracks[index]['lon'].values[1]

            # convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])


            distance = haversine_formula(lon1, lon2, lat1, lat2)
            #print(distance)
            #get the incremental range
            for i in range(tracks[index]['depth'].count()):

                tmp_distance_range.append(i*distance)

        distance_range.append(tmp_distance_range)

    return distance_range, cum_distances, incremental_distances

def haversine_formula(lon1, lon2, lat1, lat2):
# haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    distance = c*r

    return distance

def _find_nearest(array, value):
    """Find the index in array whose element is nearest to value.

    Parameters
    ----------
    array : np.array
      The array.

    value : number
      The value.

    Returns
    -------
    integer
      The index in array whose element is nearest to value.

    """
    if array.argmax() == array.size - 1 and value > array.max():
        return 0#array.size
    return (np.abs(array - value)).argmin()

def _find_nearest_temp(array, value):
    """Find the index in array whose element is nearest to value.

    Parameters
    ----------
    array : np.array
      The array.

    value : number
      The value.

    Returns
    -------
    integer
      The index in array whose element is nearest to value.

    """
    if array.argmax() == array.size - 1 and value > array.max():
        return array.size
    return (np.abs(array - value)).argmin()
