import pygplates
import numpy as np
import glob

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

MODELDIR = '/Applications/GPlates-2.2.0/SampleData/FeatureCollections'
RotFile_List = ['%s/Rotations/Matthews_etal_GPC_2016_410-0Ma_GK07.rot' % MODELDIR]
GPML_List = ['%s/DynamicPolygons/Matthews_etal_GPC_2016_MesozoicCenozoic_PlateTopologies.gpmlz' % MODELDIR,\
             '%s/DynamicPolygons/Matthews_etal_GPC_2016_Paleozoic_PlateTopologies.gpmlz' % MODELDIR]

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
handle_splits = False
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

def get_intersecting_values(intersecting_lines, cross_section_line):
    '''
    now that we have our intersecting lines, we can find the containing segment of the line, in order
    to access the correct points (and then the correct depths, variables etc.)
    '''

    intersecting_points = []
    interpolated_variables = []
    interpolated_depths = []

    for ind, line in enumerate(intersecting_lines):

        #for clarity we will enunciate each iso-subchron
        age_of_subdcution = line[0]
        iso_subchron = line[1]
        depth = line[2]
        variable_data = line[3]

        #get intersect points, and starting indices of the segments that contain intersect points of our two lines
        #(cross section line, and iso-subchron)
        closest_point_data = \
        pygplates.GeometryOnSphere.distance(iso_subchron, cross_section_line,
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
        variables = variable_data[tmp_sub_isochron_segment_index:tmp_sub_isochron_segment_index+2]
        depths = depth[tmp_sub_isochron_segment_index:tmp_sub_isochron_segment_index+2]

        interpolated_variable = sxs.get_intercept_values(tmp_sub_isochron_intercept, iso_subchron, variables)
        interpolated_variables.append(interpolated_variable)

        #NB (check)
        #because we are plotting depths and explicit distances along, we inherently correct for true/
        #apparent dip
        interpolated_depth = sxs.get_intercept_values(tmp_sub_isochron_intercept, iso_subchron, depths)
        interpolated_depths.append(interpolated_depth)

        tmp_values = age_of_subdcution, \
                     iso_subchron[tmp_sub_isochron_segment_index], \
                     depth[tmp_sub_isochron_segment_index], \
                     variable_data[tmp_sub_isochron_segment_index]

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
        if grid_filename is not None:
            grdfile = '%s%d%s' % (grid_filename[0],time,grid_filename[1])
            lut = slab.make_age_interpolator(grdfile)

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

            #print len(tessellated_line.get_points())

            if grid_filename is not None:
                x = tessellated_line.to_lat_lon_array()[:,1]
                y = tessellated_line.to_lat_lon_array()[:,0]
                subduction_ages = lut.ev(np.radians(y+90.),np.radians(x+180.))
            else:
                # if no age grids, just fill the ages with zero
                subduction_ages = [0. for point in tessellated_line.to_lat_lon_array()[:,1]]

            # CALL THE MAIN WARPING FUNCTION
            (points,
             point_depths,
             polyline,dips) = slab.warp_subduction_segment(tessellated_line,
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

            output_data.append([time,polyline,point_depths,subduction_ages])


    # write out the features for GPlates
    #output_features = pygplates.FeatureCollection(point_features)

    ### write results to file
    slab.write_subducted_slabs_to_xyz(output_filename,output_data)
    #close dataset

    return output_data
