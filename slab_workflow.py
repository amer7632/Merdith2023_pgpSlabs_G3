# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:12:04 2023

@author: Andrew Merdith, comments by Kev Wong
"""

from gplately import pygplates
import numpy as np
import slab_tracker_utils as slab
import splits_and_merges as snm
from scipy import spatial
from scipy.interpolate import interp1d
from slabdip import SlabDipper


def find_with_list(myList, target): 
    '''
    Given a list and a target value in list, return index of that target value

    Paramters
    -----------
    myList : list
        List of values you want the index from
    target : object (float/int/string)
        Item that you want to match
    '''
    inds = []
    for i in range(len(myList)):
        if myList[i] == target:
            inds += i,
    return inds


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

def calc_point_pressure(depth):
    """
    Calculate the pressure at a given point, assuming that continental crust thickness and layering
    is equivalent to that of Kaban and Mooney, 2001 (see their Eq. 1 and adjacent text).

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
    
    pressure = ((upper_crust_thickness * rho_upper_crust) \
                + (lower_crust_thickness * rho_lower_crust) \
                + (upper_mantle_thickness * rho_upper_mantle)) * 9.8
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
    print(rotation_model)
    for ind in range(len(subduction_lats)):
        # print(subduction_lats[ind],subduction_lons[ind])
        points[ind] = pygplates.PointOnSphere(subduction_lats[ind],
                                              subduction_lons[ind])
        no_dip_points[ind] = pygplates.PointOnSphere(subduction_lats[ind],
                                                     subduction_lons[ind])
        stage_rotations[ind] = rotation_model.get_rotation(
            int(subducting_plate_disappearance_times[ind]+time_step),
            int(subducting_plate_ids[ind]), int(subducting_plate_disappearance_times[ind]))

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
                    int(subducting_plate_disappearance_time+time_step),
                    int(subducting_plate_ids[ind]),
                    int(subducting_plate_disappearance_time))
                # print('here, 1')

            # Else we use the rotation at the current time.
            else:
                # The stage rotation that describes the motion of the subducting plate, with
                # respect to the fixed plate for the rotation model.
                # print('here, 2')
                stage_rotations[ind] = rotation_model.get_rotation(int(warped_time-time_step),
                                                                   int(subducting_plate_ids[ind]),
                                                                   int(warped_time))

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
            model.topology_features, model.rotation_model, times, verbose=False)

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

        subducting_plate_ids = np.copy(subduction_pids_sub)
        overriding_plate_ids = np.copy(subduction_pids_over)

        # Get variables from NetCDF grids
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
         distances) = warp_points(subduction_lats, subduction_lons, model.rotation_model,
                                  subducting_plate_ids, subduction_normals, reconstruction_time,
                                  end_time, time_step, dips, subducting_plate_disappearance_times)

        output_data.append([reconstruction_time, points, point_depths, variable_results, dips,
                            subduction_lengths, overriding_plate_ids, subducting_plate_ids,
                            point_convergence_rates, no_dip_points, subduction_lons,
                            subduction_lats, distances])
    return output_data
