# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:12:04 2023

@author: Andrew Merdith, comments by Kev Wong
"""


from __future__ import print_function
from gplately import pygplates
import numpy as np
import sys
import xarray as xr
import inpaint
import scipy.interpolate as spi

# Quick bug fix (although I don't think this appears anywhere)
bug_fix_arc_dir = -1.0 if pygplates.Version.get_imported_version() < pygplates.Version(14) else 1.0


def transform_coordinates(coords):
    """
    Transform coordinates from geodetic to Cartesian, taken from:
    https://notes.stefanomattia.net/2017/12/12/The-quest-to-find-the-closest-ground-pixel/

    Parameters
    ----------
    coords : tuple or np.array of tuples
        A set of latitude-longitude coordinates.

    Returns
    -------
    np.array of Cartesian coordinates.

    """

    # WGS 84 reference coordinate system parameters
    A = 6378.137  # major axis in km
    E2 = 6.69437999014e-3  # eccentricity squared

    coords = np.asarray(coords).astype(np.float)

    # Is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])

    # Convert to radians
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])

    # Convert to Cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))


def get_dip_angle_from_slab2(lat, lon, KDtree):
    """
    Given a single latitude/longitude point, find the nearest dip angle given a kd-tree from the
    Hayes et al. (2018) Slab2 model.

    Parameters
    ----------
    lat : float
        Latitude point.
    lon : float
        Longitude point.
    KDtree : scipy kd-tree object
        Scipy kd-tree object - an index into a set of k-dimensional points for nearest-neighbour
        lookup. (double check - KW)

    Returns
    -------
    nearest_index : tuple of nearest slab dip index

    """

    # convert lons to 0–360
    if lon < 0:
        lon = lon + 360

    lat_lon_point = (lat, lon)

    # Get 8 nearest dips around the chosen latitude/longitude in Cartesian space and take the mean
    # of the dips
    KDtree_results = KDtree.query(transform_coordinates(lat_lon_point), 8)
    nearest_dips = KDtree_results[0]  # The distances to the nearest neighbours
    nearest_indices = KDtree_results[1]  # The indices of the neighbours in the data.

    # Find the indices of the point with the nearest dip.
    index_of_nearest_dip = (np.abs(nearest_dips - np.mean(nearest_dips))).argmin()
    nearest_index = nearest_indices[0][index_of_nearest_dip]

    return (nearest_index)


def make_age_interpolator(grdfile, interp='Spherical'):
    """
    function that takes a netcdf grid, fills dummy values, then creates
    an interpolator object that can be evaluated later at specified points

    Parameters
    ----------
    grdfile : netcdf4 grid file
        netcdf4 grid file containing ages(? - KW).
    interp : string, optional
        String describing the geometry of the interpolation (? - KW). The default is 'Spherical'.

    Returns
    -------
    lut : TYPE (Not too sure about this - KW)
        DESCRIPTION.

    """

    ds_disk = xr.open_dataset(grdfile)

    try:
        data_array = ds_disk['z']
    except KeyError:
        data_array = ds_disk['peridotite_thickness_post_serpentinisation']

    coord_keys = list(data_array.coords.keys())
    gridX = data_array.coords[coord_keys[0]].data
    gridY = data_array.coords[coord_keys[1]].data
    gridZ = data_array.data

    # handle grids that are in range 0_360 instead of -180_180
    if gridX.max() > 180.:
        # This line is not needed
        # index = np.where(gridX) > 180.
        gridX = np.hstack

    gridZ_filled = inpaint.fill_ndimage(gridZ)

    # spherical interpolation
    # Note the not-ideal method for avoiding issues with points at the edges of the grid
    if interp == 'Spherical':
        lut = spi.RectSphereBivariateSpline(np.radians(gridY[1:-1]+90.),
                                            np.radians(gridX[1:-1]+180.),
                                            gridZ_filled[1:-1, 1:-1])

    # flat earth interpolation
    elif interp == 'FlatEarth':
        lut = spi.RectBivariateSpline(gridX, gridY, gridZ_filled.T)

    return lut


def getSubductionBoundarySections(topology_features, rotation_model, time):
    """
    Given files to make topological polygons, returns the features of type 'SubductionZone'
    (initial code here said 'MidOceanRidge' but I assume that this has since been edited/changed
     - KW) and get the first and last point from each one, along with the plate pairs.

    Parameters
    ----------
    topology_features : gpml file
        GPlates markup language file containing GPlates features.
    rotation_model : rot file
        GPlates rotation file.
    time : integer or float
        Time at which features are resolved.

    Returns
    -------
    subduction_boundary_sections : list of pygplates.ResolvedTopologicalSharedSubSegment objects
        A list of all subduction boundary shared subsegments given the conditions above.

    """

    # List to keep our boundary sections
    subduction_boundary_sections = []

    # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
    # We generate both the resolved topology boundaries and the boundary sections between them.
    # These two empty lists here are to store calculation results
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time,
                                 shared_boundary_sections)

    for shared_boundary_section in shared_boundary_sections:

        # Find all subduction zones
        if (
                shared_boundary_section.get_feature().get_feature_type() ==
                pygplates.FeatureType.create_gpml('SubductionZone')
                ):

            # Append all subduction zones segments into the list generated at the start
            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                subduction_boundary_sections.append(shared_sub_segment)

    return subduction_boundary_sections


def find_overriding_and_subducting_plates(subduction_shared_sub_segment, time):
    """
    Determine the overriding and subducting plates of the subduction shared sub-segment. There is
    a similarly named function for ResolvedTopologicalSharedSubSection objects in pygplates
    documentation with similar functionality but I'm unsure if it is exactly the same.

    Parameters
    ----------
    subduction_shared_sub_segment : pygplates.ResolvedTopologicalSharedSubSection object
        Subduction zone segment shared between two pygplates topologies.
    time : integer or float
        Time at which features are resolved.

    Returns
    -------
    overriding_plate : pygplates.ReconstructionGeometry object
        pygplates object corresponding to the overriding plate.
    subducting_plate : pygplates.ReconstructionGeometry object
        pygplates object corresponding to the subducting plate.
    subduction_polarity : string
        Polarity of subduction when viewed above the Earth. Either "left" or "right".

    """

    # Get the subduction polarity of the nearest subducting line.
    subduction_polarity = subduction_shared_sub_segment.get_feature().get_enumeration(
        pygplates.PropertyName.gpml_subduction_polarity)

    # If a subduction polarity does not exist:
    if (not subduction_polarity) or (subduction_polarity == 'Unknown'):
        print(
            ('Unable to find the overriding plate of the subducting shared sub-segment "{0}"'
             ).format(subduction_shared_sub_segment.get_feature().get_name()), file=sys.stderr)

        print('    subduction zone feature is missing subduction polarity property or it' +
              ' is set to "Unknown".', file=sys.stderr)
        return

    # There should be two sharing topologies - one is the overriding plate and the other the
    # subducting plate. Else I guess we would be at a triple junction of sorts? ~KW
    sharing_resolved_topologies = subduction_shared_sub_segment.get_sharing_resolved_topologies()
    if len(sharing_resolved_topologies) != 2:
        print('Unable to find the overriding and subducting plates of the subducting shared ' +
              'sub-segment "{0}" at {1}Ma'.format(
                  subduction_shared_sub_segment.get_feature().get_name(), time), file=sys.stderr)
        print('    there are not exactly 2 topologies sharing the sub-segment.', file=sys.stderr)
        print(str(sharing_resolved_topologies[0].get_resolved_feature(
            ).get_reconstruction_plate_id()), file=sys.stderr)
        return

    # If these checks are passed, then we can start fiding the plateIDs of the two plates involved
    # in subduction. These variables are placeholder while the plates are identified:
    overriding_plate = None
    subducting_plate = None

    # This function returns a list of flags that indicate if a copy of the subsegment geometry
    # gets reversed when contributing to the each resolved topology sharing this subsegment
    geometry_reversal_flags = (
        subduction_shared_sub_segment.get_sharing_resolved_topology_geometry_reversal_flags())

    # Iterate over the two plates here
    for index in range(2):

        # Get individual resolved subduction subsegments and their reversal flags
        sharing_resolved_topology = sharing_resolved_topologies[index]
        geometry_reversal_flag = geometry_reversal_flags[index]

        if (
                sharing_resolved_topology.get_resolved_boundary().get_orientation() ==
                pygplates.PolygonOnSphere.Orientation.clockwise
                ):
            # The current topology sharing the subducting line has clockwise orientation (when
            # viewed from above the Earth). If the overriding plate is to the 'left' of the
            # subducting line (when following its vertices in order) and the subducting line is
            # reversed when contributing to the topology then that topology is the overriding
            # plate. A similar test applies to the 'right' but with the subducting line not
            # reversed in the topology.
            if (
                    (subduction_polarity == 'Left' and geometry_reversal_flag) or
                    (subduction_polarity == 'Right' and not geometry_reversal_flag)
                    ):
                overriding_plate = sharing_resolved_topology
            else:
                subducting_plate = sharing_resolved_topology
        else:
            # The current topology sharing the subducting line has counter-clockwise orientation
            # (when viewed from above the Earth). If the overriding plate is to the 'left' of the
            # subducting line (when following its vertices in order) and the subducting line is not
            # reversed when contributing to the topology then that topology is the overriding
            # plate. A similar test applies to the 'right' but with the subducting line reversed in
            # the topology.
            if (
                    (subduction_polarity == 'Left' and not geometry_reversal_flag) or
                    (subduction_polarity == 'Right' and geometry_reversal_flag)
                    ):
                overriding_plate = sharing_resolved_topology
            else:
                subducting_plate = sharing_resolved_topology

    # If despite all this we still can't find the plates then more error messages are printed...
    if overriding_plate is None:
        print('Unable to find the overriding plate of the subducting shared sub-segment ' +
              '"{0}" at {1}Ma'.format(subduction_shared_sub_segment.get_feature().get_name(),
                                      time), file=sys.stderr)
        print('    both sharing topologies are on subducting side of subducting line.',
              file=sys.stderr)
        return

    if subducting_plate is None:
        print('Unable to find the subducting plate of the subducting shared sub-segment ' +
              '"{0}" at {1}Ma'.format(subduction_shared_sub_segment.get_feature().get_name(),
                                      time), file=sys.stderr)
        print('    both sharing topologies are on overriding side of subducting line.',
              file=sys.stderr)
        return

    return (overriding_plate, subducting_plate, subduction_polarity)


def warp_subduction_segment(tessellated_line,
                            rotation_model,
                            subducting_plate_id,
                            overriding_plate_id,
                            subduction_polarity,
                            time,
                            end_time,
                            time_step,
                            clean_dips,
                            ground_pixel_tree,
                            subducting_plate_disappearance_time=-1,
                            use_small_circle_path=False):
    """
    Function to warp polyline based on starting subduction segment location (this is a big one).
    In other words, this polyline tracks the position of a subducting front as it is subducted.

    Parameters
    ----------
    tessellated_line : list of points in (latitude, longitude)
        A line to tesselate.
    rotation_model : rot file
        GPlates rotation file.
    subducting_plate_id : int
        PlateID of the subducting plate.
    overriding_plate_id : int
        PlateID of the overriding plate.
    subduction_polarity : string
        Polarity of subduction when viewed above the Earth. Either "left" or "right".
    time : integer or float
        Time at which features are resolved.
    end_time : integer or float
        Time at which features are finalised.
    time_step : integer or float
        Time interval over which to iterate over.
    clean_dips : TYPE  # Not too sure what this does, but it is mentioned in the code below (KW)
        DESCRIPTION.
    ground_pixel_tree : TYPE  # Don't think this is actually needed (KW)
        DESCRIPTION.
    subducting_plate_disappearance_time : integer or float, optional
        Disappearing time of the subducting plate in pygplates. The default is -1.
    use_small_circle_path : boolean, optional
        Not too sure what this does (KW). The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # We need to reverse the subducting_normal vector direction if overriding plate is to
    # the right of the subducting line since great circle arc normal is always to the left.
    if subduction_polarity == 'Left':
        subducting_normal_reversal = 1
    else:
        subducting_normal_reversal = -1

    # Tessellate the line, and create an array with the same length as the tessellated points with
    # zero as the starting depth for each point at t=0. Get velocity vectors at each point along
    # polyline.
    points = [point for point in tessellated_line]
    stage_rotation = rotation_model.get_rotation(subducting_plate_disappearance_time + time_step,
                                                 subducting_plate_id,
                                                 subducting_plate_disappearance_time)
    # This variable here is just never called. ~KW
    # present_day_velocity_vectors = pygplates.calculate_velocities(
    #             points,
    #             stage_rotation,
    #             time_step,
    #             pygplates.VelocityUnits.kms_per_my)

    # I have rewritten this part:
    point_depths = [0.] * len(points)
    point_dips = [0.] * len(points)
    point_convergence_rates = [0.] * len(points)

    # Original code here:
    # point_depths = [0. for point in points]
    # point_dips = [0. for point in points]
    # #point_convergence_rates = [i.get_magnitude() for i in relative_velocity_vectors]
    # point_convergence_rates = [0. for point in points]

    # Need at least two points for a polyline. Otherwise, return None for all results
    if len(points) < 2:
        points = None
        point_depths = None
        polyline = None
        return points, point_depths, polyline

    # Create polyline from those points:
    polyline = pygplates.PolylineOnSphere(points)

    # Add original unwarped polyline first to a list of warped polylines
    warped_polylines = []
    warped_polylines.append(polyline)

    # Some legacy code here:
    # warped_end_time = time - warped_time_interval

    # The end time for the warping process:
    warped_end_time = end_time

    # No trying to predict the future:
    if warped_end_time < 0:
        warped_end_time = 0

    # Iterate over each time in the range defined by the input parameters
    for warped_time in np.arange(time, warped_end_time-time_step, -time_step):

        # Lots of legacy code here:
        # if warped_time<=23. and subducting_plate_id==902:
        #     if overriding_plate_id in [224,0,101]:
        #         print('forcing Farallon to become Cocos where overriding plate id is ' +
        #               '%d' % overriding_plate_id)
        #         subducting_plate_id = 909
        #     else:
        #         print('forcing Farallon to become Nazca where overriding plate id is ' +
        #               '%d' % overriding_plate_id)
        #         subducting_plate_id = 911

        # If the time at which the warping occurs is somehow after the time our subducting plate
        # disappears from the gplates model, use the stage pole for the time immediately before it
        # disappears:
        if warped_time <= subducting_plate_disappearance_time:
            print('Using %0.2f to %0.2f Ma stage pole for plate %d' % (
                subducting_plate_disappearance_time + time_step,
                subducting_plate_disappearance_time,
                subducting_plate_id)
                )
            stage_rotation = rotation_model.get_rotation(
                subducting_plate_disappearance_time + time_step,
                subducting_plate_id,
                subducting_plate_disappearance_time)

        # Else use the stage rotation that describes the motion of the subducting plate, with
        # respect to the fixed plate for the rotation model.
        else:
            stage_rotation = rotation_model.get_rotation(warped_time - time_step,
                                                         subducting_plate_id, warped_time)

        # If use_small_circle_path is True, the pole and angle representing finite rotation are
        # returned in addition to the stuff above.
        if use_small_circle_path:
            stage_pole, stage_pole_angle_radians = stage_rotation.get_euler_pole_and_angle()

        # Get velocity vectors at each point along polyline
        relative_velocity_vectors = pygplates.calculate_velocities(
                points,
                stage_rotation,
                time_step,
                pygplates.VelocityUnits.kms_per_my)

        # Get subducting normals for each segment of tessellated polyline.
        # Also add an imaginary normal prior to first and post last points.
        # (Makes it easier to later calculate average normal at tessellated points).
        # The number of normals will be one greater than the number of points.
        subducting_normals = []

        # Imaginary segment prior to first point.
        subducting_normals.append(None)

        # Iterate over each segment in the polyline to find their unit vector normal directions
        for segment in polyline.get_segments():
            if segment.is_zero_length():
                subducting_normals.append(None)
            else:
                # The normal to the subduction zone in the direction of subduction (towards
                # overriding plate).
                subducting_normals.append(
                    subducting_normal_reversal * segment.get_great_circle_normal())

        # Append one final imaginary segment after the last point.
        subducting_normals.append(None)

        # Get vectors of normals and parallels for each segment, use these to get a normal and
        # parallel at each point location
        normals = []
        parallels = []

        # Iterate over points in the tessellated polyline:
        for point_index in range(len(points)):

            # Collect normals from the list that we have jsut generated:
            prev_normal = subducting_normals[point_index]
            next_normal = subducting_normals[point_index + 1]

            # If both adjoining segments are zero-length, skip the point altogether.
            if prev_normal is None and next_normal is None:
                continue

            # If point is at the start or end of the polyline, use the normal of the next/previous
            # points respectively. Else create a normalised averaged unit vector using the function
            # pygplates.Vector3D.to_normalised().
            if prev_normal is None:
                normal = next_normal
            elif next_normal is None:
                normal = prev_normal
            else:
                normal = (prev_normal + next_normal).to_normalised()

            # This chap here gets the normalised cross product vector between the point (in
            # Cartesian space) and the normal vector in order to find the parallel to that point
            # (in the direction of the subduction zone).
            parallel = pygplates.Vector3D.cross(points[point_index].to_xyz(),
                                                normal).to_normalised()
            normals.append(normal)
            parallels.append(parallel)

        # Iterate over each point to determine the incremented position based on plate motion and
        # subduction dip. Start with some empty lists for storage:
        warped_points = []
        warped_point_depths = []
        warped_dips = []
        warped_convergence_rates = []

        # Iterate over the point indices and points in the points list:
        for point_index, point in enumerate(points):

            # Get dip angle using function defined above:
            point_lat = point.to_lat_lon_array()[0][0]
            point_lon = point.to_lat_lon_array()[0][1]
            dip_index = get_dip_angle_from_slab2(point_lat, point_lon, ground_pixel_tree)

            # print(dip_index)

            # The dip_index returns [distance][index of array] - get all data corresponding to that
            # one polyline point index.
            dip_angle_degrees = clean_dips.values[dip_index]  # not too sure what clean_dips is
            dip_angle_radians = np.radians(dip_angle_degrees)
            normal = normals[point_index]
            parallel = parallels[point_index]
            velocity = relative_velocity_vectors[point_index]

            # If the velocity vector is a zero vector:
            if velocity.is_zero_magnitude():
                # Point hasn't moved, so the warped point is the same as the old point.
                warped_points.append(point)
                warped_point_depths.append(point_depths[point_index])
                warped_dips.append(point_dips[point_index])
                warped_convergence_rates.append(point_convergence_rates[point_index])
                continue

            # Reconstruct the tracked point from position at current time to position at the next
            # time step. First we obtain the trench-orthogonal and parallel components of velocity:
            normal_angle = pygplates.Vector3D.angle_between(velocity, normal)
            parallel_angle = pygplates.Vector3D.angle_between(velocity, parallel)

            # Trench parallel and normal components of velocity vector.
            velocity_normal = np.cos(normal_angle) * velocity.get_magnitude()
            velocity_parallel = np.cos(parallel_angle) * velocity.get_magnitude()
            normal_vector = normal.to_normalised() * velocity_normal
            parallel_vector = parallel.to_normalised() * velocity_parallel

            # Adjust velocity based on subduction vertical dip angle (i.e., make sure the vertical
            # component of subduction is accounted for).
            velocity_dip = parallel_vector + np.cos(dip_angle_radians) * normal_vector
            warped_dips.append(dip_angle_degrees)
            warped_convergence_rates.append(velocity)

            # deltaZ is the amount that this point increases in depth within the time step.
            deltaZ = np.sin(dip_angle_radians) * velocity.get_magnitude()

            # Angle between normal and parallel vector should be 90 degrees always.
            # print(np.degrees(np.arccos(pygplates.Vector3D.dot(normal_vector, parallel_vector))))

            if use_small_circle_path:
                # Rotate original stage pole by the same angle that effectively
                # rotates the velocity vector to the dip velocity vector.
                dip_stage_pole_rotate = pygplates.FiniteRotation(
                        point,
                        pygplates.Vector3D.angle_between(velocity_dip, velocity))
                dip_stage_pole = dip_stage_pole_rotate * stage_pole
            else:
                # Get the unnormalised vector perpendicular to both the point and velocity vector.
                dip_stage_pole_x, dip_stage_pole_y, dip_stage_pole_z = pygplates.Vector3D.cross(
                        point.to_xyz(), velocity_dip).to_xyz()

                # PointOnSphere requires a normalised (ie, unit length) vector (x, y, z).
                dip_stage_pole = pygplates.PointOnSphere(
                        dip_stage_pole_x, dip_stage_pole_y, dip_stage_pole_z, normalise=True)

            # Get angle that velocity will rotate seed point along great circle arc
            # over 'time_step' My (if velocity in Kms / My).
            dip_stage_angle_radians = velocity_dip.get_magnitude() * (
                    time_step / pygplates.Earth.mean_radius_in_kms)

            if use_small_circle_path:
                # Increase rotation angle to adjust for fact that we're moving a
                # shorter distance with small circle (compared to great circle).
                dip_stage_angle_radians /= np.abs(np.sin(
                        pygplates.Vector3D.angle_between(
                                dip_stage_pole.to_xyz(), point.to_xyz())))
                # Use same sign as original stage rotation.
                if stage_pole_angle_radians < 0:
                    dip_stage_angle_radians = -dip_stage_angle_radians

            # get the stage rotation that describes the lateral motion of the
            # point taking the dip into account
            dip_stage_rotation = pygplates.FiniteRotation(dip_stage_pole, dip_stage_angle_radians)

            # increment the point long,lat and depth
            warped_point = dip_stage_rotation * point
            warped_points.append(warped_point)
            warped_point_depths.append(point_depths[point_index] + deltaZ)

        # finished warping all points in polyline
        # --> increment the polyline for this time step
        warped_polyline = pygplates.PolylineOnSphere(warped_points)
        warped_polylines.append(warped_polyline)
        # print('h', warped_dips)

        # For next warping iteration.
        points = warped_points
        polyline = warped_polyline
        point_depths = warped_point_depths
        point_dips = warped_dips
        point_convergence_rates = warped_convergence_rates
        # print(point_dips)
    return points, point_depths, polyline, point_dips, point_convergence_rates


def write_subducted_slabs_to_xyz(output_filename, output_data):

    with open(output_filename, 'w') as output_file:
        output_file.write('Long,Lat,Depth,AgeAtSubduction,TimeOfSubduction\n')
        for output_segment in output_data:
            for index in range(len(output_segment[2])):
                output_file.write('%0.6f,%0.6f,%0.6f,%0.2f,%0.2f\n' % (
                    output_segment[1].to_lat_lon_array()[index, 1],
                    output_segment[1].to_lat_lon_array()[index, 0],
                    output_segment[2][index],
                    output_segment[3][index],
                    output_segment[0]))


#######################################################################
# Function from here downwards are largely deprecated (I have not touched them - KW)
def getRidgeEndPoints(topology_features, rotation_model, time):
    # given files to make topological polygons, returns the features of type 'MidOceanRidge'
    # and get the first and last point from each one, along with the plate pairs
    MorEndPointArrays = []
    MorEndPointGeometries = []
    MorPlatePairs = []
    subduction_boundary_sections = []

    # Resolve our topological plate polygons (and deforming networks) to the current 'time'.
    # We generate both the resolved topology boundaries and the boundary sections between them.
    resolved_topologies = []
    shared_boundary_sections = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time,
                                 shared_boundary_sections)

    for shared_boundary_section in shared_boundary_sections:
        if (
                shared_boundary_section.get_feature().get_feature_type() ==
                pygplates.FeatureType.create_gpml('MidOceanRidge')
                ):

            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():

                if len(shared_sub_segment.get_sharing_resolved_topologies()) == 2:
                    plate_pair = [shared_sub_segment.get_sharing_resolved_topologies(
                        )[0].get_feature().get_reconstruction_plate_id(),
                        shared_sub_segment.get_sharing_resolved_topologies(
                            )[1].get_feature().get_reconstruction_plate_id()]
                else:
                    plate_pair = np.array((-1, -1))
                # print 'skipping bad topological segment....'

                tmp = shared_sub_segment.get_geometry()

                MorEndPointArrays.append(tmp.to_lat_lon_array())
                MorEndPointGeometries.append(pygplates.PointOnSphere(tmp.get_points()[0]))
                MorEndPointGeometries.append(pygplates.PointOnSphere(tmp.get_points()[-1]))
                MorPlatePairs.append(plate_pair)
                MorPlatePairs.append(plate_pair)

        elif (shared_boundary_section.get_feature().get_feature_type() ==
              pygplates.FeatureType.create_gpml('SubductionZone')):

            for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():
                subduction_boundary_sections.append(shared_sub_segment)

    return MorEndPointArrays, MorEndPointGeometries, MorPlatePairs, subduction_boundary_sections


def getMor2szDistance2(MorEndPointGeometries, MorPlatePairs, subduction_boundary_sections):
    # Get distance between end points of resolved mid-ocean ridge feature lines
    # and subduction zones, using subduction segments from resolve_topologies

    Mor2szDistance = []
    sz_opid = []

    for MorPoint, PlatePair in zip(MorEndPointGeometries, MorPlatePairs):

        min_distance_to_all_features = np.radians(180)
        # nearest_sz_point = None
        opid = 0

        for subduction_boundary_section in subduction_boundary_sections:

            if MorPoint is not None:
                min_distance_to_feature = pygplates.GeometryOnSphere.distance(
                    MorPoint,
                    subduction_boundary_section.get_geometry(),
                    min_distance_to_all_features)

                # If the current geometry is nearer than all previous geometries then
                # its associated feature is the nearest feature so far.
                if min_distance_to_feature is not None:
                    min_distance_to_all_features = min_distance_to_feature
                    opid = subduction_boundary_section.get_feature().get_reconstruction_plate_id()

        Mor2szDistance.append(min_distance_to_all_features*pygplates.Earth.mean_radius_in_kms)
        sz_opid.append(opid)

    return Mor2szDistance, sz_opid


def track_point_to_present_day(seed_geometry, PlateID, rotation_model, start_time, end_time,
                               time_step):
    # Given a seed geometry at some time in the past, return the locations of this point
    # at a series of times between that time and present-day

    point_longitude = []
    point_latitude = []

    for time in np.arange(start_time, end_time, -time_step):

        stage_rotation = rotation_model.get_rotation(time-time_step, PlateID, time)

        # use the stage rotation to reconstruct the tracked point from position at current time
        # to position at the next time step
        incremented_geometry = stage_rotation * seed_geometry

        # replace the seed point geometry with the incremented geometry in preparation for next
        # iteration
        seed_geometry = incremented_geometry

        point_longitude.append(seed_geometry.to_lat_lon_point().get_longitude())
        point_latitude.append(seed_geometry.to_lat_lon_point().get_latitude())

    return point_longitude, point_latitude


def rotate_point_to_present_day(seed_geometry, PlateID, rotation_model, start_time):
    # Given a seed geometry at some time in the past, return the locations of this point
    # at present-day (only)

    point_longitude = []
    point_latitude = []

    stage_rotation = rotation_model.get_rotation(0, PlateID, start_time, anchor_plate_id=1)

    # use the stage rotation to reconstruct the tracked point from position at current time
    # to position at the next time step
    incremented_geometry = stage_rotation * seed_geometry

    # replace the seed point geometry with the incremented geometry in preparation for next
    # iteration
    seed_geometry = incremented_geometry

    point_longitude.append(seed_geometry.to_lat_lon_point().get_longitude())
    point_latitude.append(seed_geometry.to_lat_lon_point().get_latitude())

    return point_longitude, point_latitude


def create_seed_point_feature(plat, plon, plate_id, conjugate_plate_id, time):
    # Create a gpml point feature given some attributes

    point = pygplates.PointOnSphere(plat, plon)
    point_feature = pygplates.Feature()
    point_feature.set_geometry(point)
    point_feature.set_valid_time(time, 0.)
    point_feature.set_reconstruction_plate_id(plate_id)
    point_feature.set_conjugate_plate_id(conjugate_plate_id)
    point_feature.set_name('Slab Edge | plate %d | rel. plate %d' % (plate_id, conjugate_plate_id))

    return point_feature
