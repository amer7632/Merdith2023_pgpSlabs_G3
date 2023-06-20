# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:12:04 2023

@author: Andrew Merdith, comments by Kev Wong
"""


from gplately import pygplates


def get_topologies_and_plate_id_list(topology_features, rotation_model, time):
    """
    Function to obtain resolved topologies from a GPlates feature collection and rotation model
    (output 0). It then takes the IDs of the new topological plate features and plate areas and
    appends them into two lists (outputs 1 and 2).

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
    resolved_topologies : list of pygplates topology objects
        List of resolved topologies.
    plate_id_list : list of integers
        List of IDs for plate topologies.
    plate_area_list : list of floats
        List of areas of plate topologies.

    """
    resolved_topologies = []
    pygplates.resolve_topologies(topology_features, rotation_model, resolved_topologies, time)

    plate_id_list = []
    for topology in resolved_topologies:
        plate_id_list.append(topology.get_feature().get_reconstruction_plate_id())

    plate_area_list = []
    for topology in resolved_topologies:
        plate_area_list.append(topology.get_resolved_geometry().get_area())

    return resolved_topologies, plate_id_list, plate_area_list


def find_plate_change_for_time_step(topology_features, rotation_model, time_from, time_to,
                                    verbose=True):
    """
    Function to find changes in plates between one time step and another. Returns two lists of
    tuples corresponding to information concerning the original plate setup, any new plates,
    and any disappearing plates.

    Parameters
    ----------
    topology_features : gpml file
        GPlates markup language file containing GPlates features.
    rotation_model : rot file
        GPlates rotation file.
    time_from : integer or float
        Time from which to find plate change.
    time_to : integer or float
        Time to which to find plate change.
    verbose : boolean, optional
        Print results or not? The default is True.

    Returns
    -------
    appearing_plate_map : list of tuples
        List of tuples of size 2 comprising the original appearing plate ID and the plate ID in
        which its centroid was found.
    disappearing_plate_map : list of tuples
        List of tuples of size 2 comprising the original disappearing plate ID and the plate ID in
        which its centroid is found.

    """

    # This first part uses the previous function to obtain the list of topologies, plate IDs, and
    # plate areas for the chosen gpml features given the rot file.
    (resolved_topologies_t1,
     plate_id_list_t1,
     plate_area_list_t1) = get_topologies_and_plate_id_list(topology_features,
                                                            rotation_model,
                                                            time_to)
    (resolved_topologies_t2,
     plate_id_list_t2,
     plate_area_list_t2) = get_topologies_and_plate_id_list(topology_features,
                                                            rotation_model,
                                                            time_from)

    # The following parts compares the two plate ID lists generated from the two snapshots of the
    # two times (time_to and time_from).
    # common_plates = plates that are common to both lists
    # appearing_plates = plates that exist only in the 'time_to' (first) setup
    # disappearing_plates = plates that exist only in the 'time_from' (second) setup
    common_plates = set(plate_id_list_t1).intersection(plate_id_list_t2)
    appearing_plates = set(plate_id_list_t1).difference(plate_id_list_t2)
    disappearing_plates = set(plate_id_list_t2).difference(plate_id_list_t1)

    # This code prints these common/different plates out:
    if verbose:
        # plate_ids in both lists
        print('plates that persist between %0.2f Ma and %0.2f Ma are: \n %s\n' %
              (time_from, time_to, common_plates))

        # plates ids that are not in the t2 list
        print('plates that appeared between %0.2f Ma and %0.2f Ma are: \n %s\n' %
              (time_from, time_to, appearing_plates))

        # plate ids that are not in the t1 list
        print('plates that disappeared between %0.2f Ma and %0.2f Ma are: \n %s' %
              (time_from, time_to, disappearing_plates))

        print('\nWorking on appearing plates')

    # This portion depends on the next function (SPOILERS!). This first one maps the new appearing
    # plates within the context of the previous plate setup. The first tuple output will be the new
    # plate and the second output will be the old plate.
    appearing_plate_map = plate_change_mapping(appearing_plates,
                                               resolved_topologies_t1,
                                               resolved_topologies_t2,
                                               verbose=verbose)

    if verbose:
        print('\nWorking on disappearing plates')

    # This second one maps the location of the disappearing plates within the context of the new
    # plate setup.
    disappearing_plate_map = plate_change_mapping(disappearing_plates,
                                                  resolved_topologies_t2,
                                                  resolved_topologies_t1,
                                                  verbose=verbose)

    # The output is two lists of tuples, each comprising a plate ID and the plate ID of the plate
    # in which it is now mapped.
    return appearing_plate_map, disappearing_plate_map


def plate_change_mapping(plate_list_at_time, topologies_at_time, topologies_at_delta_time,
                         verbose=True):
    """
    Function to find plate centroids at a time before/after another time.

    Parameters
    ----------
    plate_list_at_time : list of integers
        List of GPlates plate IDs.
    topologies_at_time : list of GPlates topology objects
        List of GPlates topologies at a time.
    topologies_at_delta_time : list of GPlates topology objects
        List of GPlates topologies at any other time.
    verbose : boolean, optional
        Print results or not? The default is True.

    Returns
    -------
    plate_map : list of tuples
        List of tuples of size 2 comprising the original plate ID and the plate ID in which its
        centroid is found.

    """

    plate_map = []

    # Iterate over the plates IDs resolved at a time (I assume 'time_from'):
    for plate in plate_list_at_time:

        # Get the interior centroid of all the plates and plates only at time 'time_from'
        for topology in topologies_at_time:
            if topology.get_feature().get_reconstruction_plate_id() == plate:
                centroid = topology.get_resolved_geometry().get_interior_centroid()

        # Does the centroid of the plate exist at time 'time_to' within a new/the same plate?
        for topology in topologies_at_delta_time:
            if topology.get_resolved_geometry().is_point_in_polygon(centroid):

                # This bit just prints stuff out
                if verbose:
                    print('Centroid for plate %d mapped to plate %d at delta time' %
                          (plate, topology.get_feature().get_reconstruction_plate_id()))

                # Append a tuple to a list for output
                plate_map.append((plate, topology.get_feature().get_reconstruction_plate_id()))

    return plate_map

# This is some legacy code which I am not touching

# def determine(moving_plate_id, fixed_plate_id, time_from, time_to, rotation_model):
#     angle = rotation_model.get_rotation(time_to, moving_plate_id, time_from, fixed_plate_id)
#     return angle.represents_identity_rotation()
# plate_id_list_copy_True[:] = [tup for tup in plate_id_list_copy_True
#                               if determine(tup, fixed_plate_id, time_from, time_to,
#                                            rotation_model)]


def match_splits_to_origin(appearing_plate_map):
    """
    Function to match groups of appearing plates from a single plate.

    Parameters
    ----------
    appearing_plate_map : list of tuples
        List of tuples, each comprising their original plate ID and the plate ID of where their
        centroids were originally placed.

    Returns
    -------
    new_plates_grouped_by_origin_plate : list of tuples
        List of tuples, each comprising an original plate and their successors.

    """
    # AM: This function only cares about appearing plates, and aims to find pairs of plates that
    # split from the same common plate.
    # KW: I believe plates[0] is the old plates and plates[1] is the new plates

    origin_plates_for_new_plates = []

    # Append the plate IDs of the new appearing plates into a list
    for plates in appearing_plate_map:
        origin_plates_for_new_plates.append(plates[1])

    # This next part over here finds where both plate IDs in the appearing_map tuples. This
    # iterates over all original plates.
    new_plates_grouped_by_origin_plate = []
    for origin_plate in set(origin_plates_for_new_plates):
        successor_plates = []

        # Finds all successors for a single plate ID, and appends the precursors into a list.
        for plates in appearing_plate_map:

            # The successor plates will all share the same precursor ID.
            if plates[1] == origin_plate:
                successor_plates.append(plates[0])

        # Append a tuple of the origin plate and a list of its successors
        new_plates_grouped_by_origin_plate.append((origin_plate, successor_plates))

    return new_plates_grouped_by_origin_plate


def get_great_circle_along_plate_split(plate_split, shared_boundary_sections):
    """
    Function to get a pyGPlates great circle arc object that divides a splitting plate into two
    new plates.

    Parameters
    ----------
    plate_split : tuple
        Tuple comprising plate ID of original plate and plate IDs of successors.
    shared_boundary_sections : list of shared_boundary_section objects
        List of shared_boundary_section objects obtained from resolving pyGPlates topologies

    Returns
    -------
    great_circle_arc : TYPE
        DESCRIPTION.

    """
    great_circle_arc = None

    print(plate_split)
    geometries_along_new_split = []

    # For every shared boundary section in the topologies...
    for shared_boundary_section in shared_boundary_sections:

        # For every sub segment within that shared boundary section...
        for shared_sub_segment in shared_boundary_section.get_shared_sub_segments():

            # Get the topology objects sharing that one segment
            sharing_topologies = shared_sub_segment.get_sharing_resolved_topologies()

            # This assumes that the plates always split into two (and not three, four.....)
            if len(sharing_topologies) != 2:
                # print('bad topological segment')
                continue
            else:
                sub_segment_plate_pair = []
                for topology in sharing_topologies:
                    # Get the plate IDs of the topology objects sharing that segment
                    sub_segment_plate_pair.append(
                        topology.get_feature().get_reconstruction_plate_id())

                # print(sub_segment_plate_pair)

                # If the split plates match that particular sub segment we are probing...
                if (
                    plate_split[1][0] in sub_segment_plate_pair and
                    plate_split[1][1] in sub_segment_plate_pair
                        ):

                    # Add this new shared sub segment geometry to the list.
                    geometries_along_new_split.append(shared_sub_segment.get_geometry())

    if len(geometries_along_new_split) != 1:
        print('Only works for new splits with one geometry so far')

    else:
        # print(geometries_along_new_split)

        # This bit draws a great circle arc along the split between the segments
        great_circle_arc = pygplates.GreatCircleArc(
            geometries_along_new_split[0].get_points()[0],
            geometries_along_new_split[0].get_points()[-1])

    # This returns a GreatCircleArc object that divides the original plate into two.
    return great_circle_arc


def get_plate_disappearance_time_lut(topology_features, rotation_model, time_list, verbose=False):
    """
    Function to find when a plate disappears.

    Parameters
    ----------
    topology_features : gpml file
        GPlates markup language file containing GPlates features.
    rotation_model : rot file
        GPlates rotation file.
    time_list : list of integers or floats
        Time at which features are resolved.
    verbose : boolean, optional
        Print results or not? The default is True.

    Returns
    -------
    plate_disappearance_time_lut : list of tuples
        List of tuples comprising a plate ID of the disappearing plate and the time it disappears.

    """

    plate_disappearance_time_lut = []

    # Iterate over the list of times...
    for time in time_list:
        time_from = time+1
        time_to = time

        (appearing_plate_map,
         disappearing_plate_map) = find_plate_change_for_time_step(topology_features,
                                                                   rotation_model,
                                                                   time_from,
                                                                   time_to,
                                                                   verbose)
        print(disappearing_plate_map)
        # If there are plates disappearing...
        if len(list(disappearing_plate_map)) > 0:
            print('here', list(zip(*disappearing_plate_map))[0])
            for plate_id in list(zip(*disappearing_plate_map))[0]:
                # plate_id # Not too sure what this is doing here

                # Append to that list a list of plate IDs corresponding to the original
                # disappearing plate ad when it disappears
                plate_disappearance_time_lut.append((plate_id, time))

    return plate_disappearance_time_lut
