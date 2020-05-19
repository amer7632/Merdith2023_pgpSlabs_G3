import pygplates
import numpy as np

def get_containing_segment(point, line):
    '''
Given a point on a line, find the lat,lons of the segment of a line that the point sits on.

Both point and line should be a pygplates class. (NB it might work with raw lat/lons)
    '''
    #make sure both are pygplates features
    point = pygplates.PointOnSphere(point)
    line = pygplates.PolylineOnSphere(line)

    #get closest point
    closest_point_data = \
    pygplates.GeometryOnSphere.distance(point, line,
                                        return_closest_indices=True)

    #get segment containing closest point
    closest_segment_on_polyline = line.get_segments()[closest_point_data[2]]

    return closest_segment_on_polyline

def get_distance_between_points(point, segment):

    '''
Given a point and a segment (i.e. a line between two points) get the distance between the
point and segment start/end points
    '''

    #the point in question
    from_point = point

    #start and end points
    #NB start point corresponds to the index returned from GeometryOnSphere.distance (return index)
    start_point = pygplates.PointOnSphere(segment.get_start_point().to_lat_lon_array()[0])
    end_point = pygplates.PointOnSphere(segment.get_end_point().to_lat_lon_array()[0])

    #get distance, we will keep it in radians for the moment because depth dependency will change
    #radius of the earth
    distance_to_start = pygplates.GeometryOnSphere.distance(from_point, start_point)
    distance_to_end = pygplates.GeometryOnSphere.distance(from_point, end_point)

    return distance_to_start, distance_to_end

def awkward_linear_interp(variable_start, variable_end, distance_start, distance_end):
    '''
As said, awkward basic linear interp between two values using distances.

In this case variable values are at each point. Distances are in radians between start/end points of an arc
segment and the intercept point previously determined.
    '''

    dVariable = variable_end - variable_start
    dDistance = distance_start + distance_end
    intercept_variable = variable_start + distance_start * dVariable/dDistance

    return intercept_variable

def get_intercept_values(intercept_point, iso_subchron, variables):

    '''
Get intercept values of iso-subchrons along a defined cross section line.
Uses simple linear interpolation.

intercept_point: lat/lon of intercept between two lines
iso_subchron: the line containing the variable data that intercepts the point
variables: the variable we need to interpolate
    '''

    variable_at_start, variable_at_end = variables[0], variables[1]
    segment_of_point = get_containing_segment(intercept_point, iso_subchron)
    distance_to_start, distance_to_end = get_distance_between_points(intercept_point, segment_of_point)
    intercept_variable = awkward_linear_interp(variable_at_start, variable_at_end, distance_to_start, distance_to_end)

    return intercept_variable
