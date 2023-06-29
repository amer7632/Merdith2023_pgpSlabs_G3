import numpy as np
from scipy import spatial
import splits_and_merges as snm
import slab_tracker_utils as slab



class Subducted_slabs_class(object):

    def __init__(self,
                 reconstruction_model,
                 times,
                 dip_data,
                 tesselation_distance,
                 grid_filenames,
                 handle_splits=False):

        '''
        Reconstruction_model: reconstruction model class from GPRM
        Times: array of times for anlaysis to occur, going forwards (i.e. old to young)
        Dip data: what data to use for dip, can be either a single uniform float, or pass in a map of values
                  such as from Slab2
        Tesselation: float, in radians, of spacing for analysis to occur at.
        Data grid filenames: Default False, otherwise a series of xarray netcdf files that containt data to
                             propogate onto slabs. Need to have one per timelsice.
        Handle splits: how to approach slabs that change their plate ID during subduction, default is False
                       (becomes more important as one goes further back in time)
        '''

        self.slab_times = times
        self.dip_data = self.clean_dip_data(dip_data)
        self.tesselation_distance = tesselation_distance
        self.handle_splits = handle_splits

        self.reconstruction_model = reconstruction_model

        self.grid_filenames = grid_filenames

        if self.handle_splits:
            self.disappearing_plates = self.get_disappearing_plates()


        self.output_data = self.get_subducted_slabs()

        self.slab_time, self.slab_polyline, self.slab_point_depths, \
        self.slab_interpolated_values, self.slab_dips, \
        self.slab_overriding_plate_id, self.slab_subducting_plate_id, \
        self.slab_convergence_rate = [i[0] for i in self.output_data], [i[1] for i in self.output_data], \
                                     [i[2] for i in self.output_data], [i[3] for i in self.output_data], \
                                     [i[4] for i in self.output_data], [i[5] for i in self.output_data], \
                                     [i[6] for i in self.output_data], [i[7] for i in self.output_data]

    def clean_dip_data(self, dip_data):

        if isinstance(dip_data, float):
            return dip_data

        else:
            clean_dips = dip_data.dip[dip_data.dip.notnull()]

            coords = np.column_stack((clean_dips.latitude.values.ravel(),
                                      clean_dips.longitude.values.ravel()))

            dip_ground_pixel_tree = spatial.cKDTree(slab.transform_coordinates(coords))

            return clean_dips, dip_ground_pixel_tree

    def get_disappearing_plates(self):
        '''
        Get plates that disappear during analysis. Default is False.

        Returns a tuple of (plate_id, time of disappearance) for each plate that disappears/
        '''

        plate_disappearance_time_lut = snm.get_plate_disappearance_time_lut(
                                             self.reconstruction_model,
                                             self.slab_times,
                                             verbose=True)

        print (plate_disappearance_time_lut)

        return plate_disappearance_time_lut

    def get_subducted_slabs(self):
        output_data = []
        # loop over a series of times at which we want to extract trench iso-sub-chrons
        for time in self.slab_times:

            print( 'time %0.2f Ma' % time)
            tmp_plate_snapshot = self.reconstruction_model.plate_snapshot(time)

            # Get subduction boundary segments
            resolved_topological_segments = tmp_plate_snapshot.resolved_topological_sections

            # call function to get subduction boundary segments
            subduction_boundary_sections = slab.getSubductionBoundarySections(resolved_topological_segments)

            # get interpalation of our grids
            lut = self.grid_interpolator(time)

            for segment_index,subduction_segment in enumerate(subduction_boundary_sections):
                # find the overrding plate id (and only continue if we find it)
                overriding_and_subducting_plates = slab.find_overriding_and_subducting_plates(subduction_segment,time)

                if not overriding_and_subducting_plates:
                    continue
                overriding_plate, subducting_plate, subduction_polarity = overriding_and_subducting_plates

                overriding_plate_id = overriding_plate.get_resolved_feature().get_reconstruction_plate_id()
                subducting_plate_id = subducting_plate.get_resolved_feature().get_reconstruction_plate_id()
                subducting_plate_disappearance_time = -1.

                if self.handle_splits:
                    for plate_disappearance in plate_disappearance_time_lut:
                        if plate_disappearance[0]==subducting_plate_id:
                            subducting_plate_disappearance_time = plate_disappearance[1]

                tessellated_line = subduction_segment.get_resolved_geometry().to_tessellated(self.tesselation_distance)


                interpolated_values = self.get_interpolated_values_at_sub_zone(time, tessellated_line, lut)
                            # CALL THE MAIN WARPING FUNCTION
                (points,
                 point_depths,
                 polyline, dips, convergence_rates) = slab.warp_subduction_segment(
                                                            tessellated_line,
                                                            self.reconstruction_model.rotation_model,
                                                            subducting_plate_id,
                                                            overriding_plate_id,
                                                            subduction_polarity,
                                                            time,
                                                            self.slab_times[-1],
                                                            np.diff(self.slab_times)[0],
                                                            self.dip_data,
                                                            subducting_plate_disappearance_time)
                #print(dips)
                output_data.append([time, polyline,point_depths,interpolated_values, dips,
                                    overriding_plate_id, subducting_plate_id, convergence_rates])

        return output_data

    def get_interpolated_values_at_sub_zone(self, time, tessellated_line, lut):

        grid_interpolations = []
        if lut is not None:
            x = tessellated_line.to_lat_lon_array()[:,1]
            y = tessellated_line.to_lat_lon_array()[:,0]
            for ind,interpolator in enumerate(lut):
                grid_interpolations.append(interpolator.ev(np.radians(y+90.),np.radians(x+180.)))
        else:
            # if no  grids, we will record time of subduction
            grid_interpolations = [time for point in tessellated_line.to_lat_lon_array()[:,1]]

        return grid_interpolations

    def grid_interpolator(self, time):

        lut = []
        if self.grid_filenames is not None:
            for ind,i in enumerate(self.grid_filenames):
                #print('%s%d%s' % (grid_filenames[ind][0], time, grid_filenames[ind][1]))
                grdfile = '%s%d%s' % (self.grid_filenames[ind][0], time, self.grid_filenames[ind][1])
                lut.append(slab.make_grid_interpolator(grdfile))

        return lut
