# pgpslabs

Code to make simple reconstructions of slab geometries using gplates-format topological plate reconstructions. 

The idea is principally inspired by various papers of Derek Thorkelson to map the extent of slab windows through geological time. The original code was designed by Simon Williams and John Cannon, associated with the paper McGirr et al. (2021). The code relies heavily on pyGPlates (www.pygplates.org), as well as some preconstructed routines for pyGPlates, GPlately, by Ben Mather.

In current version, raster maps of of various ocean-crust properties (e.g. seafloor age, spreading rate, peridotite content, carbon content) are intersected against known subduction zones. Points along the subduction zones are used to sample the rasters, and the kinematic-forward evolution is used to 'subduct' these points. Dip is calculaued using a GPlately function (I understand a paper is in advanced revision?), and use alongside convergence rate to approximate depth. At the next time-step (going forward in time) a new set of parameters are extracted along subduction zones and these points (in addition to the previously 'subducted' pionts are further subducted. This continues until end time (in our analysis, present-day).

All data required to rerun the analysis should be present (at least, that's the intention), so if something is missing let me know.

1_make_tracks.ipynb - heavy lifter of the 6 notebookes, this is where the magic actually happens and points are subducted and linked to both SLAB2 and Syracuse et al. 2010 thermal data

2_RAP_in_subduction_zones.ipynb - simple notebook to plot P-T distributions (this will be recreated in future notebooks)

3_antigorite_dehydration.ipynb - as (2_RAP_in_subduction_zones.ipynb) but also plotting it against antigorite stability data. Also basic calculations of H2 flux here

4_tectonic_drivers.ipynb - notebook to compare different parameters against results (spreading rate/convergence rate/age/depth/peridotite content etc.)

5_map_of_sub_zones.ipynb - notebook to plot global and regional maps of results

6_chekcing_tracks.ipynb - OLD, no longer used and likely to be deleted (was used to investigate single subduction zones in the previous method)

McGirr, R., Seton, M. and Williams, S., 2021. Kinematic and geodynamic evolution of the Isthmus of Panama region: Implications for Central American Seaway closure. GSA Bulletin, 133(3-4), pp.867-884.

Two forked python libraries need to be installed, located at: (to-do, roll into published version, still bug-shooting...)

https://github.com/amer7632/PlateTectonicTools

https://github.com/amer7632/Slab-Dip

