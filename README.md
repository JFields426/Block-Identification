# Block-Identification
Code for identifying atmospheric blocking events using AGP.

File Descriptions:
1. block_tag.py - Creates a datset of instantaneous blocking for each grid point using the AGP identification method.
2. StitchBlobs.sh - Defines large-scale blocking events by stitching instantaneously blocked grid points.
3. blocktag_file.txt - Lists the input file for StitchBlobs (block_tag output).
4. blockid_file.txt - Lists the output file for StitchBlobs.
5. block_filter.py - Filters blocks by the area of their maximum spatial extent.
6. GPH_optimizer.py - Combines and resamples GPH files for memory optimization.
7. blockinfo.py - Creates a .csv file containing information for the blocking dataset.

[OPTIONAL FILES (Filter nonconforming blocking events based on local wave activity (Liu and Wang, 2025))]
8. lwa.py - Creates a dataset of cyclonic and anticyclonic local wave activity.
9. blockinfo_lwa - Creates the same .csv file as blockinfo.py with added maximum LWA value for each blocking event.
10. lwa_filter.py - Filters out all blocking events that do not conform to 'ridge' and 'dipole' flow patterns.
