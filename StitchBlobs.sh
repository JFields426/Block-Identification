#======================================================
#THIS CODE REQUIRES THE TEMPEST EXTREMES PACKAGE TO RUN
#conda install -c conda-forge tempest-extremes
#======================================================

StitchBlobs --in_list "blocktag_file.txt" \
            --out_list "blockid_file.txt" \
            --var "block_tag" \
            --latname "latitude" \
            --lonname "longitude" \
            --mintime "4d" \
            --min_overlap_prev 20 \
