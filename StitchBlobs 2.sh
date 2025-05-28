StitchBlobs --in_list "hgt.19502022_blocktag_files.txt" \
            --out_list "hgt.19502022_blockid_files.txt" \
            --var "block_tag" \
            --latname "latitude" \
            --lonname "longitude" \
            --mintime "4d" \
            --min_overlap_prev 20 \

#            --in_list "hgt.19912020_blocktag_files.txt" \
#           --out_list "hgt.19912020_blockid_files.txt" \