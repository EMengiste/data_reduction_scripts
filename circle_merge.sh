#!/bin/bash
#

convert funda_region_clean.png -fill none -stroke red -strokewidth 20 -draw "circle 1050,1250 990,1250" pic.png   
convert pic.png -fill none -stroke red -strokewidth 20 -draw "circle 2100,890 2040,890" funda_region_annotated.png


convert +append funda_region_zoomed_grain_id_574.png funda_region_zoomed_grain_id_545.png grains_zoomed.png
convert grains_zoomed.png -crop 2700x1250+0+400 grains_zoomed.png
#convert -gravity center -append pic.png grains_zoomed.png combined.png
exit 0
#exit 0 
convert funda_region_zoomed_grain_id_574.png -pointsize 90 -font DejaVu-Serif  -fill black -stroke black -draw "text  780,200 'Grain 1' " grain_id_574.png
#
convert funda_region_zoomed_grain_id_545.png -pointsize 90 -font DejaVu-Serif  -fill black -stroke black -draw "text  780,200 'Grain 2' " grain_id_545.png
