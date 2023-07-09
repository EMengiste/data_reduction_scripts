#!/bin/bash
#

convert funda_region_grain_id_545_grain_id_592.png -fill none -stroke red -strokewidth 20 -draw "circle 850,1250 790,1250" pic.png   
convert pic.png -pointsize 80 -font DejaVu-Serif  -fill black -stroke black -draw "text 400,1250 'Grain 1' " pic.png
convert pic.png -fill none -stroke red -strokewidth 20 -draw "circle 1900,890 1840,890" pic.png  
convert pic.png -pointsize 80 -font DejaVu-Serif  -fill black -stroke black -draw "text  1990,850 'Grain 2' " pic.png

convert funda_region_zoomed_grain_id_574.png -pointsize 80 -font DejaVu-Serif  -fill black -stroke black -draw "text  420,100 'Grain 1' " grain_id_574.png
#
convert funda_region_zoomed_grain_id_545.png -pointsize 80 -font DejaVu-Serif  -fill black -stroke black -draw "text  420,100 'Grain 2' " grain_id_545.png

convert +append grain_id_574.png grain_id_545.png grains_zoomed.png
#convert -gravity center -append pic.png grains_zoomed.png combined.png
exit 0