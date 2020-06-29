# lane-detection
This software is my take on lane detection. 

Each frame is resized, yellow and white colors are extracted and transformed into grayscale. Then, we take the ROI and apply Canny.
After we have a Canny frame, HoughP is applied and the lines on the road are separated into left or right lines (which depends on the inclination 'm').
Depending on the type of frame, we can more than one left line (I call it outside line) which is the right line relative to the opposite driving lane.
In this case, those are separated and then we can simplify the lines and draw.

