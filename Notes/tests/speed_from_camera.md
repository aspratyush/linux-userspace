Vehicle Speed Estimation
============================


## https://arxiv.org/pdf/1702.06441.pdf

pg2
* [16] used cross correlation to compute number of pixels which vehicles passed between consecutive frames
* [17] vehicles detected by background subtraction and tracked by normalized cross-correlation

pg3
* [12] Dubska : detect vanishing point 1 (VP1) from tracked features using __min eigenvalue detector & KLT tracker__.
* VP2 using stromg edges in horizontal direction
* VP1 and VP2 give camera calibration
* speed from measuring travel distance between the bounding boxes
