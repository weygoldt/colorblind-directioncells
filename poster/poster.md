poster is about Color-blindness of direction selective units in the zebrafish optic tectum 

# Intruduction 
Based on behaviour experiments conducted by orger and baier, that stated chromaticity, playes a role in the perception of motion 

Optomotor response, swimming in the dircetion of the movement of a stimulus 

and with a chromatic contrast, meaning with a contrast of equal intesity the fish didt show a optomotor response 
hinting that the fish is colorblind for this moving stimulus 

and we wanted to investigate how this information is processed in the hindbrain of the fish 
wit two photon microscopy and calcium imaging 

and for this we recorded from three fishes with a stimulus 
containing different contrast levels,
moving directions and between those moving directions we had pauses without any movement 

# Preprocessing
Registration and Segmentation 

because the recording was a littel bit scattered in time we had to first alignem them oacross time and them computed the changing flurecence in the picture, an that is what we see here. 

so we get our ROIs wich corresponds to cells with genetically encoded cacium indicators 
and thats whats shown here, some reacting some are not 

and we showed our stiumulus protocoll 3 times. 

Now we had figure out wich region of intresst is direction selective
with a autocorrelation between 3 repeats, so if the cell is reacting of all 3 repeats we get a good correlation and responding ROI 

then we correlated these responing ROIs with a direction regressor for example a clockwise regressor is 1 for only this stimulie and 0 for everything else and with these correlations we made an threshold and took with a higher correlation coefficiant higher than 0.3

# Results calcium imaging

we plottet here a heatmap for different ROIs across time and  sorted for the different colorgratings and direction 

from the bottom up we see non responding, resopding and clockwise and counterclockwise ROIs 

and we see at the chromatic contratst were the the both colors are equal lower activity of these cells 

to get a better picture of this effect we can look at these line plots 

red contrast on the bottom and only with one green contrast and we can see the throughs going through the different green contrast 

to summerize these results we ploted a heatmap with the all data pooled for one contrast and we can see a at the diagonal that there the chromatic respinse is the lowest 

and for the behavoiral we can see the same we conducted eye movements from the recorded zebrafish and there we cann see both the dips and in the heatmap the minma at the chromatic levels 

# conclusion 
wee cann see that the direction selective neurosn in the optiv tectum show lowest response to chromatic red an green contrast and might be therefore color blind 