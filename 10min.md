# Intro

- Orger & Baier 2004: OMR to gratings only with chromatic contrast
- OMR: Fish swim with moving grating i.e. translational movement (when they can perceive it!)
- Changed red & green contrast in moving stimulus
- Explain on stimulus plot
- Fish stopped following grating when chromatic but not achromatic contrast was 0
- Can we see the same on the physiological level?
- Goal: Look for direction selective units in the optic tectum
- See if they respond to only chromatic contrasts.
- Method: Two photon microscopy & calcium imaging

# Workflow

## Recording

- Stimulated the fish with spherical gratings of different color contrasts (see stim matrix).
- Grating had different contrasts, rotation direction, pauses between movement, repeated the same stim 3 times.
- Recorded calcium activity in optic tectum via 2 photon microscopy
- Recorded behavior (eye movements) with camera

## Processing

- Python application for preprocessing caimg data: Suite2p
- Registration: Fisch moves, we have to align images
- Segmentation: Detect single cells to monitor their activity (ROIs)

## ROIs:

- Fluoresce due to a genetically induced protein
- Fluorescence increases with Calcium activity
- Fluorescence is excited by IR laser
- Raw data for each ROI: Some have peaks in activity, some do not respond

## ROI selection pipeline

- Filtered out 'responding' ROIs by autocorrelation across 3 stimulus repeats
- Show PDF of autocorrelation
- Filtered out direction selective ROIs by correlation with direction regressor
- Show clockwise regressor with ROI activity
- Result: Clockwise and counterclockwise selective cells

## Big plot

- Recorded from 3 zebrafish larvae
- Activity heatmap for ROIs over time, seperated by rotation direction
- Bottom: Non-responding, no pattern
- Above: Respondig ROIs, strong autocorrelation but not direction selective
- Above: Seperated clockwise and counterclockwise selective units
- Plottet above: Stimulus, different levels of green and red contrasts
- We see: Many cells clearly direction selective
- How do they respond to different color contrasts?

# Results

## Calcium imaging

- To see how the cells responded to chromatic contrasts we had to pool the data according to contrasts category
- If e.g. one stripe had 50% red intensity, the corresponding second stripe should have 50% green intensity, otherwise there is not only difference in the chromaticity but also intensity (i.e. one is simply brighter than the other).
- In other words: We expect to see lowest activity, when the intensity of the red and green stripe is the same.
- Tested 6 different lewels of red and green - 36 possible combinations (matrix)

### Line plot

- Calcium activity for every single green stripe against all red levels.
- We should see troughs where red and green are the same
- E.g. where red and green are 0
- This is also roughly what we see:
- For each level of green, the trough is approximately where red is the same level as green.
- (Where the vertical dashed line is)
- Pattern is conserved for both direction selective population

### Heatmap

- We can compute the mean activity for every possible combination and plot the in the same way as the stimulus
- Since the diagonal stimuli are only chromatic, no achromatic contrasts, we expect lower activity on the diagonal
- This is roughly what we see: Minima are on diagonal

## Behavior

- Recorded optokinetic respone: Eye movements and saccades in response to rotational movement
- Recorded eye positions & computed eye velocity in °/s
- Recorded in parrallel to calcium imaging, so same stimulus regime
- If we plot eye velocity in the same way, we would expect the same pattern
- This is what we see:

### Line plot

- Troughs at points where contrast of both stripes are the same

### Heatmap

- Diagonal line of lowest activity corresponding to the diagonal of chromatic contrast on the stimulus matrix

# Conclusion

- Direction selective neurons in the optic tectum show the lowest response to exclusively chromatic red-green contrasts and might therefore be color blind
- Behavioral readout results in the same pattern, strenghtening our results.
