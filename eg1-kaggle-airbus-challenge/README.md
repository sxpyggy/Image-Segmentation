# [Airbus ship detection challenge](https://www.kaggle.com/c/airbus-ship-detection/overview/description)

Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. 

This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.

A lot of work has been done over the last 10 years to automatically extract objects from satellite images with significative results but no effective operational effects. Now Airbus is turning to Kagglers to increase the accuracy and speed of automatic ship detection.

## [Run-length encoding](https://www.kaggle.com/inversion/run-length-decoding-quick-start)

**Run-length encoding** is used to code the aligned bounding box segment around the ship.
The `train_ship_segmentations.csv` file provides **the ground truth** (true segment box) in run-length encoding format for the training images. 

## [Data visualization](https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization)

We are required to locate ships in images, and put **an aligned bounding box segment** around the ships we locate. Many images do not contain ships, and those that do may contain multiple ships. Ships within and across images may differ in size (sometimes significantly) and be located in open sea, at docks, marinas, etc.

Note that object segments **cannot overlap**. There were a small percentage of images in both the train and test set that had slight overlap of object segments when ships were directly next to each other. Any segments overlaps were removed by setting them to background (i.e., non-ship) encoding. Therefore, some images have a ground truth which may be an aligned bounding box with some pixels removed from an edge of the segment. These small adjustments will have a minimal impact on scoring, since the scoring evaluates over increasing overlap thresholds.

We draw a sample of images in Figure \@ref(fig:images) and the same sample images with the ship segments in Figure \@ref(fig:ground-truth). 

```{r images, fig.cap='Images',echo=F}
knitr::include_graphics('data/0.png')
```

```{r ground-truth, fig.cap='Images with ship segments',echo=F}
knitr::include_graphics('data/1.png')
```

