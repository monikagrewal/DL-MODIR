# Deep Learning based Multi-Objective Deformable Image Registration
## Set up
You may need to run `pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117` to install torch in `galaxy01`


# Conclusions thus far
- voxelMorph is good, voxelMorph with UNet may give additional modelling capability
- 2 loss gives nice Pareto front
- Constraining in one objective by shifting the reference point works intuitively, but we should test more sophisticated approach
- To decide: constraint in NCCLoss or in additional guidance? median of 1.5 times the best rule?
- khead gives less diversity than deep ensemble, deep ensemble fits in one gpu for 10 solutions
- additional SegSimilarityLoss doesn't give advantage until Fixed_image_seg is not given as input

- khead for entire decoder doesn't loose diversity.
- PRNet training is slow compared to dual-stream. PRNet seems to find more spread out deformations i.e., to model bladder enlargening, it moves voxels in the entire bladder rather than just on the edge.
- small vs big PRNet doesn't seem any different.

# Suggestions
- Try on Lung CT dataset: does not have organ contours, so effect of additional guidance can not be visualized
- Try making the dataset simpler e.g., by cropping to bladder region
- Get Henrike's opinion on one dataset
- Visualize dataset in 3D Slicer

## To increase the effect of additional guidance
- remove background in loss
- combine rectum and sigmoid

# The Problem
It is difficult to assess the added benefit of SegSimilarity loss, because
    - existing segmentations are not perfect.
    - DIR task seems too difficult i.e., the images seem too complex or seem to have undergone complex deformations

# Something to debug/check
- warping segmentation with nearest neighbour causes one-sided partial derivatives in HV gradients calculation