# Issues to resolve
- images may have in-plane sizes not compatible with neural network
- images have uneven depth e.g., 55 in image 1, 60 in image 2

# Prerequisites
- implement deep learning based affine (or rigid) registration
- decide point of forking individual solutions

# More TODO
- use segmentation
- implement multi-resolution
- find more training data
- increase number of points on the Pareto front
- DVF visualization
- Pareto front visualization in the end


# Deatiled TODO
- SimpleITK registration
- input in-plane size suitable with deep learning by padding or cropping
- implement three types of net ensembles:
----- a set of neural networks (MO)
----- multi-headed neural network (MO)
----- single neural network (SO)
- implement Shape3D Dataset
- Train UNet for MRI images

- implement DeepDIRNet
- implement MRDeepDIRNet-A (multi resolution version), spatial tranformer at each level in upsampling path
- implement MRDeepDIRNet-B (multi resolution version), final dvf is some of dvf from different levels

- Design and set up comparison experiments with Elastix, DIRNet, VoxelMorph
- implement MO-MRDeepDIRNet

- check voxelmorph NCC and gradient loss implementations: DONE and USED

- Find out if providing segmentation mask as input had additional benefit on the registration performance: YES

- weight averaging in mo optimizer after a warmup period: LEAVE IT for the time being

# TO FOCUS NOW - 03 August, 2023
- commit everything in this branch, make a new cleaner branch

- Write KheadVoxelMorph

- Compare no seg input vs seg input in LS, Compare LS with MO on Validation data

# TODO: 3 August, 2023
- check `mo_voxelmorph` if it works with K=1 also.
- implement both `ConvBlock` capability: custom and original. DONE
- check `mo_voxelmorph` with `K=1`, and original `ConvBlock` should be original `VoxelMorph`.
- remove remaining `style-transfer` related customizations.
- test run
- In data: check split, remove scans with missing labels DONE

- test data LUMC has landmarks and contours in one RTSTRUCT
- test data AMC has only landmarks in the given RTSTRUCT
- Prepare test data (annotated) by affine registration and applying it on the contours and landmarks
- Save deformation to sitk image, calculate det of jacobian
- make comparison Pareto front visualization 
        -- (TRE, percent folding, 1 - dice)
        -- same color for all predictions from one approach
        -- two projections
- To compare:
        -- HV deep vs HV k-decoders
        -- LS vs HV

# TODO - 22 Feb, 2024
- no guidance vs guidance plot with varying colors for each run
- table with max dice and folding
- write experiment
