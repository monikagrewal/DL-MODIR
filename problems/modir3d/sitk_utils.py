import os, glob
import numpy as np, cv2
import SimpleITK as sitk


def resample_voxel_spacing(image, output_spacing, output_size="image"):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputSpacing(output_spacing)

    if output_size=="image":
        output_spacing = np.array(output_spacing)
        org_size = np.array(image.GetSize())
        org_spacing = np.array(image.GetSpacing())
        output_size = ((org_size*org_spacing)/output_spacing).astype(np.int32).tolist()
    else:
        org_size = np.array(image.GetSize())
        org_spacing = np.array(image.GetSpacing())
        output_size_cal = ((org_size*org_spacing)/output_spacing).astype(np.int32).tolist() 
        output_size[2] = output_size_cal[2]
    resample.SetSize(output_size)

    image = resample.Execute(image)
    return image


def rescale_intensity(image, max_val=1.0, min_val=0.):
    rescaling = sitk.RescaleIntensityImageFilter()
    rescaling.SetOutputMaximum(max_val)
    rescaling.SetOutputMinimum(min_val)
    image = rescaling.Execute(image)
    return image


def read_image(filepath, output_spacing=(2.0, 2.0, 2.0), output_size="image", \
                                 windowing=True, rescaling=True, crop_depth=True, max_depth=200):
    """
    - reads a raw image given filepath
    - applies WW and WL and rescales intensities to range 0 to 1
    - resamples to given output spacing
    - only changes output depth based on output spacing
    """
    if isinstance(filepath, str):
        image = sitk.ReadImage(filepath, sitk.sitkFloat32)
    elif isinstance(filepath, list): #list of dicoms
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(filepath)
        reader.SetOutputPixelType(sitk.sitkFloat32)
        image = reader.Execute()
    else:
        raise ValueError("unknown data type for filepath. found {} expected one from: {}".format(type(filepath), [str, list]))

    # windowing
    if windowing:
        windowing = sitk.IntensityWindowingImageFilter()
        ww = 450
        wl = 40
        windowMaximum = wl + ww//2
        windowMinimum = wl - ww//2
        windowing.SetWindowMaximum(windowMaximum)
        windowing.SetWindowMinimum(windowMinimum)
        windowing.SetOutputMaximum(1.0)
        windowing.SetOutputMinimum(0.)
        image = windowing.Execute(image)

    # resampling
    if output_spacing is not None:
        image = resample_voxel_spacing(image, output_spacing)

    # rescaling
    if rescaling:
        image = rescale_intensity(image)

    if crop_depth:
        image = image[:, :, :max_depth]
    return image


def command_iteration(method) :
    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
                                   method.GetMetricValue(),
                                   method.GetOptimizerPosition()))


def command_multiresolution_iteration(method):
    print("\tStop Condition: {0}".format(method.GetOptimizerStopConditionDescription()))
    print("============= Resolution Change =============")


def resample(image, ref_im, Tx):
    """
    resample image according to given Tx
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_im)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(Tx)

    transformed_image = resampler.Execute(image)
    return transformed_image


def affine_registration(fixed_image, moving_image):
    R = sitk.ImageRegistrationMethod()
    R.SetShrinkFactorsPerLevel([8,4,2,1])
    R.SetSmoothingSigmasPerLevel([8,4,2,1])
    R.SetMetricAsJointHistogramMutualInformation()
    # R.SetMetricAsMattesMutualInformation()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2)
    R.MetricUseFixedImageGradientFilterOff()
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                              numberOfIterations=400,
                                              convergenceMinimumValue=1e-6,
                                              convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    initialTx = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(fixed_image.GetDimension()))
    R.SetInitialTransform(initialTx, inPlace=True)
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
    R.AddCommand( sitk.sitkMultiResolutionIterationEvent, lambda: command_multiresolution_iteration(R) )

    outTx = R.Execute(fixed_image, moving_image)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    affine_aligned_image = resample(moving_image, fixed_image, outTx)

    status = True
    if R.GetMetricValue() > -0.5:
        status = False

    return affine_aligned_image, status


def save_sitk_image(image, filepath):
    sitk.WriteImage(image, filepath)


def affine_parameter_map():
    p = sitk.ParameterMap()

    # Metric parameters
    p["Transform"] = ["AffineTransform"]
    p["Metric"] = ["AdvancedMattesMutualInformation"]
    p["NumberOfResolutions"] = ["4"]
    p["Registration"] = ["MultiResolutionRegistration"]
    
    p["AutomaticTransformInitialization"] =  ["true"]
    p["AutomaticTransformInitializationMethod"] = ["Origins"]
    p["AutomaticParameterEstimation"] = ["true"]
    p["CheckNumberOfSamples"] = ["true"]

    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]

    p["DefaultPixelValue"] = ["0"]
    p["Interpolator"] = ["LinearInterpolator"]
    p["FinalBSplineInterpolationOrder"] = ["1"]
    p["Resampler"] = ["DefaultResampler"]
    p["ResampleInterpolator"] = ["FinalBSplineInterpolator"]

    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["MaximumNumberOfIterations"] = ["1024"]

    p["ImageSampler"] = ["RandomCoordinate"]
    p["NumberOfSamplesForExactGradient"] = ["4096"]
    p["NumberOfSpatialSamples"] = ["4096"]
    p["MaximumNumberOfSamplingAttempts"] = ["8"]
    p["NewSamplesEveryIteration"] = ["true"]

    p["WriteIterationInfo"] = ["false"]
    # sitk.PrintParameterMap(p)
    return p

def elastix_affine_registration(fixed_image, moving_image):
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    
    rigid_pm = affine_parameter_map()
    elastixImageFilter.SetParameterMap(rigid_pm)
    elastixImageFilter.SetLogToFile(True)
    elastixImageFilter.SetOutputDirectory('./')

    elastixImageFilter.Execute()
    transformed_image = elastixImageFilter.GetResultImage()
    # Tx = elastixImageFilter.GetTransformParameterMap()

    # get final metric value
    f = open("elastix.log")
    out = f.readlines()[-20:]
    f.close()
    metric_line = [item for item in out if "Final metric" in item][0]
    metric = float(metric_line.split(" ")[-1].replace("\n", ""))
    print("Extracted final metric value: ", metric)
    status = True
    if metric > -0.5:
        print("final metric less than 0.5")
        status = False

    return transformed_image, status