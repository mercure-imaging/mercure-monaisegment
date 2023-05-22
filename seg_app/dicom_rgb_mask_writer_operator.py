# This code is adapted from the MONAI deploy dicom_seg_writer_operator.py code 
# distributed under the Apache 2.0 license as described below :

# Copyright 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The code has been modified to output an RGB DICOM file to visualize segmentations 
# rather than DICOM SEG format in the original operator.

import datetime
import logging
import os
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import numpy as np
from typeguard import typechecked

from monai.deploy.utils.importutil import optional_import
from monai.deploy.utils.version import get_sdk_semver
from monai.deploy.operators.dicom_utils import EquipmentInfo, ModelInfo, save_dcm_file, write_common_modules
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pydicom
from pydicom.uid import generate_uid
from PIL import Image as PIL_Image
from PIL import ImageOps

dcmread, _ = optional_import("pydicom", name="dcmread")
generate_uid, _ = optional_import("pydicom.uid", name="generate_uid")
ImplicitVRLittleEndian, _ = optional_import("pydicom.uid", name="ImplicitVRLittleEndian")
Dataset, _ = optional_import("pydicom.dataset", name="Dataset")
FileDataset, _ = optional_import("pydicom.dataset", name="FileDataset")
sitk, _ = optional_import("SimpleITK")
codes, _ = optional_import("pydicom.sr.codedict", name="codes")

if TYPE_CHECKING:
    import highdicom as hd
    from pydicom.sr.coding import Code
else:
    Code, _ = optional_import("pydicom.sr.coding", name="Code")
    hd, _ = optional_import("highdicom")

import monai.deploy.core as md
from monai.deploy.core import DataPath, ExecutionContext, Image, InputContext, IOType, Operator, OutputContext
from monai.deploy.core.domain.dicom_series import DICOMSeries
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries



@md.input("seg_image", Image, IOType.IN_MEMORY)
@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("dicom_seg_instance", DataPath, IOType.DISK)
@md.env(pip_packages=["pydicom >= 2.3.0", "highdicom >= 0.18.2"])
class DICOMRGBMaskWriterOperator(Operator):
    """
    This operator writes out a DICOM RGB image with segmentations overlaid
    """

    # Supported input image format, based on extension.
    SUPPORTED_EXTENSIONS = [".nii", ".nii.gz", ".mhd"]
    # DICOM instance file extension. Case insensitive in string comparison.
    DCM_EXTENSION = ".dcm"

    def __init__(
        self,
        custom_tags: Optional[Dict[str, str]] = None,
        omit_empty_frames: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        """Instantiates the DICOM RGB Mask Writer instance with optional list of segment label strings.

        Each unique, non-zero integer value in the segmentation image represents a segment. A colormap is
        created and each integer will be assigned a different color in the mask. 

        For example, in the CT Spleen Segmentation application, the whole image background has a value
        of 0, and the Spleen segment of value 1. This then only requires a single color from the colormap.

        Note: For a large number of labels the colormap settings may need to be modified.

        Args:
            custom_tags: Optional[Dict[str, str]], optional
                Dictionary for setting custom DICOM tags using Keywords and str values only
            omit_empty_frames: bool, optional
                Whether to omit frames that contain no segmented pixels from the output segmentation.
        """

        self._custom_tags = custom_tags
        self._omit_empty_frames = omit_empty_frames
        self.copy_tags = True
        self.model_info =  ModelInfo()
        self.equipment_info = EquipmentInfo()
        self.modality_type = "OT" #Other
        self.sop_class_uid = "1.2.840.10008.5.1.4.1.1.7" #secondary capture

    def compute(self, op_input: InputContext, op_output: OutputContext, context: ExecutionContext):
        """Performs computation for this operator and handles I/O.

        For now, only a single segmentation image object or file is supported and the selected DICOM
        series for inference is required, because the DICOM Seg IOD needs to refer to original instance.
        When there are multiple selected series in the input, the first series' containing study will
        be used for retrieving DICOM Study module attributes, e.g. StudyInstanceUID.

        Raises:
            FileNotFoundError: When image object not in the input, and segmentation image file not found either.
            ValueError: Neither image object nor image file's folder is in the input, or no selected series.
        """

        # Gets the input, prepares the output folder, and then delegates the processing.
        study_selected_series_list = op_input.get("study_selected_series_list")
        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing input, list of 'StudySelectedSeries'.")
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")

        seg_image = op_input.get("seg_image")
        # In case the Image object is not in the input, and input is the seg image file folder path.
        if not isinstance(seg_image, Image):
            if isinstance(seg_image, DataPath):
                seg_image, _ = self.select_input_file(seg_image.path)
            else:
                raise ValueError("Input 'seg_image' is not Image or DataPath.")

        output_dir = op_output.get().path
        output_dir.mkdir(parents=True, exist_ok=True)

        self.process_images(seg_image, study_selected_series_list, output_dir)

    def process_images(
        self, image: Union[Image, Path], study_selected_series_list: List[StudySelectedSeries], output_dir: Path
    ):
        """ """
        # Get the seg image in numpy, and if the image is passed in as object, need to fake a input path.
        seg_image_numpy = None
        input_path = "dicom_seg"

        if isinstance(image, Image):
            seg_image_numpy = image.asnumpy()
        elif isinstance(image, Path):
            input_path = str(image)  # It is expected that this is the image file path.
            seg_image_numpy = self._image_file_to_numpy(input_path)
        else:
            raise ValueError("'image' is not an Image object or a supported image file.")

        # Pick DICOM Series that was used as input for getting the seg image.
        # For now, first one in the list.
        for study_selected_series in study_selected_series_list:
            if not isinstance(study_selected_series, StudySelectedSeries):
                raise ValueError("Element in input is not expected type, 'StudySelectedSeries'.")
            selected_series = study_selected_series.selected_series[0]
            dicom_series = selected_series.series
            self.create_dicom_rgb(seg_image_numpy, dicom_series, output_dir)
            break

    def create_dicom_rgb(self, image: np.ndarray, dicom_series: DICOMSeries, output_dir: Path):
        
        #generate segmentation mask colormap
        new_prism= mpl.colormaps['prism']
        newcolors = new_prism(np.linspace(0, 1, 80)) # set to 80 colors
        black = np.array([0, 0, 0, 1])
        newcolors[0, :] = black
        newcmp = ListedColormap(newcolors)
        cm.register_cmap('newcmp',newcmp)
        cm.get_cmap('newcmp')
        
        
        
        if not output_dir.is_dir():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                raise ValueError("output_dir {output_dir} does not exist and failed to be created.") from None
        

        
        slices = dicom_series.get_sop_instances()
        
        ds = write_common_modules(
            dicom_series, self.copy_tags, self.modality_type, self.sop_class_uid , self.model_info, self.equipment_info
        )

        
        vol_data = np.stack([s.get_pixel_array() for s in slices], axis=0)
        vol_data = vol_data.astype(np.float32)
        
        # DICOM series setting
        series_UID= generate_uid()
        series_number=random_with_n_digits(4)
        series_desc =  "RGB_MASK(" + ds.SeriesDescription[:53] + ")"

        zdim = image.shape[0]
        for i in range(zdim):
            seg_sop_instance_uid = generate_uid()
            output_path = output_dir /  f"{seg_sop_instance_uid}{DICOMRGBMaskWriterOperator.DCM_EXTENSION}"

            raw_img = vol_data[i,:,:]
            seg_img = image[i,:,:]

            dcm = ds.copy()

            # Normalize the background (input) image
            background = 255 * ( 1.0 / raw_img.max() * (raw_img - raw_img.min()) )
            background = background.astype(np.ubyte)
            background_image = PIL_Image.fromarray(background).convert("RGB")

            seg_array = seg_img
            mask_image = PIL_Image.fromarray(np.uint8(newcmp(seg_array)*255)).convert("RGB")
        

            # Blend the two images
            final_image = PIL_Image.blend(mask_image, background_image, 0.75)
            final_array = np.array(final_image).astype(np.uint8) 
        
            
            #calc window settings for DICOM display
            window_min = np.amin(final_array)
            window_max =np.amax(final_array)
            window_middle = (window_max + window_min) / 2
            window_width = window_max - window_min
                
            # Write the final image back to a new DICOM (color) image 
            dcm.WindowCenter=f"{window_middle:.2f}" 
            dcm.WindowWidth=f"{window_width:.2f}"
            dcm.PixelSpacing= slices[i].get_native_sop_instance().PixelSpacing
            dcm.InstanceNumber = slices[i].get_native_sop_instance().InstanceNumber      
            dcm.SeriesInstanceUID = series_UID
            dcm.SeriesNumber = series_number
            dcm.SOPInstanceUID =seg_sop_instance_uid
            dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            dcm.Rows = final_image.height
            dcm.Columns = final_image.width
            dcm.PhotometricInterpretation = "RGB"
            dcm.SamplesPerPixel = 3
            dcm.BitsStored = 8
            dcm.BitsAllocated = 8
            dcm.HighBit = 7
            dcm.add_new(0x00280006, 'US', 0)
            dcm.is_little_endian = True
            dcm.fix_meta_info() 
            dcm.PixelData = final_array.tobytes()
            dcm.SeriesDescription = series_desc
            dcm.ImageType = "DERIVED\\SECONDARY"
            
            # Instance file name is the same as the new SOP instance UID with '_RGB' suffix
            save_dcm_file(dcm,output_path)
            try:
                # Test reading back
                _ = self._read_from_dcm(str(output_path))
            except Exception as ex:
                print("DICOM RBG mask creation failed. Error:\n{}".format(ex))
                raise

    def _read_from_dcm(self, file_path: str):
        """Read dcm file into pydicom Dataset

        Args:
            file_path (str): The path to dcm file
        """
        return dcmread(file_path)

    def select_input_file(self, input_folder, extensions=SUPPORTED_EXTENSIONS):
        """Select the input files based on supported extensions.

        Args:
            input_folder (string): the path of the folder containing the input file(s)
            extensions (array): the supported file formats identified by the extensions.

        Returns:
            file_path (string) : The path of the selected file
            ext (string): The extension of the selected file
        """

        def which_supported_ext(file_path, extensions):
            for ext in extensions:
                if file_path.casefold().endswith(ext.casefold()):
                    return ext
            return None

        if os.path.isdir(input_folder):
            for file_name in os.listdir(input_folder):
                file_path = os.path.join(input_folder, file_name)
                if os.path.isfile(file_path):
                    ext = which_supported_ext(file_path, extensions)
                    if ext:
                        return file_path, ext
            raise IOError("No supported input file found ({})".format(extensions))
        elif os.path.isfile(input_folder):
            ext = which_supported_ext(input_folder, extensions)
            if ext:
                return input_folder, ext
        else:
            raise FileNotFoundError("{} is not found.".format(input_folder))

    def _image_file_to_numpy(self, input_path: str):
        """Converts image file to numpy"""

        img = sitk.ReadImage(input_path)
        data_np = sitk.GetArrayFromImage(img)
        if data_np is None:
            raise RuntimeError("Failed to convert image file to numpy: {}".format(input_path))
        return data_np.astype(np.uint8)


def random_with_n_digits(n):
    assert isinstance(n, int), "Argument n must be a int."
    n = n if n >= 1 else 1
    range_start = 10 ** (n - 1)
    range_end = (10**n) - 1
    return randint(range_start, range_end)


def test():
    from monai.deploy.operators.dicom_data_loader_operator import DICOMDataLoaderOperator
    from monai.deploy.operators.dicom_series_selector_operator import DICOMSeriesSelectorOperator
    from monai.deploy.operators.dicom_series_to_volume_operator import DICOMSeriesToVolumeOperator

    current_file_dir = Path(__file__).parent.resolve()
    data_path = current_file_dir.joinpath("/input_data")
    out_dir = Path("/output_data").absolute()

    loader = DICOMDataLoaderOperator()
    series_selector = DICOMSeriesSelectorOperator()
    dcm_to_volume_op = DICOMSeriesToVolumeOperator()
    dicom_rgb_mask_writer = DICOMRGBMaskWriterOperator()

    # Testing with more granular functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    series = study_list[0].get_all_series()[0]

    dcm_to_volume_op.prepare_series(series)
    voxels = dcm_to_volume_op.generate_voxel_data(series)
    metadata = dcm_to_volume_op.create_metadata(series)
    image = dcm_to_volume_op.create_volumetric_image(voxels, metadata)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)

    dicom_rgb_mask_writer.create_dicom_rgb(image_numpy, series, out_dir)

    # Testing with the main entry functions
    study_list = loader.load_data_to_studies(data_path.absolute())
    study_selected_series_list = series_selector.filter(None, study_list)
    image = dcm_to_volume_op.convert_to_image(study_selected_series_list)
    # Very crude thresholding
    image_numpy = (image.asnumpy() > 400).astype(np.uint8)
    image = Image(image_numpy)
    dicom_rgb_mask_writer.process_images(image, study_selected_series_list, out_dir)


if __name__ == "__main__":
    test()