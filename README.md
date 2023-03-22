# mercure-monaisegment
mercure module for rapid deployment of segmentation models hosted in the MONAI Model Zoo.

### Installation
1. Clone repo.
2. Build Docker container locally by running make (modify makefile with new docker tag as needed).
3. Test container :\
`docker run -it -v /input_data:/input -v /output_data:/output --env MERCURE_IN_DIR=/input  --env MERCURE_OUT_DIR=/output *docker-tag*`
### Output
Segmentations are written to specified output directory in the DICOM SEG format.
### Notes
- Default model 'spleen_ct_segmentation' bundle included in Dockerfile settings.
- To select a different model, set MONAI_BUNDLE_URL to MONAI Model Zoo bundle link address.
- Current requirements:
    * MONAI bundle must support torchscript i.e. include model.ts file
    * MONAI bundle must conform to spec. 
    * MONAI bundle must contain a segmentation model
    * May require further preprocessing or transform operators prior to inference to function correctly.

