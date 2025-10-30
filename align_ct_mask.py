"""
Orientation Code
Direction
Origin
Spacing
Affine Matrix
orthonormal direction vectors or not
"""
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
import logging
import numpy as np
# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_affine.log'),
        # logging.StreamHandler()  # This will still print to console
    ]
)
logger = logging.getLogger(__name__)


cases = subdirs('/mnt/d/projectsD/datasets/AbdomenAtlas', join=False, prefix='BDMAP')
# path = "/mnt/d/projectsD/datasets/AbdomenAtlas/BDMAP_00004991"
for case in cases:
    path = f"/mnt/d/projectsD/datasets/AbdomenAtlas/{case}"
    if not isfile(join(path, 'ct.nii.gz')) or not isfile(join(path, 'combined_labels.nii.gz')):
        logger.warning(f'Skipping case {case} due to missing image or label')
        continue
    logger.info(f"Processing case: {case}")

    ct  = nib.load(path + "/ct.nii.gz")
    mask = nib.load(path + "/combined_labels.nii.gz")

    # print("CT affine:",  ct.affine)
    # print("Mask affine:", mask.affine)
    if not np.allclose(ct.affine, mask.affine):
        logger.error(f"Nibabel Affine mismatch for case {case}")
        logger.error(f"CT affine: {ct.affine}")
        logger.error(f"Mask affine: {mask.affine}")

    # print("CT orientation codes:",
    #     nib.orientations.aff2axcodes(ct.affine))
    # print("Mask orientation codes:",
    #     nib.orientations.aff2axcodes(mask.affine))
    if not nib.orientations.aff2axcodes(ct.affine) == nib.orientations.aff2axcodes(mask.affine):
        logger.error(f"Nibabel orientation codes mismatch for case {case}")
        logger.error(f"CT orientation codes: {nib.orientations.aff2axcodes(ct.affine)}")
        logger.error(f"Mask orientation codes: {nib.orientations.aff2axcodes(mask.affine)}")

    try:
        ct_sitk  = sitk.ReadImage(path + "/ct.nii.gz")
        mask_sitk = sitk.ReadImage(path + "/combined_labels.nii.gz")
        # print(ct_sitk.GetOrigin(), ct_sitk.GetSpacing(), ct_sitk.GetDirection())
        # print(mask_sitk.GetOrigin(), mask_sitk.GetSpacing(), mask_sitk.GetDirection())

        if not np.allclose(ct_sitk.GetOrigin(), mask_sitk.GetOrigin()):
            logger.error(f"SimpleITK origin mismatch for case {case}")
            logger.error(f"CT origin: {ct_sitk.GetOrigin()}")
            logger.error(f"Mask origin: {mask_sitk.GetOrigin()}")
        if not np.allclose(ct_sitk.GetSpacing(), mask_sitk.GetSpacing()):
            logger.error(f"SimpleITK spacing mismatch for case {case}")
            logger.error(f"CT spacing: {ct_sitk.GetSpacing()}")
            logger.error(f"Mask spacing: {mask_sitk.GetSpacing()}")
        if not np.allclose(ct_sitk.GetDirection(), mask_sitk.GetDirection()):
            logger.error(f"SimpleITK direction mismatch for case {case}")
            logger.error(f"CT direction: {ct_sitk.GetDirection()}")
            logger.error(f"Mask direction: {mask_sitk.GetDirection()}")
    except Exception as e:
        logger.error(f"Error processing SimpleITK for case {case}: {e}")
        continue