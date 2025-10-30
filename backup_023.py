from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import random
import os
import logging
import numpy as np
import nibabel as nib
import SimpleITK as sitk



if __name__ == '__main__':
    # Set up logger
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('DatasetConversion_023.log'),
        # logging.StreamHandler()  # This will still print to console
    ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Dataset Conversion for AbdomenAtlas1.1Mini")
    """
    Download the dataset from huggingface:
    https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini#3--download-the-dataset
    
    IMPORTANT
    cases 5196-9262 currently do not have images, just the segmentation. This seems to be a mistake 
    """
    base = '/mnt/d/projectsD/datasets/AbdomenAtlas'
    target_dataset_id = 23
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_AbdomenAtlas1.1Mini'

    cases = subdirs(base, join=False, prefix='BDMAP')

    random.seed(42)  
    random.shuffle(cases)

    train_ratio = 0.8
    split_index = int(len(cases) * train_ratio)
    train_cases = cases[:split_index]


    maybe_mkdir_p(join(nnUNet_raw, target_dataset_name))
    imagesTr = join(nnUNet_raw, target_dataset_name, 'imagesTr')
    labelsTr = join(nnUNet_raw, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)
    imagesTs = join(nnUNet_raw, target_dataset_name, 'imagesTs')
    labelsTs = join(nnUNet_raw, target_dataset_name, 'labelsTs')
    maybe_mkdir_p(imagesTs)
    maybe_mkdir_p(labelsTs)
    missing_cases = 0
    for case in cases:
        if not isfile(join(base, case, 'ct.nii.gz')) or not isfile(join(base, case, 'combined_labels.nii.gz')):
            print(f'Skipping case {case} due to missing image')
            missing_cases += 1
            continue
        # Check Alignment of CT and Mask
        try:
            ct_sitk  = sitk.ReadImage(join(base, case, 'ct.nii.gz'))
            mask_sitk = sitk.ReadImage(join(base, case, 'combined_labels.nii.gz'))
            ct  = nib.load(join(base, case, 'ct.nii.gz'))
            mask = nib.load(join(base, case, 'combined_labels.nii.gz'))
        except Exception as e:
            logger.error(f"Error loading images for case {case}: {e}")
            continue
        
        # Check for any misalignments
        misaligned = False
        if not np.allclose(ct.affine, mask.affine):
            logger.warning(f"Affine mismatch for case {case}")
            misaligned = True
        
        if not nib.orientations.aff2axcodes(ct.affine) == nib.orientations.aff2axcodes(mask.affine):
            logger.warning(f"Orientation codes mismatch for case {case}")
            misaligned = True
            
        if not np.allclose(ct_sitk.GetOrigin(), mask_sitk.GetOrigin()):
            logger.warning(f"Origin mismatch for case {case}")
            misaligned = True
            
        if not np.allclose(ct_sitk.GetSpacing(), mask_sitk.GetSpacing()):
            logger.warning(f"Spacing mismatch for case {case}")
            diff_spacing = True
            continue
            
        if not np.allclose(ct_sitk.GetDirection(), mask_sitk.GetDirection()):
            logger.warning(f"Direction mismatch for case {case}")
            misaligned = True

        # If misaligned, resample mask to match CT
        if misaligned:
            logger.info(f"Resampling mask to align with CT for case {case}")
            mask_to_copy = nib.Nifti1Image(mask.get_fdata().astype(np.int64), ct.affine, mask.header)

        # Copy CT and Mask to NNunet Raw Directory
        if case in train_cases:
            shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
            if misaligned:
                nib.save(mask_to_copy, join(labelsTr, case + '.nii.gz'))
                ct_realigned  = nib.load(join(imagesTr, case + '_0000.nii.gz'))
                mask_realigned = nib.load(join(labelsTr, case + '.nii.gz'))
                logger.info(f"Alignment verification for case {case}:{np.allclose(ct_realigned.affine, mask_realigned.affine)}")
            else:
                shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))
        else:
            shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTs, case + '_0000.nii.gz'))
            if misaligned:
                nib.save(mask_to_copy, join(labelsTs, case + '.nii.gz'))
                ct_realigned  = nib.load(join(imagesTs, case + '_0000.nii.gz'))
                mask_realigned = nib.load(join(labelsTs, case + '.nii.gz'))
                logger.info(f"Alignment verification for case {case}:{np.allclose(ct_realigned.affine, mask_realigned.affine)}")
            else:
                shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTs, case + '.nii.gz'))

    class_map = {1: 'aorta', 2: 'gall_bladder', 3: 'kidney_left', 4: 'kidney_right', 5: 'liver',
                 6: 'pancreas', 7: 'postcava', 8: 'spleen', 9: 'stomach', 10: 'adrenal_gland_left',
                 11: 'adrenal_gland_right', 12: 'bladder', 13: 'celiac_trunk', 14: 'colon', 15: 'duodenum',
                 16: 'esophagus', 17: 'femur_left', 18: 'femur_right', 19: 'hepatic_vessel', 20: 'intestine',
                 21: 'lung_left', 22: 'lung_right', 23: 'portal_vein_and_splenic_vein',
                 24: 'prostate', 25: 'rectum'}
    labels = {
        j: i for i, j in class_map.items()
    }
    labels['background'] = 0

    generate_dataset_json(
        join(nnUNet_raw, target_dataset_name),
        {0: 'CT'},
        labels,
        len(train_cases) - missing_cases,
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        reference='https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini',
        license='Creative Commons Attribution Non Commercial Share Alike 4.0; see reference'
    )