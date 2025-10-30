from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os
import random

if __name__ == '__main__':
    """
    Download the dataset from github:
    https://github.com/JunMa11/AbdomenCT-1K

    """
    base = '/mnt/d/projectsD/datasets/AbdomenCT-1K'

    target_dataset_id = 334
    target_dataset_name = f'Dataset{target_dataset_id:03.0f}_AbdomenCT-1K'

    cases = subfiles(join(base, 'images'), join=False, prefix='Case')

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

    for case in cases:
        image_old = join(base, 'images', case)
        label_old = join(base, 'labels', case[:-12] + '.nii.gz')
        print("check label: ",label_old)
        print("check image: ",image_old)
        if not isfile(label_old):
            print(f'Skipping case {case} due to missing labels')
            continue
        if case in train_cases:
            shutil.copy(image_old, join(imagesTr, case))
            shutil.copy(label_old, join(labelsTr, case[:-12] + '.nii.gz'))
        else:
            shutil.copy(image_old, join(imagesTs, case))
            shutil.copy(label_old, join(labelsTs, case[:-12] + '.nii.gz'))

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
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient',
        reference='https://huggingface.co/datasets/AbdomenAtlas/_AbdomenAtlas1.1Mini',
        license='Creative Commons Attribution Non Commercial Share Alike 4.0; see reference'
    )