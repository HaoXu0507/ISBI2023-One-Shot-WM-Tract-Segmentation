"""
Convert a trk streamline file to a binary map.

Arguments:
    trk_file_in
    mask_file_out
    reference_file (file with same affine as DWI file, e.g. brain mask)

Example:
    python trk_2_binary.py CST_right.trk CST_right.nii.gz nodif_brain_mask.nii.gz
"""
import argparse
import threading
import multiprocessing
import os, sys, inspect
from dipy.tracking import utils as utils_trk
# from nibabel import trackvis
import nibabel as nib
import numpy as np
from scipy import ndimage
import logging


def list_all_files(rootdir,exten):
    _files =[]
    file_add =[]
    list_file = os.listdir(rootdir)

    for i in range(len(list_file)):
        path = os.path.join(rootdir,list_file[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path,exten))
        if os.path.isfile(path):
             _files.append(path)
    print(_files)
    for file in _files:
        if file.endswith(exten):
            file_add.append(file)
    return file_add


def get_number_of_points(streamlines):
    count = 0
    for sl in streamlines:
        count += len(sl)
    return count


def remove_small_blobs(img, threshold=1):
    """
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.
    """
    # if using structure=np.ones((3, 3, 3): Also considers diagonal elements for determining if a element
    # belongs to a blob -> not good, because leaves hardly any small blobs we can remove
    mask, number_of_blobs = ndimage.label(img)
    logging.debug('Number of blobs before filtering: ' + str(number_of_blobs))
    counts = np.bincount(mask.flatten())  # number of pixels in each blob
    logging.debug(counts)

    remove = counts <= threshold
    remove_idx = np.nonzero(remove)[0]

    for idx in remove_idx:
        mask[mask == idx] = 0  # set blobs we remove to 0
    mask[mask > 0] = 1  # set everything else to 1

    mask_after, number_of_blobs_after = ndimage.label(mask)
    logging.debug('Number of blobs after filtering: ' + str(number_of_blobs_after))
    return mask

def task_trk2binary(files,out_files,ref_path):

    for i in range(len(files)):
        file_in = files[i]
        file_out = out_files[i]
        ref_img_path = ref_path
        """    
        args = sys.argv[1:]
        file_in = args[0]
        file_out = args[1]
        ref_img_path = args[2]
        """
        HOLE_CLOSING = 0

        # choose from "trk" or "trk_legacy"
        #  Use "trk_legacy" for zenodo dataset v1.1.0 and below
        #  Use "trk" for zenodo dataset v1.2.0
        tracking_format = "trk"

        ref_img = nib.load(ref_img_path)
        # ref_affine = ref_img.get_affine()
        # ref_shape = ref_img.get_data().shape
        ref_affine = ref_img.affine
        ref_shape = ref_img.get_fdata().shape

        # streams, hdr = trackvis.read(file_in)
        # streamlines = [s[0] for s in streams]  # list of 2d ndarrays
        #
        # if tracking_format == "trk_legacy":
        #     streams, hdr = trackvis.read(file_in)
        #     streamlines = [s[0] for s in streams]
        # else:
        sl_file = nib.streamlines.load(file_in)
        streamlines = sl_file.streamlines

        # Upsample Streamlines (very important, especially when using DensityMap Threshold. Without upsampling eroded results)
        max_seq_len = abs(ref_affine[0, 0] / 4)
        streamlines = list(utils_trk.subsegment(streamlines, max_seq_len))

        # Remember: Does not count if a fibers has no node inside of a voxel -> upsampling helps, but not perfect
        # Counts the number of unique streamlines that pass through each voxel -> oversampling does not distort result
        dm = utils_trk.density_map(streamlines, affine=ref_affine, vol_dims=ref_shape)

        # Create Binary Map
        dm_binary = dm > 0  # Using higher Threshold problematic, because tends to remove valid parts (sparse fibers)
        dm_binary_c = dm_binary

        # Filter Blobs (might remove valid parts) -> do not use
        # dm_binary_c = remove_small_blobs(dm_binary_c, threshold=10)

        # Closing of Holes (not ideal because tends to remove valid holes, e.g. in MCP) -> do not use
        # size = 1
        # dm_binary_c = ndimage.binary_closing(dm_binary_c, structure=np.ones((size, size, size))).astype(dm_binary.dtype)

        # Save Binary Mask
        dm_binary_img = nib.Nifti1Image(dm_binary_c.astype("uint8"), ref_affine)
        nib.save(dm_binary_img, file_out)
        print("Generate",file_out)
        
class multi_trk_2_bin(threading.Thread):
    def __init__(self,trk_files):
        super(multi_trk_2_bin, self).__init__()
        self.files = trk_files


    def run(self):
        task_trk2binary(self.files)


if __name__ == '__main__':

    parse=argparse.ArgumentParser()
    parse.add_argument("--tract_dir",type=str,default="/home/hao/PycharmProjects/pythonProject/TractSeg-master/data/HCP105_Zenodo_NewTrkFormat/")
    parse.add_argument("--ref_dir",type=str,default="/home/hao/HCP/")
    
    args = parse.parse_args()
    
    subjects = [
    '992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574',\
    '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241',\
    '907656', '904044', '901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', \
    '889579', '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', \
    '859671', '857263', '856766', '849971', '845458', '837964', '837560', '833249', '833148', '826454', \
    '826353', '816653', '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', \
    '779370', '771354', '770352', '765056', '761957', '759869', '756055', '753251', '751348', '749361', \
    '748662', '748258', '742549', '734045', '732243', '729557', '729254', '715647', '715041', '709551', \
    '705341', '704238', '702133', '695768', '690152', '687163', '685058', '683256', '680957', '679568', \
    '677968', '673455', '672756', '665254', '654754', '645551', '644044', '638049', '627549', '623844', \
    '622236', '620434', '613538', '601127', '599671', '599469']
      
    for subject in subjects:
        files = list_all_files(os.path.join(args.tract_dir,subject+"/tracts"), ".trk")
        out_files = []
        for file in files:
            out_name = file.replace(".trk", ".nii.gz")
            out_files.append(out_name)
        
        ref_path =os.path.join(args.ref_dir,subject,"nodif_brain_mask.nii.gz")
        # logging.basicConfig(format='%(levelname)s: %(message)s')  # set formatting of output
        # logging.getLogger().setLevel(logging.INFO)
    
        # print(len(files))
        task_trk2binary(files,out_files,ref_path)
        # print()