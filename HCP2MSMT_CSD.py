import os
if __name__ == '__main__':

     parse=argparse.ArgumentParser()
     parse.add_argument("--hcp_dir",type=str)
     parse.add_argument("--out_dir",type=str)
    
     args = parse.parse_args()
     subjects = [
          '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574',\
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
     
     hcp_dir = args.hcp_dir
     output_root = args.out_dir
     for subject in subjects:

         input_file = os.path.join(hcp_dir, subject)
         output_dir = os.path.join(output_root, subject)
         if not os.path.exists(output_dir):
             os.mkdir(output_dir)
             print("Create dir:", output_dir)
         t1_file = os.path.join(input_file, "data.nii.gz")
         bvecs = os.path.join(input_file, "bvecs")
         bvals = os.path.join(input_file, "bvals")
         brain_mask = os.path.join(input_file, "nodif_brain_mask.nii.gz")

         # MSMT DHollander    (only works with msmt_csd, not with csd)
         # (Dhollander does not need a T1 image to estimate the response function)
         print("Creating peaks (1 of 3)...")
         os.system("dwi2response dhollander -mask " + brain_mask + " " + t1_file + " " + output_dir + "/RF_WM.txt " +
                   output_dir + "/RF_GM.txt " + output_dir + "/RF_CSF.txt -fslgrad " + bvecs + " " + bvals +
                   " -mask " + brain_mask )
         print("Creating peaks (2 of 3)...")
         os.system("dwi2fod msmt_csd " + t1_file + " " +
                   output_dir + "/RF_WM.txt " + output_dir + "/WM_FODs.nii.gz " +
                   output_dir + "/RF_GM.txt " + output_dir + "/GM_FODs.nii.gz " +
                   output_dir + "/RF_CSF.txt " + output_dir + "/CSF_FODs.nii.gz " +
                   "-fslgrad " + bvecs + " " + bvals + " -mask " + brain_mask )
         print("Creating peaks (3 of 3)...")
         os.system("sh2peaks " + output_dir + "/WM_FODs.nii.gz " + output_dir + "/peaks.nii.gz -quiet" )
