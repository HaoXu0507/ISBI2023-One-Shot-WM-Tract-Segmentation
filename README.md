# A REGISTRATION- AND UNCERTAINTY-BASED FRAMEWORK FOR WHITE MATTER TRACT SEGMENTATION WITH ONLY ONE ANNOTATED SUBJECT
>Hao Xu, Tengfei Xue, [Dongnan Liu](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dongnan-liu.html), [Fan Zhang](https://scholar.harvard.edu/fanzhang), [Carl-Fredrik Westin](https://lmi.med.harvard.edu/people/carl-fredrik-westin), [Ron Kikinis](https://lmi.med.harvard.edu/people/ron-kikinis-md), [Lauren J. O’Donnell](https://scholar.harvard.edu/laurenjodonnell/biocv), and [Weidong Cai](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/tom-cai.html) 
>
>*The IEEE International Symposium on Biomedical Imaging (ISBI) 2023 ([arxiv](https://arxiv.org))*

![framework](/framework.png)

## Abstract
White matter (WM) tract segmentation based on diffusion magnetic resonance imaging (dMRI) plays an important role in the analysis of human health and brain diseases. However, the annotation of WM tracts is time-consuming and needs experienced neuroanatomists. In this study, to explore tract segmentation in the challenging setting of minimal annotations, we propose a novel framework utilizing only one annotated subject (subject-level one-shot) for tract segmentation. Our method is constructed by proposed registration-based peak augmentation (RPA) and uncertainty-based refining (URe) modules. Registration-based data augmentation is employed for synthesizing pseudo subjects and their corresponding labels to improve the tract segmentation performance. The proposed uncertainty-based refining module alleviates the negative influence of the low-confidence predictions on pseudo labels. Comparison results indicate the effectiveness of our proposed modules, by achieving accurate whole-brain tract segmentation with only one annotated subject.

## Get Started
Our code is based on [TractSeg](https://github.com/MIC-DKFZ/TractSeg).

## Install
* PyTorch >= 3.6
* [Mrtrix 3](https://mrtrix.readthedocs.io/en/latest/installation/build_from_source.html) >= 3.0
``conda install -c mrtrix3 mrtrix3 python=3.6``

* boto3
``pip install boto3``

* tractseg
``pip install tractseg``

* nibabel
``pip install nibabel``
## Datasets Prepare
You can prepare datasets by yourself or follow the following steps.
* Download Human Connectome Project (HCP) datasets.
1. Register a HCP account: [HCP](https://db.humanconnectome.org/)
2. Enable Amazon S3 Access: [AWS](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)
3. Download HCP datasets by running [download_HCP_1200_diffusion_mri.py](/download_HCP_1200_diffusion_mri.py):

``python /download_HCP_1200_diffusion_mri.py --id your_aws_id --key your_aws_key --out_dit your_hcp_dir``
* Download Corresponding WM tract labels from [Zenodo](https://zenodo.org/record/1477956#.ZBQ5wHZByNc).
## Data Pre-Processing
* Transform a trk streamline file to a binary map.
Transform a trk streamline file to a binary map by running [trk2bin.py](/trk2bin.p):

``python /trk2bin.py --tract_dir your_tract_dir --ref_dir your_hcp_dir``

and finally, the tract dataset directory should look like:

    $your_tract_dir
    ├─992774
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
    ├─991267
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
    .
    .
    .
    ├─599469
    │   ├─tracts
    │   │   ├─AF_left.nii.gz
    │   │   ├─AF_rgiht.nii.gz
    |   |   .
    |   |   .
    |   |   .
    │   │   ├─UF_rgiht.nii.gz
  

* Transform dMRI datasets to peak data.
Transform dMRI datasets to peak data using multi-shell multi-tissue constrained spherical deconvolution (MSMT-CSD) by running [HCP2MSMT_CSD.py](/HCP2MSMT_CSD.py):

``python /HCP2MSMT_CSD.py --hcp_dir your_hcp_dir --out_dir your_msmt_csd_dir``

and finally, the peak data directory should look like:

    $your_msmt_csd_dir
    ├─992774
    │   ├─peaks.nii.gz
    ├─991267
    │   ├─peaks.nii.gz
    .
    .
    .
    ├─599469
    │   ├─peaks.nii.gz
## Training
### Step-1. Synthesize pseudo subjects

### Step-2. Train on the only labeled subject

### Step-3. Train on pseudo subjects

## Testing

## Performance
### HCP Test Set
<table>
    <tr>
        <td></td>
        <td>Methods</td>
        <td>Dice score</td>
    </tr>
    <tr>
        <td rowspan="2">Comparison</td>
        <td>U-Net</td>
        <td>43.19±15.20%</td>
    </tr>
    <tr>
        <td>TractSeg</td>
        <td>48.85±17.58%</td>
    </tr>
    <tr>
        <td>Ablation study</td>
        <td>Ours (RPA)</td>
        <td>69.45±9.53%</td>
    </tr>
    <tr>
        <td colspan="2">Ours (RPA + UDA)</td>
        <td><b>73.01±8.14%</b></td>
    </tr>
</table>

### Visualization
![Visualization](/visualization.png)