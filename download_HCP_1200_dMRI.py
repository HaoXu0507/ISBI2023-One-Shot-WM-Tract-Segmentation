'''
This script downloads data from the Human Connectome Project - 1200 subjects release.
'''
# Import packages
import argparse
import os
import boto3
from boto3.s3.transfer import TransferConfig

# Make module executable
from tqdm import tqdm



def length_transform(length):
    if length > 1024 * 1024 * 1024:
        print("length:{:.2f}GB".format(length / 1024 / 1024 / 1024))
    elif length > 1024 * 1024:
        print("length:{:.2f}MB".format(length / 1024 / 1024))
    elif length > 1024:
        print("length:{:.2f}KB".format(length / 1024))
    else:
        print("length:{}B".format(length))


class download_process():
    def __init__(self, length):
        self.length = length
        self.bar = tqdm(range(int(length / 1024)))

    def precess(self, chunk):
        self.bar.update(int(chunk / 1024))

    def close(self):
        self.bar.close()


def download_prs(chunk):
    print(chunk)


if __name__ == '__main__':

    SERIES_MAP = {

        'bvals': 'T1w/Diffusion/bvals',
        'bvecs': 'T1w/Diffusion/bvecs',
        'data': 'T1w/Diffusion/data.nii.gz',
        'grad_dev': 'T1w/Diffusion/grad_dev.nii.gz',
        'nodif_brain_mask': 'T1w/Diffusion/nodif_brain_mask.nii.gz'

    }

    subjects = ['992774', '991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367', '959574',
                '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447', '910241', '907656',
                '904044', \
                '901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579', '887373',
                '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456', '859671', '857263',
                '856766', \
                '849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653', '814649',
                '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370', '771354', '770352',
                '765056', \
                '761957', '759869', '756055', '753251', '751348', '749361', '748662', '748258', '742549', '734045',
                '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133', '695768',
                '690152', \
                '687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754',
                '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671',
                '599469']

    # Init arparser
    parser = argparse.ArgumentParser(description=__doc__)

    # Required arguments
    parser.add_argument('--id', type=str)
    parser.add_argument('--key', type=str)
    parser.add_argument('-o', '--out_dir', required=True, type=str,
                        help='Path to local folder to download files to')

    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)

    s3_bucket_name = 'hcp-openaccess'
    s3_prefix = 'HCP_1200'

    aws_access_key_id = args.id

    aws_secret_access_key = args.key


    s3 = boto3.resource('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket('hcp-openaccess')
    # If output path doesn't exist, create it

    GB = 1024 ** 3
    config = TransferConfig(max_concurrency=500, multipart_threshold=int(0.01 * GB), multipart_chunksize=int(0.01 * GB))

    for subject in subjects:
        print("Start downloading :", subject)
        if not os.path.exists(out_dir + '/' + subject):
            print('Could not find %s, creating now...' % out_dir + '/' + subject)
            os.makedirs(out_dir + '/' + subject)
        for object in bucket.objects.filter(Prefix='HCP_1200/%s/' % subject):
            l = len('HCP_1200/%s/' % subject)
            if object.key[l:] in SERIES_MAP.values():
                print(object.key)
                length = bucket.Object(object.key).content_length
                length_transform(length)

                head, tail = os.path.split(object.key)
                if not os.path.exists(os.path.join(out_dir, subject, tail)):
                    if length < 1024 * 1024:
                        bucket.download_file(object.key, os.path.join(out_dir, subject, tail), Config=config)

                    else:
                        bar = download_process(length)
                        bucket.download_file(object.key, os.path.join(out_dir, subject, tail), Config=config,
                                             Callback=bar.precess)
                        bar.close()
