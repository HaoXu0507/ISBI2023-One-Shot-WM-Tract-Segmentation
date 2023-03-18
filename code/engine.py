import time
from peak_tract_dataloader import DataLoaderTraining
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import argparse
from tractseg.models.unet_pytorch_deepsup import UNet_Pytorch_DeepSup
from torch.optim import Adamax
import torch.nn as nn
import os

#dice score
def val_f1_score_macro(y_true, y_pred):
    """
    Macro f1. Same results as sklearn f1 macro.

    Args:
        y_true: (n_samples, n_classes)
        y_pred: (n_samples, n_classes)

    Returns:

    """

    f1s = []
    if len(y_true.shape)==4:
      for i in range(y_true.shape[-1]):
          intersect = np.sum(y_true[:,:,:, i] * y_pred[:,:,:, i])  # works because all multiplied by 0 gets 0
          denominator = np.sum(y_true[:,:,:, i]) + np.sum(y_pred[:,:,:, i])  # works because all multiplied by 0 gets 0
          f1 = (2 * intersect) / (denominator + 1e-6)
          f1s.append(f1)
    elif len(y_true.shape)==3:
      for i in range(y_true.shape[-1]):
          intersect = np.sum(y_true[:,:, i] * y_pred[:,:, i])  # works because all multiplied by 0 gets 0
          denominator = np.sum(y_true[:,:, i]) + np.sum(y_pred[:,:, i])  # works because all multiplied by 0 gets 0
          f1 = (2 * intersect) / (denominator + 1e-6)
          f1s.append(f1)
    return np.array(f1s)
# Step_1 train
def model_registration_train(args, UNet,STN, data_loader,criterion, optimizer, moving_subjects, fixed_subjects,start_epoch=0):
    UNet.train()
    STN.train()
    epoch = args.epoch
    batch_gen_train = data_loader.get_batch_generator(moving_subjects, fixed_subjects, type="train")# 默认batch=1
    sim_loss_fn = criterion["sim_loss_fn"]
    grad_loss_fn = criterion["grad_loss_fn"]

    print("Open log file:", os.path.join(args.ckpt_path, "training_log" + ".txt"))
    for i in range(start_epoch+1,epoch+1):
        start_time = time.time()
        local_time = time.asctime(time.localtime(start_time))
        print(local_time)
        for j in range(1,len(moving_subjects)+1):
            print("-" * 30)

            batch = next(batch_gen_train)
            moving_img = batch["moving_img"]
            atlas_img = batch["atlas_img"]
            subject_index = batch["subject_index"]
            moving_img = moving_img.cuda()
            atlas_img = atlas_img.cuda()

            warp_field = UNet(atlas_img,moving_img) # follow the paper where 1st para is atlas_img.
            atlas2moving = STN(atlas_img,warp_field)

            if j ==1:
                full_img_name = args.ckpt_path + "/" + "epoch_{}.png".format(i)
                save_image(moving_img, atlas_img, atlas2moving, full_img_name)

            sim_loss = sim_loss_fn(atlas2moving, moving_img)
            grad_loss = grad_loss_fn(warp_field)
            loss = sim_loss + args.alpha * grad_loss
            print("epoch %d | iteration %d | subjcet %s | loss %f | sim %f | grad %f" % (i,j, moving_subjects[subject_index],loss.item(), sim_loss.item(), grad_loss.item()), flush=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(i,hour,minute))

        if i % 10 == 0:
            torch.save(UNet, args.step_1_ckpt_dir + "/" + "epoch_{}.pth".format(i))
            print("save model " + "epoch_{}.pth".format(i))
            os.system("nvidia-smi")
# Step_1 generation
def model_pseudo_label_genenration(args,model_shape,model_shape_STN, data_loader, moving_subjects, fixed_subjects,start_epoch=0):

    batch_gen_train = data_loader.get_batch_generator(moving_subjects, fixed_subjects, type="train")  # 默认batch=1

    for j in range(1, len(moving_subjects) + 1):
        print("-" * 30)
        batch = next(batch_gen_train)
        moving_img = batch["moving_img"]
        atlas_img = batch["atlas_img"]
        atlas_label = batch["atlas_label"]
        subject_index = batch["subject_index"]
        moving_img = moving_img.cuda()
        atlas_img = atlas_img.cuda()
        atlas_label = atlas_label.cuda()

        with torch.no_grad():
            warp_field = model_shape(atlas_img,
                                     moving_img)  # approximate inverse deformation of y(i) to x as φ^-1(i) = g_θs (y(i) , x).
            atlas2moving = model_shape_STN(atlas_img, warp_field, mode="nearest")
            atlas2moving_label = model_shape_STN(atlas_label, warp_field, mode="nearest")

            save_dir_img = args.step_1_pseudo_data_dir +"/" + moving_subjects[subject_index] + ".nii.gz"
            save_dir_label = args.step_1_pseudo_data_dir + "/" + moving_subjects[subject_index] + "_label.nii.gz"
            atlas2moving = atlas2moving.detach().cpu().numpy().squeeze(0).transpose(1, 2, 3, 0)
            atlas2moving_label = atlas2moving_label.detach().cpu().numpy().squeeze(0).transpose(1, 2, 3, 0)
            affine = np.array([[-1.25, 0, 0, 0.0], [0, 1.25, 0, -126.0], [0, 0, 1.25, -72.0], [0, 0, 0, 1.0]],
                              dtype=np.float64)
            nib.Nifti1Image(atlas2moving, affine).to_filename(save_dir_img)
            nib.Nifti1Image(atlas2moving_label, affine).to_filename(save_dir_label)
        print(save_dir_img)
        print(save_dir_label)
# Step_2 train
def model_train(args, data_loader,  optimizer, train_subjects, start_epoch=0):
    model_type = ["train", "val"]

    batch_size = args.batch_size
    epoch = args.epoch
    batch_gen_train = data_loader.get_batch_generator(train_subjects, type="train")
    best_dice = 0
    best_dice_epoch = 0
    iteration = 0
    nr_batches = int(144 / batch_size)
    for epoch_nr in range(start_epoch+1, epoch + 1):
        weight_factor = float(args.LOSS_WEIGHT)-(args.LOSS_WEIGHT-1)*(epoch_nr/float(args.LOSS_WEIGHT_LEN))
        print("weight_factor =",weight_factor)
        print("best dice : {:.4f} in epoch {}".format(best_dice,best_dice_epoch))
        # train
        start_time = time.time()
        local_time=time.asctime(time.localtime(start_time))
        print(local_time)
        for batch_index in range(len(train_subjects)):
            print("-" * 20)
            for i in range(nr_batches):
                    batch = next(batch_gen_train)
                    x = batch["data"]  # (bs, nr_of_channels, x, y)
                    y = batch["seg"]  # (bs, nr_of_classes, x, y)
                    direction= batch["slice_dir"]
                    subject_index = batch["subject_index"]
                    x = x.to(device)
                    y = y.to(device)


                    outputs = model(x)

                    bce_weight = torch.ones((y.shape[0],y.shape[1],y.shape[2],y.shape[3],)).cuda()
                    bundle_mask = y>0
                    bce_weight[bundle_mask.data] *=weight_factor
                    bce_loss = nn.BCEWithLogitsLoss(weight=bce_weight)(outputs, y)

                    loss = bce_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    outputs_sigmoid = torch.sigmoid(outputs.detach())
                    predict = torch.where(outputs_sigmoid.detach() > 0.5, torch.ones_like(outputs_sigmoid), torch.zeros_like(outputs_sigmoid))
                    iteration += 1

                    print("Train: epoch {} batch {} tract_name {} iteration {} direction {} loss = {:.4f} ".format(epoch_nr,batch_index+1, train_subjects[subject_index], iteration, direction,float(loss)))

                    f1_macro=val_f1_score_macro(predict.cpu().numpy().transpose(0,2,3,1), y.detach().cpu().numpy().transpose(0,2,3,1))

                    print("f1_macro = {:.4f}".format(np.nanmean(f1_macro)))
                    print("f1_per_class =")

                    for f1_index in range(len(f1_macro)):
                      if (f1_index+1)%12==0 :
                        print("{:.4f}".format(f1_macro[f1_index]),end="\n")
                      else:
                        print("{:.4f}".format(f1_macro[f1_index]),end=" ")

    # save model
        if epoch_nr % 1 == 0:
          torch.save(model, args.ckpt_path + "/" + "epoch_{}.pth".format(epoch_nr))
          print("save model " + "epoch_{}.pth".format(epoch_nr))
          os.system("nvidia-smi")
        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(epoch_nr,hour,minute))
# Step_3 train

def model_pseudo_label_train_weight(args, data_loader,  optimizer, train_subjects, start_epoch=0):

    batch_size = args.batch_size
    epoch = args.epoch

    batch_gen_train = data_loader.get_batch_generator(train_subjects, type="train")
    # best_dice=0.7703889651651645
    # best_dice_epoch=85
    best_dice = 0
    best_dice_epoch = 0

    iteration = 0
    nr_batches = int(144 / batch_size)

    print("Open log file:", os.path.join(args.ckpt_path, "training_log" + ".txt"))

    for epoch_nr in range(start_epoch+1, epoch + 1):
        weight_factor = float(args.LOSS_WEIGHT)-(args.LOSS_WEIGHT-1)*(epoch_nr/float(args.LOSS_WEIGHT_LEN))
        print("weight_factor =",weight_factor)
        print("best dice : {:.4f} in epoch {}".format(best_dice,best_dice_epoch))
        f.write("weight_factor = {}".format(weight_factor) )
        f.write("best dice : {:.4f} in epoch {}".format(best_dice,best_dice_epoch))
        # train
        start_time = time.time()
        local_time=time.asctime(time.localtime(start_time))
        print(local_time)
        for batch_index in range(len(train_subjects)):
            print("-" * 20)
            # for type in model_type:

            for i in range(nr_batches):
                batch = next(batch_gen_train)
                x = batch["data"]  # (bs, nr_of_channels, x, y)
                y = batch["seg"]  # (bs, nr_of_classes, x, y)

                direction= batch["slice_dir"]
                subject_index = batch["subject_index"]
                x = x.to(device)
                y = y.to(device)


                outputs = model(x)


                with torch.no_grad():
                    one_shot_weight = torch.sigmoid(one_shot_model(x))

                    one_shot_weight = torch.where(one_shot_weight>0.5, (one_shot_weight-0.5)*2, (1.0-one_shot_weight-0.5)*2)

                    one_shot_weight =one_shot_weight*weight_factor
                    bce_weight = torch.ones((y.shape[0], y.shape[1], y.shape[2], y.shape[3],)).cuda()
                    bundle_mask = y > 0
                    bce_weight[bundle_mask.data] += one_shot_weight[bundle_mask.data]


                loss =nn.BCELoss(weight=bce_weight)(torch.sigmoid(outputs), y)
 

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                outputs_sigmoid = torch.sigmoid(outputs.detach())
                predict = torch.where(outputs_sigmoid.detach() > 0.5, torch.ones_like(outputs_sigmoid), torch.zeros_like(outputs_sigmoid))
                
                iteration += 1

                print("Train: epoch {} batch {} tract_name {} iteration {} direction {} loss = {:.4f} ".format(epoch_nr,batch_index+1, train_subjects[subject_index], iteration, direction,float(loss)))
                
                f1_macro=val_f1_score_macro(predict.cpu().numpy().transpose(0,2,3,1), y.detach().cpu().numpy().transpose(0,2,3,1))

              
                print("f1_macro = {:.4f}".format(np.nanmean(f1_macro)))
                print("f1_per_class =")

                for f1_index in range(len(f1_macro)):
                  if (f1_index+1)%12==0 :
                    print("{:.4f}".format(f1_macro[f1_index]),end="\n")
                  else:
                    print("{:.4f}".format(f1_macro[f1_index]),end=" ")


        #save model
        if epoch_nr % 1 == 0:
              torch.save(model, args.ckpt_path + "/" + "epoch_{}.pth".format(epoch_nr))
              print("save model " + "epoch_{}.pth".format(epoch_nr))
              os.system("nvidia-smi")
        end_time=time.time()
        run_time=end_time-start_time
        hour = int(run_time/3600)
        minute = int((run_time-hour*3600)/60)
        print("epoch {} running time {}h {}min".format(epoch_nr,hour,minute))
        
    f.close()
# test
def model_test(args, data_loader,  test_subjects):
    print("*" * 40)
    print("Test")

    subjects_dice = []
    subjects_dice_x = []
    subjects_dice_y = []
    subjects_dice_z = []
    nr_batches = int(144 / args.batch_size)
    for subject_idx in range(len(test_subjects)):

        batch_gen_val = data_loader.get_batch_generator(test_subjects, subject_idx, type="val")

        global_seg_x = torch.tensor([]).to(device)
        global_seg_y = torch.tensor([]).to(device)
        global_seg_z = torch.tensor([]).to(device)

        global_predict_x = torch.tensor([]).to(device)
        global_predict_y = torch.tensor([]).to(device)
        global_predict_z = torch.tensor([]).to(device)

        predict_three_batch = torch.tensor([]).to(device)
        seg = torch.tensor([]).to(device)

        for i in range(nr_batches):
            batch = next(batch_gen_val)

            # "data_x", "seg_x", "data_y", "seg_y", "data_z", "seg_z"
            data_x = batch["data_x"]
            seg_x = batch["seg_x"]
            data_y = batch["data_y"]
            seg_y = batch["seg_y"]
            data_z = batch["data_z"]
            seg_z = batch["seg_z"]



            if i == 0:
                seg = seg_x.to(device)
            else:
                seg = torch.cat((seg, seg_x.to(device)), dim=0)

            with torch.no_grad():
                data_x = data_x.to(device)
                seg_x = seg_x.to(device)
                data_y = data_y.to(device)
                seg_y = seg_y.to(device)
                data_z = data_z.to(device)
                seg_z = seg_z.to(device)

             
                outputs_x = model(data_x)
                outputs_x = torch.sigmoid(outputs_x)
               
                outputs_y = model(data_y)
                outputs_y = torch.sigmoid(outputs_y)
                

                outputs_z = model(data_z)
                outputs_z = torch.sigmoid(outputs_z)
               
                if i == 0:
                    global_predict_x = outputs_x
                    global_predict_y = outputs_y
                    global_predict_z = outputs_z

                    global_seg_x = seg_x
                    global_seg_y = seg_y
                    global_seg_z = seg_z

                else:
                    global_predict_x = torch.cat((global_predict_x, outputs_x), dim=0)
                    global_predict_y = torch.cat((global_predict_y, outputs_y), dim=0)
                    global_predict_z = torch.cat((global_predict_z, outputs_z), dim=0)
                    global_seg_x = torch.cat((global_seg_x, seg_x),dim=0)
                    global_seg_y = torch.cat((global_seg_y, seg_y), dim=0)
                    global_seg_z = torch.cat((global_seg_z, seg_z), dim=0)
        
        seg = seg.permute(0, 2, 3, 1)
       
        global_predict_x = global_predict_x.permute(0, 2, 3, 1)
       
        global_predict_y = global_predict_y.permute(2, 0, 3, 1)
       
        global_predict_z = global_predict_z.permute(2, 3, 0, 1)
      
        predict_three_batch = torch.add(torch.add(global_predict_x, global_predict_y), global_predict_z) / 3.0
      
        predict_three_batch = torch.where(predict_three_batch > 0.5, torch.ones_like(predict_three_batch),
                                          torch.zeros_like(predict_three_batch))

        epoch_dice = np.nanmean(val_f1_score_macro(predict_three_batch.cpu().numpy(), seg.cpu().numpy()))
        print("Num.", subject_idx, "subject =", test_subjects[subject_idx], "dice =", epoch_dice)
        global_predict_x_where = torch.where(global_predict_x > 0.5, torch.ones_like(global_predict_x),
                                             torch.zeros_like(global_predict_x))
        global_predict_y_where = torch.where(global_predict_y > 0.5, torch.ones_like(global_predict_y),
                                             torch.zeros_like(global_predict_y))
        global_predict_z_where = torch.where(global_predict_z > 0.5, torch.ones_like(global_predict_z),
                                             torch.zeros_like(global_predict_z))
        epoch_dice_x = np.nanmean(val_f1_score_macro(global_predict_x_where.cpu().numpy(), seg.cpu().numpy()))
        epoch_dice_y = np.nanmean(val_f1_score_macro(global_predict_y_where.cpu().numpy(), seg.cpu().numpy()))
        epoch_dice_z = np.nanmean(val_f1_score_macro(global_predict_z_where.cpu().numpy(), seg.cpu().numpy()))

        print("dice_x =", epoch_dice_x)
        print("dice_y =", epoch_dice_y)
        print("dice_z =", epoch_dice_z)
        subjects_dice.append(epoch_dice)
        subjects_dice_x.append(epoch_dice_x)
        subjects_dice_y.append(epoch_dice_y)
        subjects_dice_z.append(epoch_dice_z)
    average_subject_dice = float(sum(subjects_dice) / len(subjects_dice))
    average_subject_dice_x = float(sum(subjects_dice_x) / len(subjects_dice_x))
    average_subject_dice_y = float(sum(subjects_dice_y) / len(subjects_dice_y))
    average_subject_dice_z = float(sum(subjects_dice_z) / len(subjects_dice_z))
    print("mean dice =", average_subject_dice)
    print("mean dice_x =", average_subject_dice_x)
    print("mean dice_y =", average_subject_dice_y)
    print("mean dice_z =", average_subject_dice_z)

def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)  
    torch.cuda.manual_seed(SEED)  
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    my_seed = 2022
    set_seed(my_seed)
    torch.set_printoptions(profile="full")
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)

    tract_name = ['AF_left', 'AF_right', 'ATR_left', 'ATR_right', 'CA', 'CC_1', 'CC_2', 'CC_3', 'CC_4', 'CC_5', 'CC_6',
                  'CC_7', 'CG_left', 'CG_right', 'CST_left', 'CST_right', 'MLF_left', 'MLF_right', 'FPT_left',
                  'FPT_right', 'FX_left', 'FX_right', 'ICP_left', 'ICP_right', 'IFO_left', 'IFO_right', 'ILF_left',
                  'ILF_right', 'MCP', 'OR_left', 'OR_right', 'POPT_left', 'POPT_right', 'SCP_left', 'SCP_right',
                  'SLF_I_left', 'SLF_I_right', 'SLF_II_left', 'SLF_II_right', 'SLF_III_left', 'SLF_III_right',
                  'STR_left', 'STR_right', 'UF_left', 'UF_right', 'CC', 'T_PREF_left', 'T_PREF_right', 'T_PREM_left',
                  'T_PREM_right', 'T_PREC_left', 'T_PREC_right', 'T_POSTC_left', 'T_POSTC_right', 'T_PAR_left',
                  'T_PAR_right', 'T_OCC_left', 'T_OCC_right', 'ST_FO_left', 'ST_FO_right', 'ST_PREF_left',
                  'ST_PREF_right', 'ST_PREM_left', 'ST_PREM_right', 'ST_PREC_left', 'ST_PREC_right', 'ST_POSTC_left',
                  'ST_POSTC_right', 'ST_PAR_left', 'ST_PAR_right', 'ST_OCC_left', 'ST_OCC_right']

    fixed_subject = ['992774']
    moving_subjects = ['991267', '987983', '984472', '983773', '979984', '978578', '965771', '965367',
                       '959574', '958976', '957974', '951457', '932554', '930449', '922854', '917255', '912447',
                       '910241', '907656', '904044', \
                       '901442', '901139', '901038', '899885', '898176', '896879', '896778', '894673', '889579',
                       '887373', '877269', '877168', '872764', '872158', '871964', '871762', '865363', '861456',
                       '859671', '857263', '856766', \
                       '849971', '845458', '837964', '837560', '833249', '833148', '826454', '826353', '816653',
                       '814649', '802844', '792766', '792564', '789373', '786569', '784565', '782561', '779370',
                       '771354', '770352', '765056', '761957', '759869', '756055', '753251', '751348', '749361',
                       '748662', '748258', '742549',
                       '734045',
                       '732243', '729557', '729254', '715647', '715041', '709551', '705341', '704238', '702133',
                       '695768',
                       '690152']
    test_subjects = ['687163', '685058', '683256', '680957', '679568', '677968', '673455', '672756', '665254', '654754',
                     '645551', '644044', '638049', '627549', '623844', '622236', '620434', '613538', '601127', '599671',
                     '599469']

    parse = argparse.ArgumentParser()


    parse.add_argument("--data_dir", type=str, default="/home/hao/data/MSMT_CSD/")
    parse.add_argument("--label_dir", type=str, default="/home/hao/data/HCP105_Zenodo_NewTrkFormat/")
    parse.add_argument("--action", type=str, default="step_1_train")

    args = parse.parse_args()
    device = torch.device("cuda")


    if args.action == "step_1_train":
        from voxel_morph_base_model import U_Network as U_Net_VoxelMorph
        from voxel_morph_base_model import SpatialTransformer, save_image
        from torch.optim import Adam
        import voxel_morph_base_model as vmbm
        from dataloader_registration import DataLoaderTraining as Dataloader_Registration

        parse.add_argument("--batch_size", type=int, default=1)
        parse.add_argument("--lr", type=float, default=0.001)
        parse.add_argument("--epoch", type=int, help='num of epoch', default=100)
        parse.add_argument("--model", type=str, help="voxelmorph 1 or 2",
                           dest="model", choices=['vm1', 'vm2'], default='vm2')
        parse.add_argument("--sim_loss", type=str, help="image similarity loss: mse or ncc",
                           dest="sim_loss", default='ncc')
        parse.add_argument("--alpha", type=float, help="regularization parameter",
                           dest="alpha", default=0.12)  # recommend 1.0 for ncc, 0.02 for mse
        parse.add_argument("--step_1_ckpt_dir", type=str)
        args = parse.parse_args()

        vol_size = [144, 144, 144]

        nf_enc = [16, 32, 32, 32]
        if args.model == "vm1":
            nf_dec = [32, 32, 32, 32, 8, 8]
        else:
            nf_dec = [32, 32, 32, 32, 32, 16, 16]

        sim_loss_fn = vmbm.ncc_loss if args.sim_loss == "ncc" else vmbm.mse_loss
        grad_loss_fn = vmbm.gradient_loss
        criterion = {"sim_loss_fn": sim_loss_fn, "grad_loss_fn": grad_loss_fn}

        UNet = U_Net_VoxelMorph(len(vol_size), nf_enc, nf_dec).to(device)
        STN = SpatialTransformer(vol_size).to(device)
        data_loader = Dataloader_Registration(args, tract_name)
        optimizer = Adam(UNet.parameters(), lr=args.lr)
        model_registration_train(args, UNet, STN, data_loader, criterion, optimizer, moving_subjects, fixed_subject)

    elif args.action == "step_1_generation":

        from voxel_morph_base_model import U_Network as U_Net_VoxelMorph
        from voxel_morph_base_model import SpatialTransformer, save_image, save_image_intensity
        from torch.optim import Adam
        import voxel_morph_base_model as vmbm
        from dataloader_registration import DataLoaderTraining as Dataloader_Registration
        import nibabel as nib



        parse.add_argument("--step_1_ckpt_dir", type=str)
        parse.add_argument("--step_1_pseudo_data_dir", type=str)

        args = parse.parse_args()



        vol_size = [144, 144, 144]

        data_loader = Dataloader_Registration(args, tract_name)

        model_shape = torch.load(args.step_1_ckpt_dir + '/epoch_200.pth', map_location=torch.device('cuda'))
        model_shape_STN = SpatialTransformer(vol_size).to(device)
        print("load shape model", args.step_1_ckpt_dir + '/epoch_200.pth')

        model_pseudo_label_genenration(args, model_shape, model_shape_STN,
                                            data_loader, moving_subjects, fixed_subject,
                                            start_epoch=0)

        # --data_dir
        # /home/hao/data/MSMT_CSD/
        # --label_dir
        # /home/hao/data/HCP105_Zenodo_NewTrkFormat/
        # --ckpt_path
        # /home/hao/PycharmProjects/TractSeg-master/ckpt/registration_mse_0.02/train_pseudo_label
        # --batch_size
        # 1
        # --lr
        # 0.001
        # --action
        # pseudo_label_generation

    elif args.action == "step_2_train":

        parse.add_argument("--batch_size", type=int, default=48)
        parse.add_argument("--lr", type=float, default=0.002)
        parse.add_argument("--epoch", type=int, help='num of epoch', default=200)
        parse.add_argument("--LOSS_WEIGHT", type=int, default=10)
        parse.add_argument("--LOSS_WEIGHT_LEN", type=int, default=200)

        args = parse.parse_args()
        model = UNet_Pytorch_DeepSup(n_input_channels=9, n_classes=72, n_filt=64).cuda()
        data_loader = DataLoaderTraining(args, tract_name)
        optimizer = Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model_train(args, data_loader, optimizer, fixed_subject+moving_subjects,  start_epoch=0)

    elif args.action == "step_3_train":

        device = torch.device("cuda")
        from dataloader_pseudo_label import DataLoaderTraining as DataLoaderTraining_Pseudo_label


        parse.add_argument("--batch_size", type=int, default=48)
        parse.add_argument("--lr", type=float, default=0.002)
        parse.add_argument("--epoch", type=int, help='num of epoch', default=200)
        parse.add_argument("--step_1_pseudo_data_dir", type=str)
        parse.add_argument("--step_2_ckpt_dir", type=str)
        parse.add_argument("--LOSS_WEIGHT", type=int, default=10)
        parse.add_argument("--LOSS_WEIGHT_LEN", type=int, default=200)
        args = parse.parse_args()
        # model = Unet(in_ch=9, out_ch=72).to(device)
        # model = torch.load('/home/hao/PycharmProjects/TractSeg-master/ckpt/registration_mse_0.02' + '/epoch_25.pth', map_location=torch.device('cuda'))

        # model = UNet_Pytorch_DeepSup( n_input_channels=9, n_classes=72, n_filt=64,dropout=True,batchnorm=False).cuda()
        model = torch.load("/home/hao/PycharmProjects/TractSeg-master/ckpt/teacher_model_with_weight/epoch_28.pth").to(
            device)
        # model = torch.load("/home/hao/PycharmProjects/TractSeg-master/ckpt/registration_mse_0.02/train_pseudo_label/epoch_60.pth").to(device)

        model.train()
        one_shot_model = torch.load("/home/hao/PycharmProjects/TractSeg-master/ckpt/one-shot/epoch_192.pth").to(device)
        one_shot_model.eval()

        # model = torch.load("/home/hao/PycharmProjects/TractSeg-master/ckpt/03/epoch_best.pth").cuda()

        # model = UNet_Pytorch( n_input_channels=9, n_classes=72, n_filt=64, batchnorm=False, dropout=False).to(device)
        data_loader = DataLoaderTraining_Pseudo_label(args, tract_name)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
        # criterion = BinaryDiceLoss()
        optimizer = Adamax(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        model_pseudo_label_train_weight(args, data_loader, criterion, optimizer, moving_subjects, start_epoch=0)

    elif args.action == "test":

        model = torch.load("/home/hao/PycharmProjects/TractSeg-master/ckpt/teacher_model_with_weight/" \
                           + '/epoch_50.pth', map_location=torch.device('cuda'))
        model.eval()
        print("load epoch_best.pth")
        data_loader = DataLoaderTraining(args, tract_name)

        model_test(args, data_loader, moving_subjects)


