import numpy as np
from settings import loader_settings
import medpy.io
import os, pathlib
import torch, sys, gc
sys.path.insert(1, '/opt/algorithm/my_model')
from model import Model
#yilei

def pred_post(pred:np.ndarray, t=0.5):
    pred[pred<t] = 0
    pred[pred>=t] = 1
    return pred

def pred_TTA(model, image):
    pred_1 = model(image)[-1][0,0].cpu().detach().numpy() 
    pred_2 = model(torch.flip(image, [4]))[-1].cpu().detach().numpy()[0,0,...,::-1] # 2
    pred_3 = model(torch.flip(image, [3]))[-1].cpu().detach().numpy()[0,0,...,::-1,:] # 1
    pred_4 = model(torch.flip(image, [2]))[-1].cpu().detach().numpy()[0,0,...,::-1,:,:] # 0
    pred_5 = model(torch.flip(image, [3, 4]))[-1].cpu().detach().numpy()[0,0,...,::-1,::-1] # 1,2
    pred_6 = model(torch.flip(image, [2, 3]))[-1].cpu().detach().numpy()[0,0,...,::-1,::-1,:] # 0,1
    pred_7 = model(torch.flip(image, [2, 4]))[-1].cpu().detach().numpy()[0,0,...,::-1,:,::-1] # 0,2
    pred_8 = model(torch.flip(image, [2, 3, 4]))[-1].cpu().detach().numpy()[0,0,...,::-1,::-1,::-1] # 0,1,2

    pred_TTA = pred_post((pred_1 + pred_2 + pred_3 + pred_4 + pred_5 + pred_6 + pred_7 + pred_8) / 8)
    del pred_1
    del pred_2
    del pred_3
    del pred_4
    del pred_5
    del pred_6
    del pred_7
    del pred_8

    gc.collect()
    return pred_TTA

def get_the_largest_two_volume_pred(all_preds:list):

    volumes = []
    for each in all_preds:
        volumes.append(sum(each.flatten()))
    
    volumes_sorted = sorted(volumes)
    idx1 = volumes.index(volumes_sorted[-1])
    idx2 = volumes.index(volumes_sorted[-2])
    
    pred1 = all_preds[idx1]
    pred2 = all_preds[idx2]
    
    f_pred = (pred1 + pred2) / 2

    f_pred[f_pred < 0.5] = 0
    f_pred[f_pred >= 0.5] = 1

    print(volumes, max(volumes), sum(f_pred.flatten()))
    return f_pred

class Seg():
    def __init__(self):
        # super().__init__(
        #     validators=dict(
        #         input_image=(
        #             UniqueImagesValidator(),
        #             UniquePathIndicesValidator(),
        #         )
        #     ),
        # )
        return
        
    def process(self):
        inp_path = loader_settings['InputPath']  # Path for the input
        out_path = loader_settings['OutputPath']  # Path for the output
        file_list = os.listdir(inp_path)  # List of files in the input
        file_list = [os.path.join(inp_path, f) for f in file_list]
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #yilei
        for fil in file_list:
            dat, hdr = medpy.io.load(fil)  # dat is a numpy array
            im_shape = dat.shape
            dat = dat.reshape(1, 1, *im_shape)  # reshape to Pytorch standard
            # Convert 'dat' to Tensor, or as appropriate for your model.
            ###########
            ### Replace this section with the call to your code.
            dat = dat/dat.mean()
            dat = dat[:, :, 19:-18, 21:-20, 15:-14]
            dat = torch.from_numpy(dat).to(device)
            model_fold0 = Model().to(device)
            model_fold1 = Model().to(device)
            model_fold2 = Model().to(device)
            model_fold3 = Model().to(device)
            model_fold4 = Model().to(device)
            model_fold0.load_state_dict(torch.load('/opt/algorithm/my_model/weights/fold0.pth', map_location=device))
            model_fold1.load_state_dict(torch.load('/opt/algorithm/my_model/weights/fold1.pth', map_location=device))
            model_fold2.load_state_dict(torch.load('/opt/algorithm/my_model/weights/fold2.pth', map_location=device))
            model_fold3.load_state_dict(torch.load('/opt/algorithm/my_model/weights/fold3.pth', map_location=device))
            model_fold4.load_state_dict(torch.load('/opt/algorithm/my_model/weights/fold4.pth', map_location=device))
            with torch.no_grad():
                # v2_fold0_pred = np.pad(model_fold0(dat)[-1][0,0].cpu().numpy(), ((19,18), (21,20), (15,14)), 'constant')
                # v2_fold1_pred = np.pad(model_fold1(dat)[-1][0,0].cpu().numpy(), ((19,18), (21,20), (15,14)), 'constant')
                # v2_fold2_pred = np.pad(model_fold2(dat)[-1][0,0].cpu().numpy(), ((19,18), (21,20), (15,14)), 'constant')
                # v2_fold3_pred = np.pad(model_fold3(dat)[-1][0,0].cpu().numpy(), ((19,18), (21,20), (15,14)), 'constant')
                # v2_fold4_pred = np.pad(model_fold4(dat)[-1][0,0].cpu().numpy(), ((19,18), (21,20), (15,14)), 'constant')
                v2_fold0_pred = pred_TTA(model_fold0, dat)
                v2_fold1_pred = pred_TTA(model_fold1, dat)
                v2_fold2_pred = pred_TTA(model_fold2, dat)
                v2_fold3_pred = pred_TTA(model_fold3, dat)
                v2_fold4_pred = pred_TTA(model_fold4, dat)
                print(fil, v2_fold0_pred.shape)
            
            dat = (v2_fold0_pred + v2_fold1_pred + v2_fold2_pred + v2_fold3_pred + v2_fold4_pred) / 5
            dat[dat>=0.5] = 1
            dat[dat<0.5] = 0
            if sum(dat.flatten()) < 1000: # trigger smart ensemble
                dat = get_the_largest_two_volume_pred([v2_fold0_pred, v2_fold1_pred, v2_fold2_pred, v2_fold3_pred, v2_fold4_pred])
            dat = np.pad(dat, ((19,18), (21,20), (15,14)), 'constant')
            dat = dat.astype(int)
            ###
            ###########
            dat = dat.reshape(*im_shape)
            out_name = os.path.basename(fil)
            out_filepath = os.path.join(out_path, out_name)
            print(f'=== saving {out_filepath} from {fil} ===')
            medpy.io.save(dat, out_filepath, hdr=hdr)
        return
    


if __name__ == "__main__":
    pathlib.Path("/output/images/stroke-lesion-segmentation/").mkdir(parents=True, exist_ok=True)
    Seg().process()
