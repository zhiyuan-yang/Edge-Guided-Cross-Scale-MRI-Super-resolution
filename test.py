import numpy as np
import torch
from torch.utils.data import DataLoader
from config import args
from datasets.load_dataset import RefSRDataset
from models.EGMSSR import DCAMSR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from torchvision.utils import save_image


batch_size = 1
device = device = torch.device('cpu' if args.cpu else 'cuda')
test_lr_path = args.test_lr_path
test_hr_path=args.test_hr_path
test_ref_path=args.test_ref_path
weight = '.\weight\epoch_45.pt'


def batch_to_device(batch):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    return batch


def main():
    testDataset = RefSRDataset(lr_path=test_lr_path,
                               hr_path=test_hr_path,
                               ref_path=test_ref_path,
                               mode='test')
    testDataLoader = DataLoader(testDataset, batch_size=batch_size,shuffle=False)
    model = DCAMSR(scale=2).to(device)
    model.load_state_dict(torch.load(weight))
    model.eval()
    test_SSIM = 0
    test_PSNR = 0
    idx = 0
    
    
    with torch.no_grad():
        for test_batch in testDataLoader:
            test_batch = batch_to_device(test_batch)
            SR_t, _ = model(lr = test_batch['LR'],
                         ref=test_batch['Ref'],
                         lr_gradient = test_batch['LR_s'])
            SR = SR_t.cpu().numpy()
            HR = test_batch['HR'].cpu().numpy()
            for i in range(SR.shape[0]):
                test_PSNR += PSNR(HR[i,0,:,:], SR[i,0,:,:], data_range=1)
                test_SSIM += SSIM(HR[i,0,:,:], SR[i,0,:,:], data_range=1)
            idx += 1
            save_image(SR_t, './output/' + str(idx) + '.jpg')
        test_PSNR = test_PSNR/len(testDataLoader.dataset)
        test_SSIM = test_SSIM/len(testDataLoader.dataset)
        idx = idx + 1
    
    print(f'PSNR: {test_PSNR}, SSIM: {test_SSIM}')                        

if __name__ == '__main__':
    main()