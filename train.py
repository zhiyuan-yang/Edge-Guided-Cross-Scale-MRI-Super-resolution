import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from config import args
from datasets.load_dataset import RefSRDataset
from models.EGMSSR import DCAMSR
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import datetime
from utils.loss.loss import get_loss_dict
from utils.logger import Logger
from utils.set_random_seed import set_random_seed
import logging
import os


# recording setting
logger = Logger(log_file_name=args.log_file_name, logger_name=args.logger_name, log_level=logging.DEBUG).get_log()
log_dir = './TensorboardLog/' + datetime.datetime.now().strftime('%Y-%m-%d') + '/'
writer = SummaryWriter(log_dir=log_dir)

# training setting
batch_size = args.batch_size
num_workers = args.num_workers
loss = get_loss_dict(args=args,logger=logger)
rec_w = args.rec_w #weight of reconstruction loss
struct_w = args.struct_w
learning_rate = args.lr_rate
num_epochs = args.num_epochs
beta1 = args.beta1
beta2 = args.beta2
eps = args.eps
device = torch.device('cpu' if args.cpu else 'cuda')
train_lr_path = args.train_lr_path
train_hr_path = args.train_hr_path
train_ref_path=args.train_ref_path


# validation setting
val_freq = args.val_freq
val_lr_path = args.val_lr_path
val_hr_path = args.val_hr_path
val_ref_path=args.val_ref_path


def batch_to_device(batch):
    for key in batch.keys():
        batch[key] = batch[key].to(device)
    return batch


def main():
    trainDataset = RefSRDataset(lr_path=train_lr_path,
                                hr_path=train_hr_path,
                                ref_path=train_ref_path,
                                mode='train')
    trainDataLoader = DataLoader(dataset = trainDataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    valDataset = RefSRDataset(lr_path=val_lr_path,
                              hr_path=val_hr_path,
                              ref_path=val_ref_path,
                              mode='test')
    valDataLoader = DataLoader(valDataset,
                                batch_size=batch_size,
                                shuffle=False)  
    model = DCAMSR(scale=2).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate,
                                 betas=(beta1, beta2), 
                                 eps=eps)
    start =time.time()
    max_PSNR = -np.Inf
    max_SSIM = -np.Inf
    max_PSNR_epoch = 0
    max_SSIM_epoch = 0
    min_val_loss = np.Inf


    logger.info(f'Start Training\n')
 
    #training stage
    for i in range(num_epochs):
        model.train()
        curr_epoch_train_loss = 0
        for idx, batch in enumerate(trainDataLoader):
            optimizer.zero_grad()
            batch = batch_to_device(batch)
            
            #SR_t, grad_sr_t = model(ref = batch['Ref'], lr=batch['LR'])
            SR_t, grad_sr_t = model(ref = batch['Ref'], lr=batch['LR'], lr_gradient = batch['LR_s'])
            #SR_t, grad_sr_t = model(lr=batch['LR'], lr_gradient = batch['LR_s'])  ## w/o reference
            rec_loss = loss['rec_loss'](SR_t, batch['HR'])
            struc_loss = loss['struct_loss'](grad_sr_t, batch['HR_s']) 
            curr_batch_train_loss = rec_w * rec_loss + struct_w * struc_loss
            curr_batch_train_loss.backward()
            optimizer.step()
            curr_epoch_train_loss += curr_batch_train_loss.item() * trainDataLoader.batch_size
        
        curr_epoch_train_loss = curr_epoch_train_loss/len(trainDataLoader.dataset)


        if (i+1) % val_freq == 0:
            model.eval()
            curr_epoch_val_loss = 0
            curr_val_SSIM = 0
            curr_val_PSNR = 0
            with torch.no_grad():
                for val_batch in valDataLoader:
                    val_batch = batch_to_device(val_batch)
                    SR_val_t, grad_val_sr_t = model(lr = val_batch['LR'], lr_gradient=val_batch['LR_s'])
                    struc_loss = loss['struct_loss'](grad_val_sr_t, val_batch['HR'])
                    rec_loss = loss['rec_loss'](SR_val_t,val_batch['HR']) 
                    curr_batch_val_loss = rec_w * rec_loss + struct_w * struc_loss
                    curr_epoch_val_loss += curr_batch_val_loss.item() * valDataLoader.batch_size
                    SR_val = SR_val_t.cpu().numpy()
                    HR_val = val_batch['HR'].cpu().numpy()
                    for j in range(SR_val.shape[0]):
                        curr_val_SSIM += SSIM(SR_val[j,0,:,:], HR_val[j,0,:,:], data_range=1)
                        curr_val_PSNR += PSNR(HR_val[j,0,:,:], SR_val[j,0,:,:], data_range=1) 

            curr_val_SSIM = curr_val_SSIM/len(valDataLoader.dataset)
            curr_val_PSNR = curr_val_PSNR/len(valDataLoader.dataset)
            curr_epoch_val_loss =curr_epoch_val_loss/len(valDataLoader.dataset)
            
            if curr_val_PSNR > max_PSNR:
                max_PSNR_epoch = i + 1
                max_PSNR = curr_val_PSNR
            if curr_val_SSIM > max_SSIM:
                max_SSIM_epoch = i + 1
                max_SSIM = curr_val_SSIM
            if curr_epoch_val_loss < min_val_loss:
                min_val_loss = curr_epoch_val_loss
            
            if os.path.exists('./weight') is False:
                os.makedirs('./weight')    
            torch.save(model.state_dict(), './weight/epoch_'+ str(i+1)+'.pt')
            
            writer.add_scalar("validation loss", curr_epoch_val_loss, (i+1)/val_freq)
            writer.add_scalar("validation PSNR", curr_val_PSNR, (i+1)/val_freq)
            writer.add_scalar("validation SSIM", curr_val_SSIM, (i+1)/val_freq)
            logger.info(f'Current Epoch: {i+1} Validation Loss: {curr_epoch_val_loss}, Validation PSNR: {curr_val_PSNR}, Validation SSIM: {curr_val_SSIM}\n')
        
        logger.info(f'Training Epoch:{i+1} Epoch Loss:{curr_epoch_train_loss}\n')
        writer.add_scalar("train loss", curr_epoch_train_loss, i+1)    


    end = time.time()
    logger.info(f'MAX PSNR: {max_PSNR}, at epoch {max_PSNR_epoch}; MAX SSIM: {max_SSIM}, at epoch {max_SSIM_epoch}\n')
    logger.info(f'Running Time: {(end-start)/3600}h')


if __name__ == '__main__':
    set_random_seed(0)
    main()