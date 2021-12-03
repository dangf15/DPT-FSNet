import torch
from AudioData import TrainingDataset, EvalDataset
from torch.utils.data import DataLoader
from model import Net
from metric import get_pesq, get_stoi
from criteria import  stftm_loss, mag_loss
from utils import Checkpoint
from tqdm import tqdm
import os
import warnings

# hyperparameter
frame_size = 512
overlap = 0.5
frame_shift = int(512 * (1 - overlap))
max_epochs = 200
batch_size = 8
lr_init = 64 ** (-0.5)
eval_steps = 50000
weight_delay = 1e-7
batches_per_epoch = 40000

# lr scheduling
step_num = 0
warm_ups = 4000

sr = 16000

resume_model    = None#'./checkpoints/temp/latest.model-3.model'
model_save_path = './checkpoints/temp/'

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

early_stop = False

# file path
train_file_list_path = './train_file_list'
validation_file_list_path = './validation_file_list'

# data and data_loader
train_data = TrainingDataset(train_file_list_path, frame_size=512, frame_shift=256)
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4,)

validation_data = EvalDataset(validation_file_list_path, frame_size=512, frame_shift=256)
validation_loader = DataLoader(validation_data,
                               batch_size=1,
                               shuffle=False,
                               num_workers=4,)

# define model
model = Net()
model = torch.nn.DataParallel(model)
model = model.cuda()
print('Number of learnable parameters: %d' % numParams(model))

optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_delay)

mag_loss = mag_loss()
freq_loss = stftm_loss()

def validate(net, eval_loader, test_metric=False):
    net.eval()
    if test_metric:
        print('********Starting metrics evaluation on val dataset**********')
        total_stoi = 0.0
        total_snr = 0.0
        total_pesq = 0.0

    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for k, (features, labels) in enumerate(eval_loader):
            features = features.cuda()  # [1, 1, num_frames, frame_size]
            labels = labels.cuda()  # [signal_len, ]

            output = net(features) # [1, 1, sig_len_recover]
            output = output.squeeze()  # [sig_len_recover,]

            #print(output.shape)
            labels = labels[:,:output.shape[-1]].squeeze()  # keep length same (output label)
            #print(labels.shape)
            #eval_loss = torch.mean((output - labels) ** 2) * 1000

            eval_loss = net.module.loss(output, labels, loss_mode='SI-SNR')#torch.mean((output - labels) ** 2) * 1000
            total_eval_loss += eval_loss.item()
            ##print(k)     

            est_sp = output.cpu().numpy()
            cln_raw = labels.cpu().numpy()
            if test_metric:
                st = get_stoi(cln_raw, est_sp, sr)
                pe = get_pesq(cln_raw, est_sp, sr)
                sn = eval_loss
                total_pesq += pe

                total_snr += sn
                total_stoi += st
            count += 1
#            print(count)
        avg_eval_loss = total_eval_loss / count
    net.train()
    if test_metric:
        return avg_eval_loss, total_stoi / count, total_pesq / count, total_snr / count
    else:
        return avg_eval_loss

# train model
if resume_model:
    print('Resume model from "%s"' % resume_model)
    checkpoint = Checkpoint()
    checkpoint.load(resume_model)

    warm_ups = 0
    start_epoch = checkpoint.start_epoch + 1
    best_val_loss = checkpoint.best_val_loss
    prev_val_loss = checkpoint.prev_val_loss
    num_no_improv = checkpoint.num_no_improv
    half_lr = checkpoint.half_lr
    model.load_state_dict(checkpoint.state_dict)
    optimizer.load_state_dict(checkpoint.optimizer)

else:
    print('Training from scratch.')
    start_epoch = 0
    best_val_loss = float("inf")
    prev_val_loss = float("inf")
    num_no_improv = 0
    half_lr = False

for epoch in range(start_epoch, max_epochs):
    model.train()
    total_train_loss, count, ave_train_loss = 0.0, 0, 0.0

    pbar = tqdm(enumerate(train_loader))
    for index, (features, labels,sig_len ) in enumerate(train_loader):
        step_num += 1

        if step_num <= warm_ups:

            lr = 0.2 * lr_init * min(step_num ** (-0.5),
                                     step_num * (warm_ups ** (-1.5))) 
        else:
            lr = 0.0004 * (0.98 ** ((epoch - 1) // 2)) 

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
 

        # feature -- [batch_size, 2, nframes, frame_size]
        features = features.cuda()
        # label -- [batch_size, 1, signal_length]
        labels = labels.cuda()


        optimizer.zero_grad()

        output = model(features)  # output -- [batch_size, 1, sig_len_recover]


        output = output[:, :, :labels.shape[-1]]  # [batch_size, 1, sig_len]
        labels = labels[...,:output.shape[-1]]


        #loss_time = model.module.loss(output, labels, loss_mode='SI-SNR')
        loss_mag = mag_loss(output, labels)   
        loss_freq = freq_loss(output, labels, loss_mask)
        loss =  loss_mag +loss_freq #+ 0.05 * loss_time


        ref_loss = model.module.loss(output, labels, loss_mode='SI-SNR')        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)


        optimizer.step()

        train_loss = loss.item()
        total_train_loss += train_loss

        count += 1

        #del loss, output, features, labels

        del loss, loss_mag, loss_freq, output, features, labels

        pbar.set_description('iter = {}/{}, epoch = {}/{}, loss = {:.5f}, ref_loss = {:.5f}, lr = {:.6f}'.format(index + 1, len(train_loader), epoch + 1, max_epochs, train_loss, ref_loss,lr))

        if (index + 1) % eval_steps == 0:
            ave_train_loss = total_train_loss / count

            # validation
            avg_eval_loss = validate(model, validation_loader)
            model.train()

            print('Epoch [%d/%d], Iter [%d/%d],  ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
            epoch + 1, max_epochs, index + 1, len(train_loader), ave_train_loss, avg_eval_loss))

            count = 0
            total_train_loss = 0.0


        if (index + 1) % batches_per_epoch == 0:
            break

    # validate metric
    avg_eval, avg_stoi, avg_pesq, avg_snr = validate(model, validation_loader, test_metric=True)
    model.train()
    print('#' * 50)
    print('')
    print('After {} epoch the performance on validation score is a s follows:'.format(epoch + 1))
    print('')
    print('Avg_loss: {:.4f}'.format(avg_eval))
    print('STOI: {:.4f}'.format(avg_stoi))
    print('SNR: {:.4f}'.format(avg_snr))
    print('PESQ: {:.4f}'.format(avg_pesq))

    
    store_to_file =  'After {} epoch the performance on validation score is a s follows:'.format(epoch + 1) +\
                     'Avg_loss: {:.4f}'.format(avg_eval) +\
                     'STOI: {:.4f}'.format(avg_stoi) +\
                     'SNR: {:.4f}'.format(avg_snr) +\
                     'PESQ: {:.4f}'.format(avg_pesq)

    doc = store_to_file + '\n'
    file_name = './log/train_log'
    with open(file_name, "a") as myfile:
        myfile.write(doc)


    # adjust learning rate and early stop
    if avg_eval >= prev_val_loss:
        num_no_improv += 1
        if num_no_improv == 2:
            half_lr = True
        if num_no_improv >= 10 and early_stop is True:
            print("No improvement and apply early stop")
            break
    else:
        num_no_improv = 0

    prev_val_loss = avg_eval

    if avg_eval < best_val_loss:
        best_val_loss = avg_eval
        is_best_model = True
    else:
        is_best_model = False

    # save model
    latest_model = 'latest.model'
    best_model = 'best.model'

    checkpoint = Checkpoint(start_epoch=epoch,
                            best_val_loss=best_val_loss,
                            prev_val_loss=prev_val_loss,
                            state_dict=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            num_no_improv=num_no_improv,
                            half_lr=half_lr)
    checkpoint.save(is_best=is_best_model,
                    filename=os.path.join(model_save_path, latest_model + '-{}.model'.format(epoch + 1)),
                    best_model=os.path.join(model_save_path, best_model))
