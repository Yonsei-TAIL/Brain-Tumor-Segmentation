import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from medpy.metric.binary import sensitivity, specificity, dc, hd95

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.transforms import decode_preds

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")

    if isinstance(net, list):
        net, net_D = net
        net.train()
        net_D.train()

        optimizer, optimizer_D = optimizer
        criterion, criterion_D = criterion
    
    else:
        net.train()

    losses, ce_dices, necro_dices, peri_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, mask) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, mask = torch.Tensor(img).float(), torch.Tensor(mask).float()
        if opt.use_gpu:
            img, mask = img.cuda(non_blocking=True), mask.cuda(non_blocking=True)

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Train Generator
        if opt.discriminator:
            valid = Variable(torch.FloatTensor(img.shape[0], 1).fill_(1.0), requires_grad=False)
            if opt.use_gpu:
                valid = valid.cuda(non_blocking=True)

            loss_G = criterion_D(net_D(pred, mask), valid)
            loss = loss + loss_G * 0.5

        # Backward and step
        loss.backward()
        optimizer.step()

        # Train Discriminator
        if opt.discriminator:
            optimizer_D.zero_grad()

            fake = Variable(torch.FloatTensor(img.shape[0], 1).fill_(0.0), requires_grad=False)
            if opt.use_gpu:
                fake = fake.cuda(non_blocking=True)
            
            real_loss = criterion_D(net_D(img, mask), valid)
            fake_loss = criterion_D(net_D(pred.detach(), mask), fake)
            loss_D = 0.5 * (real_loss + fake_loss)

            loss_D.backward()
            optimizer_D.step()

        # Calculation Dice Coef Score
        pred_decoded = torch.stack(decode_preds(pred), 0)
        ce_dice, necro_dice, peri_dice = DiceCoef(return_score_per_channel=True)(pred_decoded, mask[:,1:])
        total_dice = (ce_dice + necro_dice + peri_dice) / 3

        ce_dices.update(ce_dice.item(), img.size(0))
        necro_dices.update(necro_dice.item(), img.size(0))
        peri_dices.update(peri_dice.item(), img.size(0))
        total_dices.update(total_dice.item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if (it==0) or (it+1) % 10 == 0:
            print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice : CE %.4f Necro %.4f Peri %.4f Total %.4f'
                % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, ce_dices.avg, necro_dices.avg, peri_dices.avg, total_dices.avg))

    print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice : CE %.4f Necro %.4f Peri %.4f Total %.4f\n"
        % (epoch+1, opt.max_epoch, losses.avg, ce_dices.avg, necro_dices.avg, peri_dices.avg, total_dices.avg))


def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_dice, best_epoch):
    print("Start Evaluation...")
    if isinstance(net, list):
        net, _ = net
        net.eval()

        optimizer, _ = optimizer
        criterion, _ = criterion
    else:
        net.eval()

    # 'PatientID - Array' Dictionary
    ce_dict_GT, ce_dict_pred = dict(), dict()
    necro_dict_GT, necro_dict_pred = dict(), dict()
    peri_dict_GT, peri_dict_pred = dict(), dict()

    # Result containers
    losses, ce_dices, necro_dices, peri_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    for it, (img, masks_resized, masks_org, meta) in enumerate(dataset_val):
        # Load Data
        img, masks_resized = [torch.Tensor(tensor).float() for tensor in [img, masks_resized]]
        if opt.use_gpu:
            img, masks_resized = [tensor.cuda(non_blocking=True) for tensor in [img, masks_resized]]

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Loss Calculation
        loss = criterion(pred, masks_resized)

        # Save GT and Pred Array to Dictionary
        pred_decoded = decode_preds(pred, meta, refine=True)
        for pred, gt, patID in zip(pred_decoded, masks_org, meta['patientID']):
            pred_ce, pred_necro, pred_peri = pred.cpu().data.numpy()
            gt_ce, gt_necro, gt_peri = gt.cpu().data.numpy()
            patID = int(patID[0].item())
            
            if patID not in ce_dict_GT:
                ce_dict_GT[patID] = gt_ce[None, ...]
                ce_dict_pred[patID] = pred_ce[None, ...]
                necro_dict_GT[patID] = gt_necro[None, ...]
                necro_dict_pred[patID] = pred_necro[None, ...]
                peri_dict_GT[patID] = gt_peri[None, ...]
                peri_dict_pred[patID] = pred_peri[None, ...]
                
            else:
                ce_dict_GT[patID] = np.concatenate([ce_dict_GT[patID], gt_ce[None, ...]], 0)
                ce_dict_pred[patID] = np.concatenate([ce_dict_pred[patID], pred_ce[None, ...]], 0)
                necro_dict_GT[patID] = np.concatenate([necro_dict_GT[patID], gt_necro[None, ...]], 0)
                necro_dict_pred[patID] = np.concatenate([necro_dict_pred[patID], pred_necro[None, ...]], 0)
                peri_dict_GT[patID] = np.concatenate([peri_dict_GT[patID], gt_peri[None, ...]], 0)
                peri_dict_pred[patID] = np.concatenate([peri_dict_pred[patID], pred_peri[None, ...]], 0)

        # Stack Results
        losses.update(loss.item(), img.size(0))
    
    # Evaluation Metrics Calculation
    for patID in ce_dict_GT:
        gt_ce = ce_dict_GT[patID]
        pred_ce = ce_dict_pred[patID]
        
        gt_necro = necro_dict_GT[patID]
        pred_necro = necro_dict_pred[patID]
        
        gt_peri = peri_dict_GT[patID]
        pred_peri = peri_dict_pred[patID]

        ce_dice = dc(pred_ce, gt_ce)
        necro_dice = dc(pred_necro, gt_necro)
        peri_dice = dc(pred_peri, gt_peri)
        total_dice = (ce_dice + necro_dice + peri_dice) / 3

        ce_dices.update(ce_dice, 1)
        necro_dices.update(necro_dice, 1)
        peri_dices.update(peri_dice, 1)
        total_dices.update(total_dice, 1)

    print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice : CE %.4f Necro %.4f Peri %.4f Total %.4f"
        % (epoch+1, opt.max_epoch, losses.avg, ce_dices.avg, necro_dices.avg, peri_dices.avg, total_dices.avg))

    # Update Result
    if total_dices.avg > best_dice:
        print('Best Score Updated...')
        best_dice = total_dices.avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('%s/*.pth' % opt.exp):
            os.remove(path)

        model_filename = '%s/epoch_%04d_dice%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_dice, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Dice: %.8f in %3d epoch\n' % (best_dice, best_epoch+1))
    
    return best_dice, best_epoch


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")
    if isinstance(net, list):
        net, _ = net
        net.eval()

    else:
        net.eval()

    # 'PatientID - Array' Dictionary
    ce_dict_GT, ce_dict_pred = dict(), dict()
    necro_dict_GT, necro_dict_pred = dict(), dict()
    peri_dict_GT, peri_dict_pred = dict(), dict()

    # Result containers
    ce_dices, necro_dices, peri_dices = AverageMeter(), AverageMeter(), AverageMeter()
    ce_hausdorff95s, necro_hausdorff95s, peri_hausdorff95s = AverageMeter(), AverageMeter(), AverageMeter()
    ce_sensitivity, necro_sensitivity, peri_sensitivity = AverageMeter(), AverageMeter(), AverageMeter()
    ce_specificity, necro_specificity, peri_specificity = AverageMeter(), AverageMeter(), AverageMeter()

    for img, _, masks_org, meta in tqdm(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()
        if opt.use_gpu:
            img = img.cuda(non_blocking=True)

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Save GT and Pred Array to Dictionary
        pred_decoded = decode_preds(pred, meta, refine=True)
        for pred, gt, patID in zip(pred_decoded, masks_org, meta['patientID']):
            pred_ce, pred_necro, pred_peri = pred.cpu().data.numpy()
            gt_ce, gt_necro, gt_peri = gt.cpu().data.numpy()
            patID = int(patID[0].item())
            
            if patID not in ce_dict_GT:
                ce_dict_GT[patID] = gt_ce[None, ...]
                ce_dict_pred[patID] = pred_ce[None, ...]
                necro_dict_GT[patID] = gt_necro[None, ...]
                necro_dict_pred[patID] = pred_necro[None, ...]
                peri_dict_GT[patID] = gt_peri[None, ...]
                peri_dict_pred[patID] = pred_peri[None, ...]
                
            else:
                ce_dict_GT[patID] = np.concatenate([ce_dict_GT[patID], gt_ce[None, ...]], 0)
                ce_dict_pred[patID] = np.concatenate([ce_dict_pred[patID], pred_ce[None, ...]], 0)
                necro_dict_GT[patID] = np.concatenate([necro_dict_GT[patID], gt_necro[None, ...]], 0)
                necro_dict_pred[patID] = np.concatenate([necro_dict_pred[patID], pred_necro[None, ...]], 0)
                peri_dict_GT[patID] = np.concatenate([peri_dict_GT[patID], gt_peri[None, ...]], 0)
                peri_dict_pred[patID] = np.concatenate([peri_dict_pred[patID], pred_peri[None, ...]], 0)

    # Calculate Metrics
    print('\nCalculating Evaluation Metrics...')
    for patID, gt_ce in ce_dict_GT.items():
        pred_ce = ce_dict_pred[patID]
        ce_dices.update(dc(pred_ce, gt_ce), 1)
        ce_hausdorff95s.update(hd95(gt_ce, pred_ce), 1)
        ce_sensitivity.update(sensitivity(gt_ce, pred_ce), 1)
        ce_specificity.update(specificity(gt_ce, pred_ce), 1)

    for patID, gt_necro in necro_dict_GT.items():
        pred_necro = necro_dict_pred[patID]
        necro_dices.update(dc(pred_necro, gt_necro), 1)
        necro_hausdorff95s.update(hd95(gt_necro, pred_necro), 1)
        necro_sensitivity.update(sensitivity(gt_necro, pred_necro), 1)
        necro_specificity.update(specificity(gt_necro, pred_necro), 1)
        
    for patID, gt_peri in peri_dict_GT.items():
        pred_peri = peri_dict_pred[patID]
        peri_dices.update(dc(pred_peri, gt_peri), 1)
        peri_hausdorff95s.update(hd95(gt_peri, pred_peri), 1)
        peri_sensitivity.update(sensitivity(gt_peri, pred_peri), 1)
        peri_specificity.update(specificity(gt_peri, pred_peri), 1)

    print("Evaluate Result\
           \n>>>> Dice : CE %.4f Necro %.4f Peri %.4f\
           \n>>>> Hausdorff95 : CE %.4f Necro %.4f Peri %.4f\
           \n>>>> Sensitivity : CE %.4f Necro %.4f Peri %.4f\
           \n>>>> Specificity : CE %.4f Necro %.4f Peri %.4f\
           "
        % (ce_dices.avg, necro_dices.avg, peri_dices.avg,
           ce_hausdorff95s.avg, necro_hausdorff95s.avg, peri_hausdorff95s.avg,
           ce_sensitivity.avg, necro_sensitivity.avg, peri_sensitivity.avg,
           ce_specificity.avg, necro_specificity.avg, peri_specificity.avg,))