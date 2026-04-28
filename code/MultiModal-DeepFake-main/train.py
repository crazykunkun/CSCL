import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb

import argparse
# import ruamel_yaml as yaml
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import logging
from types import MethodType
from tools.env import init_dist
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from models import box_ops
from tools.multilabel_metrics import AveragePrecisionMeter, get_multi_label
from models.CSCL import CSCL
from transformers import RobertaTokenizerFast


def set_trainable_scope(model, scope):
    if scope == 'all':
        for param in model.parameters():
            param.requires_grad = True
        return

    if scope == 'freq_only':
        trainable_prefixes = (
            'freq_fusion',
            'img_intra_model',
            'img_extra_model',
            'bbox_head',
        )
    elif scope == 'freq_iou':
        trainable_prefixes = (
            'freq_fusion',
            'img_intra_model',
            'img_extra_model',
            'bbox_head',
        )
    elif scope == 'freq_aux_iou':
        trainable_prefixes = (
            'freq_fusion',
            'img_intra_model',
            'img_extra_model',
            'bbox_head',
        )
    elif scope == 'freq_image_heads':
        trainable_prefixes = (
            'freq_fusion',
            'img_intra_model',
            'img_extra_model',
            'fusion_head',
            'itm_head',
            'bbox_head',
            'cls_head_img',
            'cls_head_text',
        )
    else:
        raise ValueError(f'Unsupported train scope: {scope}')

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)


def freeze_meter_backbone(model):
    frozen_count = 0
    frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith('meter.'):
            param.requires_grad = False
            frozen_count += 1
            frozen_params += param.numel()
    return frozen_count, frozen_params


def filter_meter_state_dict(state_dict):
    return {name: param for name, param in state_dict.items() if name.startswith('meter.')}


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def write_progress(progress_path, **kwargs):
    serializable = {}
    for key, value in kwargs.items():
        if isinstance(value, (np.floating, np.integer)):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    with open(progress_path, "w") as f:
        json.dump(serializable, f)


def format_eta(seconds):
    seconds = max(0, int(seconds))
    return str(datetime.timedelta(seconds=seconds))

def setlogger(log_file):
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    def epochInfo(self, set, idx, loss, acc):
        self.info('{set}-{idx:d} epoch | loss:{loss:.8f} | auc:{acc:.4f}%'.format(
            set=set,
            idx=idx,
            loss=loss,
            acc=acc
        ))

    logger.epochInfo = MethodType(epochInfo, logger)

    return logger


def text_input_adjust(text_input, fake_word_pos, device):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    # fake_token_pos adaptation
    fake_token_pos_batch = []
    for i in range(len(fake_word_pos)):
        fake_token_pos = []

        fake_word_pos_decimal = np.where(fake_word_pos[i].numpy() == 1)[0].tolist() # transfer fake_word_pos into numbers
        
        subword_idx = text_input.word_ids(i)
        subword_idx_rm_CLSSEP = subword_idx[1:-1]
        subword_idx_rm_CLSSEP_array = np.array(subword_idx_rm_CLSSEP) # get the sub-word position (token position)

        # transfer the fake word position into fake token position
        for i in fake_word_pos_decimal: 
            fake_token_pos.extend(np.where(subword_idx_rm_CLSSEP_array == i)[0].tolist())
        
        fake_token_pos_batch.append(fake_token_pos)

    return text_input, fake_token_pos_batch


def train(args, model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, summary_writer, scaler, progress_path=None, logger=None):
    # train
    model.train()
    if hasattr(model, 'meter'):
        meter_params = list(model.meter.parameters())
        if meter_params and not any(param.requires_grad for param in meter_params):
            model.meter.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_BIC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_MLC', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Loss_sim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('Loss_freq', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 100   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    global_step = epoch*len(data_loader)
    total_steps = config['schedular']['epochs'] * len(data_loader)
    epoch_start_time = time.time()

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, image_path) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        step_start_time = time.time()
        should_log_batch = (i == 0 or i % print_freq == 0)
        if progress_path is not None and (i == 0 or i % 50 == 0):
            elapsed_epoch = time.time() - epoch_start_time
            avg_step_time = elapsed_epoch / max(i + 1, 1)
            epoch_eta_seconds = avg_step_time * max(len(data_loader) - i - 1, 0)
            total_done_steps = epoch * len(data_loader) + i + 1
            total_eta_seconds = avg_step_time * max(total_steps - total_done_steps, 0)
            write_progress(
                progress_path,
                stage="train",
                epoch=epoch,
                batch=i,
                total_batches=len(data_loader),
                epoch_eta=format_eta(epoch_eta_seconds),
                total_eta=format_eta(total_eta_seconds),
            )

        if config['schedular']['sched'] == 'cosine_in_step':
            scheduler.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args, config)        

        optimizer.zero_grad()

        image = image.to(device,non_blocking=True)

        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)

        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        with autocast(dtype=torch.bfloat16):
            model_outputs = model(image, label, text_input, fake_image_box, fake_token_pos)
            if len(model_outputs) == 5:
                loss_BIC, loss_bbox, loss_giou, loss_MLC, Loss_sim = model_outputs
                Loss_freq = Loss_sim * 0.0
            else:
                loss_BIC, loss_bbox, loss_giou, loss_MLC, Loss_sim, Loss_freq = model_outputs

            loss = config['loss_BIC_wgt']*loss_BIC \
                 + config['loss_bbox_wgt']*loss_bbox \
                 + config['loss_giou_wgt']*loss_giou \
                 + config['loss_MLC_wgt']*loss_MLC \
                 + config['Loss_sim_wgt']*Loss_sim \
                 + config.get('Loss_freq_wgt', 0)*Loss_freq

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()    
        
        metric_logger.update(loss_BIC=loss_BIC.item())
        metric_logger.update(loss_bbox=loss_bbox.item())
        metric_logger.update(loss_giou=loss_giou.item())
        metric_logger.update(loss_MLC=loss_MLC.item())
        metric_logger.update(Loss_sim=Loss_sim.item())
        metric_logger.update(Loss_freq=Loss_freq.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])         
        if logger is not None and should_log_batch:
            elapsed_epoch = time.time() - epoch_start_time
            avg_step_time = elapsed_epoch / max(i + 1, 1)
            epoch_eta_seconds = avg_step_time * max(len(data_loader) - i - 1, 0)
            total_done_steps = epoch * len(data_loader) + i + 1
            total_eta_seconds = avg_step_time * max(total_steps - total_done_steps, 0)
            logger.info(
                'Train Epoch [{}/{}] Batch [{}/{}] | lr {:.8f} | '
                'loss_BIC {:.4f} | loss_bbox {:.4f} | loss_giou {:.4f} | '
                'loss_MLC {:.4f} | Loss_sim {:.4f} | Loss_freq {:.4f} | loss {:.4f} | avg_loss {:.4f} | '
                'epoch_eta {} | total_eta {}'.format(
                    epoch + 1,
                    config['schedular']['epochs'],
                    i,
                    len(data_loader),
                    optimizer.param_groups[0]["lr"],
                    loss_BIC.item(),
                    loss_bbox.item(),
                    loss_giou.item(),
                    loss_MLC.item(),
                    Loss_sim.item(),
                    Loss_freq.item(),
                    loss.item(),
                    metric_logger.meters['loss'].global_avg,
                    format_eta(epoch_eta_seconds),
                    format_eta(total_eta_seconds),
                )
            )
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations and config['schedular']['sched'] != 'cosine_in_step': 
            scheduler.step(i//step_size)   

        global_step+=1
        

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'lr': optimizer.param_groups[0]["lr"],                                                                                                                                                                                              
                'loss_BIC': loss_BIC.item(),                                                                                                  
                'loss_bbox': loss_bbox.item(),                                                                                                  
                'loss_giou': loss_giou.item(),                                                                                                                                                                                               
                'loss_MLC': loss_MLC.item(),  
                'Loss_sim': Loss_sim.item(),  
                'Loss_freq': Loss_freq.item(),  
                'loss': loss.item(),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, global_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.log:
        print("Averaged stats:", metric_logger.global_avg(), flush=True)     
    if logger is not None:
        epoch_elapsed = time.time() - epoch_start_time
        logger.info(
            'Train Epoch [{}/{}] averaged | loss_BIC {} | loss_bbox {} | loss_giou {} | loss_MLC {} | Loss_sim {} | Loss_freq {} | loss {} | epoch_time {}'.format(
                epoch + 1,
                config['schedular']['epochs'],
                "{:.6f}".format(metric_logger.meters['loss_BIC'].global_avg),
                "{:.6f}".format(metric_logger.meters['loss_bbox'].global_avg),
                "{:.6f}".format(metric_logger.meters['loss_giou'].global_avg),
                "{:.6f}".format(metric_logger.meters['loss_MLC'].global_avg),
                "{:.6f}".format(metric_logger.meters['Loss_sim'].global_avg),
                "{:.6f}".format(metric_logger.meters['Loss_freq'].global_avg),
                "{:.6f}".format(metric_logger.meters['loss'].global_avg),
                format_eta(epoch_elapsed),
            )
        )
    if progress_path is not None:
        write_progress(
            progress_path,
            stage="train_done",
            epoch=epoch,
            batch=len(data_loader),
            total_batches=len(data_loader),
        )
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    



@torch.no_grad()
def evaluation(args, model, data_loader, tokenizer, device, config, logger=None):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    print('Computing features for evaluation...')
    start_time = time.time()   
    print_freq = 200 

    y_true, y_pred, IOU_pred, IOU_50, IOU_75, IOU_95 = [], [], [], [], [], []
    cls_nums_all = 0
    cls_acc_all = 0   
    
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0

    multi_label_meter = AveragePrecisionMeter(difficult_examples=False)
    multi_label_meter.reset()

    for i, (image, label, text, fake_image_box, fake_word_pos, W, H, image_path) in enumerate(metric_logger.log_every(args, data_loader, print_freq, header)):
        if logger is not None and (i == 0 or i % print_freq == 0):
            logger.info(
                'Evaluation Epoch progress Batch [{}/{}]'.format(
                    i,
                    len(data_loader),
                )
            )
        
        image = image.to(device,non_blocking=True) 
        
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 
        
        text_input, fake_token_pos = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)

        ##================= real/fake cls ========================## 
        cls_label = torch.ones(len(label), dtype=torch.long).to(image.device) 
        real_label_pos = np.where(np.array(label) == 'orig')[0].tolist()
        cls_label[real_label_pos] = 0

        y_pred.extend(F.softmax(logits_real_fake,dim=1)[:,1].cpu().flatten().tolist())
        y_true.extend(cls_label.cpu().flatten().tolist())

        pred_acc = logits_real_fake.argmax(1)
        cls_nums_all += cls_label.shape[0]
        cls_acc_all += torch.sum(pred_acc == cls_label).item()

        # ----- multi metrics -----
        target, _ = get_multi_label(label, image)
        multi_label_meter.add(logits_multicls, target)
        
        ##================= bbox cls ========================## 
        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)

        IOU, _ = box_ops.box_iou(boxes1, boxes2.to(device), test=True)

        IOU_pred.extend(IOU.cpu().tolist())

        IOU_50_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_75_bt = torch.zeros(IOU.shape, dtype=torch.long)
        IOU_95_bt = torch.zeros(IOU.shape, dtype=torch.long)

        IOU_50_bt[IOU>0.5] = 1
        IOU_75_bt[IOU>0.75] = 1
        IOU_95_bt[IOU>0.95] = 1

        IOU_50.extend(IOU_50_bt.cpu().tolist())
        IOU_75.extend(IOU_75_bt.cpu().tolist())
        IOU_95.extend(IOU_95_bt.cpu().tolist())

        ##================= token cls ========================##  
        token_label = text_input.attention_mask[:,1:].clone() # [:,1:] for ingoring class token
        token_label[token_label==0] = -100 # -100 index = padding token
        token_label[token_label==1] = 0

        for batch_idx in range(len(fake_token_pos)):
            fake_pos_sample = fake_token_pos[batch_idx]
            if fake_pos_sample:
                for pos in fake_pos_sample:
                    token_label[batch_idx, pos] = 1
                    
        logits_tok_reshape = logits_tok.view(-1, 2)
        logits_tok_pred = logits_tok_reshape.argmax(1)
        token_label_reshape = token_label.view(-1)
        
        # F1
        TP_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 1)).item()
        TN_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 0)).item()
        FP_all += torch.sum((token_label_reshape == 0) * (logits_tok_pred == 1)).item()
        FN_all += torch.sum((token_label_reshape == 1) * (logits_tok_pred == 0)).item()

    ##================= real/fake cls ========================## 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ACC_cls = cls_acc_all / cls_nums_all
    unique_classes = np.unique(y_true)
    if len(unique_classes) >= 2:
        AUC_cls = roc_auc_score(y_true, y_pred)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
        if np.isnan(fpr).any() or np.isnan(tpr).any():
            EER_cls = float('nan')
        else:
            EER_cls = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    else:
        AUC_cls = float('nan')
        EER_cls = float('nan')
    
    ##================= multi-label cls ========================## 
    MAP = multi_label_meter.value().mean()
    OP, OR, OF1, CP, CR, CF1 = multi_label_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = multi_label_meter.overall_topk(3)
    
    ##================= bbox cls ========================##
    IOU_score = sum(IOU_pred)/len(IOU_pred)
    IOU_ACC_50 = sum(IOU_50)/len(IOU_50)
    IOU_ACC_75 = sum(IOU_75)/len(IOU_75)
    IOU_ACC_95 = sum(IOU_95)/len(IOU_95)

    # ##================= token cls========================##
    ACC_tok = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
    Precision_tok = TP_all / (TP_all + FP_all)
    Recall_tok = TP_all / (TP_all + FN_all)
    F1_tok = 2*Precision_tok*Recall_tok / (Precision_tok + Recall_tok)

    return AUC_cls, ACC_cls, EER_cls, \
           MAP.item(), OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
           IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
           ACC_tok, Precision_tok, Recall_tok, F1_tok
    
def main_worker(gpu, args, config):

    if gpu is not None:
        args.gpu = gpu

    if args.distributed:
        init_dist(args)
    else:
        args.gpu = 0
        args.rank = 0
        args.log = True
        torch.cuda.set_device(0)
    log_dir = os.path.join(args.output_dir, 'log'+ args.log_num)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'shell.txt')
    progress_file = os.path.join(log_dir, 'progress.txt')
    logger = setlogger(log_file)
    yaml.dump(config, open(os.path.join(log_dir, 'config.yaml'), 'w')) 
    
    if args.log:
        summary_writer = SummaryWriter(log_dir)
    else:
        summary_writer = None

    if args.log:
        logger.info('******************************')
        logger.info(args)
        logger.info('******************************')
        logger.info(config)
        logger.info('******************************')

    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best_iou_score = float('-inf')
    best_epoch = -1

    #### Dataset #### 
    if args.log:
        print("Creating dataset")
    train_dataset, val_dataset = create_dataset(config)
    
    if args.distributed:
        samplers = create_sampler([train_dataset], [True], args.world_size, args.rank) + [None]    
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],
                                samplers,
                                batch_size=[config['batch_size_train']]+[config['batch_size_val']], 
                                num_workers=[8, 4],
                                is_trains=[True, False], 
                                collate_fns=[None, None])

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    #### Model #### 
    if args.log:
        print(f"Creating CSCL")
    model = CSCL(args=args, config=config)
    model = model.to(device)   
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        if args.load_meter_only:
            state_dict = filter_meter_state_dict(state_dict)
        if args.log:
            if args.load_meter_only:
                print('load meter-only checkpoint weights from %s'%args.checkpoint)
                print(f'loaded_meter_keys={len(state_dict)}')
            else:
                print('load checkpoint from %s'%args.checkpoint)  
        msg = model.load_state_dict(state_dict, strict=False)
        if args.log:
            print(msg)  

    set_trainable_scope(model, args.train_scope)
    frozen_meter_tensors, frozen_meter_params = freeze_meter_backbone(model)
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    if args.log:
        print(f'train_scope={args.train_scope}')
        print(f'frozen_meter_tensors={frozen_meter_tensors}')
        print(f'frozen_meter_params={frozen_meter_params}')
        print(f'trainable_params={trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)')

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    if config['schedular']['sched'] == 'cosine_in_step':
        args.lr = config['optimizer']['lr']

    if args.checkpoint and args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    scaler = GradScaler()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.log:
        print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):
        if args.log:
            logger.info(f'===== Epoch {epoch + 1}/{max_epoch} started =====')
        write_progress(
            progress_file,
            stage="epoch_start",
            epoch=epoch,
            batch=0,
            total_batches=len(train_loader),
            max_epoch=max_epoch,
        )
            
        train_stats = train(args, model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config, summary_writer, scaler, progress_path=progress_file, logger=logger)
        if args.log:
            logger.info(f'===== Epoch {epoch + 1}/{max_epoch} training finished, starting evaluation =====')
        write_progress(
            progress_file,
            stage="evaluation",
            epoch=epoch,
            batch=len(train_loader),
            total_batches=len(train_loader),
            max_epoch=max_epoch,
        )
        AUC_cls, ACC_cls, EER_cls, \
        MAP, OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, \
        IOU_score, IOU_ACC_50, IOU_ACC_75, IOU_ACC_95, \
        ACC_tok, Precision_tok, Recall_tok, F1_tok \
        = evaluation(args, model_without_ddp, val_loader, tokenizer, device, config, logger=logger)
        if args.log:
            logger.info(
                '===== Epoch {}/{} done | AUC {:.4f} | EER {:.4f} | ACC {:.4f} | mAP {:.4f} | '
                'IOU {:.4f} | IOU@50 {:.4f} | IOU@75 {:.4f} | IOU@95 {:.4f} | '
                'ACC_tok {:.4f} | Precision {:.4f} | Recall {:.4f} | F1 {:.4f} ====='.format(
                    epoch + 1,
                    max_epoch,
                    AUC_cls * 100,
                    EER_cls * 100,
                    ACC_cls * 100,
                    MAP * 100,
                    IOU_score * 100,
                    IOU_ACC_50 * 100,
                    IOU_ACC_75 * 100,
                    IOU_ACC_95 * 100,
                    ACC_tok * 100,
                    Precision_tok * 100,
                    Recall_tok * 100,
                    F1_tok * 100,
                )
            )

        #============ tensorboard train log info ============#
        if args.log:
            lossinfo = {
                'AUC_cls': round(AUC_cls*100, 4),                                                                                                  
                'ACC_cls': round(ACC_cls*100, 4),                                                                                                  
                'EER_cls': round(EER_cls*100, 4),                                                                                                  
                'MAP': round(MAP*100, 4),                                                                                                  
                'OP': round(OP*100, 4),                                                                                                  
                'OR': round(OR*100, 4), 
                'OF1': round(OF1*100, 4), 
                'CP': round(CP*100, 4), 
                'CR': round(CR*100, 4), 
                'CF1': round(CF1*100, 4), 
                'OP_k': round(OP_k*100, 4), 
                'OR_k': round(OR_k*100, 4), 
                'OF1_k': round(OF1_k*100, 4), 
                'CP_k': round(CP_k*100, 4), 
                'CR_k': round(CR_k*100, 4), 
                'CF1_k': round(CF1_k*100, 4), 
                'IOU_score': round(IOU_score*100, 4),                                                                                                  
                'IOU_ACC_50': round(IOU_ACC_50*100, 4),                                                                                                  
                'IOU_ACC_75': round(IOU_ACC_75*100, 4),                                                                                                  
                'IOU_ACC_95': round(IOU_ACC_95*100, 4),                                                                                                  
                'ACC_tok': round(ACC_tok*100, 4),                                                                                                  
                'Precision_tok': round(Precision_tok*100, 4),                                                                                                  
                'Recall_tok': round(Recall_tok*100, 4),                                                                                                  
                'F1_tok': round(F1_tok*100, 4),                                                                                                  
                    } 
            for tag, value in lossinfo.items():
                summary_writer.add_scalar(tag, value, epoch)

        #============ evaluation info ============#
        val_stats = {"AUC_cls": "{:.4f}".format(AUC_cls*100),
                     "ACC_cls": "{:.4f}".format(ACC_cls*100),
                     "EER_cls": "{:.4f}".format(EER_cls*100),
                     "MAP": "{:.4f}".format(MAP*100),
                     "OP": "{:.4f}".format(OP*100),
                     "OR": "{:.4f}".format(OR*100),
                     "OF1": "{:.4f}".format(OF1*100),
                     "CP": "{:.4f}".format(CP*100),
                     "CR": "{:.4f}".format(CR*100),
                     "CF1": "{:.4f}".format(CF1*100),
                     "OP_k": "{:.4f}".format(OP_k*100),
                     "OR_k": "{:.4f}".format(OR_k*100),
                     "OF1_k": "{:.4f}".format(OF1_k*100),
                     "CP_k": "{:.4f}".format(CP_k*100),
                     "CR_k": "{:.4f}".format(CR_k*100),
                     "CF1_k": "{:.4f}".format(CF1_k*100),
                     "IOU_score": "{:.4f}".format(IOU_score*100),
                     "IOU_ACC_50": "{:.4f}".format(IOU_ACC_50*100),
                     "IOU_ACC_75": "{:.4f}".format(IOU_ACC_75*100),
                     "IOU_ACC_95": "{:.4f}".format(IOU_ACC_95*100),
                     "ACC_tok": "{:.4f}".format(ACC_tok*100),
                     "Precision_tok": "{:.4f}".format(Precision_tok*100),
                     "Recall_tok": "{:.4f}".format(Recall_tok*100),
                     "F1_tok": "{:.4f}".format(F1_tok*100),
        }
        
        if utils.is_main_process(): 
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'val_{k}': v for k, v in val_stats.items()},
                            'epoch': epoch,
                        }             
            with open(os.path.join(log_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            if config['schedular']['sched'] != 'cosine_in_step':
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
            else:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr': optimizer.param_groups[0]["lr"],
                    'config': config,
                    'epoch': epoch,
                }                    

            torch.save(save_obj, os.path.join(log_dir, 'checkpoint_latest.pth'))

            if IOU_score > best_iou_score:
                best_iou_score = IOU_score
                best_epoch = epoch
                save_obj['best_iou_score'] = best_iou_score
                save_obj['best_epoch'] = best_epoch
                torch.save(save_obj, os.path.join(log_dir, 'checkpoint_best.pth'))
                if args.log:
                    logger.info(
                        '===== New best model at epoch {}/{} | IOU {:.4f} ====='.format(
                            epoch + 1,
                            max_epoch,
                            IOU_score * 100,
                        )
                    )

            write_progress(
                progress_file,
                stage="epoch_done",
                epoch=epoch,
                batch=len(train_loader),
                total_batches=len(train_loader),
                max_epoch=max_epoch,
                AUC_cls=round(AUC_cls * 100, 4),
                MAP=round(MAP * 100, 4),
                IOU_score=round(IOU_score * 100, 4),
                F1_tok=round(F1_tok * 100, 4),
                best_IOU_score=round(best_iou_score * 100, 4),
                best_epoch=best_epoch + 1,
            )

        if config['schedular']['sched'] != 'cosine_in_step':
            lr_scheduler.step(epoch+warmup_steps+1)  
        if args.distributed:
            dist.barrier() 

    if utils.is_main_process():
        save_obj['best_iou_score'] = best_iou_score
        save_obj['best_epoch'] = best_epoch
        torch.save(save_obj, os.path.join(log_dir, 'checkpoint_%02d.pth'%epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.log:
        print('Training time {}'.format(total_time_str))
        logger.info(f'Training time {total_time_str}')
    write_progress(progress_file, stage="finished", max_epoch=max_epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train.yaml')
    parser.add_argument('--checkpoint', default=False) 
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--text_encoder', default='/root/autodl-tmp/model/roberta-base')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='world size for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23459', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none',
                        help='job launcher')
    parser.add_argument('--log_num', default='new', type=str)
    parser.add_argument('--train_scope', choices=['all', 'freq_image_heads', 'freq_only', 'freq_iou', 'freq_aux_iou'], default='all')
    parser.add_argument('--load_meter_only', action='store_true')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # main(args, config)
    if args.launcher == 'none':
        args.launcher = 'pytorch'
        main_worker(0, args, config)
    else:
        ngpus_per_node = torch.cuda.device_count()
        args.ngpus_per_node = ngpus_per_node
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args, config))
