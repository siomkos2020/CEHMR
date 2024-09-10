#! usr/bin/python
import torch
import argparse
import numpy as np
import dill
import time
import random
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, AdamW
import os
import torch.nn.functional as F
import torch.nn as nn
from collections import defaultdict

import sys

sys.path.append("..")

from models.cehmr_net import CEHMRNet
from utils import llprint, multi_label_metric, ddi_rate_score, get_n_params

MODEL_NAME = 'HCEMRNet'

def set_random_seed(seed=1203):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed(1203)
def print_to_log(string, log_fp):
    llprint(string)
    print(string, file=log_fp)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', default=False, help="eval mode")
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help="model name")
    parser.add_argument('--ddi', action='store_true', default=False, help="using ddi")
    parser.add_argument('--voc_path', type=str, default='../data/voc_final.pkl', help='Vocab path to load.')
    parser.add_argument('--ddi_path', type=str, default='../data/ddi_A_final.pkl', help='DDI table to look up.')
    parser.add_argument('--upper_med_path', type=str, default='../data/hire_data/med_h.pkl', help='Vocab for upper level medications.')
    parser.add_argument('--data_dir', type=str, default='../data/hire_data/', help='Dataset dir.')
    parser.add_argument('--num_epoch', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Running device.')
    parser.add_argument('--weight_path', type=str, default='', help='Model weight path for predicting.')
    parser.add_argument('--TS_type', type=str, default='exp', help='Training scheduler type.')
    parser.add_argument('--CL_Train', action='store_true', default=False, help='Set true to start SGCL algorithm.')
    args = parser.parse_args()

    return args


## Scoring functions
def bootstrap_score(model, weight_path, data_path, voc_size, save_path, device=torch.device('cpu')):
    model.load_state_dict(torch.load(open(weight_path, 'rb'), map_location=device))
    train_data = dill.load(open(data_path, 'rb'))

    model.eval()
    new_data = []
    adm_ja_dist = []
    for step, input in enumerate(train_data):
        new_pat_data = []
        for adm_idx, adm in enumerate(input):
            target_output = model(input[:adm_idx + 1])
            ## Get level3 res
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1

            target_output1 = target_output[:, voc_size[3] + voc_size[4]:].detach().cpu().numpy()[0]
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            adm_ja = (y_gt_tmp * y_pred_tmp).sum() / (y_pred_tmp.sum() + y_gt_tmp.sum() - (y_gt_tmp * y_pred_tmp).sum())
            adm_ja_dist.append(adm_ja.item())
            new_pat_data.append(adm + [adm_ja.item()])
            llprint('\rMeasuring %d/%d, ADM: %.4f' % (step + 1, adm_idx + 1, adm_ja.item()))
        new_data.append(new_pat_data)

    dill.dump(new_data, open(save_path, 'wb'))


## Pacing functions
def pacing_function(current_epoch, mode='single_step'):
    if mode == 'single_step':
        if current_epoch < 7:
            return 0.55
        else:
            return 1
    elif mode == 'exp':
        return (np.exp(0.11 * current_epoch - 0.5))
    elif mode == 'linear':
        return 0.09 * current_epoch + 0.1


def eval(model, data_eval, voc_size, epoch, log_fp):
    # evaluate
    print('')
    model.eval()
    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    ja2, ja3 = [], []
    case_study = defaultdict(dict)
    med_cnt = 0
    visit_cnt = 0
    for step, input in enumerate(data_eval):
        # Level3
        y_gt = []
        y_pred = []
        y_pred_prob = []
        y_pred_label = []
        # Level 2
        y_gt2 = []
        y_pred2 = []
        y_pred_prob2 = []
        y_pred_label2 = []
        # Level 1
        y_gt3 = []
        y_pred3 = []
        y_pred_prob3 = []
        y_pred_label3 = []

        for adm_idx, adm in enumerate(input):
            target_output = model(input[:adm_idx + 1])
            # Get level3 res
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            target_output1 = target_output[:, voc_size[3] + voc_size[4]:].detach().cpu().numpy()[0]
            y_pred_prob.append(target_output1)
            y_pred_tmp = target_output1.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
            # Get level2 res
            y_gt_tmp2 = np.zeros(voc_size[3])
            y_gt_tmp2[adm[3]] = 1
            y_gt2.append(y_gt_tmp2)

            target_output2 = target_output[:, voc_size[4]:voc_size[3] + voc_size[4]].detach().cpu().numpy()[0]
            y_pred_prob2.append(target_output2)
            y_pred_tmp = target_output2.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred2.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label2.append(sorted(y_pred_label_tmp))
            ## Get level1 res
            y_gt_tmp3 = np.zeros(voc_size[4])
            y_gt_tmp3[adm[4]] = 1
            y_gt3.append(y_gt_tmp3)

            target_output3 = target_output[:, 0:voc_size[4]].detach().cpu().numpy()[0]
            y_pred_prob3.append(target_output3)
            y_pred_tmp = target_output3.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred3.append(y_pred_tmp)
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label3.append(sorted(y_pred_label_tmp))

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred),
                                                                                 np.array(y_pred_prob))
        case_study[adm_ja] = {'ja': adm_ja, 'patient': input, 'y_label': y_pred_label}
        adm_ja2, _, _, _, _ = multi_label_metric(np.array(y_gt2), np.array(y_pred2), np.array(y_pred_prob2))
        adm_ja3, _, _, _, _ = multi_label_metric(np.array(y_gt3), np.array(y_pred3), np.array(y_pred_prob3))

        ja.append(adm_ja)
        ja2.append(adm_ja2)
        ja3.append(adm_ja3)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rEval--Epoch: %d, Step: %d/%d' % (epoch, step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record)

    print_to_log(
        '\tDDI Rate: %.4f, JaL1: %.4f, JaL2:%.4f, JaL3: %.4f,  PRAUC: %.4f, AVG_PRC: %.4f, AVG_RECALL: %.4f, AVG_F1: %.4f AVG_CNT: %d\n' % (
            ddi_rate, np.mean(ja3), np.mean(ja2), np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r),
            np.mean(avg_f1), med_cnt / visit_cnt
        ), log_fp=log_fp)

    return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1)


def main():
    # Load settings
    args = get_args()
    voc = dill.load(open(args.voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    med_h1_voc, med_h2_voc = dill.load(open(args.upper_med_path, 'rb'))
    voc_size = (len(diag_voc.id2word), len(pro_voc.id2word), len(med_voc.id2word), len(med_h1_voc), len(med_h2_voc))
    tree_size = len(med_voc.id2word) + len(med_h1_voc) + len(med_h2_voc)

    # Create log file.
    model_name = args.model_name
    if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))
    log_path = os.path.join("saved", model_name,
                            '%s.log' % time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    log_fp = open(log_path, 'w')
    print_to_log(str(args) + '\n', log_fp)


    # Generate parent index for each medication code.
    tree_adj = np.zeros((tree_size, tree_size))
    tree_self_adj = np.zeros((tree_size, tree_size))
    for idx, med in med_voc.id2word.items():
        i = len(med_h1_voc) + len(med_h2_voc) + idx
        j1 = len(med_h2_voc) + med_h1_voc[med[:-1]]
        j2 = med_h2_voc[med[0]]
        tree_adj[i, j1] = 1
        tree_adj[j1, j2] = 1
        tree_self_adj[i, i] = 1
        tree_self_adj[j1, j1] = 1

    # Load data.
    data_train = dill.load(open(os.path.join(args.data_dir, 'data_train.pkl'), 'rb'))
    data_test = dill.load(open(os.path.join(args.data_dir, 'data_test.pkl'), 'rb'))
    data_eval = dill.load(open(os.path.join(args.data_dir, 'data_eval.pkl'), 'rb'))

    EPOCH = args.num_epoch
    LR = args.lr
    TEST = args.eval
    device = torch.device(args.device)

    model = CEHMRNet(
        vocab_size=voc_size,
        emb_dim=64,
        device=device
    )

    tree_adj = torch.FloatTensor(tree_adj).to(device)
    tree_self_adj = torch.FloatTensor(tree_self_adj).to(device)

    if TEST:
        model.load_state_dict(torch.load(open(args.weight_path, 'rb')))
    model.to(device=device)

    # Scoring process
    CL_TRAINING = args.CL_Train
    data_measured_path = os.path.join(args.data_dir, 'data_train_plus.pkl')
    if os.path.exists(data_measured_path):
        data_train = dill.load(open(data_measured_path, 'rb'))
    else:
        dm_weight_path = os.path.join("saved", 'pretrained', 'pretrained_model.model')
        if os.path.exists(dm_weight_path):
            try:
                print('Measuring Starts!')
                bootstrap_score(model, dm_weight_path, os.path.join(args.data_dir, 'data_train.pkl'),
                                voc_size, data_measured_path, device)
                print('Measuring Ends!')
                exit()
            except Exception:
                exit('Difficulty measuring wrong!')
        else:
            CL_TRAINING = not CL_TRAINING
            if not os.path.exists(os.path.join("saved", 'pretrained')):
                os.mkdir(os.path.join("saved", 'pretrained'))

    if CL_TRAINING:
        print_to_log('Traing mode: CL', log_fp)
    else:
        print_to_log('NO CL', log_fp)

    print_to_log('Number of learnable parameters: %d' % get_n_params(model.hmnc_f), log_fp)
    bce_loss = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    if TEST:
        ddi_rate_list = []
        ja_list = []
        prauc_list = []
        avg_p_list = []
        avg_r_list = []
        avg_f1_list = []
        sampled_ratio = int(0.8*len(data_eval))
        for _ in range(10):
            to_sample_ids = np.random.randint(len(data_eval), size=sampled_ratio)
            data_to_eval = [data_eval[idx] for idx in to_sample_ids]
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = \
                eval(model, data_to_eval, voc_size, 0, log_fp)
            print(ddi_rate, ja, prauc, avg_p, avg_r, avg_f1)
            ddi_rate_list.append(ddi_rate)
            ja_list.append(ja)
            prauc_list.append(prauc)
            avg_p_list.append(avg_p)
            avg_r_list.append(avg_r)
            avg_f1_list.append(avg_f1)
        print('-'*100)
        print(np.mean(ja_list), 'pm', np.std(ja_list))
        print(np.mean(prauc_list), 'pm', np.std(prauc_list))
        print(np.mean(avg_f1_list), 'pm', np.std(avg_f1_list))
        print(np.mean(ddi_rate_list), 'pm', np.std(ddi_rate_list))
    else:
        history = defaultdict(list)
        best_epoch = 0
        best_ja = 0
        loss_records = []
        for epoch in range(EPOCH):
            loss_record1 = []
            loss_record2 = []
            loss_record3 = []
            start_time = time.time()
            model.train()
            train_samp_cnt, total_samp_cnt = 0, 0
            for step, input in enumerate(data_train):
                for idx, adm in enumerate(input):
                    total_samp_cnt += 1
                    seq_input = input[:idx + 1]
                    loss_l1_target = np.zeros((1, len(med_h2_voc)))
                    loss_l2_target = np.zeros((1, len(med_h1_voc)))
                    loss_l3_target = np.zeros((1, voc_size[2]))

                    loss_l3_target[:, adm[2]] = 1
                    loss_l2_target[:, adm[3]] = 1
                    loss_l1_target[:, adm[4]] = 1

                    loss_global_target = np.concatenate([loss_l1_target, loss_l2_target, loss_l3_target], axis=1)

                    loss3_target = np.full((1, voc_size[2] + voc_size[3] + voc_size[4]), -1)

                    for idx, item in enumerate(adm[4]):
                        loss3_target[0][idx] = item

                    for idx, item in enumerate(adm[3]):
                        loss3_target[0][idx + len(adm[4])] = item + voc_size[4]

                    for idx, item in enumerate(adm[2]):
                        loss3_target[0][idx + len(adm[4]) + len(adm[3])] = item + voc_size[4] + voc_size[3]

                    target_output1 = model(seq_input)

                    # loss1 = F.binary_cross_entropy_with_logits(target_output1, torch.FloatTensor(loss_global_target).to(device))
                    mask_target = np.ones((target_output1.shape))
                    mask_target[:, voc_size[4]:voc_size[4]+voc_size[3]] = 0
                    mask_target = torch.FloatTensor(mask_target).to(device)
                    loss1 = bce_loss(target_output1, torch.FloatTensor(loss_global_target).to(device))

                    loss2 = -torch.matmul(tree_adj, target_output1.t()) + torch.matmul(tree_self_adj,
                                                                                       target_output1.t())
                    loss2 = 0.1 * (torch.relu(loss2) ** 2).mean()
                    loss3 = F.multilabel_margin_loss(target_output1, torch.LongTensor(loss3_target).to(device))

                    loss = 0.9 * loss1 + 0.01 * loss3 + loss2
                    #loss = 0.9 * loss1 + 0.01 * loss3

                    optimizer.zero_grad()
                    if CL_TRAINING:
                        current_w = adm[5]
                        if 1 - current_w <= pacing_function(epoch, mode=args.TS_type):
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            train_samp_cnt += 1
                    else:
                        loss_records.append(loss.item())
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        train_samp_cnt += 1

                    loss_record1.append(loss1.item())
                    loss_record2.append(loss2.item())
                    loss_record3.append(loss3.item())

                llprint('\rTrain--Epoch: %d, Step: %d/%d, Loss: %.4f, Cosistence: %.4f, Mul Loss: %.4f, Samp: %d' % (
                    epoch, step, len(data_train), np.array(loss_record1).mean().item(),
                    np.array(loss_record2).mean().item(), np.array(loss_record3).mean().item(),
                    train_samp_cnt))

            # Evaluating
            #eval(model, data_train, voc_size, epoch, log_fp)
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = eval(model, data_eval, voc_size, epoch, log_fp)

            history['ja'].append(ja)
            history['ddi_rate'].append(ddi_rate)
            history['avg_p'].append(avg_p)
            history['avg_r'].append(avg_r)
            history['avg_f1'].append(avg_f1)
            history['prauc'].append(prauc)

            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60
            print_to_log(
                '\tEpoch: %d, Loss1: %.4f, Consistence: %.4f,  One Epoch Time: %.2fm, Appro Left Time: %.2fh, Samp Ratio: %.4f\n' % (
                epoch,
                np.array(loss_record1).mean().item(),
                np.array(loss_record2).mean().item(),
                elapsed_time,
                elapsed_time * (EPOCH - epoch - 1) / 60,
                train_samp_cnt / total_samp_cnt), log_fp=log_fp)

            if CL_TRAINING:
                torch.save(model.state_dict(), open(
                    os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))
            else:
                if ja > best_ja:
                    torch.save(model.state_dict(), open(os.path.join('saved', model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, ja, ddi_rate)), 'wb'))

            if epoch != 0 and best_ja < ja:
                best_epoch = epoch
                best_ja = ja
                best_prauc = prauc
                best_f1 = avg_f1


        dill.dump(history, open(os.path.join('saved', model_name, 'history.pkl'), 'wb'))
        dill.dump(loss_records, open('loss_records.pkl', 'wb'))

        print_to_log('best_epoch at %d, Ja: %.4f, PRAUC: %.4f, AVG_F1: %.4f' % (best_epoch, best_ja, best_prauc, best_f1), log_fp)
        log_fp.close()


if __name__ == '__main__':
    main()
