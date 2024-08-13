"""
Train and test
"""
from torchnet import meter
from tqdm import tqdm
import torch.nn as nn
import torch
from torchsummary import summary
import os
from torch.utils.data import DataLoader


# use models_deepvanet_paper for transformer_eeg and conv_ for face
# use models_ for transformer_eeg and transformer_face
from models_Copy_resmit_copy_backup import DeepVANetBio, DeepVANetVision, DeepVANet
from dataset import DEAP, MAHNOB, DEAPAll, MAHNOBAll
from utils import out_put

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

import torch.nn as nn

import torch
import torch.nn as nn


def custom_init_weights(model):
    # Conv layers
    if hasattr(model.features.rnn, 'conv_embedding'):

        torch.nn.init.xavier_uniform(model.features.rnn.conv_embedding.weight)

    if hasattr(model.features.rnn, 'conv_embedding2'):
        torch.nn.init.xavier_uniform(model.features.rnn.conv_embedding2.weight)

    if hasattr(model.features.rnn, 'lstm'):
        for name, param in model.features.rnn.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)  # Xavier uniform initialization for weights
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # Initialize biases to zero as before
                if 'forget' in name:  # Assuming forget gate bias customization
                    nn.init.constant_(
                        param[model.features.rnn.lstm.hidden_size:(2 * model.features.rnn.lstm.hidden_size)], 1.0)

    # Fully connected layers
    if hasattr(model.features, 'fc'):
        nn.init.xavier_normal_(model.features.fc.weight)
        nn.init.constant_(model.features.fc.bias, 0.01)

    # Classifier layers
    if hasattr(model, 'classifier'):
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.01)




def train(modal, dataset, subject, k, l, epoch, lr, batch_size, file_name, indices, face_feature_size=16, bio_feature_size=64, use_gpu=False, pretrain=True):
    '''
    Train and test the model. Output the results.
    :param modal: data modality
    :param dataset: used dataset
    :param subject: subject id
    :param k: kth fold
    :param l: emotional label
    :param epoch: the number of epoches
    :param lr: learn rate
    :param batch_size: training batach size
    :param file_name: result file name
    :param indices: a list of index of the dataset
    :param face_feature_size: face feature size
    :param bio_feature_size: bio-sensing feature size
    :param use_gpu: use gpu or not
    :param pretrain: use pretrained cnn nor not
    :return: the best test accuracy
    '''

    # Xavier/Glorot initialization

    if use_gpu:
        print("using GPU")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    directory = file_name.split('/')[-2]
    print(directory, "here", batch_size)

    # if not os.path.exists(f'./results/{dataset}/{modal}/'+directory):
    #     os.mkdir(f'./results/{dataset}/{modal}/'+directory)

    if not os.path.exists(f'./results/{dataset}/{modal}/s{subject}/'+directory):
        os.mkdir(f'./results/{dataset}/{modal}/s{subject}/'+directory)

    file_name = f'./results/{dataset}/{modal}/s{subject}/{directory}/' + directory

    if dataset == 'DEAP':
        ############## inter-subjects ##############
        if subject == 0:
            train_data = DEAPAll(modal=modal, k=k, kind='train', indices=indices, label=l)
            val_data = DEAPAll(modal=modal, k=k, kind='val', indices=indices, label=l)
        ############## per-subjects ##############
        else:
            train_data = DEAP(modal=modal,subject=subject,k=k,kind='train',indices=indices, label=l, type="MIT")
            val_data = DEAP(modal=modal,subject=subject,k=k,kind='val',indices=indices, label=l, type = "MIT")
        bio_input_size = 40
        peri_input_size = 8
    if dataset == 'MAHNOB':
        ############## inter-subjects  ##############
        if subject == 0:
            train_data = MAHNOBAll(modal=modal, k=k, kind='train', indices=indices, label=l)
            val_data = MAHNOBAll(modal=modal, k=k, kind='val', indices=indices, label=l)
        ############## per-subject #################
        else:
            train_data = MAHNOB(modal=modal,subject=subject,k=k,kind='train',indices=indices, label=l)
            val_data = MAHNOB(modal=modal,subject=subject,k=k,kind='val',indices=indices, label=l)
        bio_input_size = 38
        peri_input_size = 6

    # model
    if modal == 'face':

        model = DeepVANetVision(feature_size=face_feature_size,pretrain=pretrain).to(device)
    if modal == 'bio':
        model = DeepVANetBio(input_size=bio_input_size, feature_size=bio_feature_size).to(device)
    if modal == 'eeg':
        model = DeepVANetBio(input_size=32, feature_size=bio_feature_size).to(device)
    if modal == 'peri':
        model = DeepVANetBio(input_size=peri_input_size, feature_size=bio_feature_size).to(device)
    if modal == 'faceeeg':
        model = DeepVANet(bio_input_size=32, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)
    if modal == 'faceperi':
        model = DeepVANet(bio_input_size=peri_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)
    if modal == 'facebio':
        print("here. dont worry")
        model = DeepVANet(bio_input_size=bio_input_size, face_feature_size=face_feature_size, bio_feature_size=bio_feature_size, pretrain=pretrain).to(device)

    # to print the parameters of the model
    count_parameters(model)

    # model.apply(init_xavier_all)
    # model.apply(init_xavier_all)
    # Apply the custom initialization
    # custom_init_weights(model)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


    # criterion and optimizer
    criterion = torch.nn.BCELoss()
    # criterion = torch.nn.BCEWithLogitsLoss()
    lr = lr

    optimizer = torch.optim.SGD(model.parameters(),lr=lr, weight_decay=1e-5,momentum=0.9)

    # meters
    loss_meter = meter.AverageValueMeter()

    best_accuracy = 0
    best_epoch = 0


    # train
    for epoch in range(epoch):
        pred_label = []
        true_label = []

        skip = 0
        loss_meter.reset()

        for ii, (data,label) in enumerate(train_loader):


            # print(ii)

            # print(data[0].shape, data[1].shape)
            # train model
            if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
                input = (data[0].float().to(device), data[1].float().to(device))
            else:
                input = data.float().to(device)


            label = label.float().to(device)

            optimizer.zero_grad()
            pred = model(input).float()

            # debu

            pred = pred.view((-1))
            # try:

            # print(pred, label, ii)

            loss = criterion(pred, label)
            # print(loss, "loss")

            # print(pred, label)
            # print("het", loss)

            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            # meters update
            loss_meter.add(loss.item())

            # print(pred, "pred")
            pred = (pred >= 0.5).float().to(device).data
            pred_label.append(pred)
            true_label.append(label)
            #
            #
            # print(label, "label")
            # print(loss, "loss")


        pred_label = torch.cat(pred_label,0)
        true_label = torch.cat(true_label,0)

        # print(pred_label, "hey")
        # print(torch.sum(pred_label == 1).item(), "pred")
        # print(torch.sum(true_label == 1).item())
        # print(true_label, "hoya")

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_mean = param.grad.data.abs().mean()
        #         grad_std = param.grad.data.std()
        #         print(f"{name}: grad_mean={grad_mean:.5f}, grad_std={grad_std:.5f}")
        #         if grad_mean < 1e-4:
        #             print(f"Warning: Potential vanishing gradient in {name}")
        #         elif grad_mean > 10:
        #             print(f"Warning: Potential exploding gradient in {name}")
        #
        # #
        # print(
        #     "---------------------------------------------------------------------------------------------------------")

        train_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
        out_put('Epoch: ' + 'train' + str(epoch) + '| train accuracy: ' + str(train_accuracy.item()), file_name)

        val_accuracy = val(modal, model, val_loader, use_gpu)

        out_put('Epoch: ' + 'train' + str(epoch) + '| train loss: ' + str(loss_meter.value()[0]) +
              '| val accuracy: ' + str(val_accuracy.item()), file_name)

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            model.save(f"{file_name}_best.pth")

        print(f"epoch {epoch}: successful")

    model.save(f'{file_name}.pth')

    perf = f"best accuracy is {best_accuracy} in epoch {best_epoch}" + "\n"
    out_put(perf,file_name)

    return best_accuracy


@torch.no_grad()
def val(modal, model, dataloader, use_gpu):
    model.eval()
    if use_gpu:
      device = torch.device('cuda')
    else:
      device = torch.device('cpu')

    pred_label = []
    true_label = []

    for ii, (data, label) in enumerate(dataloader):
        if modal == 'faceeeg' or modal == 'faceperi' or modal == 'facebio':
            input = (data[0].float().to(device), data[1].float().to(device))
        else:
            input = data.float().to(device)
        label = label.to(device)
        pred = model(input).float()

        pred = (pred >= 0.5).float().to(device).data
        # print(pred)
        pred = pred.view(-1)
        # print(pred, "pred")
        # print(label, "label")
        pred_label.append(pred)
        true_label.append(label)


    pred_label = torch.cat(pred_label, 0)
    true_label = torch.cat(true_label, 0)

    # print(pred_label, "preder")
    # print(true_label, "trueer")

    val_accuracy = torch.sum(pred_label == true_label).type(torch.FloatTensor) / true_label.size(0)
    # print(val_accuracy, "val accuracy")
    model.train()

    return val_accuracy
