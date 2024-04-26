"""
Demo for per-subject experiment
"""
import train
from train import train
from decision_level_fusion import decision_fusion
import random
import argparse
import os

# deap subject_id: sample number
deap_indices_dict = {1: 2400,
             2: 2400,
             3: 2340,
             4: 2400,
             5: 2340,
             6: 2400,
             7: 2400,
             8: 2400,
             9: 2400,
             10: 2400,
             11: 2220,
             12: 2400,
             13: 2400,
             14: 2340,
             15: 2400,
             16: 2400,
             17: 2400,
             18: 2400,
             19: 2400,
             20: 2400,
             21: 2400,
             22: 2400,
             23: 2400,
             24: 2400,
             25: 2400,
             26: 2400,
             27: 2400,
             28: 2400,
             29: 2400,
             30: 2400,
             31: 2400,
             32: 2400}

seed_indices_dict = {
    1: 47001,
    2: 46601,
    3: 41201,
    4: 47601,
    5: 37001,
    6: 39001,
    7: 47401,
    8: 43201,
    9: 53001,
    10: 47401,
    11: 47001,
    12: 46601,
    13: 47001,
    14: 47601,
    15: 41201
}
# seed_indices_dict = {
#     1: 47000,
#     2: 46600,
#     3: 41200,
#     4: 47600,
#     5: 37000,
#     6: 3900,
#     7: 47400,
#     8: 43200,
#     9: 53000,
#     10: 47400,
#     11: 47000,
#     12: 46600,
#     13: 47000,
#     14: 47600,
#     15: 41200
# }

seed_labels = {
1:1,2:0,3:-1,4:-1,5:0,6:1,7:-1,8:0,9:1,10:1,11:0,12:-1,13:0,14:1,15:-1
}


# mahnob subject_id: sample number
mahnob_indices_dict = {1: 1611,
             2: 1611,
             3: 1305,
             4: 1611,
             5: 1611,
             6: 1611,
             7: 1611,
             8: 1611,
             9: 1124,
             10: 1611,
             11: 1611,
             13: 1611,
             14: 1611,
             16: 1370,
             17: 1611,
             18: 1611,
             19: 1611,
             20: 1611,
             21: 1611,
             22: 1611,
             23: 1611,
             24: 1611,
             25: 1611,
             27: 1611,
             28: 1611,
             29: 1611,
             30: 1611}

def demo():
    parser = argparse.ArgumentParser(description='Per-subject experiment')
    parser.add_argument('--dataset', '-d', default='SEED', help='The dataset used for evaluation', type=str)
    parser.add_argument('--fusion', default='feature', help='Fusion strategy (feature or decision)', type=str)
    parser.add_argument('--epoch', '-e', default=20, help='The number of epochs in training', type=int)
    parser.add_argument('--batch_size', '-b', default=64, help='The batch size used in training', type=int)
    parser.add_argument('--learn_rate', '-l', default=0.001, help='Learn rate in training', type=float)
    parser.add_argument('--gpu', '-g', default='True', help='Use gpu or not', type=str)
    # parser.add_argument('--file', '-f', default='./results/results.txt', help='File name to save the results', type=str)
    parser.add_argument('--modal', '-m', default='eeg', help='Type of data to train', type=str)
    parser.add_argument('--subject', '-s', default=1, help='Subject id', type=int)
    parser.add_argument('--face_feature_size', default=16, help='Face feature size', type=int)
    parser.add_argument('--bio_feature_size', default=64, help='Bio feature size', type=int)
    parser.add_argument('--label', default='valence', help='Valence or arousal', type=str)
    parser.add_argument('--pretrain',default='True', help='Use pretrained CNN', type=str)

    args = parser.parse_args()

    use_gpu = True if args.gpu == 'True' else False
    pretrain = True if args.pretrain == 'True' else False



    if not os.path.exists(f'./results/'):
        os.mkdir(f'./results/')
    if not os.path.exists(f'./results/{args.dataset}/'):
        os.mkdir(f'./results/{args.dataset}/')
    if not os.path.exists(f'./results/{args.dataset}/{args.modal}/'):
        os.mkdir(f'./results/{args.dataset}/{args.modal}/')

    for subject in range(1,33):
        if args.dataset == 'DEAP':
            indices = list(range(deap_indices_dict[subject]))
        if args.dataset == 'MAHNOB':
            indices = list(range(mahnob_indices_dict[subject]))
        if args.dataset == "SEED":
            indices = list(range(10182))
            # args.label = seed_indices_dict
            # args.label = seed_indices_dict
        # shuffle the dataset
        random.shuffle(indices)
        subject = 15
        # we are running the for loop for 10 times, why ?
        for k in range(1, 11):
            if args.fusion == 'feature':
                train(modal=args.modal, dataset=args.dataset, epoch=args.epoch, lr=args.learn_rate, use_gpu=use_gpu,
                            file_name=f'./results/{args.dataset}/{args.modal}/{args.dataset}_{args.modal}_{args.label}_s{subject}_k{k}_{args.face_feature_size}_{args.bio_feature_size}/{args.dataset}_{args.modal}_{args.label}_s{args.subject}_k{k}_{args.face_feature_size}_{args.bio_feature_size}',
                            batch_size=args.batch_size, subject=subject, k=k, l=args.label, indices=indices,
                            face_feature_size=args.face_feature_size, bio_feature_size=args.bio_feature_size, pretrain=pretrain)
            if args.fusion == 'decision':
                decision_fusion(args.dataset, args.modal, args.subject, k, args.label, indices, use_gpu, pretrain)
        break

if __name__ == '__main__':
    demo()