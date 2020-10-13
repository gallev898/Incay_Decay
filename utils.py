from enum import Enum

import torch
import argparse
import torch.nn.functional as F

from models.frezze_layers import resnet18_sim_soft_freeze_layers
from models.embedding_resnet import resnet18_sim_soft


class PATHS(Enum):
    LOCAL_LOAD_PATH = '/Users/gallevshalev/Desktop/incay_decay_trained_models'
    SERVER_LOAD_PATH = '/yoav_stg/gshalev/gal/incay_decay'
    WANDB = '/yoav_stg/gshalev/wandb'
    MODEL_SAVE_DIR = '/yoav_stg/gshalev/gal/incay_decay/{}'
    COLLECTOR_SAVE_DIR = '/yoav_stg/gshalev/gal/incay_decay/{}'



def classification_layer(out, out_pre_norm, embeddings, bias, args):
    if args.normalize:
        normalized_embeddings = F.normalize(embeddings, p=2, dim=0)
        logits = out.matmul(normalized_embeddings) * args.scale + bias
    else:
        logits = out_pre_norm.matmul(embeddings) * args.scale + bias

    output = F.log_softmax(logits, dim=1)
    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

    return output, pred, logits


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--runname', type=str, default='noname')
    parser.add_argument('--train_dataset', type=str, default='cifar10')
    parser.add_argument('--exit_layer', type=str, default='layer4')

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--embedding_size', default=512, type=int)

    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--beta', default=0.0, type=float)
    parser.add_argument('--alpha', default=0.0, type=float)
    parser.add_argument('--log_interval', default=100, type=float)

    parser.add_argument('--wd', default=True, action='store_false')
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--collect_per_sample', default=False, action='store_true')
    parser.add_argument('--freeze', default=False, action='store_true')
    parser.add_argument('--fixed', default=False, action='store_true')

    args = parser.parse_args()

    return args


def load_trained_embeddings_and_bias(args, device):
    print('Load pretrained embeddings and bias from: {}'.format(args.load_path))
    embedding_path = '{}/{}/embedding'.format(args.load_path, args.runname)
    bias_path = '{}/{}/bias'.format(args.load_path, args.runname)
    embeddings = torch.load(embedding_path, map_location=device)
    bias = torch.load(bias_path, map_location=device)
    return embeddings, bias


def load_trained_model(args, device):
    model_path = '{}/{}/{}.ckpt'.format(args.load_path, args.runname, args.runname)
    model = get_model(args.embedding_size, device)
    torch_load = torch.load(model_path, map_location=device)
    print(
        'model loaded with epoch:{} loss:{} acc:{}'.format(torch_load['epoch'], torch_load['loss'], torch_load['acc']))
    model.load_state_dict(torch_load['model_state'])
    model.eval()

    return model


def get_device(args):
    return torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")


def get_model(EMBEDDING_SIZE, device, args):

    if args.freeze:
        model = resnet18_sim_soft_freeze_layers(EMBEDDING_SIZE)
    else:
        model = resnet18_sim_soft(EMBEDDING_SIZE)
    model.to(device)

    return model

# utils.py
