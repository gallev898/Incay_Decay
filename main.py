import os
import wandb
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from utils import *
from DataAggregator import DataAggregator
from dataloader.dataloader import load
from torch.optim.lr_scheduler import ReduceLROnPlateau


# wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca


class KEYS(Enum):
    correctly_norm_through_epoch = 'correctly_norm_through_epoch'
    wrongly_norm_through_epoch = 'wrongly_norm_through_epoch'
    correct_logits_through_epoch = 'correct_logits_through_epoch'
    wrong_logits_through_epoch = 'wrong_logits_through_epoch'
    correct_cosine_through_epoch = 'correct_cosine_through_epoch'
    wrong_cosine_through_epoch = 'wrong_cosine_through_epoch'
    correct_conf_through_epoch = 'correct_conf_through_epoch'
    wrong_conf_through_epoch = 'wrong_conf_through_epoch'
    min_correct_through_epoch = 'min_correct_through_epoch'
    min_wrong_through_epoch = 'min_wrong_through_epoch'
    aggregate_incay_loss = 'aggregate_incay_loss'
    aggregate_decay_loss = 'aggregate_decay_loss'
    aggregated_ce_loss = 'aggregated_ce_loss'


class PER_SAMPLE_KEYS(Enum):
    pred = 'pred'
    logit = 'logit'
    cosine = 'cosine'
    norm = 'norm'
    conf = 'conf'


def get_embeddings(embedding_size, fixed):
    word2vec_dictionary = dict()

    for cls_idx in range(NUM_OF_CLASSES):
        v = np.random.randint(low=-100, high=100, size=embedding_size)
        v = v / np.linalg.norm(v)
        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    bias = torch.ones(NUM_OF_CLASSES) * 0.01

    w2v_matrix = w2v_matrix.clone().to(device).requires_grad_(True)
    bias = bias.clone().to(device).requires_grad_(True)

    if fixed:
        print('Fixed embeddings- NOTICE!')
        w2v_matrix.requires_grad = False
        bias.requires_grad = False

    return w2v_matrix, bias


def initialize_wandb(model):
    if not args.run_local:
        wandb.init(project="incay_decay", name=args.runname, dir=PATHS.WANDB.value)
        wandb.watch(model)
        wandb.run.save()


def get_optimizer(args, embeddings, bias):
    if args.wd and args.normalize:  # TODO: WTF?!
        raise Exception('Do NOT use wd when training cosine (args.normalize = True & args.wd= True)')

    if args.wd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    if not args.fixed:
        optimizer.add_param_group({'params': embeddings})
        optimizer.add_param_group({'params': bias})
    return optimizer


def collect_representations_norm(target, pred, out_pre_norm, collector, device, per_sample_collector, samples_names):
    if per_sample_collector is not None:
        for o, s_n in zip(out_pre_norm, samples_names):
            per_sample_collector[s_n][PER_SAMPLE_KEYS.norm.value].append(torch.norm(o, p=2).item())

    # create masks
    correct_mask_vec = pred.eq(target.view_as(pred)).float().to(device)
    ones = torch.ones(correct_mask_vec.shape).to(device)
    wrong_mask_vec = ones - correct_mask_vec

    # separate correct from wrong for norm calc
    correctly_classified = torch.mul(out_pre_norm, correct_mask_vec)
    wrongly_classified = torch.mul(out_pre_norm, wrong_mask_vec)
    assert torch.all(torch.eq(correctly_classified + wrongly_classified, out_pre_norm))

    # calc norm
    correctly_classified_norm = torch.norm(correctly_classified, p=2, dim=1)
    wrongly_classified_norm = torch.norm(wrongly_classified, p=2, dim=1)

    # remove zeros
    correctly_classified_norm = correctly_classified_norm[correctly_classified_norm > 0]
    wrongly_classified_norm = wrongly_classified_norm[wrongly_classified_norm > 0]

    collector[KEYS.correctly_norm_through_epoch].append(correctly_classified_norm.mean().item())
    collector[KEYS.wrongly_norm_through_epoch].append(wrongly_classified_norm.mean().item() if
                                                      wrongly_classified_norm.nelement() != 0 else 0)
    return correctly_classified_norm, wrongly_classified_norm


def incay_c_decay_w_loss(args, correctly_classified_norm, wrongly_classified_norm, collector):
    if correctly_classified_norm.nelement() != 0:
        incay_loss = torch.div(torch.ones(correctly_classified_norm.shape).to(device), correctly_classified_norm).mean()
    else:
        incay_loss = torch.zeros(1).to(device)

    decay_loss = wrongly_classified_norm.mean() if wrongly_classified_norm.nelement() != 0 else torch.zeros(1).to(
        device)

    loss = incay_loss * args.alpha + decay_loss * args.beta

    collector[KEYS.aggregate_incay_loss] += incay_loss.item()
    collector[KEYS.aggregate_decay_loss] += decay_loss.item()

    return loss


def incay_w_decay_c_loss(args, correctly_classified_norm, wrongly_classified_norm, collector):
    if wrongly_classified_norm.nelement() != 0:
        incay_loss = torch.div(torch.ones(wrongly_classified_norm.shape).to(device), wrongly_classified_norm).mean()
    else:
        incay_loss = torch.zeros(1).to(device)

    decay_loss = correctly_classified_norm.mean() if correctly_classified_norm.nelement() != 0 else torch.zeros(1).to(
        device)

    loss = incay_loss * args.alpha + decay_loss * args.beta

    collector[KEYS.aggregate_incay_loss] += incay_loss.item()
    collector[KEYS.aggregate_decay_loss] += decay_loss.item()

    return loss


def collect_correctly_and_wrongly_logits(logits, pred, target, collector, per_sample_collector, samples_names):
    correctly, wrongly = [], []
    min_correct, min_wrong = [], []

    append = lambda key, var: collector[key].append(np.average(var))

    for l, p, t, s_n in zip(logits, pred, target, samples_names):
        logit = l[p.item()].item()
        if per_sample_collector is not None:
            per_sample_collector[s_n][PER_SAMPLE_KEYS.logit.value].append(logit)
        if p.item() == t.item():  # if correctly classified
            correctly.append(logit)
            min_correct.append(min(l).item())
        else:  # if wrongly classified
            wrongly.append(logit)
            min_wrong.append(min(l).item())

    if len(correctly) > 0:
        append(KEYS.correct_logits_through_epoch, correctly)
    if len(wrongly) > 0:
        append(KEYS.wrong_logits_through_epoch, wrongly)
    if len(min_correct) > 0:
        append(KEYS.min_correct_through_epoch, min_correct)
    if len(min_wrong) > 0:
        append(KEYS.min_wrong_through_epoch, min_wrong)


def cosine_similarity(o, p):
    o = o.detach().cpu().numpy()
    p = p.detach().cpu().numpy()

    return np.dot(o, p) / (np.linalg.norm(o) * np.linalg.norm(p))


def collect_cosine_similarity(embeddings, output, acc_vec, pred, collector, per_sample_collector, samples_names):
    for o, a, p, s_n in zip(output, acc_vec, pred, samples_names):
        cosA = cosine_similarity(o, embeddings.t()[p].squeeze(0))
        if per_sample_collector is not None:
            per_sample_collector[s_n][PER_SAMPLE_KEYS.cosine.value].append(cosA)
        if a:
            collector[KEYS.correct_cosine_through_epoch].append(cosA)
        else:
            collector[KEYS.wrong_cosine_through_epoch].append(cosA)


def get_collector():
    collector = {
        KEYS.correctly_norm_through_epoch: [], KEYS.wrongly_norm_through_epoch: [],
        KEYS.correct_conf_through_epoch: [], KEYS.wrong_conf_through_epoch: [],
        KEYS.correct_logits_through_epoch: [], KEYS.wrong_logits_through_epoch: [],
        KEYS.correct_cosine_through_epoch: [], KEYS.wrong_cosine_through_epoch: [],
        KEYS.min_correct_through_epoch: [], KEYS.min_wrong_through_epoch: [],
        KEYS.aggregate_incay_loss: 0,
        KEYS.aggregate_decay_loss: 0,
        KEYS.aggregated_ce_loss: 0
    }
    return collector


def fill_dataAggregator(collector, batch_idx):
    dataAggregator = DataAggregator()

    dataAggregator.avg_cor_norm = np.average(collector[KEYS.correctly_norm_through_epoch])
    dataAggregator.avg_wrong_norm = np.average(collector[KEYS.wrongly_norm_through_epoch])

    dataAggregator.avg_correct_logits = np.average(collector[KEYS.correct_logits_through_epoch])
    dataAggregator.avg_wrong_logits = np.average(collector[KEYS.wrong_logits_through_epoch])

    dataAggregator.avg_correct_cosine = np.average(collector[KEYS.correct_cosine_through_epoch])
    dataAggregator.avg_wrong_cosine = np.average(collector[KEYS.wrong_cosine_through_epoch])

    dataAggregator.avg_correct_conf = np.average(collector[KEYS.correct_conf_through_epoch])
    dataAggregator.avg_wrong_conf = np.average(collector[KEYS.wrong_conf_through_epoch])

    dataAggregator.avg_min_correct_logits = np.average(collector[KEYS.min_correct_through_epoch])
    dataAggregator.avg_min_wrong_logits = np.average(collector[KEYS.min_wrong_through_epoch])

    dataAggregator.avg_ce_loss = collector[KEYS.aggregated_ce_loss] / batch_idx
    dataAggregator.avg_incay_loss = collector[KEYS.aggregate_incay_loss] / batch_idx
    dataAggregator.avg_decay_loss = collector[KEYS.aggregate_decay_loss] / batch_idx

    return dataAggregator


def get_added_loss(args, correctly_classified_norm, wrongly_classified_norm, collector):
    added_loss = 0

    if args.loss == 'incay_c_decay_w':
        added_loss = incay_c_decay_w_loss(args, correctly_classified_norm, wrongly_classified_norm, collector)
    elif args.loss == 'incay_w_decay_c':
        added_loss = incay_w_decay_c_loss(args, correctly_classified_norm, wrongly_classified_norm, collector)

    return added_loss


def collect_conf(conf, pred, target, collector, per_sample_collector, samples_names):
    correct_conf, wrong_conf = [], []

    for c, p, t, s_n in zip(conf, pred, target, samples_names):
        confidence = c[p.item()]
        confidence_for_target = c[t.item()]
        if per_sample_collector is not None:
            per_sample_collector[s_n][PER_SAMPLE_KEYS.conf.value].append(confidence_for_target)
        if p.item() == t.item():
            correct_conf.append(confidence)
        else:
            wrong_conf.append(confidence)

    if len(correct_conf) > 0:
        collector[KEYS.correct_conf_through_epoch].append(np.average(correct_conf))
    if len(wrong_conf) > 0:
        collector[KEYS.wrong_conf_through_epoch].append(np.average(wrong_conf))


def collect_acc_per_sample(acc_vec, per_sample_collector, samples_names):
    for s_name, acc in zip(samples_names, acc_vec):
        per_sample_collector[s_name][PER_SAMPLE_KEYS.pred.value].append(acc.item())


def run_epoch(loader, model, loss_func, epoch, device, embeddings, bias, args, train=True, optimizer=None,
              per_sample_collector=None):
    aggregate_loss, correct_smpl_cnt = 0, 0

    collector = get_collector()

    for batch_idx, (data, target, samples_names) in tqdm(enumerate(loader)):
        # if batch_idx ==1:
        #     break
        if train:
            optimizer.zero_grad()

        data, target = data.to(device), target.to(device)

        # forward
        if args.freeze:
            out, out_pre_norm = model(data, args.exit_layer)  # normalized f(x) ,f(x)
        else:
            out, out_pre_norm = model(data)  # normalized f(x) ,f(x)

        output, pred, logits = classification_layer(out, out_pre_norm, embeddings, bias, args)  # after softmax

        # acc
        acc_vec = pred.eq(target.view_as(pred)).squeeze(1)
        correct_smpl_cnt += sum(acc_vec).item()

        if per_sample_collector is not None:
            collect_acc_per_sample(acc_vec, per_sample_collector, samples_names)

        # Collect: logits, cosine, norms, conf
        collect_correctly_and_wrongly_logits(logits, pred, target, collector, per_sample_collector, samples_names)
        collect_cosine_similarity(embeddings, out_pre_norm, acc_vec, pred, collector, per_sample_collector,
                                  samples_names)
        collect_conf(np.exp(output.detach().cpu().numpy()), pred, target, collector, per_sample_collector,
                     samples_names)
        correctly_classified_norm, wrongly_classified_norm = collect_representations_norm(target, pred,
                                                                                          out_pre_norm, collector,
                                                                                          device, per_sample_collector,
                                                                                          samples_names)

        # loss
        ce_loss = loss_func(output, target)
        collector[KEYS.aggregated_ce_loss] += ce_loss.item()

        added_loss = get_added_loss(args, correctly_classified_norm, wrongly_classified_norm, collector)
        loss = ce_loss + added_loss

        if train:
            loss.backward()
            optimizer.step()

        aggregate_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format('Train' if train else 'Test',
                                                                          epoch,
                                                                          batch_idx * len(data),
                                                                          len(loader.dataset),
                                                                          100. * batch_idx / len(loader),
                                                                          loss.item()
                                                                          ))

    dataAggregator = fill_dataAggregator(collector, batch_idx)
    avg_loss = aggregate_loss / batch_idx
    acc = (100 * correct_smpl_cnt / len(loader.dataset))

    return avg_loss, acc, dataAggregator, collector


def set_exit_layer_strategy(args, epoch):
    last_layer = args.exit_layer
    if epoch <= 10:
        args.exit_layer = 'layer1'
    elif epoch <= 20:
        args.exit_layer = 'layer2'
    elif epoch <= 30:
        args.exit_layer = 'layer3'
    else:
        args.exit_layer = 'layer4'

    if last_layer != args.exit_layer:
        print('************************************************************')
        print('Updated exit_layer to: {}'.format(args.exit_layer))
        print('************************************************************')




def iteration(epoch, model, loss_func, device, loader, optimizer, args, embeddings, bias, train=True,
              per_sample_collector=None):
    if train:
        model.train()
        set_exit_layer_strategy(args, epoch)
    else:
        model.eval()

    return run_epoch(loader=loader,
                     model=model,
                     loss_func=loss_func,
                     epoch=epoch,
                     device=device,
                     embeddings=embeddings,
                     bias=bias,
                     args=args,
                     train=train,
                     optimizer=optimizer,
                     per_sample_collector=per_sample_collector)


def save_model(path, runname, model, epoch, loss, acc):
    ckpt_dir = os.path.join(path, '{}.ckpt'.format(runname))

    print('Saving the model to {}'.format(path))

    torch.save(dict(model_state=model.state_dict(), epoch=epoch, loss=loss, acc=acc), ckpt_dir)


def get_save_dir(args):
    save_dir = "." if args.run_local else PATHS.MODEL_SAVE_DIR.value.format(args.runname)
    if not os.path.exists(save_dir) and not args.run_local:
        os.mkdir(save_dir)

    return save_dir


def get_data_loader(args):
    return load(args.train_dataset, args.run_local, args.batch_size, args.num_workers, edit=True)[:2]


def get_sample_collector(collect_per_sample, train_loader):
    per_sample_collector = None

    if collect_per_sample:
        if args.run_local: # NOTICE  if args.run_local:
            print('loading collector')
            per_sample_collector = torch.load('per_sample_collector')
        else:
            print('Collecting data per sample ')
            per_sample_collector = dict()

            for batch_idx, (data, target, samples_names) in tqdm(enumerate(train_loader)):
                for i in samples_names:
                    per_sample_collector[i] = dict()
                    per_sample_collector[i][PER_SAMPLE_KEYS.pred.value] = []
                    per_sample_collector[i][PER_SAMPLE_KEYS.logit.value] = []
                    per_sample_collector[i][PER_SAMPLE_KEYS.conf.value] = []
                    per_sample_collector[i][PER_SAMPLE_KEYS.norm.value] = []
                    per_sample_collector[i][PER_SAMPLE_KEYS.cosine.value] = []

        assert len(per_sample_collector) == 49999, 'len(per_sample_collector) = {}'.format(len(per_sample_collector))
    return per_sample_collector


def dummy_run(loader, model, device):
    while True:
        for batch_idx, (data, target, samples_names) in tqdm(enumerate(loader)):
            data, target = data.to(device), target.to(device)
            model(data)
        print('Dummy print \n¯\_༼ᴼل͜ᴼ༽_/¯\n')


if __name__ == '__main__':

    args = get_args()

    save_dir = get_save_dir(args)
    device = get_device(args)

    if args.train_dataset == 'cifar10':
        NUM_OF_CLASSES = 10
    elif args.train_dataset == 'cifar100':
        NUM_OF_CLASSES = 100
    else:
        raise Exception('Number of classes is undefined')

    EPOCHS = args.epochs
    EMBEDDING_SIZE = args.embedding_size

    model = get_model(EMBEDDING_SIZE, device, args)
    embeddings, bias = get_embeddings(EMBEDDING_SIZE, args.fixed)

    initialize_wandb(model)

    loss_func = F.nll_loss
    optimizer = get_optimizer(args, embeddings, bias)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3,
                                  verbose=True, threshold_mode='abs')

    train_loader, test_loader = get_data_loader(args)

    per_sample_collector = get_sample_collector(args.collect_per_sample, train_loader)

    best_acc = 0
    for epoch in range(1, EPOCHS + 1):

        print('Start Train')
        loss_sum, acc, train_data, _ = iteration(epoch, model, loss_func, device, train_loader, optimizer, args,
                                                 embeddings, bias)
        with torch.no_grad():
            print('Evaluating on the train set...')
            train_loss, train_acc, train_val_data, _ = iteration(epoch, model, loss_func, device,
                                                                 train_loader,
                                                                 optimizer, args, embeddings, bias, train=False,
                                                                 per_sample_collector=per_sample_collector)
            print('Train Avg Loss: {}   Accuracy: {}'.format(train_loss, train_acc))

            print('Evaluating on the test set...')
            test_loss, test_acc, test_val_data, _ = iteration(epoch, model, loss_func, device, test_loader,
                                                              optimizer, args, embeddings, bias, train=False)
            print('Test Avg Loss: {}   Accuracy: {}'.format(test_loss, test_acc))

        if best_acc < test_acc:
            print('Saving new best model')
            save_model(save_dir, args.runname, model, epoch, test_loss, test_acc)
            best_acc = test_acc
            if not args.run_local:
                torch.save(embeddings, os.path.join(save_dir, 'embedding'))
                torch.save(bias, os.path.join(save_dir, 'bias'))

        wandb_dic = {'train_loss': train_loss,
                     'train_acc': train_acc,
                     'train_avg_cor_norm': train_val_data.avg_cor_norm,
                     'train_avg_wrong_norm': train_val_data.avg_wrong_norm,
                     'train_avg_correct_logits': train_val_data.avg_correct_logits,
                     'train_avg_wrong_logits': train_val_data.avg_wrong_logits,
                     'train_avg_correct_cosine': train_val_data.avg_correct_cosine,
                     'train_avg_wrong_cosine': train_val_data.avg_wrong_cosine,
                     'train_avg_min_correct_logits': train_val_data.avg_min_correct_logits,
                     'train_avg_min_wrong_logits': train_val_data.avg_min_wrong_logits,
                     'train_avg_incay_loss': train_val_data.avg_incay_loss,
                     'train_avg_decay_loss': train_val_data.avg_decay_loss,
                     'train_avg_ce_loss': train_val_data.avg_ce_loss,

                     'test_loss': test_loss,
                     'test_acc': test_acc,
                     'test_avg_cor_norm': test_val_data.avg_cor_norm,
                     'test_avg_wrong_norm': test_val_data.avg_wrong_norm,
                     'test_avg_correct_logits': test_val_data.avg_correct_logits,
                     'test_avg_wrong_logits': test_val_data.avg_wrong_logits,
                     'test_avg_correct_cosine': test_val_data.avg_correct_cosine,
                     'test_avg_wrong_cosine': test_val_data.avg_wrong_cosine,
                     'test_avg_min_correct_logits': test_val_data.avg_min_correct_logits,
                     'test_avg_min_wrong_logits': test_val_data.avg_min_wrong_logits,
                     'test_avg_incay_loss': test_val_data.avg_incay_loss,
                     'test_avg_decay_loss': test_val_data.avg_decay_loss,
                     'test_avg_ce_loss': test_val_data.avg_ce_loss,
                     }

        if not args.run_local:
            wandb.log(wandb_dic)
        scheduler.step(test_acc)

    torch.save(per_sample_collector, os.path.join(save_dir, 'per_sample_collector'))
    print('Successfully saved per_sample_collector to : {}'.format(save_dir))
    print('**************************************************************************\n'
          'End Of Run \n'
          ' (｡♡‿♡｡) \n'
          '**************************************************************************\n')
    dummy_run(train_loader, model, device)

# main.py
