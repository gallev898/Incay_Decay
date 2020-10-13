import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from dataloader.cifar_edit import CIFAR10, CIFAR100


def cifar10_loader(run_local, batch_size, num_workers, edit):
    path = '/yoav_stg/gshalev/semantic_labeling/cifar10' if not run_local else '/Users/gallevshalev/Desktop/datasets/cifar10'

    # todo: better understand the transform operation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if edit:
        trainset = CIFAR10(root=path, train=True,
                           download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    if edit:
        testset = CIFAR10(root=path, train=False,
                          download=True, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                               download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes, testset


def cifar100_loader(run_local, batch_size, num_workers, edit):
    path = '/yoav_stg/gshalev/semantic_labeling/cifar100' if not run_local else '/Users/gallevshalev/Desktop/datasets/cifar100'
    # todo: better understand the transform operation
    transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if edit:
        trainset = CIFAR100(root=path, train=True,
                            download=True, transform=transform)
    else:

        trainset = torchvision.datasets.CIFAR100(root=path, train=True,
                                                 download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)

    if edit:
        testset = CIFAR100(root=path, train=False,
                           download=True, transform=transform_test)
    else:
        testset = torchvision.datasets.CIFAR100(root=path, train=False,
                                                download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = ('apple',  # id 0
               'aquarium_fish',
               'baby',
               'bear',
               'beaver',
               'bed',
               'bee',
               'beetle',
               'bicycle',
               'bottle',
               'bowl',
               'boy',
               'bridge',
               'bus',
               'butterfly',
               'camel',
               'can',
               'castle',
               'caterpillar',
               'cattle',
               'chair',
               'chimpanzee',
               'clock',
               'cloud',
               'cockroach',
               'couch',
               'crab',
               'crocodile',
               'cup',
               'dinosaur',
               'dolphin',
               'elephant',
               'flatfish',
               'forest',
               'fox',
               'girl',
               'hamster',
               'house',
               'kangaroo',
               'computer_keyboard',
               'lamp',
               'lawn_mower',
               'leopard',
               'lion',
               'lizard',
               'lobster',
               'man',
               'maple_tree',
               'motorcycle',
               'mountain',
               'mouse',
               'mushroom',
               'oak_tree',
               'orange',
               'orchid',
               'otter',
               'palm_tree',
               'pear',
               'pickup_truck',
               'pine_tree',
               'plain',
               'plate',
               'poppy',
               'porcupine',
               'possum',
               'rabbit',
               'raccoon',
               'ray',
               'road',
               'rocket',
               'rose',
               'sea',
               'seal',
               'shark',
               'shrew',
               'skunk',
               'skyscraper',
               'snail',
               'snake',
               'spider',
               'squirrel',
               'streetcar',
               'sunflower',
               'sweet_pepper',
               'table',
               'tank',
               'telephone',
               'television',
               'tiger',
               'tractor',
               'train',
               'trout',
               'tulip',
               'turtle',
               'wardrobe',
               'whale',
               'willow_tree',
               'wolf',
               'woman',
               'worm')

    return trainloader, testloader, classes, testset


def cifar10_1_loader(run_local, batch_size, num_workers):
    cifar10_gen_path = '/Users/gallevshalev/Desktop/datasets/cifar10.1' if run_local else '/yoav_stg/gshalev/semantic_labeling/cifar10.1'
    cifar_extention = np.load("{}/cifar10.1_v6_data.npy".format(cifar10_gen_path))
    cifar_extention_lbl = np.load("{}/cifar10.1_v6_labels.npy".format(cifar10_gen_path)).tolist()
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    data_loader = []
    with torch.no_grad():
        for idx, (img, lbl) in enumerate(zip(cifar_extention, cifar_extention_lbl)):
            img2 = transform_test(img).unsqueeze(0)
            data_loader.append((img2, torch.from_numpy(np.array(lbl))))
    return data_loader


def load(dataset, run_local, batch_size, num_workers, edit=False):
    if dataset == 'cifar10':
        return cifar10_loader(run_local, batch_size, num_workers, edit)
    elif dataset == 'cifar100':
        return cifar100_loader(run_local, batch_size, num_workers, edit)
    elif dataset == 'cifar10.1':
        return cifar10_1_loader(run_local, batch_size, num_workers)


# dataloader.py