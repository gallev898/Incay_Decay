from dataloader.dataloader import load
from main import run_epoch
from utils import *
import os
import torch

if __name__ == '__main__':
    args = get_args()
    if not args.batch_size == 1:
        raise Exception('--batch_size must be 1')
    args.load_path = PATHS.LOCAL_LOAD_PATH.value if args.run_local else PATHS.SERVER_LOAD_PATH.value
    device = get_device(args)

    model = load_trained_model(args, device)
    embeddings, bias = load_trained_embeddings_and_bias(args, device)

    print('Loading {} dataset...'.format(args.train_dataset))
    _, testloader, _, _ = load(args.train_dataset, args.run_local, args.batch_size, args.num_workers)

    loss_func = F.nll_loss

    avg_loss, acc, dataAggregator, collector = run_epoch(loader=testloader,
                                                         model=model,
                                                         loss_func=loss_func,
                                                         epoch=1,
                                                         device=device,
                                                         embeddings=embeddings,
                                                         bias=bias,
                                                         args=args,
                                                         train=False,
                                                         optimizer=None)

    torch.save(collector, os.path.join(PATHS.COLLECTOR_SAVE_DIR.value, 'test_collector'))
    print('Successfully saved test_collector to :{}'.format(PATHS.COLLECTOR_SAVE_DIR.value))
