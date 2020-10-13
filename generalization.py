from dataloader.dataloader import load
from utils import *
import torch
from tqdm import tqdm

print(torch.__version__)

if __name__ == '__main__':
    args = get_args()
    args.load_path = PATHS.LOCAL_LOAD_PATH.value if args.run_local else PATHS.SERVER_LOAD_PATH.value
    device = get_device(args)
    DATA_SET = 'cifar10.1'

    data_loader = load(DATA_SET, args.run_local, args.batch_size, args.num_workers)

    model = load_trained_model(args, device)
    embeddings, bias = load_trained_embeddings_and_bias(args, device)

    with torch.no_grad():
        acc = 0
        for idx, (img, lbl) in enumerate(tqdm(data_loader)):
            data, target = img.to(device), lbl.to(device)
            out, out_pre_norm = model(data)  # normalized f(x) ,f(x)
            output, pred, logits = classification_layer(out, out_pre_norm, embeddings, bias, args)  # after softmax
            if pred == lbl:
                acc += 1

        f = open('generalization_results.txt', 'a')
        f.write('Acc for: {} is : {}\n'.format(args.runname, acc / len(data_loader)))
        f.close()
