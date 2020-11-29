import argparse

import torch
import numpy as np
import os
from tensorboardX import SummaryWriter

from utils.dataloader import IEMOCAPDataset, get_IEMO_dataloaders
from utils.train_eval_model import EarlyStopping, Train_Eval_Model


def init_seed(seed=6):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            '[Error] Unsupported value encountered.')


def main(args):

    init_seed()

    epoch_num = args.epoch_num
    class_num = args.class_num
    patience = args.patience
    batch_size = args.batch_size
    lr = args.lr
    lr_scheduler = args.lr_scheduler
    weighted_loss = args.weighted_loss
    optimizer = args.optimizer

    gate_his = args.gate_his
    stream = args.stream
    print('[Info] Stream:', stream)

    Model = import_class('models.' + args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[Info] Device: {}'.format(device))

    graph_args = {
        'strategy': args.graph_strategy,
        'max_dis_connect': args.max_dis_connect
    }

    if (stream == 'joint') or (stream == 'bone'):
        index = [0]
    elif stream == '2s':
        index = [0, 1]
    else:
        raise ValueError('[Error] Stream not existing.')

    kw = {
        'attbranch': args.attbranch,
        'gate': args.gate,
        'n_head': args.n_head,
        'd_kc': args.d_kc,
        'd_vc': args.d_vc
    }
    model = Model(3, class_num, graph_args, **kw)

    model_name = model.get_model_name()

    model_path = os.path.join('results', model_name)

    his_path = os.path.join(model_path, model_name + '0')
    i = 0
    while True:
        if os.path.exists(his_path):
            i = i + 1
            his_path = os.path.join(model_path, model_name + str(i))
        else:
            os.makedirs(his_path)
            print('[Info] History path is {}'.format(his_path))
            break
    best_model_path = os.path.join(his_path, 'model.pkl')

    earlystop = EarlyStopping(patience=patience,
                              save_path=best_model_path,
                              descend_mode=False)
    writer = SummaryWriter(log_dir=his_path)

    dataset = import_class('utils.dataloader.' + args.dataset)
    train_loader, val_loader, test_loader = get_IEMO_dataloaders(
        dataset=dataset, batch_size=batch_size, stream=stream)

    if weighted_loss:
        loss_weights = torch.FloatTensor(
            [5492 / 1102, 5492 / 1606, 5492 / 1081, 5492 / 1703])
        lossfunc = torch.nn.CrossEntropyLoss(loss_weights.to(device))
    else:
        lossfunc = torch.nn.CrossEntropyLoss()

    train_eval = Train_Eval_Model(model,
                                  optim=optimizer,
                                  lr=lr,
                                  loss_func=lossfunc,
                                  device=device,
                                  lr_scheduler=lr_scheduler)
    _, _, _, _, _, _, labels, preds = train_eval.train_model(
        train_loader,
        val_loader,
        test_loader,
        index,
        epoch_num,
        earlystop=earlystop,
        writer=writer,
        gate_his=gate_his)


if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--epoch_num', type=int, default=500)
    p.add_argument('--class_num', type=int, default=4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_scheduler', type=str2bool, default=True)
    p.add_argument('--weighted_loss', type=str2bool, default=False)
    p.add_argument('--optimizer', type=str, default='adam')

    p.add_argument('--model', type=str, default='sstgcn.SSTGCN')
    p.add_argument('--dataset', type=str, default='IEMOCAPDataset')
    p.add_argument('--stream', type=str, default='joint')
    p.add_argument('--gate_his', type=str2bool, default=True)

    p.add_argument('--graph_strategy', type=str, default='spatial')
    p.add_argument('--max_dis_connect', type=int, default=1)

    p.add_argument('--attbranch', type=str2bool, default=True)
    p.add_argument('--gate', type=str2bool, default=True)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--d_kc', type=float, default=0.25)
    p.add_argument('--d_vc', type=float, default=0.25)

    p.add_argument('--comment', type=str, required=False)

    args = p.parse_args()

    main(args)