import argparse
import torch

def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def main(args):

    epoch_num = args.epoch_num
    class_num = args.class_num
    patience = args.patience
    batch_size = args.batch_size
    lr = args.lr
    gate_his = args.gate_his

    Model = import_class('models.'+args.model)

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--epoch_num', type=int, default=200)
    p.add_argument('--class_num', type=int, default=4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)

    # p.add_argument('--model', type=str, default='st_gcn.Multi_stream')
    p.add_argument('--model', type=str, default='att_st_gcn.Model')
    p.add_argument('--dataset', type=str, default='IEMOCAPDataset')
    p.add_argument('--motion_extend', type=str2bool, default=False)
    p.add_argument('--stream', type=str, default='Joint')
    p.add_argument('--gate_his', type=str2bool, default=True)

    p.add_argument('--graph', type=str, default='model.st_gcn.Graph')

    p.add_argument('--comment', type=str, required=False)

    args = p.parse_args()

    main(args)