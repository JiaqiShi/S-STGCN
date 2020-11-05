import argparse
import torch

def init_seed(seed):
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

def main(args):

    epoch_num = args.epoch_num
    class_num = args.class_num
    patience = args.patience
    batch_size = args.batch_size
    lr = args.lr
    gate_his = args.gate_his
    stream = args.stream
    print('[Info] Stream:',stream)

    Model = import_class('models.'+args.model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('[Info] Device: {}'.format(device))

    graph_args = {'strategy':args.graph_strategy,'max_dis_connect':args.max_dis_connect}

    if (stream == 'joint') or (stream == 'bone'):
        index = [0]
    elif stream == '2s':
        index = [0,1]
    else:
        raise ValueError('[Error] Stream not existing.')
    
    kw = {'attbranch':}
    model = Model(3,class_num,graph_args)

if __name__ == '__main__':

    p = argparse.ArgumentParser()

    p.add_argument('--epoch_num', type=int, default=200)
    p.add_argument('--class_num', type=int, default=4)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)

    # p.add_argument('--model', type=str, default='st_gcn.Multi_stream')
    p.add_argument('--model', type=str, default='SSTGCN')
    p.add_argument('--dataset', type=str, default='IEMOCAPDataset')
    p.add_argument('--motion_extend', type=str2bool, default=False)
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