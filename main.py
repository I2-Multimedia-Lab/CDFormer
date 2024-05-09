from option import args
import torch
import utility
import data
import model
import loss
from trainer import Trainer
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count
def calc_params(model, res=False):
    from thop import profile
    from thop import clever_format

    inp = torch.randn(1, 1, 3, 192, 192).cuda()
    macs, params = profile(model.cuda(), inputs=inp)
    macs, params = clever_format([macs, params], "%.3f")
    print(f'Params(M): {params}, FLOPs(G): {macs}')
if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            epoch = t.train()
        checkpoint.done()
