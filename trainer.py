import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util
import torch.nn as nn
from model.loss_ssim import SSIMLoss

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.G_lossfn_weight = args.G_lossfn_weight
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        self.E_decay = args.E_decay

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        # ----------------------------------------
        # define loss
        # ----------------------------------------
        G_lossfn_type = self.args.G_lossfn_type
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.args.G_lossfn_weight
        print('G_lossfn_weight')
        print(self.G_lossfn_weight)


    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_diffuison, losses_sr  = utility.AverageMeter(), utility.AverageMeter()

        i = 0
        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()  # b, n, c, h, w
            lr, b_kernels = degrade(hr)  # bn, c, h, w

            self.optimizer.zero_grad()

            timer.tic()
            # forward
            ## train stage1
            if epoch <= self.args.epochs_encoder:
                lr = lr[:,0,...]
                hr = hr[:,0,...]
                z, dwt2, sr = self.model((lr,hr,False))
                l_sr = self.loss(sr, hr)
                loss = l_sr
                losses_sr.update(l_sr.item())
            ## train stage2
            else:
                lr = lr[:,0,...]
                hr = hr[:,0,...]
                z, dwt2, sr = self.model((lr, hr,True))
                l_diffusion = self.loss(z, dwt2)
                l_sr = self.loss(sr, hr)
                loss = l_diffusion + l_sr * 0.01
                losses_diffuison.update(l_diffusion.item())
                losses_sr.update(l_sr.item())

            # backward
            loss.backward()
            self.optimizer.step()
            timer.hold()



            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [sr loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [diffusion loss:{:.6f}] [sr loss:{:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_diffuison.avg,losses_sr.avg,
                            timer.release(),
                        ))


        # save model
        if epoch >= 0:
            target = self.model.get_model()
            model_dict = target.state_dict()
            keys = list(model_dict.keys())
            for key in keys:
                if 'encoder' in key :
                    del model_dict[key]
            torch.save(
                model_dict,
                os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
            )
        return epoch

    def test(self, epoch):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    hr = hr.cuda()  # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, _ = degrade(hr, random=False)  # b, 1, c, h, w
                    timer_test.tic()
                    hr = hr[:, 0, ...]                  # b, c, h, w
                    lr = lr[:, 0, ...]
                    if self.args.data_test == 'FreeData':
                        sr = self.model((hr, hr,True))
                        sr = utility.quantize(sr, self.args.rgb_range)
                    else:
                        sr = self.model((lr, hr,True))
                        timer_test.hold()

                        sr = utility.quantize(sr, self.args.rgb_range)
                        hr = utility.quantize(hr, self.args.rgb_range)

                        psnr = utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        ssim = utility.calc_ssim(
                            sr, hr, scale,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_psnr += psnr
                        eval_ssim += ssim
                        print('Testing {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f};'.
                            format(filename[0], psnr, ssim))
                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale)
                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
        return eval_psnr / len(self.loader_test), eval_ssim / len(self.loader_test)

    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h // scale * scale), :int(w // scale * scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

