import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer
from utils import util


class DefaultTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config, model, loss, metrics, optimizer, lr_scheduler, resume, data_loader,
                 valid_data_loader=None, train_logger=None, **extra_args):
        super(DefaultTrainer, self).__init__(config, model, loss, metrics, optimizer, lr_scheduler, resume,
                                             train_logger)

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self._load_pretrained_weights(**extra_args)  # load the pretrained weights of each phase

    def _load_pretrained_weights(self, **extra_args):
        phase1_checkpoint_path = extra_args.get('phase1_checkpoint_path')
        phase2_checkpoint_path = extra_args.get('phase2_checkpoint_path')
        if phase1_checkpoint_path:
            phase1_checkpoint = torch.load(phase1_checkpoint_path)
            if self.data_parallel:
                self.model.module.Phase1.load_state_dict(phase1_checkpoint['model'])
            else:
                self.model.Phase1.load_state_dict(phase1_checkpoint['model'])
            print('load phase1_checkpoint from {} ...'.format(phase1_checkpoint_path))
        if phase2_checkpoint_path:
            phase2_checkpoint = torch.load(phase2_checkpoint_path)
            if self.data_parallel:
                self.model.module.Phase2.load_state_dict(phase2_checkpoint['model'])
            else:
                self.model.Phase2.load_state_dict(phase2_checkpoint['model'])
            print('load phase2_checkpoint from {} ...'.format(phase2_checkpoint_path))

    def _eval_metrics(self, S0_out, S0, DoP_out, DoP, AoP_out, AoP):
        current_metrics = dict()
        for variable_name, mets in self.metrics.items():
            # manually mapping
            if variable_name == 'S0':
                pred, gt = S0_out / 2, S0 / 2
            elif variable_name == 'DoP':
                pred, gt = DoP_out.mean(dim=1, keepdim=True), DoP.mean(dim=1, keepdim=True)
            elif variable_name == 'AoP':
                pred, gt = AoP_out.mean(dim=1, keepdim=True) / torch.pi, AoP.mean(dim=1, keepdim=True) / torch.pi
            else:
                raise Exception(f'variable_name: {variable_name} not found in metrics!')

            # calculate and record current_metrics
            current_metrics[variable_name] = np.zeros(len(mets))
            for i, met in enumerate(mets):
                m = met(pred, gt)
                current_metrics[variable_name][i] += m
                self.writer.add_scalar(f'{met.__name__}_{variable_name}', m)

        return current_metrics

    @staticmethod
    def _update_total_metrics(total_metrics, current_metrics):
        for variable_name in total_metrics:
            total_metrics[variable_name] += current_metrics[variable_name]

    @staticmethod
    def _avg_metrics(total_metrics, l):
        return {key: value / l for key, value in total_metrics.items()}

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        # set the model to train mode
        self.model.train()

        total_loss = 0
        total_metrics = dict()
        for variable_name, mets in self.metrics.items():
            total_metrics[variable_name] = np.zeros(len(mets))

        # start training
        for batch_idx, sample in enumerate(self.data_loader):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

            # get data and send them to GPU
            # (N, 3, H, W) GPU tensor
            L1 = sample['L1'].to(self.device)
            L2 = sample['L2'].to(self.device)
            L3 = sample['L3'].to(self.device)
            L4 = sample['L4'].to(self.device)
            S0_L = (L1 + L2 + L3 + L4) / 2
            S1_L = L3 - L1
            S2_L = L4 - L2

            B1 = sample['B1'].to(self.device)
            B2 = sample['B2'].to(self.device)
            B3 = sample['B3'].to(self.device)
            B4 = sample['B4'].to(self.device)
            mask = sample['mask'].to(self.device)
            S0_B = (B1 + B2 + B3 + B4) / 2
            S1_B = B3 - B1
            S2_B = B4 - B2

            # (N, 3, H, W) GPU tensor
            I1 = sample['I1'].to(self.device)
            I2 = sample['I2'].to(self.device)
            I3 = sample['I3'].to(self.device)
            I4 = sample['I4'].to(self.device)
            S0, S1, S2, DoP, AoP = util.compute_Si_from_Ii(I1, I2, I3, I4)

            x_B = S1_B / (S0_B + 1e-7)
            y_B = S2_B / (S0_B + 1e-7)
            x = S1 / (S0 + 1e-7)
            y = S2 / (S0 + 1e-7)

            # get network output
            # (N, 3, H, W) GPU tensor
            S0_temp, x_out, y_out, I1_out, I2_out, I3_out, I4_out = self.model(S0_B, mask, S0_L, S1_L, S2_L, x_B, y_B)
            S0_out, S1_out, S2_out, DoP_out, AoP_out = util.compute_Si_from_Ii(I1_out, I2_out, I3_out, I4_out)

            # visualization
            if batch_idx % 200 == 0:
                # save images to tensorboardX
                with torch.no_grad():
                    self.writer.add_image('mask', make_grid(mask))
                    self.writer.add_image('S0_out', make_grid(S0_out / 2))
                    self.writer.add_image('DoP_out', make_grid(util.convert_DoP(DoP_out)))
                    self.writer.add_image('AoP_out', make_grid(util.convert_AoP(AoP_out)))
                    self.writer.add_image('S0', make_grid(S0 / 2))
                    self.writer.add_image('DoP', make_grid(util.convert_DoP(DoP)))
                    self.writer.add_image('AoP', make_grid(util.convert_AoP(AoP)))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(S0_temp, S0, x_out, x, y_out, y, I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            current_metrics = self._eval_metrics(S0_out, S0, DoP_out, DoP, AoP_out, AoP)
            self._update_total_metrics(total_metrics, current_metrics)

            # show current training step info
            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                    )
                )

        # turn the learning rate
        self.lr_scheduler.step()

        # get batch average loss/metrics as log and do validation
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': self._avg_metrics(total_metrics, len(self.data_loader))
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        # set the model to validation mode
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = dict()
        for variable_name, mets in self.metrics.items():
            total_val_metrics[variable_name] = np.zeros(len(mets))

        # start validating
        with torch.no_grad():
            for batch_idx, sample in enumerate(self.valid_data_loader):
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')

                # get data and send them to GPU
                # (N, 3, H, W) GPU tensor
                L1 = sample['L1'].to(self.device)
                L2 = sample['L2'].to(self.device)
                L3 = sample['L3'].to(self.device)
                L4 = sample['L4'].to(self.device)
                S0_L = (L1 + L2 + L3 + L4) / 2
                S1_L = L3 - L1
                S2_L = L4 - L2

                B1 = sample['B1'].to(self.device)
                B2 = sample['B2'].to(self.device)
                B3 = sample['B3'].to(self.device)
                B4 = sample['B4'].to(self.device)
                mask = sample['mask'].to(self.device)
                S0_B = (B1 + B2 + B3 + B4) / 2
                S1_B = B3 - B1
                S2_B = B4 - B2

                # (N, 3, H, W) GPU tensor
                I1 = sample['I1'].to(self.device)
                I2 = sample['I2'].to(self.device)
                I3 = sample['I3'].to(self.device)
                I4 = sample['I4'].to(self.device)
                S0, S1, S2, DoP, AoP = util.compute_Si_from_Ii(I1, I2, I3, I4)

                x_B = S1_B / (S0_B + 1e-7)
                y_B = S2_B / (S0_B + 1e-7)
                x = S1 / (S0 + 1e-7)
                y = S2 / (S0 + 1e-7)

                # get network output
                # (N, 3, H, W) GPU tensor
                S0_temp, x_out, y_out, I1_out, I2_out, I3_out, I4_out = self.model(S0_B, mask, S0_L, S1_L, S2_L, x_B, y_B)
                S0_out, S1_out, S2_out, DoP_out, AoP_out = util.compute_Si_from_Ii(I1_out, I2_out, I3_out, I4_out)
                loss = self.loss(S0_temp, S0, x_out, x, y_out, y, I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                current_metrics = self._eval_metrics(S0_out, S0, DoP_out, DoP, AoP_out, AoP)
                self._update_total_metrics(total_val_metrics, current_metrics)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': self._avg_metrics(total_val_metrics, len(self.valid_data_loader))
        }
