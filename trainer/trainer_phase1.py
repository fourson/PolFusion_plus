import numpy as np
import torch
from torchvision.utils import make_grid

from base.base_trainer import BaseTrainer


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

    def _eval_metrics(self, S0_temp, S0):
        current_metrics = dict()
        for variable_name, mets in self.metrics.items():
            # manually mapping
            if variable_name == 'S0':
                pred, gt = S0_temp / 2, S0 / 2
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

            # (N, 3, H, W) GPU tensor
            B1 = sample['B1'].to(self.device)
            B2 = sample['B2'].to(self.device)
            B3 = sample['B3'].to(self.device)
            B4 = sample['B4'].to(self.device)
            mask = sample['mask'].to(self.device)
            S0_B = (B1 + B2 + B3 + B4) / 2

            # (N, 3, H, W) GPU tensor
            I1 = sample['I1'].to(self.device)
            I2 = sample['I2'].to(self.device)
            I3 = sample['I3'].to(self.device)
            I4 = sample['I4'].to(self.device)
            S0 = (I1 + I2 + I3 + I4) / 2

            # get network output
            # (N, 3, H, W) GPU tensor
            S0_temp = self.model(S0_B, mask, S0_L, S1_L, S2_L)

            if batch_idx % 200 == 0:
                # save images to tensorboardX
                with torch.no_grad():
                    self.writer.add_image('mask', make_grid(mask))
                    self.writer.add_image('S0_B', make_grid(S0_B / 2))
                    self.writer.add_image('S0_L', make_grid(S0_L / 2))
                    self.writer.add_image('S0_temp', make_grid(S0_temp / 2))
                    self.writer.add_image('S0', make_grid(S0 / 2))

            # train model
            self.optimizer.zero_grad()
            model_loss = self.loss(S0_temp, S0)
            model_loss.backward()
            self.optimizer.step()

            # calculate total loss/metrics and add scalar to tensorboard
            self.writer.add_scalar('loss', model_loss.item())
            total_loss += model_loss.item()
            current_metrics = self._eval_metrics(S0_temp, S0)
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

                # (N, 3, H, W) GPU tensor
                B1 = sample['B1'].to(self.device)
                B2 = sample['B2'].to(self.device)
                B3 = sample['B3'].to(self.device)
                B4 = sample['B4'].to(self.device)
                mask = sample['mask'].to(self.device)
                S0_B = (B1 + B2 + B3 + B4) / 2

                # (N, 3, H, W) GPU tensor
                I1 = sample['I1'].to(self.device)
                I2 = sample['I2'].to(self.device)
                I3 = sample['I3'].to(self.device)
                I4 = sample['I4'].to(self.device)
                S0 = (I1 + I2 + I3 + I4) / 2

                # get network output and compute loss
                # (N, 3, H, W) GPU tensor
                S0_temp = self.model(S0_B, mask, S0_L, S1_L, S2_L)
                loss = self.loss(S0_temp, S0)

                # calculate total loss/metrics and add scalar to tensorboardX
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                current_metrics = self._eval_metrics(S0_temp, S0)
                self._update_total_metrics(total_val_metrics, current_metrics)

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': self._avg_metrics(total_val_metrics, len(self.valid_data_loader))
        }
