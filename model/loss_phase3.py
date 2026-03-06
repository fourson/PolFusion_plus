import torch

from .loss_utils.l12 import l1, l2

tag = 'loss_phase3'


def l1_and_dop_and_aop(I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4, **kwargs):
    I_out_cat = torch.cat((I1_out, I2_out, I3_out, I4_out), dim=1)
    I_cat = torch.cat((I1, I2, I3, I4), dim=1)

    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = l1(I_out_cat, I_cat) * l1_loss_lambda
    print(f'in {tag}, l1_loss: {l1_loss.item()}')

    pr_loss_lambda_a = kwargs.get('pr_loss_lambda_a', 1)
    pr_loss_a = l2(I1_out + I3_out, I2_out + I4_out) * pr_loss_lambda_a
    print(f'in {tag}, pr_loss_a: {pr_loss_a.item()}')

    pr_loss_lambda_b = kwargs.get('pr_loss_lambda_b', 1)
    pr_loss_b = l2((I4_out - I2_out) * (I3 - I1), (I3_out - I1_out) * (I4 - I2)) * pr_loss_lambda_b
    print(f'in {tag}, pr_loss_b: {pr_loss_b.item()}')

    return l1_loss + pr_loss_a
