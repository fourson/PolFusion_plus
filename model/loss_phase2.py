from .loss_utils.l12 import l1, l2
from .loss_utils.total_variation import total_variation

tag = 'loss_phase2'


def l1_and_tv_and_ratio(x_out, x, y_out, y, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = (l1(x_out, x) + l1(y_out, y)) * l1_loss_lambda
    print(f'in {tag}, l1_loss: {l1_loss.item()}')

    tv_loss_lambda = kwargs.get('tv_loss_lambda', 1)
    tv_loss = (total_variation((x_out + 1) / 2) + total_variation((y_out + 1) / 2)) * tv_loss_lambda
    print(f'in {tag}, tv_loss: {tv_loss.item()}')

    ratio_loss_lambda = kwargs.get('ratio_loss_lambda', 1)
    ratio_loss = l2(x_out * y, y_out * x) * ratio_loss_lambda
    print(f'in {tag}, ratio_loss: {ratio_loss.item()}')

    return l1_loss + tv_loss + ratio_loss
