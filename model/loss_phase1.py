from .loss_utils.l12 import l1
from .loss_utils.perceptual import perceptual

tag = 'loss_phase1'


def l1_and_perceptual(S0_temp, S0, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = l1(S0_temp, S0) * l1_loss_lambda
    print(f'in {tag}, l1_loss: {l1_loss.item()}')

    perceptual_loss_lambda = kwargs.get('perceptual_loss_lambda', 1)
    perceptual_loss = perceptual(S0_temp, S0) * perceptual_loss_lambda
    print(f'in {tag}, perceptual_loss: {perceptual_loss.item()}')

    return l1_loss + perceptual_loss
