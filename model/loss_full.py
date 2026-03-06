from .loss_phase1 import l1_and_perceptual as Phase1Loss
from .loss_phase2 import l1_and_tv_and_ratio as Phase2Loss
from .loss_phase3 import l1_and_dop_and_aop as Phase3Loss

tag = 'full_loss'


def default_loss(S0_temp, S0, x_out, x, y_out, y, I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4, **kwargs):
    phase1_lambda = kwargs.get('phase1_lambda', 1)
    phase1_loss = Phase1Loss(S0_temp, S0, **kwargs['phase1_args']) * phase1_lambda
    print(f'in {tag}, phase1_loss: {phase1_loss.item()}')

    phase2_lambda = kwargs.get('phase2_lambda', 1)
    phase2_loss = Phase2Loss(x_out, x, y_out, y, **kwargs['phase2_args']) * phase2_lambda
    print(f'in {tag}, phase2_loss: {phase2_loss.item()}')

    phase3_lambda = kwargs.get('phase3_lambda', 1)
    phase3_loss = Phase3Loss(I1_out, I1, I2_out, I2, I3_out, I3, I4_out, I4, **kwargs['phase3_args']) * phase3_lambda
    print(f'in {tag}, phase3_loss: {phase3_loss.item()}')

    return phase1_loss + phase2_loss + phase3_loss
