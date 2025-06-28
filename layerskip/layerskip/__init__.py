from .early_exit_loss import (
    uniform_loss_scale,
    linear_l_loss_scale,
    sum_l_loss_scale,
    sqrt_l_loss_scale,
    inv_l_loss_scale,
    inv_sqrt_l_loss_scale,
)

EARLY_EXIT_LOSS_SCALE_FUNCTIONS = {
    "uniform": uniform_loss_scale,
    "linear": linear_l_loss_scale,
    "sum": sum_l_loss_scale,
    "sqrt": sqrt_l_loss_scale,
    "inv": inv_l_loss_scale,
    "inv_sqrt": inv_sqrt_l_loss_scale,
}