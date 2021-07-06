import mindspore as ms
import numpy as np
from tensor_diff import show_tensor_diff

from lamb import _update_run_op


def lamb():
    size = (512, 256)

    param_np = np.random.normal(size=size)
    param_ms = ms.Tensor(param_np, ms.float32)
    momentum_np = np.random.normal(size=size)
    momentum_ms = ms.Tensor(momentum_np, ms.float32)
    velocity_np = np.random.normal(size=size)
    velocity_ms = ms.Tensor(velocity_np, ms.float32)
    gradient_np = np.random.normal(size=size)
    gradient_ms = ms.Tensor(gradient_np, ms.float32)

    beta1 = ms.Tensor(0.9, ms.float32)
    beta2 = ms.Tensor(0.999, ms.float32)
    eps = ms.Tensor(1e-6, ms.float32)
    global_step = ms.Tensor(1, ms.int64)
    lr = ms.Tensor(1e-3, ms.float32)
    weight_decay = ms.Tensor(0.0, ms.float32)
    param = ms.Parameter(param_ms,
                         name="weight",
                         requires_grad=True)
    m = ms.Parameter(momentum_ms,
                     "momentum")
    v = ms.Parameter(velocity_ms,
                     "velocity")
    gradient = gradient_ms
    decay_flag = False
    optim_filter = True

    gradient_updated_1 = _update_run_op(beta1, beta2, eps, global_step, lr,
                                        weight_decay, param, m, v, gradient,
                                        decay_flag, optim_filter)

    gradient_updated_2 = _update_run_op(beta1, beta2, eps, global_step, lr,
                                        weight_decay, param, m, v, gradient,
                                        decay_flag, optim_filter)

    show_tensor_diff(gradient_updated_1.asnumpy(), gradient_updated_2.asnumpy())


def main():
    lamb()


if __name__ == "__main__":
    #  ms.common.set_seed(1)
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")
    main()
