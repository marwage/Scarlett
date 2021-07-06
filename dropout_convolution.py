import mindspore as ms
import numpy as np


def dropout():
    input_np = np.random.normal(size=(512, 256))
    input_ms = ms.Tensor(input_np, ms.float32)

    network_1 = ms.nn.Dropout(keep_prob=0.8)
    network_1.set_train()
    output_1 = network_1(input_ms)
    output_1_np = output_1.asnumpy()

    print(output_1_np[:5, :5])

    network_2 = ms.nn.Dropout(keep_prob=0.8)
    network_2.set_train()
    output_2 = network_2(input_ms)
    output_2_np = output_2.asnumpy()

    print(output_2_np[:5, :5])

    sum_diff = np.sum(np.abs(output_1_np - output_2_np))
    print("Dropout")
    print("Summed absolute difference: {}".format(sum_diff))
    print("---+++---")


def convolution():
    batch_size = 8
    num_channels = 3
    height = 256
    width = 512

    input_np = np.random.normal(size=(batch_size, num_channels, height, width))
    input_ms = ms.Tensor(input_np, ms.float32)

    weight_np = np.random.normal(size=(3, 3, 3, 3))
    weight_ms = ms.Tensor(weight_np, ms.float32)

    network_1 = ms.nn.Conv2d(num_channels, num_channels, (3, 3), weight_init=weight_ms)
    network_1.set_train()
    output_1 = network_1(input_ms)
    output_1_np = output_1.asnumpy()

    print(output_1_np[0, 0, :5, :5])

    network_2 = ms.nn.Conv2d(num_channels, num_channels, (3, 3), weight_init=weight_ms)
    network_2.set_train()
    output_2 = network_2(input_ms)
    output_2_np = output_2.asnumpy()

    print(output_2_np[0, 0, :5, :5])

    sum_diff = np.sum(np.abs(output_1_np - output_2_np))
    print("Convolution")
    print("Summed absolute difference: {}".format(sum_diff))
    print("---+++---")


def main():
    dropout()
    convolution()


if __name__ == "__main__":
    #  ms.common.set_seed(1)
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")
    main()
