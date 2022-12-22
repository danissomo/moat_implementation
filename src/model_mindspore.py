import mindspore.nn as nn
import mindspore
from mindspore import context
import wandb
wandb.init(project="mindspore")

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
from easydict import EasyDict as edict

cfg = edict({
    'num_classes': 10,
    'learning_rate': 0.002,
    'momentum': 0.9,
    'epoch_size': 2, # remember that you are using a Docker with only 1 CPU 
    'batch_size': 32,
    'buffer_size': 1000,
    'image_height': 227,
    'image_width': 227,
    'save_checkpoint_steps': 1562,
    'keep_checkpoint_max': 10,
})

data_path="data_mr/cifar-10-batches-bin"
import mindspore.dataset.vision as CV
import mindspore.dataset.transforms as C
from mindspore.common import dtype as mstype
import mindspore.dataset as ds

def create_dataset(data_path, batch_size=32, repeat_size=1, status="train"):
    """
    create dataset for train or test
    """
    cifar_ds = ds.Cifar10Dataset(data_path)
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if status == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(input_columns="label", operations=typecast_op)
    if status == "train":
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_crop_op)
        cifar_ds = cifar_ds.map(input_columns="image", operations=random_horizontal_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=resize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=rescale_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=normalize_op)
    cifar_ds = cifar_ds.map(input_columns="image", operations=channel_swap_op)

    cifar_ds = cifar_ds.shuffle(buffer_size=cfg.buffer_size)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds


import mindspore.ops.operations as P
import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="valid"):
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode=pad_mode)

def fc_with_initialize(input_channels, out_channels):
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    return TruncatedNormal(0.02)  # 0.02


class AlexNet(nn.Cell):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.batch_size = cfg.batch_size
        self.conv1 = conv(3, 96, 11, stride=4)
        self.conv2 = conv(96, 256, 5, pad_mode="same")
        self.conv3 = conv(256, 384, 3, pad_mode="same")
        self.conv4 = conv(384, 384, 3, pad_mode="same")
        self.conv5 = conv(384, 256, 3, pad_mode="same")
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = fc_with_initialize(6*6*256, 4096)
        self.fc2 = fc_with_initialize(4096, 4096)
        self.fc3 = fc_with_initialize(4096, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


import mindspore.nn as nn
from mindspore import context
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, Callback, RunContext

class ConfusionMatrixCallback(Callback):
    def __init__(self, ds_train, ds_eval):
        self.train = ds_train
        self.eval = ds_eval
        self.cur_answ = None
        self.cur_q = None
    def on_train_step_begin(self, run_context):
        self.cur_answ = run_context.original_args()['train_dataset_element'][1]
        self.cur_q  = run_context.original_args()['train_dataset_element'][0]
        return super().on_train_step_begin(run_context)    

    def on_train_step_end(self, run_context: RunContext):
        model  = run_context.original_args()['network']
        cur_epoch = run_context.original_args()['cur_epoch_num']
        cur_loss = run_context.original_args()['net_outputs']
        pred = model(self.cur_q)
        metric = Accuracy()
        metric.clear()
        metric.update(pred,self.cur_answ)
        acc = metric.eval()
        print(f"acc {acc}")
        wandb.log({
            f"e{cur_epoch}_acc_train" : acc,
            f"e{cur_epoch}_loss_train" : float(str(cur_loss)),
            })
        return super().on_train_step_begin(run_context)
    
    def on_eval_step_end(self, run_context):
        print(run_context.original_args().keys())

        return super().on_eval_step_begin(run_context)

network = AlexNet(cfg.num_classes)

loss = nn.SoftmaxCrossEntropyWithLogits( sparse=True, reduction="mean")
opt = nn.Momentum(network.trainable_params(), cfg.learning_rate, cfg.momentum)

model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

print("============== Starting Training ==============")
ds_train = create_dataset(data_path, cfg.batch_size, cfg.epoch_size, "train")
ds_eval = create_dataset(data_path, cfg.batch_size, cfg.epoch_size, "test")
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max) 
custom_cb = ConfusionMatrixCallback(ds_train, ds_eval)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_alexnet", config=config_ck)
# model.eval(ds_eval, custom_cb)
model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), custom_cb], dataset_sink_mode=False)
