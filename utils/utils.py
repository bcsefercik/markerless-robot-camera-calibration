import torch, glob, os, numpy as np
import sys
import random

sys.path.append("../")  # noqa

from utils import config, logger

_config = config.Config()
_logger = logger.Logger().get()

_use_cuda = torch.cuda.is_available()
_device = torch.device("cuda" if _use_cuda else "cpu")

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(
    optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6
):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = max(base_lr * (multiplier ** (epoch // step_epoch)), clip)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(
        intersection, bins=np.arange(K + 1)
    )  # area_intersection: K, indicates the number of members in each class in intersection
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def checkpoint_save(
    model, exp_path, exp_name, epoch, optimizer=None, save_freq=16, use_cuda=True
):
    f = os.path.join(exp_path, exp_name + "-%09d" % epoch + ".pth")
    _logger.info("Saving " + f)
    model.cpu()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else optimizer,
        },
        f,
    )
    if use_cuda:
        model.cuda()

    # remove previous checkpoints unless they are a power of 2 or a multiple of save_freq to save disk space
    epoch = epoch - 1
    f = os.path.join(exp_path, exp_name + "-%09d" % epoch + ".pth")
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def checkpoint_restore(
    model, exp_path="", exp_name="", optimizer=None, use_cuda=True, epoch=0, f=""
):
    if use_cuda:
        model.cpu()

    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + "-%09d" % epoch + ".pth")
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + "-*.pth")))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0 and (os.path.exists(f) or os.path.exists(os.path.join(os.path.dirname(BASE_PATH), f))):
        if not os.path.exists(f):
            f = os.path.join(
                os.path.dirname(BASE_PATH),
                f
            )
        _logger.info("Restore from " + f)
        checkpoint = torch.load(f, map_location=torch.device(_device))
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if use_cuda:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
    else:
        epoch = -100
        _logger.info("Did not restore from " + f)

    if use_cuda:
        model.cuda()
    return epoch + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def load_model_param(model, pretrained_dict, prefix=""):
    # suppose every param in model should exist in pretrain_dict, but may differ in the prefix of the name
    # For example:    model_dict: "0.conv.weight"     pretrain_dict: "FC_layer.0.conv.weight"
    model_dict = model.state_dict()
    len_prefix = 0 if len(prefix) == 0 else len(prefix) + 1
    pretrained_dict_filter = {
        k[len_prefix:]: v
        for k, v in pretrained_dict.items()
        if k[len_prefix:] in model_dict and prefix in k
    }
    assert len(pretrained_dict_filter) > 0
    model_dict.update(pretrained_dict_filter)
    model.load_state_dict(model_dict)
    return len(pretrained_dict_filter), len(model_dict)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, "w")
    for i in range(N):
        c = colors[i]
        fout.write(
            "v %f %f %f %d %d %d\n"
            % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2])
        )
    fout.close()


def get_batch_offsets(batch_idxs, bs):
    """
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    """
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def print_error(message, user_fault=False):
    sys.stderr.write("ERROR: " + str(message) + "\n")
    if user_fault:
        sys.exit(2)
    sys.exit(-1)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


torch_generator = torch.Generator()
torch_generator.manual_seed(_config.GENERAL.seed)


def remove_suffix(s, suffix):
    if s.endswith(suffix):
        return s[: -len(suffix)]
    return s
