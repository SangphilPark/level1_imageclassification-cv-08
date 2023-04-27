import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


class EarlyStopping:
    """Ï£ºÏñ¥Ïß? patience ?ù¥?õÑÎ°? validation lossÍ∞? Í∞úÏÑ†?êòÏß? ?ïä?úºÎ©? ?ïô?äµ?ùÑ Ï°∞Í∏∞ Ï§ëÏ??"""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation lossÍ∞? Í∞úÏÑ†?êú ?õÑ Í∏∞Îã§Î¶¨Îäî Í∏∞Í∞Ñ
                            Default: 7
            verbose (bool): True?ùº Í≤ΩÏö∞ Í∞? validation loss?ùò Í∞úÏÑ† ?Ç¨?ï≠ Î©îÏÑ∏Ïß? Ï∂úÎ†•
                            Default: False
            delta (float): Í∞úÏÑ†?êò?óà?ã§Í≥? ?ù∏?†ï?êò?äî monitered quantity?ùò ÏµúÏÜå Î≥??ôî
                            Default: 0
            path (str): checkpoint????û• Í≤ΩÎ°ú
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation lossÍ∞? Í∞êÏÜå?ïòÎ©? Î™®Îç∏?ùÑ ????û•?ïú?ã§.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(
        range(batch_size), k=n) if shuffle else list(range(n))
    # cautions: hardcoded, ?ù¥ÎØ∏Ï?? ?Å¨Í∏∞Ïóê ?î∞?ùº figsize Î•? Ï°∞Ï†ï?ï¥?ïº ?ï† ?àò ?ûà?äµ?ãà?ã§. T.T
    figure = plt.figure(figsize=(12, 18 + 2))
    # cautions: hardcoded, ?ù¥ÎØ∏Ï?? ?Å¨Í∏∞Ïóê ?î∞?ùº top Î•? Ï°∞Ï†ï?ï¥?ïº ?ï† ?àò ?ûà?äµ?ãà?ã§. T.T
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training runs through ", device)

    # -- dataset
    dataset_module = getattr(import_module("dataset"),
                             args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module(
        "dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module(
        "model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"),
                         args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    # early_stopping = EarlyStopping(patience=3, verbose=True)
    counter = 0
    patience = args.patience

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            inputs, (mask_labels, gender_labels, age_labels) = train_batch
            inputs = inputs.to(device)

            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            (mask_outs, gender_outs, age_outs) = torch.split(
                outs, [3, 2, 3], dim=1)

            mask_loss = criterion(mask_outs, mask_labels)
            gender_loss = criterion(gender_outs, gender_labels)
            age_loss = criterion(age_outs, age_labels)

            mask_preds = torch.argmax(mask_outs, dim=1)
            gender_preds = torch.argmax(gender_outs, dim=1)
            age_preds = torch.argmax(age_outs, dim=1)

            loss = mask_loss + gender_loss + 1.5 * age_loss

            loss.backward()
            optimizer.step()

            preds, labels = dataset.encode_multi_class(
                mask_labels, gender_labels, age_labels), dataset.encode_multi_class(mask_preds, gender_preds, age_preds)

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss,
                                  epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch)

                loss_value = 0
                matches = 0
        wandb.log({"train_loss": train_loss}, step=epoch)
        wandb.log({"train_acc": train_acc}, step=epoch)
        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []

            mask_loss_items = []
            gender_loss_items = []
            age_loss_items = []

            mask_val_acc_items = []
            gender_val_acc_items = []
            age_val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, (mask_labels, gender_labels, age_labels) = val_batch
                inputs = inputs.to(device)

                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                outs = model(inputs)
                (mask_outs, gender_outs, age_outs) = torch.split(
                    outs, [3, 2, 3], dim=1)

                mask_preds = torch.argmax(mask_outs, dim=1)
                gender_preds = torch.argmax(gender_outs, dim=1)
                age_preds = torch.argmax(age_outs, dim=1)

                mask_loss = criterion(mask_outs, mask_labels).item()
                gender_loss = criterion(gender_outs, gender_labels).item()
                age_loss = criterion(age_outs, age_labels).item()

                loss_item = mask_loss + gender_loss + age_loss

                preds, labels = dataset.encode_multi_class(
                    mask_labels, gender_labels, age_labels), dataset.encode_multi_class(mask_preds, gender_preds, age_preds)

                mask_val_acc = (mask_labels == mask_preds).sum().item()
                gender_val_acc = (gender_labels == gender_preds).sum().item()
                age_val_acc = (age_labels == age_preds).sum().item()

                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                mask_val_acc_items.append(mask_val_acc)
                gender_val_acc_items.append(gender_val_acc)
                age_val_acc_items.append(age_val_acc)

                mask_loss_items.append(mask_loss)
                gender_loss_items.append(gender_loss)
                age_loss_items.append(age_loss)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach(
                    ).cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            # mask, genderm age (?èâÍ∑? loss,acc Í≥ÑÏÇ∞)

            mask_val_loss = np.sum(mask_loss_items) / len(val_loader)
            gender_val_loss = np.sum(gender_loss_items) / len(val_loader)
            age_val_loss = np.sum(age_loss_items) / len(val_loader)

            mask_val_acc = np.sum(mask_val_acc_items) / len(val_set)
            gender_val_acc = np.sum(gender_val_acc_items) / len(val_set)
            age_val_acc = np.sum(age_val_acc_items) / len(val_set)

            # one class loss, acc Í≥ÑÏÇ∞

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            wandb.log({"valid_loss": val_loss}, step=epoch)
            wandb.log({"valid_acc": val_acc}, step=epoch)

            # mask, gender, age (loss & acc)
            wandb.log({"mask_loss": mask_val_loss}, step=epoch)
            wandb.log({"mask_acc": mask_val_acc}, step=epoch)
            wandb.log({"gender_loss": gender_val_loss}, step=epoch)
            wandb.log({"gender_acc": gender_val_acc}, step=epoch)
            wandb.log({"age_loss": age_val_loss}, step=epoch)
            wandb.log({"age_acc": age_val_acc}, step=epoch)

            if(counter >= patience):
                print("Early Stopping")
                break
            print()

        # early_stopping (val_loss, model)
        # if early_stopping .early_stop:
        #     print("at epoch : ", epoch)
        #     break


if __name__ == '__main__':
    wandb.init(project="wandb_test")
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset',
                        help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation',
                        help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int,
                        default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000,
                        help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel',
                        help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy',
                        help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20,
                        help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp',
                        help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patience', type=int, default=5,
                        help='early stop hypermarameter')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get(
        'SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    wandb.config.update(args)
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
