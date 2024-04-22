# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os
import pdb

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import logging
from pathlib import Path

import torch, platform

# 这是一个建立在PyTorch之上的库，它简化了训练过程并减少了样板代码。
# PyTorch Lightning 允许你以更高层次的抽象来定义模型和训练过程，同时提供了许多有用的功能
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from AR.data.data_module import Text2SemanticDataModule

from AR.models.t2s_lightning_module import Text2SemanticLightningModule

from AR.utils.io import load_yaml_config

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from AR.utils import get_newest_ckpt

from collections import OrderedDict
from time import time as ttime
import shutil


def my_save(fea,path):
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    tmp_path="%s.pth"%(ttime())
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if (
                self._every_n_epochs >= 1
                and (trainer.current_epoch + 1) % self._every_n_epochs == 0
            ):
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    # torch.save(
                    my_save(
                        to_save_od,
                        "%s/%s-e%s.ckpt"
                        % (
                            self.half_weights_save_dir,
                            self.exp_name,
                            trainer.current_epoch + 1,
                        ),
                    )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):

    # 导入参数
    config = load_yaml_config(args.config_file)

    # 输出路径，没有就新建
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ckpt的路径设置
    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 设置全局随机种子，以确保实验的可重复性
    seed_everything(config["train"]["seed"], workers=True)


    # 使用PyTorch Lightning框架 定义了一系列模块

    # 变量ckpt_callback 预期 是 ModelCheckpoint类型 的实例
    # 模型检查点回调（ModelCheckpoint）是一个在PyTorch Lightning训练过程中自动保存模型状态的机制。
    # 它允许在训练过程中的关键时刻自动保存模型的权重和参数，比如当验证集上的性能达到新高时。
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )

    # 训练记录
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)

    # 主节点的地址是本地机器，这通常用于单机多卡的训练设置
    os.environ["MASTER_ADDR"]="localhost"

    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],

        accelerator="gpu" if torch.cuda.is_available() else "cpu",

        limit_val_batches=0,

        devices=-1 if torch.cuda.is_available() else 1,

        benchmark=False,
        fast_dev_run=False,

        strategy = DDPStrategy(
            process_group_backend="nccl" if platform.system() != "Windows" else "gloo"
        ) if torch.cuda.is_available() else "auto",

        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
    )

    # 定义模型，参数中包含输出的路径
    model: Text2SemanticLightningModule = Text2SemanticLightningModule(
        config, output_dir
    )

    # 定义数据，数据分别为 2-text 和 6-音素
    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        train_semantic_path=config["train_semantic_path"],
        train_phoneme_path=config["train_phoneme_path"],
    )

    try:
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))

        ckpt_path = ckpt_dir / newest_ckpt_name

    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Start from here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    args = parser.parse_args()
    logging.info(str(args))
    main(args)
