import torch
import tqdm
import os
import wandb
import shutil
import numpy as np
import importlib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import auc, precision_recall_curve, average_precision_score,accuracy_score

import pytorch_lightning
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning import LightningDataModule, LightningModule
from Viodetect.datasets.builder import build_dataloader
from Viodetect.models.builder import build_model
from Viodetect.optimizers.builder import build_optimizer, build_lr_scheduler
from Viodetect.utils.callback import register_trainer

class DataModule(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return build_dataloader(self.cfg.data['train'])

    def val_dataloader(self):
        return build_dataloader(self.cfg.data['val'])

class MyModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.model = build_model(cfg.model)

        register_trainer(self)

        self.target = np.load(self.cfg.gt_label)[::16]

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.model.inference(batch)
        if 'pred' in outputs:
            return outputs['pred']
        else:
            ind_cls = torch.sigmoid(outputs['logits'])
            return ind_cls

    def training_step(self, batch, *args):
        loss, _ = self.model(batch)
        self.log('train/loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        loss, outputs = self.model(batch)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        if 'pred' in outputs:
            return outputs['pred'].mean(0).squeeze(), batch['index'][0]
        else:
            return torch.sigmoid(
                outputs['logits']).mean(0).squeeze(), batch['index'][0]

    def validation_epoch_end(self, outputs) -> None:
        index = [x[1].item() for x in outputs]

        pred = np.concatenate(
            [outputs[i][0].detach().cpu().numpy() for i in index])

        target = self.target[:len(pred)]
        AP = average_precision_score(target, pred)
        y_target = np.where(pred >= 0.5,1,0)
        acc = accuracy_score(target,y_target)
        self.log("val/ap", AP, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/acc",acc,on_epoch=True,prog_bar=True,sync_dist=True)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = build_optimizer(
            self.cfg.optimizer,
            self.model.get_parameters(self.cfg.optimizer['lr']))
        scheduler = build_lr_scheduler(self.cfg.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss"
        }


def load_model(cfg, model_path: str = None, remote: bool = False):
    model = MyModel(cfg).cuda()
    save_dir = cfg.save_dir
    if remote and model_path:
        run = wandb.init()
        artifact = run.use_artifact(model_path, type='model')
        artifact_dir = artifact.download()
        best_model_path = Path(artifact_dir) / "model.ckpt"
    elif model_path:
        best_model_path = model_path
    else:
        best_model_path = os.path.join(save_dir, 'best.ckpt')

    if os.path.exists(best_model_path):
        pkl = torch.load(best_model_path)
        model.load_state_dict(pkl['state_dict'])

    return model


def test(cfg, model_path: str = None, remote: bool = False,
         smooth: bool = False, window: int = 5, draw_plot: bool = False):
    pytorch_lightning.seed_everything(cfg.rand_seed)
    model = load_model(cfg, model_path, remote)
    test_data_loader = build_dataloader(cfg.data['val'])

    gt_label = np.load(cfg.gt_label).astype(np.int32)[::16]

    result = []
    model.eval()
    kernel = np.ones((window,)) / window
    for batch in tqdm.tqdm(test_data_loader):
        for k, v in batch.items():
            batch[k] = v.cuda()
        with torch.no_grad():
            ind_cls = model.predict_step(batch, 0)

        ind_cls = torch.mean(ind_cls, 0).detach().cpu().numpy()
        if smooth:
            ind_cls = np.convolve(ind_cls, kernel)[window // 2:-(window // 2)]
        result.extend(ind_cls.tolist())

    # pred = np.repeat(result, 16)
    precision, recall, th = precision_recall_curve(list(gt_label), result)
    pr_auc = auc(recall, precision)
    print(pr_auc)

    if draw_plot:

        output_dir = os.path.join(cfg.base_path, 'output', cfg.name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savez(os.path.join(output_dir, 'output.npz'),
                 precision=precision,
                 recall=recall,
                 th=th,
                 result=result,
                 pr_auc=pr_auc)

        plt.cla()

        plt.title('PRC_xd_vio_video')
        plt.plot(precision, recall)
        plt.xlabel('precision')
        plt.ylabel('recall')

        output_path = os.path.join(output_dir, f'{pr_auc}.png')
        plt.savefig(output_path, dpi=600)

    return pr_auc
