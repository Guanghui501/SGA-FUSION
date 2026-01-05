"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python alignn/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union
import ignite
import torch

from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
    # For different version of pytorch-ignite
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from data import get_train_val_loaders
from config import TrainingConfig
from models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import os

# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.preprocessing import StandardScaler

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    # pc = pk.load(open("pca.pkl", "rb"))
    # y_pred = torch.tensor(pc.transform(y_pred), device=device)
    # y = torch.tensor(pc.transform(y), device=device)

    # y_pred = torch.tensor(pca_sc.inverse_transform(y_pred),device=device)
    # y = torch.tensor(pca_sc.inverse_transform(y),device=device)
    # print (y.shape,y_pred.shape)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def train_dgl(config: Union[TrainingConfig, Dict[str, Any]], model: nn.Module = None, train_val_test_loaders=[], resume=0, model_config=None):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used

    Args:
        config: Training configuration
        model: Model to train
        train_val_test_loaders: Data loaders
        resume: Resume from checkpoint
        model_config: Model configuration (ALIGNNConfig) to save in checkpoint
    """
    # print(config)
    # if type(config) is dict:
    #     try:
    #         print(config)
    #         config = TrainingConfig(**config)
    #     except Exception as exp:
    #         print("Check", exp)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    checkpoint_dir = os.path.join(config.output_dir)
    deterministic = False
    classification = False
    # print("config:")
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    # pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    line_graph = False
    alignn_models = {"alignn","dense_alignn","alignn_cgcnn","alignn_layernorm"}
    if config.model.name == "clgn":
        line_graph = True
    if config.model.name == "cgcnn":
        line_graph = True
    if config.model.name == "icgcnn":
        line_graph = True
    if config.model.name in alignn_models and config.model.alignn_layers > 0:
        line_graph = True

    # print ('output_dir train', config.output_dir)
    train_loader = train_val_test_loaders[0]
    val_loader = train_val_test_loaders[1]
    test_loader = train_val_test_loaders[2]
    prepare_batch = train_val_test_loaders[3]

    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True

    # define network, optimizer, scheduler
    _model = {"alignn": ALIGNN}
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    model_number=[]
    if resume ==1:
        for f in os.listdir(config.output_dir):
            # print(config.output_dir)
            if f.startswith('checkpoint_'):
                # print(f)
                model_number.append(int(f.split('.')[0].split('_')[1]))
    # print(model_number)
    # exit()
    if resume ==1:
        checkpoint_path = config.output_dir+'checkpoint_'+str(max(model_number))+'.pt'
        print(f"\n{'='*80}")
        print(f"🔄 恢复训练")
        print(f"{'='*80}")
        print(f"加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # 显示 checkpoint 包含的内容
        print(f"Checkpoint 包含的键: {list(checkpoint.keys())}")
        if "epoch" in checkpoint:
            print(f"从 Epoch {checkpoint['epoch']} 恢复")

        # 加载模型权重
        net.load_state_dict(checkpoint["model"])
        print("✅ 已加载模型权重")
        print(f"{'='*80}\n")

    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if resume ==1:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("✅ 已加载 optimizer 状态")
        else:
            print("⚠️  Checkpoint 中没有 optimizer 状态，将使用新的 optimizer")

    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=config.learning_rate,epochs=config.epochs,steps_per_epoch=steps_per_epoch,pct_start=0.3)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer)

    if resume ==1:
        if "lr_scheduler" in checkpoint:
#            scheduler.load_state_dict(checkpoint["lr_scheduler"])
#            print("✅ 已加载 scheduler 状态")
            if config.scheduler == "onecycle":
                print("⚠️  检测到 OneCycleLR scheduler")
                print("   Resume 训练时不加载旧 scheduler 状态（避免步数冲突）")
                print("   Scheduler 将从头开始，学习率会重新 warm up")
            else:
                scheduler.load_state_dict(checkpoint["lr_scheduler"])
                print("✅ 已加载 scheduler 状态")
        else:
            print("⚠️  Checkpoint 中没有 scheduler 状态，将使用新的 scheduler")

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "bce": nn.BCELoss(),
    }
    criterion = criteria[config.criterion]

    # Check if contrastive learning is enabled
    use_contrastive = getattr(config.model, 'use_contrastive_loss', False)
    contrastive_weight = getattr(config.model, 'contrastive_loss_weight', 0.1)

    # set up metrics based on task type
    if classification:
        # Classification metrics
        def thresholded_output_transform(output):
            y_pred, y = output
            y_pred = (y_pred > 0.5).long()  # Convert to long for binary classification metrics
            y = y.long()  # Ensure targets are also long
            return y_pred, y

        metrics = {
            "loss": Loss(criterion),
            "accuracy": Accuracy(output_transform=thresholded_output_transform),
            "precision": Precision(output_transform=thresholded_output_transform, average=True),
            "recall": Recall(output_transform=thresholded_output_transform, average=True),
        }
        print(f"\n📊 分类模式已启用，使用准确率、精确率、召回率指标")
    else:
        # Regression metrics
        metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}

    # Gradient clipping for stability
    grad_clip = 1.0
    print(f"\n📎 梯度裁剪已启用: max_norm = {grad_clip}")

    if use_contrastive:
        print(f"\n🔥 对比学习已启用:")
        print(f"  - 损失权重: {contrastive_weight}")
        print(f"  - 温度参数: {getattr(config.model, 'contrastive_temperature', 0.1)}")

        # Custom training function for contrastive learning
        def custom_train_step(engine, batch):
            net.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch)

            # Forward pass
            output = net(x)

            # Handle dict output from contrastive learning
            if isinstance(output, dict):
                y_pred = output['predictions']
                task_loss = criterion(y_pred, y)

                # Add contrastive loss if available
                if 'contrastive_loss' in output:
                    contrastive_loss = output['contrastive_loss']
                    total_loss = task_loss + contrastive_weight * contrastive_loss
                else:
                    total_loss = task_loss
            else:
                y_pred = output
                total_loss = criterion(y_pred, y)

            total_loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            return total_loss.item()

        trainer = ignite.engine.Engine(custom_train_step)
    else:
        # Custom training function with gradient clipping
        def train_step_with_clip(engine, batch):
            net.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch)
            y_pred = net(x)
            if isinstance(y_pred, dict):
                y_pred = y_pred['predictions']
            loss = criterion(y_pred, y)
            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            return loss.item()

        trainer = ignite.engine.Engine(train_step_with_clip)

    if resume ==1:
        if "trainer" in checkpoint:
            trainer.load_state_dict(checkpoint["trainer"])
            print("✅ 已加载 trainer 状态")
        else:
            print("⚠️  Checkpoint 中没有 trainer 状态，将从当前 epoch 开始训练")

    # Custom output transform for contrastive learning
    if use_contrastive:
        def output_transform(output):
            """Extract predictions from dict output"""
            y_pred, y = output
            if isinstance(y_pred, dict):
                return y_pred['predictions'], y
            return y_pred, y

        # Create custom metrics with output transform
        metrics = {
            "loss": Loss(criterion, output_transform=output_transform),
            "mae": MeanAbsoluteError(output_transform=output_transform)
        }

    evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    train_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    test_evaluator = create_supervised_evaluator(net,metrics=metrics,prepare_batch=prepare_batch,device=device)

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda engine: scheduler.step())

    # Setup metric tracking based on task type
    if classification:
        # For classification, track accuracy (higher is better)
        metric_name = 'accuracy'
        best_val_metric = 0.0
        best_test_metric = 0.0
        best_loss = float('inf')
        metric_better = lambda new, old: new > old  # Higher is better
    else:
        # For regression, track MAE (lower is better)
        metric_name = 'mae'
        best_val_metric = float('inf')
        best_test_metric = float('inf')
        best_loss = float('inf')
        metric_better = lambda new, old: new < old  # Lower is better

    # Early stopping setup
    early_stopping_patience = config.n_early_stopping
    epochs_without_improvement = 0
    if early_stopping_patience:
        print(f"\n⏹️ Early Stopping 已启用: patience = {early_stopping_patience}")

    if config.write_checkpoint:
        # model checkpointing
        to_save = {
            "model": net,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "trainer": trainer,
        }
        handler = Checkpoint(
            to_save,
            DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if config.progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator,output_transform=lambda x: {"mae": x})

    history = {"train": {m: [] for m in metrics.keys()},"validation": {m: [] for m in metrics.keys()},"test": {m: [] for m in metrics.keys()}}

    if config.store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)
        test_evaluator.run(test_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        tstmetrics = test_evaluator.state.metrics

        for metric in metrics.keys():
            tm = tmetrics[metric]
            vm = vmetrics[metric]
            tstm = tstmetrics[metric]

            if isinstance(tm, torch.Tensor):
                tm = tm.cpu().numpy().tolist()
                vm = vm.cpu().numpy().tolist()
                tstm = tstm.cpu().numpy().tolist()

            history["train"][metric].append(tm)
            history["validation"][metric].append(vm)
            history["test"][metric].append(tstm)

        if config.store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data
            dumpjson(filename=os.path.join(config.output_dir, "history_val.json"),data=history["validation"])
            dumpjson(filename=os.path.join(config.output_dir, "history_train.json"),data=history["train"])
        if config.progress:
            pbar = ProgressBar()
            pbar.log_message(f"Epoch: {engine.state.epoch:.1f}")
            if classification:
                pbar.log_message(f"Train_Acc: {tmetrics['accuracy']:.4f}")
                pbar.log_message(f"Val_Acc: {vmetrics['accuracy']:.4f}")
                pbar.log_message(f"Test_Acc: {tstmetrics['accuracy']:.4f}")
                if 'precision' in tmetrics:
                    pbar.log_message(f"Val_Precision: {vmetrics['precision']:.4f}, Val_Recall: {vmetrics['recall']:.4f}")
            else:
                pbar.log_message(f"Train_MAE: {tmetrics['mae']:.4f}")
                pbar.log_message(f"Val_MAE: {vmetrics['mae']:.4f}")
                pbar.log_message(f"Test_MAE: {tstmetrics['mae']:.4f}")

        nonlocal best_loss, best_val_metric, best_test_metric, epochs_without_improvement

        # Save best validation model
        improved = False
        if metric_better(vmetrics[metric_name], best_val_metric):
            best_val_metric = vmetrics[metric_name]
            best_val_checkpoint = {
                "model": net.state_dict(),
                "config": model_config,  # Save model config (ALIGNNConfig)
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": engine.state.epoch,
                f"val_{metric_name}": best_val_metric,
            }
            torch.save(best_val_checkpoint, os.path.join(config.output_dir, "best_val_model.pt"))
            metric_display = f"{metric_name.upper()}: {best_val_metric:.4f}"
            print(f"✅ Saved best val model ({metric_display}) at epoch {engine.state.epoch}")
            improved = True
            epochs_without_improvement = 0

        # Save best test model
        if metric_better(tstmetrics[metric_name], best_test_metric):
            best_test_metric = tstmetrics[metric_name]
            best_test_checkpoint = {
                "model": net.state_dict(),
                "config": model_config,  # Save model config (ALIGNNConfig)
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": engine.state.epoch,
                f"test_{metric_name}": best_test_metric,
            }
            torch.save(best_test_checkpoint, os.path.join(config.output_dir, "best_test_model.pt"))
            metric_display = f"{metric_name.upper()}: {best_test_metric:.4f}"
            print(f"✅ Saved best test model ({metric_display}) at epoch {engine.state.epoch}")

        if vmetrics['loss'] < best_loss:
            best_loss = vmetrics['loss']

        # Early stopping check
        if not improved:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⏹️ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                print(f"Best Val {metric_name.upper()}: {best_val_metric:.4f}")
                trainer.terminate()

        print(f"Best_val_{metric_name}: {best_val_metric:.4f}, Best_test_{metric_name}: {best_test_metric:.4f}")
        if early_stopping_patience:
            print(f"Epochs without improvement: {epochs_without_improvement}/{early_stopping_patience}")
        print("\n")

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)

    # Print final summary
    print("\n" + "="*80)
    print("🎯 Training Complete!")
    print("="*80)
    if classification:
        print(f"Best Validation Accuracy: {best_val_metric:.4f}")
        print(f"Best Test Accuracy: {best_test_metric:.4f}")
    else:
        print(f"Best Validation MAE: {best_val_metric:.4f}")
        print(f"Best Test MAE: {best_test_metric:.4f}")
    print(f"\nCheckpoints saved:")
    print(f"  - best_val_model.pt")
    print(f"  - best_test_model.pt")
    print("="*80 + "\n")

    # 定义预测保存函数，避免代码重复
    def save_predictions(model, data_loader, output_file, dataset_name):
        """保存预测结果到CSV文件"""
        model.eval()
        f = open(output_file, "w")
        f.write("id,target,prediction\n")
        targets = []
        predictions = []

        with torch.no_grad():
            ids = data_loader.dataset.ids
            sample_idx = 0
            for dat in data_loader:
                g, lg, text, target = dat
                out_data = model([g.to(device), lg.to(device), text])
                if isinstance(out_data, dict):
                    out_data = out_data['predictions']
                out_data = out_data.cpu().numpy().tolist()
                target = target.cpu().numpy().flatten().tolist()

                batch_size = len(target) if isinstance(target, list) else 1
                if batch_size == 1 and not isinstance(target, list):
                    target = [target]
                    out_data = [out_data]

                for k in range(batch_size):
                    id = ids[sample_idx + k]
                    pred_value = out_data[k]
                    f.write("%s, %6f, %6f\n" % (id, target[k], pred_value))
                    targets.append(target[k])
                    predictions.append(pred_value)
                sample_idx += batch_size
        f.close()

        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(np.array(targets), np.array(predictions))
        print(f"  {dataset_name} MAE: {mae:.6f}")
        return mae

    # ==================== 1. 最后一个epoch的预测 ====================
    print("\n📊 [1/3] 保存最后一个epoch的预测结果...")
    print("-" * 60)

    # 验证集
    last_val_file = os.path.join(config.output_dir, "predictions_last_epoch_val.csv")
    save_predictions(net, val_loader, last_val_file, "Last Epoch - Validation")

    # 测试集
    last_test_file = os.path.join(config.output_dir, "predictions_last_epoch_test.csv")
    save_predictions(net, test_loader, last_test_file, "Last Epoch - Test")
    print(f"✅ 已保存: predictions_last_epoch_val.csv, predictions_last_epoch_test.csv")

    # ==================== 2. 最佳验证集模型的预测 ====================
    print("\n📊 [2/3] 加载最佳验证集模型并保存预测...")
    print("-" * 60)

    best_val_checkpoint_path = os.path.join(config.output_dir, "best_val_model.pt")
    if os.path.exists(best_val_checkpoint_path):
        best_val_checkpoint = torch.load(best_val_checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(best_val_checkpoint['model'])

        # 验证集
        best_val_val_file = os.path.join(config.output_dir, "predictions_best_val_model_val.csv")
        save_predictions(net, val_loader, best_val_val_file, "Best Val Model - Validation")

        # 测试集
        best_val_test_file = os.path.join(config.output_dir, "predictions_best_val_model_test.csv")
        save_predictions(net, test_loader, best_val_test_file, "Best Val Model - Test")
        print(f"✅ 已保存: predictions_best_val_model_val.csv, predictions_best_val_model_test.csv")
    else:
        print("⚠️  未找到best_val_model.pt，跳过")

    # ==================== 3. 最佳测试集模型的预测 ====================
    print("\n📊 [3/3] 加载最佳测试集模型并保存预测...")
    print("-" * 60)

    best_test_checkpoint_path = os.path.join(config.output_dir, "best_test_model.pt")
    if os.path.exists(best_test_checkpoint_path):
        best_test_checkpoint = torch.load(best_test_checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(best_test_checkpoint['model'])

        # 验证集
        best_test_val_file = os.path.join(config.output_dir, "predictions_best_test_model_val.csv")
        save_predictions(net, val_loader, best_test_val_file, "Best Test Model - Validation")

        # 测试集
        best_test_test_file = os.path.join(config.output_dir, "predictions_best_test_model_test.csv")
        save_predictions(net, test_loader, best_test_test_file, "Best Test Model - Test")
        print(f"✅ 已保存: predictions_best_test_model_val.csv, predictions_best_test_model_test.csv")
    else:
        print("⚠️  未找到best_test_model.pt，跳过")

    # 保留兼容性：创建默认文件链接到最佳验证集模型的预测
    print("\n📝 创建默认预测文件（链接到最佳验证集模型）...")
    if os.path.exists(best_val_checkpoint_path):
        import shutil
        default_val = os.path.join(config.output_dir, "prediction_results_val_set.csv")
        default_test = os.path.join(config.output_dir, "prediction_results_test_set.csv")
        shutil.copy(best_val_val_file, default_val)
        shutil.copy(best_val_test_file, default_test)
        print(f"✅ 已创建: prediction_results_val_set.csv (= best_val_model)")
        print(f"✅ 已创建: prediction_results_test_set.csv (= best_val_model)")

    print("\n" + "="*60)
    print("✅ 所有预测结果已保存完成！")
    print("="*60)


    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
