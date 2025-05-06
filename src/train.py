import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import logging
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.brainauth_model import P3DCNN, SiameseBrainAuth
from src.dataset import BrainAuthDataset, get_dataloaders
from src.utils import (
    set_seed, load_config, get_device, create_dirs,
    save_checkpoint, load_checkpoint, calculate_metrics,
    plot_metrics, plot_confusion_matrix, plot_roc_curve
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuth')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        
    返回:
        avg_loss: 平均损失
        all_targets: 所有真实标签
        all_predictions: 所有预测结果
        all_scores: 所有预测分数
    """

    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_scores = []

    pbar = tqdm(dataloader, desc="Training")

    for eeg1, eeg2, labels in pbar:
         # 将数据移动到设备上
        eeg1, eeg2 = eeg1.to(device), eeg2.to(device)
        labels = labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(eeg1, eeg2)
        
        # 计算损失
        if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
            # 对于二元分类，需要调整标签和输出形状
            loss = criterion(outputs.view(-1), labels.float())
            scores = outputs.view(-1).detach().cpu().numpy()
            predictions = (scores > 0.5).astype(int)
        else:
            # 对于交叉熵损失
            loss = criterion(outputs, labels)
            scores = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 更新统计信息
        running_loss += loss.item() * labels.size(0)
        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predictions)
        all_scores.extend(scores)
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item()})

    avg_loss = running_loss / len(dataloader.dataset)

    return avg_loss, np.array(all_targets), np.array(all_predictions), np.array(all_scores)

def validate(model, dataloader, criterion, device):
    """
    验证模型性能
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    返回:
        avg_loss: 平均损失
        metrics: 评估指标字典
    """

    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    all_scores = []

    with torch.no_grad():
        for eeg1, eeg2, labels in dataloader:
            # 将数据移动到设备上
            eeg1, eeg2 = eeg1.to(device), eeg2.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(eeg1, eeg2)

            # 计算损失
            if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
                loss = criterion(outputs.view(-1), labels.float())
                scores = outputs.view(-1).detach().cpu().numpy()
                predictions = (scores > 0.5).astype(int)
            else:
                loss = criterion(outputs, labels)
                scores = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            # 更新统计信息
            running_loss += loss.item() * labels.size(0)
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
            all_scores.extend(scores)

    avg_loss = running_loss / len(dataloader.dataset)

    metrics = calculate_metrics(
        y_true=np.array(all_targets),
        y_pred=np.array(all_predictions),
        y_score=np.array(all_scores)
    )

    return avg_loss, metrics

def train_model(config_path):
    """
    训练模型的主函数
    
    参数:
        config_path: 配置文件路径
    """

    config = load_config(config_path)

    set_seed(config['seed'])

    device = get_device()

    # 创建输出目录
    output_dirs = [
        config['output']['checkpoint_dir'],
        config['output']['log_dir'],
        config['output']['results_dir']
    ]
    create_dirs(output_dirs)

    # 配置文件日志
    log_file = os.path.join(
        config['output']['log_dir'],
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # 记录配置信息
    logger.info("开始训练BrainAuth模型")
    logger.info(f"配置: {config}")

     # 获取数据加载器
    logger.info("加载数据集...")
    dataloaders = get_dataloaders(config_path)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")

    # 初始化模型
    model_config = config['model']
    logger.info(f"初始化模型: {model_config['name']}")

    if model_config['name'] == 'P3DCNN':
        model = P3DCNN(
            input_shape=tuple(model_config['input_shape']),
            num_classes=model_config['num_classes']
        )
    elif model_config['name'] == 'SiameseBrainAuth':
        model = SiameseBrainAuth(
            input_shape=tuple(model_config['input_shape'])
        )
    else:
        raise ValueError(f"不支持的模型: {model_config['name']}")
    
    model = model.to(device)

    # 如果有预训练权重，加载它们
    if model_config['pretrained']:
        logger.info(f"加载预训练权重: {model_config['pretrained']}")
        load_checkpoint(model_config['pretrained'], model)
    
    # 定义损失函数
    train_config = config['train']
    loss_name = train_config['loss_function']
    logger.info(f"使用损失函数: {loss_name}")
    
    if loss_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_name == 'focal':
        from pytorch_toolbelt.losses import FocalLoss
        criterion = FocalLoss()
    elif loss_name == 'contrastive':
        from pytorch_metric_learning.losses import ContrastiveLoss
        criterion = ContrastiveLoss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")


    # 定义优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # 定义学习率调度器
    scheduler_type = train_config['lr_scheduler']
    scheduler_params = train_config['lr_scheduler_params']
    
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['epochs']
        )
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_params['gamma'],
            patience=scheduler_params['patience'],
            verbose=True
        )
    else:
        scheduler = None

    # 训练模型
    num_epochs = train_config['epochs']
    best_metric = float('-inf')  # 用于跟踪最佳验证性能
    best_metric_name = config['evaluation']['best_metric']
    early_stopping_counter = 0
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
        'val_auc': [], 'val_eer': []
    }
    
    logger.info(f"开始训练 {num_epochs} 个epochs...")

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss, train_targets, train_preds, train_scores = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 计算训练指标
        train_metrics = calculate_metrics(train_targets, train_preds, train_scores)
        
        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        # 打印指标
        logger.info(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        logger.info(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 更新学习率
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # 保存模型# 记录指标历史
        metrics_history['train_loss'].append(train_loss)
        metrics_history['val_loss'].append(val_loss)
        metrics_history['train_accuracy'].append(train_metrics['accuracy'])
        metrics_history['val_accuracy'].append(val_metrics['accuracy'])
        metrics_history['train_precision'].append(train_metrics['precision'])
        metrics_history['val_precision'].append(val_metrics['precision'])
        metrics_history['train_recall'].append(train_metrics['recall'])
        metrics_history['val_recall'].append(val_metrics['recall'])
        metrics_history['train_f1'].append(train_metrics['f1'])
        metrics_history['val_f1'].append(val_metrics['f1'])
        
        if 'auc' in val_metrics:
            metrics_history['val_auc'].append(val_metrics['auc'])
        if 'eer' in val_metrics:
            metrics_history['val_eer'].append(val_metrics['eer'])
        
        # 检查是否是最佳模型
        current_metric = val_metrics[best_metric_name]
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            logger.info(f"发现新的最佳模型! {best_metric_name}: {best_metric:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"最佳{best_metric_name}未改善, 计数器: {early_stopping_counter}/{train_config['early_stopping_patience']}")
        
        # 保存检查点
        if is_best or (epoch + 1) % config['output']['save_frequency'] == 0 or epoch == num_epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics,
                'config': config
            }
            
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            save_checkpoint(
                checkpoint,
                is_best,
                config['output']['checkpoint_dir'],
                filename=f'checkpoint_epoch_{epoch+1}.pth'
            )
        
        # 早停
        if train_config['early_stopping'] and early_stopping_counter >= train_config['early_stopping_patience']:
            logger.info(f"触发早停! {train_config['early_stopping_patience']} 个epochs未改善")
            break

    # 训练结束后绘制指标历史
    logger.info("训练完成，绘制指标图...")
    metrics_plot_path = os.path.join(config['output']['results_dir'], 'training_metrics.png')
    plot_metrics(metrics_history, save_path=metrics_plot_path)
    
    # 加载最佳模型进行测试评估
    best_model_path = os.path.join(config['output']['checkpoint_dir'], 'model_best.pth')
    logger.info(f"加载最佳模型进行评估: {best_model_path}")
    load_checkpoint(best_model_path, model)
    
    # 在测试集上评估
    logger.info("在测试集上评估模型...")
    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    
    logger.info(f"测试集性能：")
    logger.info(f"损失: {test_loss:.4f}")
    logger.info(f"准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"精确率: {test_metrics['precision']:.4f}")
    logger.info(f"召回率: {test_metrics['recall']:.4f}")
    logger.info(f"F1分数: {test_metrics['f1']:.4f}")
    
    if 'auc' in test_metrics:
        logger.info(f"AUC: {test_metrics['auc']:.4f}")
    if 'eer' in test_metrics:
        logger.info(f"EER: {test_metrics['eer']:.4f}")
    
    # 保存混淆矩阵
    cm_path = os.path.join(config['output']['results_dir'], 'confusion_matrix.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], save_path=cm_path)
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'config': config
    }
    
    torch.save(
        test_results,
        os.path.join(config['output']['results_dir'], 'test_results.pth')
    )
    
    logger.info(f"评估结果保存到 {config['output']['results_dir']}")
    
    return model, test_metrics

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser(description='训练BrainAuth模型')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='配置文件路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train_model(args.config)




