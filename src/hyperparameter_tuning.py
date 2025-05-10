import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import optuna

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.brainauth_model import LightSiameseBrainAuth, SiameseBrainAuth
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, load_config, get_device, create_dirs,
    save_checkpoint, load_checkpoint, calculate_metrics,
    plot_metrics
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuthOptuna')

class OptunaOptimizer:
    """使用Optuna框架进行超参数优化的类"""
    
    def __init__(self, config_path: str, study_name: str = None):
        """初始化Optuna优化器
        
        Args:
            config_path: 基础配置文件路径
            study_name: Optuna研究名称，如果为None则自动生成
        """
        self.base_config_path = config_path
        self.config = load_config(config_path)
        self.optuna_config = load_config(self.config.get('optuna_config', './configs/optuna_config.yaml'))
        
        # 检查模型类型，确保是我们支持的类型
        self.model_name = self.config['model']['name']
        if self.model_name not in ['LightSiameseBrainAuth', 'SiameseBrainAuth']:
            logger.warning(f"注意：当前优化主要针对LightSiameseBrainAuth，但收到的模型是{self.model_name}")
        
        # 如果没有指定study_name，则创建一个基于时间戳和模型名称的名称
        if study_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.study_name = f"{self.model_name}_{timestamp}"
        else:
            self.study_name = study_name
            
        # 创建输出目录
        self.output_dir = os.path.join(
            self.config['output']['results_dir'], 
            'optuna', 
            self.study_name
        )
        create_dirs([self.output_dir])
        
        # 配置文件日志
        log_file = os.path.join(self.output_dir, f"optuna_{self.study_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # 设置设备
        self.device = get_device()
        logger.info(f"使用设备: {self.device}")
        
        # 获取数据加载器（只需要一次）
        logger.info("预加载数据集...")
        self.dataloaders = get_dataloaders(config_path)
        
        # 设置随机种子
        set_seed(self.config['seed'])
        
        # 设置优化指标
        self.direction = self.optuna_config.get('direction', 'maximize')
        self.metric_name = self.optuna_config.get('metric', 'f1')
        logger.info(f"优化目标: {self.direction} {self.metric_name}")
    
    def _create_model(self, trial: optuna.Trial) -> torch.nn.Module:
        """根据试验参数创建模型
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            创建的PyTorch模型
        """
        model_config = self.config['model']
        model_name = model_config['name']
        
        # 获取输入形状（默认保持不变）
        input_shape = tuple(model_config['input_shape'])
        
        # 主要支持LightSiameseBrainAuth
        if model_name == 'LightSiameseBrainAuth':
            # 超参数：卷积层通道数
            conv1_channels = trial.suggest_int(
                'conv1_channels', 
                self.optuna_config['params']['conv1_channels']['min'],
                self.optuna_config['params']['conv1_channels']['max'],
                step=self.optuna_config['params']['conv1_channels'].get('step', 2)
            )
            
            conv2_channels = trial.suggest_int(
                'conv2_channels', 
                self.optuna_config['params']['conv2_channels']['min'],
                self.optuna_config['params']['conv2_channels']['max'],
                step=self.optuna_config['params']['conv2_channels'].get('step', 4)
            )
            
            conv3_channels = trial.suggest_int(
                'conv3_channels', 
                self.optuna_config['params']['conv3_channels']['min'],
                self.optuna_config['params']['conv3_channels']['max'],
                step=self.optuna_config['params']['conv3_channels'].get('step', 8)
            )
            
            conv6_channels = trial.suggest_int(
                'conv6_channels', 
                self.optuna_config['params']['conv6_channels']['min'],
                self.optuna_config['params']['conv6_channels']['max'],
                step=self.optuna_config['params']['conv6_channels'].get('step', 16)
            )
            
            # 卷积通道列表
            conv_channels = [conv1_channels, conv2_channels, conv3_channels, conv6_channels]
            
            # 超参数：全连接层大小
            hidden_size = trial.suggest_int(
                'hidden_size',
                self.optuna_config['params']['hidden_size']['min'],
                self.optuna_config['params']['hidden_size']['max'],
                step=self.optuna_config['params']['hidden_size'].get('step', 32)
            )
            
            # 超参数：Dropout率
            dropout_rate = trial.suggest_float(
                'dropout_rate', 
                self.optuna_config['params']['dropout_rate']['min'],
                self.optuna_config['params']['dropout_rate']['max']
            )
            
            # 超参数：是否使用批归一化
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            
            # 构建LightSiameseBrainAuth模型
            model = LightSiameseBrainAuth(
                input_shape=input_shape,
                conv_channels=conv_channels,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm
            )
            
        # 对其他模型提供基本兼容支持
        elif model_name == 'SiameseBrainAuth':
            # 卷积层通道列表默认值
            conv_channels = [16, 32, 64, 128, 256, 512]
            
            # 超参数：全连接层大小
            hidden_size = trial.suggest_int(
                'hidden_size',
                self.optuna_config['params']['hidden_size']['min'],
                self.optuna_config['params']['hidden_size']['max'],
                step=self.optuna_config['params']['hidden_size'].get('step', 64)
            )
            
            # 超参数：Dropout率
            dropout_rate = trial.suggest_float(
                'dropout_rate', 
                self.optuna_config['params']['dropout_rate']['min'],
                self.optuna_config['params']['dropout_rate']['max']
            )
            
            # 构建SiameseBrainAuth模型
            model = SiameseBrainAuth(
                input_shape=input_shape,
                hidden_size=hidden_size,
                dropout_rate=dropout_rate
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        return model.to(self.device)
    
    def _create_optimizer(self, trial: optuna.Trial, model: torch.nn.Module) -> torch.optim.Optimizer:
        """创建优化器
        
        Args:
            trial: Optuna试验对象
            model: 模型
            
        Returns:
            PyTorch优化器
        """
        # 超参数：学习率
        lr = trial.suggest_float(
            'learning_rate',
            self.optuna_config['params']['learning_rate']['min'],
            self.optuna_config['params']['learning_rate']['max'],
            log=True  # 对数尺度
        )
        
        # 超参数：权重衰减
        weight_decay = trial.suggest_float(
            'weight_decay',
            self.optuna_config['params']['weight_decay']['min'],
            self.optuna_config['params']['weight_decay']['max'],
            log=True  # 对数尺度
        )
        
        # 超参数：优化器类型
        optimizer_name = trial.suggest_categorical(
            'optimizer',
            ['adam', 'adamw', 'sgd']
        )
        
        # 根据优化器类型创建相应的优化器
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            # SGD额外参数: 动量
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=momentum,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def _create_loss_function(self, trial: optuna.Trial) -> torch.nn.Module:
        """创建损失函数
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            PyTorch损失函数
        """
        # 可选择不同的损失函数
        loss_name = trial.suggest_categorical(
            'loss_function',
            self.optuna_config['params']['loss_function']['choices']
        )
        
        # 创建损失函数
        if loss_name == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'bce':
            return torch.nn.BCEWithLogitsLoss()
        elif loss_name == 'focal':
            try:
                from pytorch_toolbelt.losses import FocalLoss
                # 调整Focal Loss的gamma参数
                gamma = trial.suggest_float(
                    'focal_gamma',
                    self.optuna_config['params']['focal_gamma']['min'],
                    self.optuna_config['params']['focal_gamma']['max']
                )
                return FocalLoss(gamma=gamma)
            except ImportError:
                logger.warning("未安装pytorch_toolbelt，改用BCEWithLogitsLoss")
                return torch.nn.BCEWithLogitsLoss()
        elif loss_name == 'contrastive':
            try:
                from pytorch_metric_learning.losses import ContrastiveLoss
                # 调整ContrastiveLoss的margin参数
                margin = trial.suggest_float(
                    'contrastive_margin',
                    self.optuna_config['params']['contrastive_margin']['min'],
                    self.optuna_config['params']['contrastive_margin']['max']
                )
                return ContrastiveLoss(margin=margin)
            except ImportError:
                logger.warning("未安装pytorch_metric_learning，改用BCEWithLogitsLoss")
                return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.BCEWithLogitsLoss()  # 默认回退选项
    
    def _create_scheduler(self, trial: optuna.Trial, optimizer: torch.optim.Optimizer, 
                        num_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器
        
        Args:
            trial: Optuna试验对象
            optimizer: 优化器
            num_epochs: 总训练轮数
            
        Returns:
            PyTorch学习率调度器
        """
        # 选择不同的调度器
        scheduler_name = trial.suggest_categorical(
            'lr_scheduler',
            self.optuna_config['params']['lr_scheduler']['choices']
        )
        
        # 创建调度器
        if scheduler_name == 'step':
            step_size = trial.suggest_int(
                'step_size',
                self.optuna_config['params']['step_size']['min'],
                self.optuna_config['params']['step_size']['max']
            )
            gamma = trial.suggest_float(
                'gamma',
                self.optuna_config['params']['gamma']['min'],
                self.optuna_config['params']['gamma']['max']
            )
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_name == 'plateau':
            patience = trial.suggest_int(
                'patience',
                self.optuna_config['params']['patience']['min'],
                self.optuna_config['params']['patience']['max']
            )
            gamma = trial.suggest_float(
                'gamma',
                self.optuna_config['params']['gamma']['min'],
                self.optuna_config['params']['gamma']['max']
            )
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=gamma, verbose=True
            )
        elif scheduler_name == 'one_cycle':
            # One Cycle LR - 适用于训练轮数较少的场景
            max_lr = optimizer.param_groups[0]['lr'] * 10  # 最大学习率是初始学习率的10倍
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=num_epochs
            )
        else:
            return None  # 不使用调度器
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna优化的目标函数
        
        Args:
            trial: Optuna试验对象
            
        Returns:
            验证集上的性能指标
        """
        # 记录开始
        logger.info(f"开始试验 {trial.number}")
        
        # 创建模型
        model = self._create_model(trial)
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 创建损失函数
        criterion = self._create_loss_function(trial)
        
        # 创建优化器
        optimizer = self._create_optimizer(trial, model)
        
        # 获取超参数
        epochs = self.optuna_config.get('epochs', self.config['train']['epochs'])
        patience = self.optuna_config.get('early_stopping_patience', 
                                          self.config['train']['early_stopping_patience'])
        
        # 创建调度器
        scheduler = self._create_scheduler(trial, optimizer, epochs)
        
        # 获取数据加载器
        train_loader = self.dataloaders['train']
        val_loader = self.dataloaders['val']
        
        # 记录当前试验的参数
        logger.info(f"试验 {trial.number} 参数:")
        for key, value in trial.params.items():
            logger.info(f"  {key}: {value}")
        
        try:
            # 训练循环
            best_score = float('-inf') if self.direction == 'maximize' else float('inf')
            best_metrics = None
            early_stopping_counter = 0
            
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for eeg1, eeg2, labels in train_loader:
                    eeg1, eeg2 = eeg1.to(self.device), eeg2.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(eeg1, eeg2)
                    
                    # 计算损失
                    if isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                        loss = criterion(outputs.view(-1), labels.float())
                        preds = (torch.sigmoid(outputs.view(-1)) > 0.5).int()
                    else:
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)
                    
                    # 统计正确预测数
                    train_correct += (preds == labels).sum().item()
                    train_total += labels.size(0)
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * labels.size(0)
                
                train_loss /= len(train_loader.dataset)
                train_acc = train_correct / train_total
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_preds = []
                all_labels = []
                all_scores = []
                
                with torch.no_grad():
                    for eeg1, eeg2, labels in val_loader:
                        eeg1, eeg2 = eeg1.to(self.device), eeg2.to(self.device)
                        labels = labels.to(self.device)
                        
                        # 前向传播
                        outputs = model(eeg1, eeg2)
                        
                        # 计算损失
                        if isinstance(criterion, torch.nn.BCELoss) or isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                            loss = criterion(outputs.view(-1), labels.float())
                            scores = torch.sigmoid(outputs.view(-1))
                            preds = (scores > 0.5).int()
                        else:
                            loss = criterion(outputs, labels)
                            scores = torch.softmax(outputs, dim=1)[:, 1]
                            preds = torch.argmax(outputs, dim=1)
                        
                        # 统计正确预测数
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)
                        
                        val_loss += loss.item() * labels.size(0)
                        
                        # 收集预测和标签
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        all_scores.extend(scores.cpu().numpy())
                
                val_loss /= len(val_loader.dataset)
                val_acc = val_correct / val_total
                
                # 计算其他指标
                val_metrics = calculate_metrics(
                    y_true=np.array(all_labels),
                    y_pred=np.array(all_preds),
                    y_score=np.array(all_scores)
                )
                
                # 更新学习率
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # 记录结果
                logger.info(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
                logger.info(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, F1: {val_metrics['f1']:.4f}")
                
                # 检查是否是最佳模型
                current_score = val_metrics[self.metric_name]
                is_best = False
                
                if self.direction == 'maximize':
                    is_best = current_score > best_score
                else:
                    is_best = current_score < best_score
                
                if is_best:
                    best_score = current_score
                    best_metrics = val_metrics.copy()
                    early_stopping_counter = 0
                    logger.info(f"发现新的最佳模型! {self.metric_name}: {best_score:.4f}")
                    
                    # 保存最佳模型
                    checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'trial_number': trial.number,
                        'params': trial.params,
                        'val_metrics': val_metrics
                    }
                    
                    # 保存到试验特定目录
                    trial_dir = os.path.join(self.output_dir, f"trial_{trial.number}")
                    create_dirs([trial_dir])
                    torch.save(checkpoint, os.path.join(trial_dir, 'model_best.pt'))
                else:
                    early_stopping_counter += 1
                    logger.info(f"最佳{self.metric_name}未改善, 计数器: {early_stopping_counter}/{patience}")
                
                # 早停
                if early_stopping_counter >= patience:
                    logger.info(f"触发早停! {patience} 个epochs未改善")
                    break
                
                # 报告中间值给Optuna
                trial.report(current_score, epoch)
                
                # 处理Optuna的剪枝
                if trial.should_prune():
                    logger.info(f"试验 {trial.number} 在 epoch {epoch+1} 被剪枝")
                    raise optuna.exceptions.TrialPruned()
            
            # 记录最终结果
            logger.info(f"试验 {trial.number} 完成")
            logger.info(f"最佳 {self.metric_name}: {best_score:.4f}")
            
            # 返回目标指标
            return best_score
            
        except Exception as e:
            logger.error(f"试验 {trial.number} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise e
    
    def optimize(self, n_trials: int = None, timeout: int = None) -> optuna.study.Study:
        """运行优化过程
        
        Args:
            n_trials: 试验次数，如果为None则使用配置中的值
            timeout: 超时（秒），如果为None则使用配置中的值
            
        Returns:
            Optuna Study对象
        """
        # 设置试验次数
        if n_trials is None:
            n_trials = self.optuna_config.get('n_trials', 20)
        
        # 设置超时
        if timeout is None:
            timeout = self.optuna_config.get('timeout', None)
        
        # 获取方向（最大化/最小化）
        direction = self.direction
        
        # 创建存储
        storage_name = f"sqlite:///{os.path.join(self.output_dir, f'{self.study_name}.db')}"
        
        # 创建或加载研究
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            storage=storage_name,
            load_if_exists=True
        )
        
        # 记录优化开始
        logger.info(f"开始 {self.study_name} 优化")
        logger.info(f"目标指标: {direction} {self.metric_name}")
        logger.info(f"计划试验次数: {n_trials}")
        if timeout:
            logger.info(f"超时: {timeout} 秒")
        
        # 运行优化
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        # 记录最佳结果
        logger.info("优化完成")
        logger.info(f"最佳试验: {study.best_trial.number}")
        logger.info(f"最佳值: {study.best_value:.4f}")
        logger.info(f"最佳参数: {study.best_params}")
        
        # 生成图表
        self._generate_plots(study)
        
        # 保存最佳配置
        self._save_best_config(study)
        
        return study
    
    def _generate_plots(self, study: optuna.study.Study) -> None:
        """生成优化过程的图表
        
        Args:
            study: Optuna Study对象
        """
        try:
            # 确保输出目录存在
            plots_dir = os.path.join(self.output_dir, 'plots')
            create_dirs([plots_dir])
            
            # 优化历史图
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'optimization_history.png'))
            plt.close()
            
            # 参数重要性图
            try:
                plt.figure(figsize=(10, 6))
                optuna.visualization.matplotlib.plot_param_importances(study)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'param_importances.png'))
                plt.close()
            except:
                logger.warning("无法生成参数重要性图表")
            
            # 并行坐标图
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'parallel_coordinate.png'))
            plt.close()
            
            # 切片图（对每个参数）
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_slice(study)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'slice.png'))
            plt.close()
            
            # 轮廓图（如果参数数量>=2）
            try:
                plt.figure(figsize=(10, 8))
                optuna.visualization.matplotlib.plot_contour(study)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'contour.png'))
                plt.close()
            except:
                logger.warning("无法生成轮廓图表")
            
            logger.info(f"图表保存到: {plots_dir}")
        except Exception as e:
            logger.error(f"生成图表时出错: {str(e)}")
            logger.error("继续执行不生成图表")
    
    def _save_best_config(self, study: optuna.study.Study) -> None:
        """保存最佳配置为新的YAML文件
        
        Args:
            study: Optuna Study对象
        """
        # 创建配置的深拷贝
        config = self.config.copy()
        
        # 更新模型配置
        model_config = config['model'].copy()
        model_name = model_config['name']
        best_params = study.best_params
        
        # 根据模型类型设置参数
        if model_name == 'LightSiameseBrainAuth':
            # 更新卷积通道配置
            conv_channels = [
                best_params.get('conv1_channels', 8),
                best_params.get('conv2_channels', 16),
                best_params.get('conv3_channels', 32),
                best_params.get('conv6_channels', 64)
            ]
            model_config['conv_channels'] = conv_channels
            
            # 更新其他模型参数
            model_config['hidden_size'] = best_params.get('hidden_size', 128)
            model_config['dropout_rate'] = best_params.get('dropout_rate', 0.3)
            model_config['use_batch_norm'] = best_params.get('use_batch_norm', True)
            
        elif model_name == 'SiameseBrainAuth':
            # 更新全连接层参数
            model_config['hidden_size'] = best_params.get('hidden_size', 256)
            model_config['dropout_rate'] = best_params.get('dropout_rate', 0.5)
        
        # 更新配置中的模型设置
        config['model'] = model_config
        
        # 更新训练配置
        train_config = config['train'].copy()
        
        # 学习率
        train_config['learning_rate'] = best_params.get('learning_rate', train_config['learning_rate'])
        
        # 权重衰减
        train_config['weight_decay'] = best_params.get('weight_decay', train_config['weight_decay'])
        
        # 优化器类型
        train_config['optimizer'] = best_params.get('optimizer', 'adam')
        if train_config['optimizer'] == 'sgd':
            train_config['momentum'] = best_params.get('momentum', 0.9)
        
        # 损失函数
        train_config['loss_function'] = best_params.get('loss_function', train_config['loss_function'])
        
        # 损失函数相关参数
        if train_config['loss_function'] == 'focal' and 'focal_gamma' in best_params:
            train_config['focal_gamma'] = best_params['focal_gamma']
        elif train_config['loss_function'] == 'contrastive' and 'contrastive_margin' in best_params:
            train_config['contrastive_margin'] = best_params['contrastive_margin']
        
        # 学习率调度器
        train_config['lr_scheduler'] = best_params.get('lr_scheduler', train_config['lr_scheduler'])
        
        # 调度器相关参数
        scheduler_params = {}
        if train_config['lr_scheduler'] == 'step':
            scheduler_params = {
                'step_size': best_params.get('step_size', 10),
                'gamma': best_params.get('gamma', 0.1)
            }
        elif train_config['lr_scheduler'] == 'plateau':
            scheduler_params = {
                'patience': best_params.get('patience', 5),
                'gamma': best_params.get('gamma', 0.1)
            }
        elif train_config['lr_scheduler'] == 'one_cycle':
            scheduler_params = {
                'max_lr': best_params.get('learning_rate', 0.001) * 10,
                'pct_start': 0.3
            }
        
        train_config['lr_scheduler_params'] = scheduler_params
        
        # 更新配置中的训练设置
        config['train'] = train_config
        
        # 保存配置文件
        best_config_path = os.path.join(self.output_dir, 'best_config.yaml')
        with open(best_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"最佳配置保存到: {best_config_path}")
        
        # 为了便于后续使用，额外保存一个简化的参数文件
        best_params_path = os.path.join(self.output_dir, 'best_params.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)
        
        logger.info(f"最佳参数保存到: {best_params_path}")
        
        # 保存最佳模型
        try:
            # 找到最佳试验目录
            trial_dir = os.path.join(self.output_dir, f"trial_{study.best_trial.number}")
            model_path = os.path.join(trial_dir, 'model_best.pt')
            
            if os.path.exists(model_path):
                # 创建一个最终模型目录
                final_dir = os.path.join(self.output_dir, 'final_model')
                create_dirs([final_dir])
                
                # 复制模型到最终目录
                import shutil
                shutil.copy(model_path, os.path.join(final_dir, 'model_best.pt'))
                
                logger.info(f"最佳模型保存到: {os.path.join(final_dir, 'model_best.pt')}")
        except Exception as e:
            logger.error(f"保存最佳模型时出错: {str(e)}")


def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description='Optuna超参数调优')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='基础配置文件路径')
    parser.add_argument('--optuna_config', type=str, default='./configs/optuna_config.yaml', help='Optuna配置文件路径')
    parser.add_argument('--study_name', type=str, default=None, help='Optuna研究名称')
    parser.add_argument('--n_trials', type=int, default=None, help='试验次数')
    parser.add_argument('--timeout', type=int, default=None, help='超时时间（秒）')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--test_best', action='store_true', help='测试最佳模型而不是运行优化')
    return parser


def main():
    """主函数"""
    # 解析命令行参数
    parser = setup_argparse()
    args = parser.parse_args()
    
    # 设置GPU
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 加载基础配置
    config = load_config(args.config)
    
    # 更新配置中的optuna_config路径
    config['optuna_config'] = args.optuna_config
    
    # 更新随机种子（如果指定）
    if args.seed is not None:
        config['seed'] = args.seed
        set_seed(args.seed)
    
    # 检查是测试最佳模型还是运行优化
    if args.test_best:
        if args.study_name is None:
            logger.error("测试最佳模型需要指定 --study_name")
            return
        
        # 评估最佳模型
        evaluate_best_model(args.study_name, args.config)
    else:
        # 创建优化器并运行优化
        optimizer = OptunaOptimizer(args.config, args.study_name)
        study = optimizer.optimize(n_trials=args.n_trials, timeout=args.timeout)
        
        logger.info("优化完成!")
        logger.info(f"最佳 {optimizer.metric_name}: {study.best_value:.4f}")
        logger.info(f"最佳参数: {study.best_params}")


def evaluate_best_model(study_name, config_path):
    """评估指定研究的最佳模型
    
    Args:
        study_name: Optuna研究名称
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    
    # 确定研究目录
    output_dir = os.path.join(config['output']['results_dir'], 'optuna', study_name)
    
    # 加载最佳配置
    best_config_path = os.path.join(output_dir, 'best_config.yaml')
    if not os.path.exists(best_config_path):
        logger.error(f"找不到最佳配置文件: {best_config_path}")
        return
    
    # 加载存储以获取最佳试验编号
    storage_path = os.path.join(output_dir, f'{study_name}.db')
    if not os.path.exists(storage_path):
        logger.error(f"找不到研究存储: {storage_path}")
        return
    
    try:
        study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage_path}")
        best_trial = study.best_trial
        
        # 加载最佳模型
        model_path = os.path.join(output_dir, 'final_model', 'model_best.pt')
        if not os.path.exists(model_path):
            logger.error(f"找不到最佳模型: {model_path}")
            return
        
        # 输出评估信息
        logger.info(f"评估研究 {study_name} 的最佳模型")
        logger.info(f"最佳试验: {best_trial.number}")
        logger.info(f"最佳参数: {best_trial.params}")
        
        # 使用脚本评估模型
        # 这里我们使用单独的train.py脚本运行测试，
        # 而不是在这里直接实现评估逻辑
        import subprocess
        
        cmd = [
            sys.executable, 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "train.py"), 
            "--config", best_config_path,
            "--mode", "test",
            "--checkpoint", model_path
        ]
        
        logger.info(f"运行评估命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
    except Exception as e:
        logger.error(f"评估模型时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()