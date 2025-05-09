#!/usr/bin/env python
"""
预处理并缓存BrainAuth数据
"""

import argparse
import logging
import os
import yaml
from pathlib import Path
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BrainAuth')

def main():
    parser = argparse.ArgumentParser(description='预处理并缓存BrainAuth数据')
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据根目录 (覆盖配置文件设置)')
    parser.add_argument('--cache_dir', type=str, help='HDF5缓存目录 (覆盖配置文件设置)')
    parser.add_argument('--condition', type=str, choices=['eyes_open', 'eyes_closed'], help='实验条件 (覆盖配置文件设置)')
    parser.add_argument('--reset', action='store_true', help='重置已有缓存')
    parser.add_argument('--workers', type=int, default=0, help='处理工作线程数 (0=自动)')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用命令行参数覆盖配置文件设置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.cache_dir:
        config['data']['hdf5_cache_dir'] = args.cache_dir
    if args.condition:
        config['data']['condition'] = args.condition
    if args.reset:
        config['data']['reset_cache'] = True
    if args.workers > 0:
        config['data']['num_workers'] = args.workers
    
    # 创建临时配置文件用于预处理
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name
    
    try:
        # 导入预处理函数
        from dataset import preprocess_and_cache_all_data
        
        # 运行预处理
        logger.info(f"开始预处理 {config['data']['condition']} 条件下的数据...")
        preprocess_and_cache_all_data(temp_config_path)
        logger.info("预处理完成!")
        
    except Exception as e:
        logger.error(f"预处理时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    finally:
        # 删除临时配置文件
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
    
    return 0

if __name__ == "__main__":
    exit(main())