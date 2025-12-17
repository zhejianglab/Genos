#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding提取API服务
基于embedding_extraction.py的逻辑，提供DNA序列embedding提取的API接口
支持GPU、NPU和CPU设备
"""

import torch
import os
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置OpenBLAS线程数，避免警告
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['NUMEXPR_NUM_THREADS'] = '24'

# 确定运行设备和数据类型
def get_device_and_dtype(force_cpu=False):
    """
    确定模型运行设备和数据类型
    优先级: NPU > GPU > CPU
    
    Args:
        force_cpu: 是否强制使用CPU
        
    Returns:
        tuple: (device, torch_dtype)
    """
    device = None
    torch_dtype = None
    
    # 如果强制使用CPU
    if force_cpu:
        device = "cpu"
        torch_dtype = torch.float16
        logger.info("强制使用CPU进行推理")
        return device, torch_dtype
    
    # 优先检查NPU
    try:
        if torch.npu.is_available():
            device = "npu:0"
            torch_dtype = torch.float16
            logger.info("使用华为昇腾NPU进行推理")
            return device, torch_dtype
    except Exception as e:
        logger.warning(f"检测NPU时出错: {str(e)}")
    
    # 如果NPU不可用，尝试使用GPU
    if torch.cuda.is_available():
        device = "cuda:0"
        torch_dtype = torch.bfloat16
        logger.info("使用NVIDIA GPU进行推理")
    else:
        # 使用CPU作为最后的备选
        device = "cpu"
        torch_dtype = torch.float16
        logger.info("未检测到可用的GPU/NPU，使用CPU进行推理")
    
    return device, torch_dtype

# 获取默认设备
default_device, default_dtype = get_device_and_dtype()

import asyncio
lock = asyncio.Lock()

class EmbeddingExtractor:
    """Embedding提取器类，封装序列embedding提取逻辑"""
    
    def __init__(self, model_path, model_type="flash", device=None, torch_dtype=None, model_name="1.2B", force_cpu=False, device_map=None, memory_ratio=0.9):
        """
        初始化Embedding提取器
        
        Args:
            model_path (str): 模型路径
            model_type (str): 模型类型，"flash" 或 "no_flash"
            device (str, optional): 设备类型，如果为None则自动选择。支持单设备（如"cuda:0"）或多设备列表（如["cuda:0", "cuda:1"]）
            torch_dtype (torch.dtype, optional): 数据类型，如果为None则根据设备自动选择
            model_name (str): 模型名称
            force_cpu (bool): 是否强制使用CPU
            device_map (str or dict, optional): 设备映射方式
                - "auto": 自动分配模型到可用设备
                - "balanced": 平衡分配到所有设备
                - "sequential": 按顺序分配到设备
                - dict: 手动指定每层的设备映射
                - None: 使用单设备模式（device参数指定的设备）
            memory_ratio (float): 内存分配比例，默认0.9（使用90%的内存）
        """
        # 如果未指定设备，则自动选择
        if device is None or torch_dtype is None:
            auto_device, auto_dtype = get_device_and_dtype(force_cpu)
            if device is None:
                device = auto_device
            if torch_dtype is None:
                torch_dtype = auto_dtype
        
        # 处理设备参数：支持单设备字符串或多设备列表
        if isinstance(device, str):
            self.device = torch.device(device)
            self.devices = [device]  # 设备列表，用于多设备模式
            self.multi_device = False
        elif isinstance(device, list):
            # 多设备模式
            self.devices = device
            self.device = torch.device(device[0])  # 主设备
            self.multi_device = True
            logger.info(f"多设备模式: 使用设备 {self.devices}")
        else:
            self.device = torch.device(device)
            self.devices = [str(device)]
            self.multi_device = False
            
        self.torch_dtype = torch_dtype
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = model_name
        self.device_map = device_map
        self.memory_ratio = memory_ratio
        self._npu_manual_device_map = False  # 标记是否需要手动分配NPU设备
        self.load_model()
        
    def _get_available_devices(self):
        """
        获取可用的设备列表
        
        Returns:
            list: 可用设备列表，如 ["cuda:0", "cuda:1"] 或 ["npu:0", "npu:1"]
        """
        devices = []
        
        # 检查NPU
        try:
            if torch.npu.is_available():
                npu_count = torch.npu.device_count()
                devices.extend([f"npu:{i}" for i in range(npu_count)])
        except:
            pass
        
        # 检查CUDA
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            devices.extend([f"cuda:{i}" for i in range(cuda_count)])
        
        if not devices:
            devices = ["cpu"]
        
        return devices
    
    def get_gpu_memory_info(self):
        """获取GPU内存使用情况信息"""
        if self.device.type == "cuda":
            current_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return f"当前GPU内存使用: {current_allocated:.2f}GB, 最大峰值: {max_allocated:.2f}GB, 保留内存: {reserved:.2f}GB"
        return "不适用 (非CUDA设备)"
    
    def _get_device_memory_gb(self, device_str):
        """
        获取指定设备的总内存大小（GB）
        
        Args:
            device_str (str): 设备字符串，如 "cuda:0" 或 "npu:0"
            
        Returns:
            float: 设备总内存大小（GB），如果无法获取则返回None
        """
        logger.info(f"获取设备 {device_str} 的内存大小...")
        try:
            device_obj = torch.device(device_str)
            if device_obj.type == "cuda":
                # CUDA设备
                total_memory = torch.cuda.get_device_properties(device_obj).total_memory / 1024**3
                logger.info(f"设备 {device_str} 总内存: {total_memory:.2f}GB")
                return total_memory
            elif device_obj.type == "npu":
                # NPU设备
                # 尝试获取NPU内存信息
                try:
                    # 方法1: 使用torch.npu.get_device_properties获取内存信息
                    if hasattr(torch.npu, 'get_device_properties'):
                        props = torch.npu.get_device_properties(device_obj)
                        if hasattr(props, 'total_memory'):
                            total_memory = props.total_memory / 1024**3
                            logger.info(f"设备 {device_str} 总内存: {total_memory:.2f}GB")
                            return total_memory
                        # 有些版本可能使用其他属性名
                        if hasattr(props, 'global_mem_size'):
                            total_memory = props.global_mem_size / 1024**3
                            logger.info(f"设备 {device_str} 总内存: {total_memory:.2f}GB")
                            return total_memory
                    # 方法2: 尝试使用torch.npu.max_memory_reserved获取（需要先分配一些内存）
                    # 这个方法不太可靠，因为需要先分配内存
                    # 方法3: 对于昇腾NPU 910B，通常有32GB内存
                    # 如果无法获取，使用默认值32GB
                    logger.warning(f"无法直接获取NPU {device_str} 的内存信息，使用默认值32GB")
                    return 32.0  # 默认32GB，适用于昇腾NPU 910B
                except Exception as e:
                    logger.warning(f"获取NPU {device_str} 内存信息失败: {e}，使用默认值32GB")
                    return 32.0  # 默认32GB
            else:
                # CPU或其他设备
                logger.info(f"设备 {device_str} 是CPU，不限制内存")
                return None
        except Exception as e:
            logger.warning(f"获取设备 {device_str} 内存信息时出错: {e}")
            return None
    
    def _build_max_memory_dict(self, all_devices, specified_devices):
        """
        构建max_memory字典，用于限制设备使用
        
        Args:
            all_devices (list): 所有可用设备列表
            specified_devices (list): 用户指定的设备列表
            
        Returns:
            dict: max_memory字典，格式如 {0: "28GiB", 1: "0GiB"}（使用整数索引）
                 对于NPU设备，返回None（因为accelerate库不支持NPU设备名称）
        """
        # 检查是否包含NPU设备
        has_npu = any(dev.startswith('npu:') for dev in specified_devices + all_devices)
        
        if has_npu:
            # NPU设备不支持max_memory参数，accelerate库只支持整数（GPU/XPU）、'mps'、'cpu'和'disk'
            logger.warning("检测到NPU设备，max_memory参数不支持NPU设备名称，将使用device_map列表形式限制设备")
            return None
        
        logger.info(f"构建max_memory字典，指定设备: {specified_devices}, 内存分配比例: {self.memory_ratio}")
        max_memory = {}
        
        # 将设备字符串转换为整数索引，并构建映射
        device_to_index = {}
        for dev in all_devices:
            if dev.startswith('cuda:'):
                # CUDA设备：提取索引
                index = int(dev.split(':')[1])
                device_to_index[dev] = index
            elif dev.startswith('npu:'):
                # NPU设备：提取索引（虽然NPU不支持，但为了完整性）
                index = int(dev.split(':')[1])
                device_to_index[dev] = index
            else:
                # 其他设备（如cpu），跳过
                continue
        
        # 构建max_memory字典，使用整数索引
        for dev in all_devices:
            if dev not in device_to_index:
                continue
                
            index = device_to_index[dev]
            if dev in specified_devices:
                # 用户指定的设备，获取实际内存并乘以比例
                total_memory_gb = self._get_device_memory_gb(dev)
                if total_memory_gb is not None:
                    # 计算分配的内存（GB）
                    allocated_memory_gb = total_memory_gb * self.memory_ratio
                    # 转换为GiB格式（1GB ≈ 0.931GiB，但通常直接使用GB值）
                    max_memory[index] = f"{allocated_memory_gb:.2f}GiB"
                    logger.info(f"设备 {dev} (索引 {index}): 总内存 {total_memory_gb:.2f}GB, 分配 {allocated_memory_gb:.2f}GB ({self.memory_ratio*100:.1f}%)")
                else:
                    # 如果无法获取内存信息，使用一个较大的默认值
                    logger.warning(f"无法获取设备 {dev} 的内存信息，使用默认值 32GiB")
                    max_memory[index] = "32GiB"
            else:
                # 未指定的设备，设置为0，禁止使用
                max_memory[index] = "0GiB"
                logger.info(f"设备 {dev} (索引 {index}): 设置为0GiB，禁止使用")
        
        return max_memory
    
    def _manual_distribute_model_to_npu(self):
        """
        手动将模型分配到指定的NPU设备
        对于NPU设备，由于accelerate库不支持max_memory，我们需要手动分配模型层
        """
        logger.info(f"开始手动分配模型到NPU设备: {self.devices}")
        try:
            # 获取模型的所有命名模块
            named_modules = dict(self.model.named_modules())
            
            # 计算每个设备应该分配的层数
            num_devices = len(self.devices)
            
            # 获取所有需要分配的层（通常是transformer的层）
            layers_to_distribute = []
            for name, module in named_modules.items():
                # 识别transformer层（通常包含layer、block等关键字）
                if any(keyword in name.lower() for keyword in ['layer', 'block', 'transformer']):
                    # 排除顶层模块，只选择实际的层
                    if '.' in name and not name.startswith('model.'):
                        layers_to_distribute.append((name, module))
            
            # 如果找不到合适的层，尝试使用model.layers或model.blocks等
            if not layers_to_distribute:
                # 尝试直接访问model.layers或model.blocks
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                    layers = self.model.model.layers
                    for i, layer in enumerate(layers):
                        layers_to_distribute.append((f'model.layers.{i}', layer))
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'blocks'):
                    blocks = self.model.model.blocks
                    for i, block in enumerate(blocks):
                        layers_to_distribute.append((f'model.blocks.{i}', block))
                else:
                    # 如果还是找不到，使用所有命名模块（排除顶层）
                    layers_to_distribute = [(name, module) for name, module in named_modules.items() 
                                          if '.' in name and name.count('.') >= 2]
            
            if not layers_to_distribute:
                # 如果仍然找不到层，直接分配到第一个设备
                logger.warning("无法识别模型层结构，将整个模型分配到第一个设备")
                device_obj = torch.device(self.devices[0])
                self.model = self.model.to(device_obj)
                logger.info(f"模型已分配到设备: {self.devices[0]}")
                return
            
            # 按设备分配层
            layers_per_device = len(layers_to_distribute) // num_devices
            remainder = len(layers_to_distribute) % num_devices
            
            current_idx = 0
            for device_idx, device_str in enumerate(self.devices):
                device_obj = torch.device(device_str)
                # 计算这个设备应该分配的层数
                num_layers = layers_per_device + (1 if device_idx < remainder else 0)
                end_idx = current_idx + num_layers
                
                # 分配层到这个设备
                for layer_name, layer_module in layers_to_distribute[current_idx:end_idx]:
                    try:
                        layer_module.to(device_obj)
                        logger.debug(f"层 {layer_name} 已分配到 {device_str}")
                    except Exception as e:
                        logger.warning(f"将层 {layer_name} 分配到 {device_str} 失败: {e}")
                
                current_idx = end_idx
                logger.info(f"设备 {device_str} 已分配 {num_layers} 层")
            
            # 将embedding和输出层分配到第一个设备
            first_device = torch.device(self.devices[0])
            for name, module in named_modules.items():
                if any(keyword in name.lower() for keyword in ['embedding', 'embeddings', 'lm_head', 'head']):
                    try:
                        module.to(first_device)
                        logger.debug(f"模块 {name} 已分配到 {self.devices[0]}")
                    except Exception as e:
                        logger.warning(f"将模块 {name} 分配到 {self.devices[0]} 失败: {e}")
            
            logger.info(f"✅ 模型已手动分配到NPU设备: {self.devices}")
        except Exception as e:
            logger.error(f"手动分配模型到NPU设备失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果手动分配失败，回退到第一个设备
            logger.warning(f"回退到第一个设备: {self.devices[0]}")
            device_obj = torch.device(self.devices[0])
            self.model = self.model.to(device_obj)
    
    def _force_release_gpu_memory(self, original_device_type):
        """
        强制释放GPU内存，使用多种方法确保彻底释放
        
        Args:
            original_device_type (str): 原始设备类型 ("cuda" 或 "npu")
        
        Returns:
            bool: 是否成功释放（通过检查内存是否减少来判断）
        """
        import gc
        import time
        
        if original_device_type != "cuda":
            return True
        
        logger.info("开始强制释放GPU内存...")
        
        # 记录释放前的内存
        before_memory = torch.cuda.memory_allocated(self.device) / 1024**3
        before_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        logger.info(f"释放前内存: 已分配 {before_memory:.2f}GB, 保留 {before_reserved:.2f}GB")
        
        # 方法1: 将模型移到CPU再删除（如果模型还存在）
        if hasattr(self, 'model') and self.model is not None:
            try:
                logger.info("将模型移到CPU以释放GPU内存...")
                # 获取模型所在的设备
                try:
                    first_param = next(self.model.parameters())
                    model_device = first_param.device
                    if model_device.type == "cuda":
                        # 将模型移到CPU
                        self.model = self.model.cpu()
                        logger.info("模型已移到CPU")
                except:
                    pass
            except Exception as e:
                logger.warning(f"将模型移到CPU时出错: {e}")
        
        # 方法2: 删除所有可能的引用
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        # 方法3: 多次垃圾回收
        for i in range(3):
            gc.collect()
            if i < 2:
                time.sleep(0.1)  # 短暂等待让GC完成
        
        # 方法4: 清理CUDA缓存（多次）
        for i in range(3):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if i < 2:
                time.sleep(0.1)  # 短暂等待让CUDA完成释放
        
        # 方法5: 再次垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        # 方法6: 重置内存统计
        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except:
            pass
        
        # 等待一段时间让CUDA真正释放内存
        time.sleep(0.5)
        
        # 再次清理
        torch.cuda.empty_cache()
        gc.collect()
        
        # 检查释放后的内存
        after_memory = torch.cuda.memory_allocated(self.device) / 1024**3
        after_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        logger.info(f"释放后内存: 已分配 {after_memory:.2f}GB, 保留 {after_reserved:.2f}GB")
        
        # 计算释放的内存
        memory_freed = before_memory - after_memory
        reserved_freed = before_reserved - after_reserved
        
        if memory_freed > 0.1 or reserved_freed > 0.1:  # 至少释放了100MB
            logger.info(f"✅ 成功释放GPU内存: 已分配内存减少 {memory_freed:.2f}GB, 保留内存减少 {reserved_freed:.2f}GB")
            return True
        else:
            logger.warning(f"⚠️ GPU内存释放不明显: 已分配内存减少 {memory_freed:.2f}GB, 保留内存减少 {reserved_freed:.2f}GB")
            # 即使释放不明显，也继续尝试，因为可能是CUDA的内存碎片化问题
            return True
    
    def load_model(self):
        """加载预训练模型和tokenizer"""
        logger.info(f"开始加载模型，路径: {self.model_path}, 设备: {self.device}, 数据类型: {self.torch_dtype}, 模型类型: {self.model_type}")
        try:
            # 释放之前的模型资源，避免显存泄漏
            if hasattr(self, 'model') and self.model is not None:
                logger.info("释放之前的模型资源...")
                
                # 记录原始设备类型，用于释放资源
                original_device_type = None
                if hasattr(self, 'model'):
                    # 尝试获取模型所在的设备
                    try:
                        # 获取模型第一个参数所在的设备
                        first_param = next(self.model.parameters())
                        original_device_type = first_param.device.type
                    except:
                        # 如果无法获取，使用self.device
                        original_device_type = self.device.type if hasattr(self, 'device') else None
                
                # 使用强制释放方法
                if original_device_type == "cuda":
                    self._force_release_gpu_memory(original_device_type)
                elif original_device_type == "npu":
                    # NPU的释放
                    import gc
                    if hasattr(self, 'model') and self.model is not None:
                        try:
                            self.model = self.model.cpu()
                        except:
                            pass
                    if hasattr(self, 'model'):
                        del self.model
                        self.model = None
                    if hasattr(self, 'tokenizer'):
                        del self.tokenizer
                        self.tokenizer = None
                    gc.collect()
                    torch.npu.empty_cache()
                    gc.collect()
            
            logger.info(f"加载模型 {self.model_path} 到 {self.device}，数据类型: {self.torch_dtype}...")
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # 配置模型加载参数
            kwargs = dict(
                output_hidden_states=True,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True
            )
            
            # Flash Attention优化（在GPU和NPU上启用）
            # 注意：多设备模式下，Flash Attention可能不支持，需要检查
            if self.model_type == "flash" and (self.device.type == "cuda" or self.device.type == "npu"):
                # 多设备模式下，Flash Attention可能不支持，使用默认实现
                if self.multi_device or self.device_map is not None:
                    logger.warning("多设备模式下，Flash Attention可能不支持，使用默认注意力实现")
                else:
                    logger.info("启用Flash Attention加速")
                    kwargs.update(dict(
                        attn_implementation="flash_attention_2"
                    ))
            else:
                logger.info("使用默认注意力实现")
            
            # 处理多设备映射
            if self.device_map is not None:
                # 使用device_map进行模型并行
                if self.device_map == "auto":
                    logger.info("使用自动设备映射（device_map='auto'）")
                    # 如果用户指定了设备列表，需要限制只使用指定的设备
                    if self.multi_device and self.devices:
                        # 检查是否包含NPU设备
                        has_npu = any(dev.startswith('npu:') for dev in self.devices)
                        if has_npu:
                            # NPU设备不支持max_memory，accelerate库无法识别NPU设备名称
                            # 对于NPU多设备，我们需要使用其他方法
                            # 方法：不使用device_map，先加载到CPU，然后手动分配到指定设备
                            logger.warning(f"NPU设备不支持max_memory参数，将先加载到CPU，然后手动分配到指定设备: {self.devices}")
                            # 不设置device_map，先加载到CPU
                            kwargs.pop('device_map', None)  # 确保不设置device_map
                            self._npu_manual_device_map = True  # 标记需要手动分配
                        else:
                            # CUDA设备可以使用max_memory
                            all_devices = self._get_available_devices()
                            max_memory = self._build_max_memory_dict(all_devices, self.devices)
                            if max_memory is not None:
                                kwargs['max_memory'] = max_memory
                                kwargs['device_map'] = "auto"
                                logger.info(f"限制设备使用，只使用指定设备: {self.devices}, max_memory: {max_memory}")
                            else:
                                kwargs['device_map'] = "auto"
                    else:
                        kwargs['device_map'] = "auto"
                elif self.device_map == "balanced":
                    # 平衡分配到所有设备
                    if self.multi_device:
                        device_list = self.devices
                    else:
                        # 自动检测可用设备
                        device_list = self._get_available_devices()
                    logger.info(f"使用平衡设备映射，分配到设备: {device_list}")
                    kwargs['device_map'] = "balanced"
                    # 如果用户指定了设备列表，需要限制只使用指定的设备
                    if self.multi_device and self.devices:
                        # 检查是否包含NPU设备
                        has_npu = any(dev.startswith('npu:') for dev in self.devices)
                        if has_npu:
                            # NPU设备不支持max_memory，accelerate库无法识别NPU设备名称
                            # 对于NPU多设备，我们需要使用其他方法
                            # 方法：不使用device_map，先加载到CPU，然后手动分配到指定设备
                            logger.warning(f"NPU设备不支持max_memory参数，将先加载到CPU，然后手动分配到指定设备: {self.devices}")
                            # 不设置device_map，先加载到CPU
                            kwargs.pop('device_map', None)  # 确保不设置device_map
                            self._npu_manual_device_map = True  # 标记需要手动分配
                        else:
                            # CUDA设备可以使用max_memory
                            all_devices = self._get_available_devices()
                            max_memory = self._build_max_memory_dict(all_devices, self.devices)
                            if max_memory is not None:
                                kwargs['max_memory'] = max_memory
                                logger.info(f"限制设备使用，只使用指定设备: {self.devices}, max_memory: {max_memory}")
                elif self.device_map == "sequential":
                    # 顺序分配到设备
                    if self.multi_device:
                        device_list = self.devices
                    else:
                        device_list = self._get_available_devices()
                    logger.info(f"使用顺序设备映射，分配到设备: {device_list}")
                    kwargs['device_map'] = "sequential"
                    # 如果用户指定了设备列表，需要限制只使用指定的设备
                    if self.multi_device and self.devices:
                        # 检查是否包含NPU设备
                        has_npu = any(dev.startswith('npu:') for dev in self.devices)
                        if has_npu:
                            # NPU设备不支持max_memory，accelerate库无法识别NPU设备名称
                            # 对于NPU多设备，我们需要使用其他方法
                            # 方法：不使用device_map，先加载到CPU，然后手动分配到指定设备
                            logger.warning(f"NPU设备不支持max_memory参数，将先加载到CPU，然后手动分配到指定设备: {self.devices}")
                            # 不设置device_map，先加载到CPU
                            kwargs.pop('device_map', None)  # 确保不设置device_map
                            self._npu_manual_device_map = True  # 标记需要手动分配
                        else:
                            # CUDA设备可以使用max_memory
                            all_devices = self._get_available_devices()
                            max_memory = self._build_max_memory_dict(all_devices, self.devices)
                            if max_memory is not None:
                                kwargs['max_memory'] = max_memory
                                logger.info(f"限制设备使用，只使用指定设备: {self.devices}, max_memory: {max_memory}")
                elif isinstance(self.device_map, dict):
                    # 手动指定设备映射
                    logger.info(f"使用手动设备映射: {self.device_map}")
                    kwargs['device_map'] = self.device_map
                else:
                    logger.warning(f"不支持的device_map值: {self.device_map}，使用单设备模式")
            elif self.multi_device:
                # 多设备模式但没有指定device_map，使用自动分配
                logger.info(f"多设备模式，使用自动设备映射分配到: {self.devices}")
                kwargs['device_map'] = "auto"
                # 使用max_memory限制只使用指定的设备
                if self.devices:
                    # 检查是否包含NPU设备
                    has_npu = any(dev.startswith('npu:') for dev in self.devices)
                    if has_npu:
                        # NPU设备不支持max_memory，accelerate库无法识别NPU设备名称
                        # 对于NPU多设备，我们需要使用其他方法
                        # 方法：不使用device_map，先加载到CPU，然后手动分配到指定设备
                        logger.warning(f"NPU设备不支持max_memory参数，将先加载到CPU，然后手动分配到指定设备: {self.devices}")
                        # 不设置device_map，先加载到CPU
                        kwargs.pop('device_map', None)  # 确保不设置device_map
                        self._npu_manual_device_map = True  # 标记需要手动分配
                    else:
                        # CUDA设备可以使用max_memory
                        all_devices = self._get_available_devices()
                        max_memory = self._build_max_memory_dict(all_devices, self.devices)
                        if max_memory is not None:
                            kwargs['max_memory'] = max_memory
                            logger.info(f"限制设备使用，只使用指定设备: {self.devices}, max_memory: {max_memory}")
            
            # 加载模型
            try:
                # 优先尝试加载因果语言模型（支持生成任务）
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
                    logger.info("成功加载因果语言模型(AutoModelForCausalLM)")
                except Exception as e:
                    # 如果失败，回退到普通模型
                    logger.warning(f"加载因果语言模型失败，回退到普通模型: {e}")
                    self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
                    logger.info("成功加载普通模型(AutoModel)")
            except RuntimeError as e:
                # 如果加载模型时内存不足，尝试强制释放后重试
                if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                    logger.warning("加载模型时GPU内存不足，尝试强制释放内存后重试...")
                    # 强制释放GPU内存（如果之前有模型的话）
                    if hasattr(self, 'model') and self.model is not None:
                        self._force_release_gpu_memory("cuda")
                    # 再次尝试加载
                    try:
                        logger.info("强制释放内存后，再次尝试加载模型...")
                        self.model = AutoModel.from_pretrained(self.model_path, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"强制释放内存后仍然无法加载模型: {retry_error}")
                        error_msg = (
                            f"GPU内存不足，无法加载模型。已尝试强制释放内存但仍失败。\n"
                            f"错误详情: {retry_error}\n"
                            f"当前GPU内存使用: {self.get_gpu_memory_info()}\n"
                            f"建议: 1) 减少其他GPU进程的内存使用\n"
                            f"      2) 使用更小的模型\n"
                            f"      3) 使用CPU模式运行（通过--force_cpu参数）"
                        )
                        raise RuntimeError(error_msg) from retry_error
                else:
                    raise e
            
            # 移到指定设备并设置为评估模式
            # 注意：如果使用了device_map，模型已经在加载时分配到各个设备，不需要再移动
            if self._npu_manual_device_map:
                # NPU多设备模式，需要手动分配模型到指定设备
                logger.info(f"手动分配NPU模型到指定设备: {self.devices}")
                self._manual_distribute_model_to_npu()
                self.model.eval()
            elif self.device_map is not None or self.multi_device:
                # 多设备模式，模型已经通过device_map分配到各个设备
                logger.info(f"模型已通过device_map分配到多个设备")
                # 设置为评估模式
                self.model.eval()
            else:
                # 单设备模式，需要手动移动模型
                try:
                    if self.device.type == "npu":
                        # 昇腾NPU设备
                        self.model = self.model.to(self.device)
                        logger.info(f"模型已成功加载到{self.device}")
                    elif self.device.type == "cuda":
                        # GPU设备
                        self.model = self.model.to(self.device)
                        logger.info(f"模型已加载到{self.device}")
                    else:
                        # CPU设备
                        self.model = self.model.eval()
                        logger.info("模型在CPU上运行")
                except RuntimeError as e:
                    # 如果移至设备时内存不足，尝试强制释放后重试
                    if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                        logger.warning("将模型移至GPU时内存不足，尝试强制释放内存后重试...")
                        # 强制释放GPU内存
                        self._force_release_gpu_memory("cuda")
                        # 再次尝试移动
                        try:
                            logger.info("强制释放内存后，再次尝试将模型移至GPU...")
                            self.model = self.model.to(self.device)
                            logger.info(f"模型已加载到{self.device}")
                        except Exception as retry_error:
                            logger.error(f"强制释放内存后仍然无法将模型移至GPU: {retry_error}")
                            error_msg = (
                                f"GPU内存不足，无法将模型移至GPU。已尝试强制释放内存但仍失败。\n"
                                f"错误详情: {retry_error}\n"
                                f"当前GPU内存使用: {self.get_gpu_memory_info()}\n"
                                f"建议: 1) 减少其他GPU进程的内存使用\n"
                                f"      2) 使用更小的模型\n"
                                f"      3) 使用CPU模式运行（通过--force_cpu参数）"
                            )
                            raise RuntimeError(error_msg) from retry_error
                    # 如果移至NPU失败，回退到GPU或CPU（仅针对NPU错误）
                    elif self.device.type == "npu":
                        logger.error(f"将模型移至NPU失败: {str(e)}")
                        logger.info("尝试回退到GPU或CPU")
                        if torch.cuda.is_available():
                            self.device = torch.device("cuda:0")
                            self.model = self.model.to(self.device)
                            logger.info(f"模型已回退到{self.device}")
                        else:
                            self.device = torch.device("cpu")
                            self.model = self.model.eval()
                            logger.info("模型在CPU上运行")
                    else:
                        raise e

            logger.info("✅ 模型加载完成")
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise e
    
    async def extract_embedding(self, sequence, pooling_method="mean"):
        """
        提取序列的embedding
        
        Args:
            sequence (str): 输入的DNA序列
            pooling_method (str): 池化方法，支持 "mean", "max", "last", "none"
                - "mean": 平均池化（默认）
                - "max": 最大池化
                - "last": 取最后一个token
                - "none": 返回完整序列embedding
            
        Returns:
            dict: 包含embedding和相关信息的字典
        """
        logger.info(f"开始提取embedding，序列长度: {len(sequence)}, 池化方法: {pooling_method}, 设备: {self.device}")
        try:
            import time
            start = time.time()
            # Tokenize输入序列
            inputs = self.tokenizer(sequence, return_tensors="pt")
            
            # 移到指定设备
            # 在多设备模式下，inputs需要移动到主设备（第一个设备）
            # 模型内部会根据device_map自动处理数据在不同设备间的传输
            try:
                if self.multi_device or self.device_map is not None:
                    # 多设备模式：移动到主设备
                    main_device = self.device
                    if self.device.type == "npu" or self.device.type == "cuda":
                        inputs = {k: v.to(main_device) for k, v in inputs.items()}
                elif self.device.type == "npu":
                    # 昇腾NPU设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                elif self.device.type == "cuda":
                    # GPU设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # CPU设备不需要移动
            except Exception as e:
                # 如果移至NPU失败，尝试使用CPU
                if self.device.type == "npu":
                    logger.warning(f"将数据移至NPU失败: {str(e)}")
                    logger.info("尝试使用CPU继续处理")

            # 前向传播获取embedding
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs)
                    # 根据设备类型同步
                    # 多设备模式下，需要同步所有使用的设备
                    if self.multi_device or self.device_map is not None:
                        # 同步所有设备
                        for device_str in self.devices:
                            device_obj = torch.device(device_str)
                            if device_obj.type == "cuda":
                                torch.cuda.synchronize(device_obj)
                            elif device_obj.type == "npu":
                                torch.npu.synchronize(device_obj)
                    elif self.device.type == "cuda":
                        torch.cuda.synchronize()
                    elif self.device.type == "npu":
                        # NPU也需要同步
                        torch.npu.synchronize()
                except RuntimeError as e:
                    # 检查是否是FlashAttention错误
                    if "FlashAttention" in str(e) and self.model_type == "flash":
                        logger.warning(f"FlashAttention不支持当前设备，错误: {e}")
                        logger.info("正在重新加载模型，使用默认注意力实现...")
                        # 记录原始设备，用于后续判断是否需要重新创建inputs
                        original_device = self.device
                        # 重新加载模型，使用默认注意力实现
                        self.model_type = "no_flash"
                        try:
                            self.load_model()
                        except Exception as reload_error:
                            logger.error(f"重新加载模型失败: {reload_error}")
                            # 如果GPU内存不足，尝试强制释放内存后重试
                            if "CUDA out of memory" in str(reload_error) and original_device.type == "cuda":
                                logger.warning("GPU内存不足，尝试强制释放内存后重试...")
                                # 强制释放GPU内存
                                self._force_release_gpu_memory("cuda")
                                # 再次尝试加载
                                try:
                                    logger.info("强制释放内存后，再次尝试加载模型...")
                                    self.load_model()
                                except Exception as retry_error:
                                    logger.error(f"强制释放内存后仍然无法加载模型: {retry_error}")
                                    error_msg = (
                                        f"GPU内存不足，无法加载模型。已尝试强制释放内存但仍失败。\n"
                                        f"错误详情: {retry_error}\n"
                                        f"当前GPU内存使用: {self.get_gpu_memory_info()}\n"
                                        f"建议: 1) 减少其他GPU进程的内存使用\n"
                                        f"      2) 使用更小的模型\n"
                                        f"      3) 使用CPU模式运行（通过--force_cpu参数）"
                                    )
                                    raise RuntimeError(error_msg) from retry_error
                            else:
                                raise reload_error
                        
                        # 确保inputs在正确的设备上
                        if self.device.type == "npu":
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        elif self.device.type == "cuda":
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        # CPU设备不需要移动
                        
                        # 再次尝试前向传播
                        outputs = self.model(**inputs)
                        # 根据设备类型同步
                        if self.device.type == "cuda":
                            torch.cuda.synchronize()
                        elif self.device.type == "npu":
                            torch.npu.synchronize()
                    else:
                        raise e
            
            # 提取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            logger.debug(f"last_hidden_state shape: {last_hidden_state.shape}")
            
            # 获取attention mask
            attention_mask = inputs.get(
                "attention_mask",
                torch.ones_like(inputs['input_ids'])
            )
            
            # 根据池化方法处理embedding
            if pooling_method == "mean":
                # 平均池化（考虑attention mask）
                mask = attention_mask.unsqueeze(-1).to(self.device)  # 使用self.device而不是全局device
                logger.debug(f"mask shape: {mask.shape}")
                pooled_embedding = (last_hidden_state * mask).sum(1) / mask.sum(1)
                
            elif pooling_method == "max":
                # 最大池化
                pooled_embedding = torch.max(last_hidden_state, dim=1)[0]
                
            elif pooling_method == "last":
                # 取最后一个有效token
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled_embedding = last_hidden_state[torch.arange(last_hidden_state.size(0)), sequence_lengths]
                
            elif pooling_method == "none":
                # 返回完整序列embedding
                pooled_embedding = last_hidden_state
            else:
                raise ValueError(f"不支持的池化方法: {pooling_method}")
            
            # 根据设备类型处理tensor转换
            # 先将tensor移至CPU
            cpu_tensor = pooled_embedding.cpu()
            
            # 检查并转换BFloat16类型，确保更好的兼容性
            if cpu_tensor.dtype == torch.bfloat16:
                # BFloat16在某些环境中可能需要转换为Float16以确保兼容性
                logger.warning(f"注意: 将BFloat16转换为Float16以确保更好的兼容性")
                cpu_tensor = cpu_tensor.to(torch.float16)
            
            # 转换为numpy数组
            embedding_array = cpu_tensor.numpy()
            
            # 构建返回结果
            result = {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "token_count": inputs['input_ids'].shape[1],
                "embedding_shape": list(embedding_array.shape),
                "embedding_dim": embedding_array.shape[-1],
                "pooling_method": pooling_method,
                "model_type": self.model_type,
                "device": str(self.device),
                "embedding": embedding_array.tolist()  # 转换为列表便于JSON序列化
            }
            processing_time = time.time() - start
            logger.info(f"✅ Embedding提取成功，处理时间: {processing_time:.4f}秒，embedding维度: {embedding_array.shape}") 
            return result
            
        except Exception as e:
            logger.error(f"❌ Embedding提取失败: {e}")
            raise e
    
    async def predict_next_bases(self, sequence, predict_length=10):
        """
        预测DNA序列的下游碱基
        
        Args:
            sequence (str): 输入的DNA序列
            predict_length (int): 预测的碱基数量
            
        Returns:
            dict: 包含原始序列、预测序列和预测碱基的字典
        """
        logger.info(f"开始预测下游碱基，序列长度: {len(sequence)}, 预测长度: {predict_length}, 设备: {self.device}")
        try:
            import time
            start = time.time()
            
            # 初始化预测序列
            predicted_sequence = sequence
            current_inputs = self.tokenizer(sequence, return_tensors="pt")
            
            # 移到指定设备
            if self.multi_device or self.device_map is not None:
                # 多设备模式：移动到主设备
                main_device = self.device
                if self.device.type == "npu" or self.device.type == "cuda":
                    current_inputs = {k: v.to(main_device) for k, v in current_inputs.items()}
            elif self.device.type == "npu":
                # 昇腾NPU设备
                current_inputs = {k: v.to(self.device) for k, v in current_inputs.items()}
            elif self.device.type == "cuda":
                # GPU设备
                current_inputs = {k: v.to(self.device) for k, v in current_inputs.items()}
            
            # 连续预测碱基
            predicted_bases = ""
            with torch.no_grad():
                for i in range(predict_length):
                    # 获取当前输入的输出
                    try:
                        current_outputs = self.model(**current_inputs)
                        # 根据设备类型同步
                        if self.multi_device or self.device_map is not None:
                            # 同步所有设备
                            for device_str in self.devices:
                                device_obj = torch.device(device_str)
                                if device_obj.type == "cuda":
                                    torch.cuda.synchronize(device_obj)
                                elif device_obj.type == "npu":
                                    torch.npu.synchronize(device_obj)
                        elif self.device.type == "cuda":
                            torch.cuda.synchronize()
                        elif self.device.type == "npu":
                            torch.npu.synchronize()
                    except RuntimeError as e:
                        logger.error(f"预测过程中出错: {e}")
                        raise e
                    
                    # 获取logits或隐藏状态
                    if hasattr(current_outputs, 'logits'):
                        # 使用模型自带的logits
                        logits = current_outputs.logits
                    elif hasattr(current_outputs, 'hidden_states') and hasattr(self.model, 'lm_head'):
                        # 使用隐藏状态和语言模型头
                        last_hidden_state = current_outputs.hidden_states[-1]
                        logits = self.model.lm_head(last_hidden_state)
                    else:
                        # 如果模型不支持预测，则抛出异常
                        raise RuntimeError("当前模型不支持碱基预测，请使用支持生成任务的模型")
                    
                    # 预测下一个碱基
                    last_token_logits = logits[:, -1, :]
                    next_token_id = torch.argmax(last_token_logits, dim=-1)
                    next_base = self.tokenizer.decode(next_token_id)
                    
                    # 添加到预测序列
                    predicted_sequence += next_base
                    predicted_bases += next_base
                    
                    # 更新输入序列
                    current_inputs = self.tokenizer(predicted_sequence, return_tensors="pt")
                    if self.device.type != "cpu":
                        if self.multi_device or self.device_map is not None:
                            main_device = self.device
                            current_inputs = {k: v.to(main_device) for k, v in current_inputs.items()}
                        else:
                            current_inputs = {k: v.to(self.device) for k, v in current_inputs.items()}
            
            end = time.time()
            logger.info(f"碱基预测完成，耗时: {end - start:.2f}秒")
            
            return {
                "original_sequence": sequence,
                "predicted_sequence": predicted_sequence,
                "predicted_bases": predicted_bases,
                "predict_length": predict_length,
                "total_length": len(predicted_sequence)
            }
        except Exception as e:
            logger.error(f"碱基预测失败: {e}")
            raise e
    
    def extract_embedding_batch(self, sequences, pooling_method="mean"):
        """
        批量提取多个序列的embedding
        
        Args:
            sequences (list): DNA序列列表
            pooling_method (str): 池化方法
            
        Returns:
            list: 每个序列的embedding结果
        """
        results = []
        for seq in sequences:
            try:
                result = self.extract_embedding(seq, pooling_method)
                results.append(result)
            except Exception as e:
                logger.error(f"序列 {seq[:50]}... 处理失败: {e}")
                results.append({
                    "sequence": seq,
                    "error": str(e)
                })
        return results


# 全局变量：存储不同配置的提取器
extractors = {}


def get_or_create_extractor(model_name, force_cpu=False, device=None, torch_dtype=None, device_map=None, memory_ratio=0.9, model_path_prefix=None):
    """
    获取或创建指定模型的提取器
    
    Args:
        model_name (str): 模型名称
        force_cpu (bool): 是否强制使用CPU
        device (str or list, optional): 设备类型，如果为None则自动选择
            - 单设备: "cuda:0" 或 "npu:0"
            - 多设备: ["cuda:0", "cuda:1"] 或 ["npu:0", "npu:1"]
        torch_dtype (torch.dtype, optional): 数据类型，如果为None则根据设备自动选择
        device_map (str or dict, optional): 设备映射方式
            - "auto": 自动分配模型到可用设备
            - "balanced": 平衡分配到所有设备
            - "sequential": 按顺序分配到设备
            - dict: 手动指定每层的设备映射
            - None: 使用单设备模式
        memory_ratio (float): 内存分配比例，默认0.9（使用90%的内存）
        model_path_prefix (str, optional): 模型路径前缀，如果为None则使用默认路径
        
    Returns:
        EmbeddingExtractor: 提取器实例
    """
    # 默认模型路径前缀
    default_prefix = '/storeData/AI_models/modelscope/hub/models/BGI-HangzhouAI/'
    
    # 模型配置（与embedding_extraction.py保持一致）
    model_configs = {
        "1.2B": {
            "type": "flash",
        },
        "10B": {
            "type": "flash",
        }
    }
    
    # 根据前缀构建完整路径
    path_prefix = model_path_prefix or default_prefix
    # 确保路径前缀以斜杠结尾
    if not path_prefix.endswith('/'):
        path_prefix += '/'
    
    if model_name not in model_configs:
        raise ValueError(f"未知的模型名称: {model_name}. 支持的模型: {list(model_configs.keys())}")
    
    # 如果提取器已存在，直接返回
    if model_name in extractors:
        return extractors[model_name]
    
    # 创建新的提取器
    config = model_configs[model_name]
    extractor = EmbeddingExtractor(
        model_path=f"{path_prefix}Genos-{model_name}",
        model_type=config["type"],
        device=device,
        torch_dtype=torch_dtype,
        model_name=model_name,
        force_cpu=force_cpu,
        device_map=device_map,
        memory_ratio=memory_ratio
    )
    extractors[model_name] = extractor
    
    return extractor


from sanic import Sanic, text
from sanic.response import json

# Create an instance of the Sanic app
app = Sanic("sanic-server")

@app.route('/extract', methods=['POST'])
async def extract_embedding(request):
    """
    提取客户端输入序列的embedding
    
    注意：此端点接收客户端提供的DNA序列，不生成随机序列
    如需生成随机序列，请使用 /generate 端点
    """
    # 获取客户端发送的请求数据
    sequence = request.json.get('sequence')
    model_name = request.json.get('model_name')
    pooling_method = request.json.get('pooling_method', 'mean')
    
    # 验证客户端输入的序列格式
    if not isinstance(sequence, str) or len(sequence) == 0:
        return json({"error": "sequence必须是非空字符串"})
    
    if not model_name:
        return json({"error": "model_name必须是非空字符串"})
    
    # 获取提取器
    try:
        extractor = get_or_create_extractor(model_name)
    except Exception as e:
        return json({"error": f"模型加载失败: {str(e)}"})
    
    # 提取客户端序列的embedding
    logger.info(f"处理客户端序列，长度: {len(sequence)}")
    result = await extractor.extract_embedding(sequence, pooling_method)
    
    return json({
        "success": True,
        "message": "客户端序列embedding提取成功",
        "result": result
    })

@app.route('/predict', methods=['POST'])
async def predict_bases(request):
    """
    预测DNA序列的下游碱基
    """
    # 获取客户端发送的请求数据
    sequence = request.json.get('sequence')
    model_name = request.json.get('model_name')
    predict_length = request.json.get('predict_length', 10)
    
    # 验证客户端输入
    if not isinstance(sequence, str) or len(sequence) == 0:
        return json({"error": "sequence必须是非空字符串"})
    
    if not model_name:
        return json({"error": "model_name必须是非空字符串"})
    
    if not isinstance(predict_length, int) or predict_length <= 0:
        return json({"error": "predict_length必须是正整数"})
    
    # 限制预测长度
    if predict_length > 1000:
        return json({"error": "predict_length不能超过1000"})
    
    # 获取提取器
    try:
        extractor = get_or_create_extractor(model_name)
    except Exception as e:
        return json({"error": f"模型加载失败: {str(e)}"})
    
    # 预测下游碱基
    logger.info(f"处理碱基预测请求，序列长度: {len(sequence)}, 预测长度: {predict_length}")
    result = await extractor.predict_next_bases(sequence, predict_length)
    
    return json({
        "success": True,
        "message": "碱基预测成功",
        "result": result
    })
   
def run_server(args):
    """
    运行embedding服务
    
    Args:
        args: 命令行参数
    """
    # 根据命令行参数确定设备设置
    force_cpu = args.force_cpu
    device = args.device if args.device else None
    device_map = args.device_map if args.device_map else None
    memory_ratio = args.memory_ratio if hasattr(args, 'memory_ratio') else 0.9
    model_path_prefix = args.model_path_prefix if hasattr(args, 'model_path_prefix') else None
    torch_dtype = None
    
    # 处理设备参数：如果device包含逗号，则转换为列表（多设备模式）
    if device and ',' in device:
        device = [d.strip() for d in device.split(',')]
        logger.info(f"检测到多设备模式: {device}")
        # 如果指定了多设备但没有指定device_map，默认使用auto
        if device_map is None:
            device_map = "auto"
            logger.info(f"多设备模式下，自动使用 device_map='auto'")
    
    # 初始化模型提取器
    logger.info(f"正在初始化模型提取器，强制CPU: {force_cpu}, 指定设备: {device}, device_map: {device_map}, 内存分配比例: {memory_ratio}, 模型路径前缀: {model_path_prefix}")
    extractor = get_or_create_extractor("1.2B", force_cpu=force_cpu, device=device, torch_dtype=torch_dtype, device_map=device_map, memory_ratio=memory_ratio, model_path_prefix=model_path_prefix)
    extractor = get_or_create_extractor("10B", force_cpu=force_cpu, device=device, torch_dtype=torch_dtype, device_map=device_map, memory_ratio=memory_ratio, model_path_prefix=model_path_prefix)
    
    # 启动服务器
    logger.info(f"正在启动服务器，监听地址: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, single_process=True)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='DNA序列Embedding提取服务')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器监听地址')
    parser.add_argument('--port', type=int, default=8000, help='服务器监听端口')
    
    # 模型配置
    parser.add_argument('--model_path_prefix', type=str, default='/AI_models/BGI-HangzhouAI/', 
                        help='模型存储路径前缀，默认: /AI_models/BGI-HangzhouAI/')
    
    # 设备配置
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU进行推理')
    parser.add_argument('--device', type=str, default=None, 
                        help='指定运行设备。单设备: npu:0, cuda:0, cpu。多设备: 用逗号分隔，如 "cuda:0,cuda:1" 或 "npu:0,npu:1"')
    parser.add_argument('--device_map', type=str, default=None,
                        help='设备映射方式: auto (自动分配), balanced (平衡分配), sequential (顺序分配)。'
                             '如果指定了多设备(--device)，将自动使用device_map="auto"')
    parser.add_argument('--memory_ratio', type=float, default=0.9,
                        help='内存分配比例，默认0.9（使用90%%的内存）。范围: 0.0-1.0')
    
    # 日志配置
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    # 运行服务器
    run_server(args)
    
