# -*- coding: utf-8 -*-
"""
共享内存管理模块

用于多进程并行处理时，避免进程间数据复制的开销。
通过 multiprocessing.shared_memory 将 42GB 的流量矩阵数据
预加载到共享内存中，所有 worker 进程直接访问同一份数据。

对应 plan.md 优化计划:
- P0: 内存预加载 + 共享内存 (预期加速 2-3x)

使用方法:
    # 主进程中初始化
    manager = SharedMemoryManager()
    manager.create_from_files(data_dir)

    # 子进程中访问
    matrices = manager.get_matrices()

    # 程序结束时清理
    manager.cleanup()
"""

import atexit
import logging
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SharedArrayInfo:
    """共享数组的元信息

    用于在子进程中重建 numpy 数组视图。

    Attributes:
        name: 共享内存块名称
        shape: 数组形状
        dtype: 数组数据类型
    """
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype


class SharedMemoryManager:
    """共享内存管理器

    负责创建、管理和清理多进程共享的流量矩阵数据。

    设计要点:
    1. 主进程负责创建共享内存并加载数据
    2. 子进程通过元信息访问共享内存中的数据
    3. 使用 atexit 确保程序退出时清理共享内存

    Example:
        >>> manager = SharedMemoryManager()
        >>> manager.create_from_files('/path/to/data')
        >>> info = manager.get_info()
        >>> # 在子进程中使用 info 重建数组
        >>> matrices = SharedMemoryManager.attach(info)
    """

    # 共享内存名称前缀
    SHM_PREFIX = "traffic_"

    # 矩阵名称
    MATRIX_NAMES = ['occupancy', 'speed', 'volume']

    def __init__(self):
        """初始化共享内存管理器"""
        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._array_info: Dict[str, SharedArrayInfo] = {}
        self._arrays: Dict[str, np.ndarray] = {}
        self._node_order: Optional[np.ndarray] = None
        self._is_owner = False  # 是否为创建者（主进程）

        # 注册清理函数
        atexit.register(self.cleanup)

    def create_from_files(
        self,
        data_dir: str,
        year: int = 2023
    ) -> Dict[str, SharedArrayInfo]:
        """从文件创建共享内存

        将 .npy 文件加载到共享内存中。仅应在主进程中调用。
        使用 mmap 模式避免双份内存占用（84GB → 42GB 峰值）。

        Args:
            data_dir: 数据目录路径
            year: 数据年份

        Returns:
            字典，包含每个矩阵的 SharedArrayInfo

        Raises:
            FileNotFoundError: 数据文件不存在
            RuntimeError: 共享内存已创建
        """
        if self._is_owner:
            raise RuntimeError("共享内存已创建，不能重复创建")

        data_dir = Path(data_dir)

        # 文件路径映射
        file_map = {
            'occupancy': data_dir / f'occupancy_{year}_all.npy',
            'speed': data_dir / f'speed_{year}_all.npy',
            'volume': data_dir / f'volume_{year}_all.npy',
        }

        # 检查文件存在
        for name, path in file_map.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} 矩阵文件不存在: {path}")

        logger.info("开始预加载流量矩阵到共享内存...")

        # 用于异常时清理已创建的共享内存
        created_shm_blocks: Dict[str, shared_memory.SharedMemory] = {}

        try:
            for name, path in file_map.items():
                logger.info(f"  加载 {name}: {path}")

                # 使用 mmap 模式读取，避免双份内存占用！
                # mmap_mode='r' 不会将整个文件读入内存，而是按需读取
                data = np.load(str(path), mmap_mode='r')
                logger.info(f"    形状: {data.shape}, 类型: {data.dtype}, "
                           f"大小: {data.nbytes / 1024**3:.2f} GB (mmap模式)")

                # 创建共享内存块
                shm_name = f"{self.SHM_PREFIX}{name}"

                # 先尝试删除可能存在的旧共享内存
                try:
                    old_shm = shared_memory.SharedMemory(name=shm_name)
                    old_shm.close()
                    old_shm.unlink()
                    logger.info(f"    清理旧共享内存: {shm_name}")
                except FileNotFoundError:
                    pass

                # 创建新的共享内存
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=data.nbytes,
                    name=shm_name
                )
                # 记录已创建的共享内存，用于异常时清理
                created_shm_blocks[name] = shm

                # 创建共享内存上的 numpy 数组视图
                shared_array = np.ndarray(
                    data.shape,
                    dtype=data.dtype,
                    buffer=shm.buf
                )

                # 分块复制数据到共享内存（内存友好）
                # 对于大数组，分块复制可以减少瞬时内存峰值
                chunk_size = 1000  # 每次复制 1000 行
                total_rows = data.shape[0]
                for start_row in range(0, total_rows, chunk_size):
                    end_row = min(start_row + chunk_size, total_rows)
                    shared_array[start_row:end_row] = data[start_row:end_row]

                # mmap 文件会自动释放，无需手动 del

                # 保存引用
                self._shm_blocks[name] = shm
                self._arrays[name] = shared_array
                self._array_info[name] = SharedArrayInfo(
                    name=shm_name,
                    shape=shared_array.shape,
                    dtype=shared_array.dtype
                )

                logger.info(f"    共享内存创建成功: {shm_name}")

        except Exception as e:
            # 创建失败时，清理所有已创建的共享内存块！
            logger.exception(f"创建共享内存失败: {e}")
            logger.info("正在清理已创建的共享内存块...")
            for name, shm in created_shm_blocks.items():
                try:
                    shm.close()
                    shm.unlink()
                    logger.info(f"  已清理: {self.SHM_PREFIX}{name}")
                except Exception as cleanup_error:
                    logger.exception(f"  清理失败 {name}: {cleanup_error}")
            # 清空内部状态
            self._shm_blocks.clear()
            self._arrays.clear()
            self._array_info.clear()
            raise

        # 加载 node_order (较小，不需要共享内存)
        node_order_path = data_dir / 'node_order.npy'
        if node_order_path.exists():
            self._node_order = np.load(str(node_order_path))
            logger.info(f"  加载 node_order: {self._node_order.shape}")

        self._is_owner = True
        logger.info("流量矩阵预加载完成!")

        return self._array_info.copy()

    def get_info(self) -> Dict[str, SharedArrayInfo]:
        """获取共享数组的元信息

        用于传递给子进程，以便子进程重建数组视图。

        Returns:
            字典，包含每个矩阵的 SharedArrayInfo
        """
        return self._array_info.copy()

    def get_node_order(self) -> Optional[np.ndarray]:
        """获取节点顺序数组

        Returns:
            节点顺序数组，如果未加载则返回 None
        """
        return self._node_order

    @staticmethod
    def attach(
        array_info: Dict[str, SharedArrayInfo]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, shared_memory.SharedMemory]]:
        """附加到已存在的共享内存

        在子进程中调用，重建 numpy 数组视图。

        Args:
            array_info: 共享数组的元信息字典

        Returns:
            (arrays, shm_handles) 元组:
            - arrays: 字典，包含重建的 numpy 数组
            - shm_handles: 字典，包含 SharedMemory 句柄（必须保留以防 GC 回收）

        Note:
            子进程不应调用 unlink()，只有主进程负责清理。
            但子进程应在退出前调用 shm.close() 释放映射。
        """
        arrays = {}
        shm_handles = {}

        for name, info in array_info.items():
            try:
                # 附加到已存在的共享内存
                shm = shared_memory.SharedMemory(name=info.name)

                # 重建 numpy 数组视图
                arr = np.ndarray(
                    info.shape,
                    dtype=info.dtype,
                    buffer=shm.buf
                )

                arrays[name] = arr
                shm_handles[name] = shm  # 保留句柄防止 GC 回收

            except Exception as e:
                # 清理已创建的句柄
                for h in shm_handles.values():
                    try:
                        h.close()
                    except Exception:
                        pass
                logger.exception(f"附加共享内存失败 {info.name}: {e}")
                raise

        return arrays, shm_handles

    def get_matrices(self) -> Dict[str, np.ndarray]:
        """获取流量矩阵数组

        主进程中直接返回已创建的数组。

        Returns:
            字典，包含 occupancy, speed, volume 数组
        """
        return self._arrays.copy()

    def cleanup(self) -> None:
        """清理共享内存

        仅主进程（创建者）应调用此方法来释放共享内存。
        """
        if not self._is_owner:
            return

        logger.info("清理共享内存...")

        for name, shm in self._shm_blocks.items():
            try:
                shm.close()
                shm.unlink()
                logger.info(f"  已清理: {self.SHM_PREFIX}{name}")
            except Exception as e:
                logger.warning(f"  清理失败 {name}: {e}")

        self._shm_blocks.clear()
        self._arrays.clear()
        self._array_info.clear()
        self._is_owner = False

    def __del__(self):
        """析构时清理"""
        self.cleanup()


class SharedTrafficMatrices:
    """共享流量矩阵数据容器

    用于替代 TrafficMatrices，支持共享内存。

    Attributes:
        occupancy: 占用率矩阵
        speed: 速度矩阵
        volume: 流量矩阵
        node_order: 节点顺序数组
        _shm_handles: 共享内存句柄（必须保留以防 GC 回收）
    """

    def __init__(
        self,
        occupancy: np.ndarray,
        speed: np.ndarray,
        volume: np.ndarray,
        node_order: Optional[np.ndarray] = None,
        shm_handles: Optional[Dict[str, shared_memory.SharedMemory]] = None
    ):
        self.occupancy = occupancy
        self.speed = speed
        self.volume = volume
        self.node_order = node_order
        # CRITICAL: 必须保留 shm_handles 引用，否则 GC 会回收导致数组失效！
        self._shm_handles = shm_handles or {}

    def close(self) -> None:
        """关闭共享内存映射（子进程退出时调用）

        Note:
            子进程只调用 close()，不调用 unlink()。
            unlink() 只由主进程负责。
        """
        for name, shm in self._shm_handles.items():
            try:
                shm.close()
                logger.debug(f"子进程关闭共享内存映射: {name}")
            except Exception as e:
                logger.warning(f"关闭共享内存映射失败 {name}: {e}")
        self._shm_handles.clear()

    def __del__(self):
        """析构时自动关闭"""
        self.close()

    @classmethod
    def from_shared_memory(
        cls,
        array_info: Dict[str, SharedArrayInfo],
        node_order: Optional[np.ndarray] = None
    ) -> 'SharedTrafficMatrices':
        """从共享内存创建实例

        Args:
            array_info: 共享数组的元信息
            node_order: 节点顺序数组

        Returns:
            SharedTrafficMatrices 实例

        Note:
            返回的实例会持有 shm_handles 引用，防止 GC 回收。
            子进程退出前应调用 close() 释放映射。
        """
        # CRITICAL: attach() 返回 (arrays, shm_handles) 元组！
        arrays, shm_handles = SharedMemoryManager.attach(array_info)

        return cls(
            occupancy=arrays['occupancy'],
            speed=arrays['speed'],
            volume=arrays['volume'],
            node_order=node_order,
            shm_handles=shm_handles  # 保留句柄引用！
        )


# 全局共享内存管理器实例（用于简化调用）
_global_manager: Optional[SharedMemoryManager] = None


def get_global_manager() -> SharedMemoryManager:
    """获取全局共享内存管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = SharedMemoryManager()
    return _global_manager


def preload_to_shared_memory(
    data_dir: str,
    year: int = 2023
) -> Tuple[Dict[str, SharedArrayInfo], Optional[np.ndarray]]:
    """便捷函数：预加载数据到共享内存

    Args:
        data_dir: 数据目录
        year: 数据年份

    Returns:
        (array_info, node_order) 元组
    """
    manager = get_global_manager()
    array_info = manager.create_from_files(data_dir, year)
    node_order = manager.get_node_order()
    return array_info, node_order


def cleanup_shared_memory() -> None:
    """便捷函数：清理共享内存"""
    global _global_manager
    if _global_manager is not None:
        _global_manager.cleanup()
        _global_manager = None
