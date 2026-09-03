import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Self, Tuple

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
)
from ucm.logger import init_logger
from ucm.sparse.blend.blockwise_rope import block_wise_rope_forward

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


@dataclass
class ChunkMetaData:
    # [start, start + len)
    start_token_dix: int
    chunk_tokens_len: int

    start_blk_idx: int
    chunk_blks_len: int

    cached_start_position: int

    vllm_blk_ids: List[int] = field(default_factory=list)
    chunk_blks_hash: List[bytes] = field(default_factory=list)
    store_hits: List[bool] = field(default_factory=list)

    @property
    def end_token_dix(self) -> int:
        return self.start_token_dix + self.chunk_tokens_len

    @property
    def end_blk_idx(self) -> int:
        return self.start_blk_idx + self.chunk_blks_len

    @property
    def cached_end_position(self) -> int:
        return self.cached_start_position + self.chunk_tokens_len

    @property
    def position_offset(self) -> int:
        return self.start_token_dix - self.cached_start_position

    @property
    def hits_vllm_blk_ids(self) -> List[int]:
        return list(itertools.compress(self.vllm_blk_ids, self.store_hits))

    @property
    def hits_chunk_blks_hash(self) -> List[bytes]:
        return list(itertools.compress(self.chunk_blks_hash, self.store_hits))

    def merge_chunk(self, temp_chunk_meta: Self) -> None:
        # current we use a fix pattern(end with a fix token id) to recognize the text token chunk
        # in some special situation, one text chunk maybe split as multi text chunk, so we should merge them into one
        self.chunk_tokens_len += temp_chunk_meta.chunk_tokens_len
        self.chunk_blks_len += temp_chunk_meta.chunk_blks_len
        self.chunk_blks_hash += temp_chunk_meta.chunk_blks_hash

    def update_meta_partial_pc(self, num_pc_part_blks: int, block_size: int) -> None:
        if num_pc_part_blks > 0:
            self.start_token_dix += num_pc_part_blks * block_size
            self.chunk_tokens_len -= num_pc_part_blks * block_size

            self.start_blk_idx += num_pc_part_blks
            self.chunk_blks_len -= num_pc_part_blks

            self.chunk_blks_hash = self.chunk_blks_hash[num_pc_part_blks:]
            self.store_hits = self.store_hits[num_pc_part_blks:]
            self.cached_start_position += num_pc_part_blks * block_size


class BlendStage(Enum):
    BUILD_CHUNK_CACHE = auto()
    BUILD_PREFIX_CACHE = auto()
    CACHE_BLEND = auto()

    def is_blend_cache(self) -> bool:
        return self == BlendStage.CACHE_BLEND

    def is_prefix_cache(self) -> bool:
        return self == BlendStage.BUILD_PREFIX_CACHE


@dataclass
class BlendRequestMeta:
    ucm_block_hashs: list[bytes] = field(default_factory=list)
    # hbm pc is not supported
    hbm_hit_block_num: int = 0
    # ucm pc is supported
    pc_hit_block_num: int = 0
    chunks_meta: List[ChunkMetaData] = field(default_factory=list)
    blend_stage: BlendStage = BlendStage.BUILD_PREFIX_CACHE


@dataclass
class BlendRequestDispatchMeta(RequestDispatchMeta):
    chunks_meta: List[ChunkMetaData]
