# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from typing import Any, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from verl.utils.device import get_device_name
from verl.workers.rollout.utils import ensure_async_iterator


def broadcast_pyobj(
    data: list[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(get_device_name() if not force_cpu_device else "cpu")

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(np.frombuffer(serialized_data, dtype=np.uint8)).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data


async def get_named_tensor_buckets(
    iterable: Iterator[tuple[str, torch.Tensor]], bucket_bytes: int
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """
    Group tensors into buckets based on a specified size in megabytes.

    Args:
        iterable: An iterator of tuples containing tensor names and tensors.
        bucket_bytes: The maximum size of each bucket in bytes.

    Yields:
        Lists of tuples, where each tuple contains a tensor name and its corresponding tensor.

    Example:
        >>> tensors = [('tensor1', torch.randn(1000, 1000)), ('tensor2', torch.randn(2000, 2000))]
        >>> for bucket in get_named_tensor_buckets(tensors, bucket_size_mb=10):
        ...     print(bucket)
        [('tensor1', tensor(...)), ('tensor2', tensor(...))]

    """
    if bucket_bytes <= 0:
        raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

    current_bucket = []
    current_size = 0
    async for name, tensor in ensure_async_iterator(iterable):
        tensor_size = tensor.element_size() * tensor.numel()
        if current_size + tensor_size > bucket_bytes:
            if current_bucket:
                yield current_bucket
            current_bucket = [(name, tensor.clone())]
            current_size = tensor_size
        else:
            current_bucket.append((name, tensor.clone()))
            current_size += tensor_size

    if current_bucket:
        yield current_bucket


async def update_sglang_weights_no_flush(
    engine,
    params_batch: list[tuple[str, torch.Tensor]],
    device_mesh_key: str,
    device_mesh: DeviceMesh,
    load_format: Optional[str] = None,
):
    """Mirror SGLang's tensor weight sync, but defer cache flush to the caller.

    SGLang's default UpdateWeightsFromTensorReqInput sets ``flush_cache=True``.
    Under async partial rollout, that server-side flush can fail while requests are
    winding down and crash the scheduler. We keep the same gather/serialize path
    but explicitly disable the in-request flush, then let the caller flush once
    after the whole sync finishes.
    """

    from torch.distributed.tensor import DTensor

    from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
    from sglang.srt.model_executor.model_runner import LocalSerializedTensor
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()

    infer_tp_size = device_mesh[device_mesh_key].mesh.size()[0]
    infer_tp_rank = device_mesh[device_mesh_key].get_local_rank()

    def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
        if isinstance(tensor, DTensor):
            return tensor.full_tensor()
        return tensor

    named_tensors_batch = [
        (
            name,
            MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor.detach())),
        )
        for name, tensor in params_batch
    ]

    if infer_tp_rank == 0:
        gathered_serialized_batches = [None for _ in range(infer_tp_size)]
    else:
        gathered_serialized_batches = None

    dist.gather_object(
        obj=named_tensors_batch,
        object_gather_list=gathered_serialized_batches,
        dst=device_mesh[device_mesh_key].mesh.tolist()[0],
        group=device_mesh[device_mesh_key].get_group(),
    )

    if infer_tp_rank == 0:
        logical_tensors = zip(*gathered_serialized_batches, strict=True)
        named_tensors = [
            (
                tensor_group[0][0],
                LocalSerializedTensor(values=[rank_part[1] for rank_part in tensor_group]),
            )
            for tensor_group in logical_tensors
        ]
        update_weights_request = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(named_tensors) for _ in range(infer_tp_size)
            ],
            load_format=load_format,
            flush_cache=False,
        )
        return await engine.update_weights_from_tensor(update_weights_request)


def ensure_sglang_flush_cache_succeeded(response: dict | None, context: str) -> None:
    """Validate that a cache flush succeeded instead of silently continuing.

    The adapter returns ``{}`` after retry exhaustion, so callers must treat an
    empty response as failure. When the server includes explicit success fields,
    validate them as well.
    """

    if not response:
        raise RuntimeError(f"SGLang flush_cache failed during {context}: empty response")

    if "cache_flushed" in response and not response["cache_flushed"]:
        raise RuntimeError(f"SGLang flush_cache failed during {context}: {response}")

    if "success" in response and not response["success"]:
        raise RuntimeError(f"SGLang flush_cache failed during {context}: {response}")

    status = response.get("status")
    if status is not None and str(status).lower() not in {"success", "ok", "flushed"}:
        raise RuntimeError(f"SGLang flush_cache failed during {context}: {response}")
