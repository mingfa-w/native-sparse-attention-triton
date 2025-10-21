# Copyright 2025 Xunhao Lai & Jianqiao Lu.
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
import math
from typing import Any, Tuple, Union
from collections import Counter
import torch
import triton
import triton.language as tl
import warnings
from native_sparse_attention.ops.triton import utils
from native_sparse_attention.ops.triton.utils import get_num_warps_stages, is_hopper_gpu


IS_HOPPER_GPU = is_hopper_gpu()


@triton.jit
def forward_kernel(
    q_ptr,  # Q: n x h x d
    k_ptr,  # K: n x h x d
    v_ptr,  # V: n x h x d
    o_ptr,  # O: n x h x d
    lse_ptr,  # LSE: h x n
    # 压缩参数
    kernel_size,
    kernel_stride,
    # 序列长度
    cu_seqlens_q,
    cu_seqlens_k,
    # 形状参数
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # 缩放因子
    sm_scale,
    # 步长参数
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_lh,
    stride_ln,
    # 当前批次的起始q块偏移（核心参数）
    q_block_start: tl.constexpr,
    # 元参数
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # 获取程序ID
    pid_b = tl.program_id(0)  # 批次ID
    pid_h = tl.program_id(1)  # 查询头ID
    pid_q_local = tl.program_id(2)  # 当前批次内的局部q块ID

    # 计算全局q块ID（局部ID + 批次起始偏移）
    global_pid_q = pid_q_local + q_block_start

    # KV头ID计算
    pid_kh = pid_h // NUM_SHARE_Q_HEADS

    # 获取当前批次的q/k序列范围（去padding后）
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    # 计算当前q块在序列中的起始位置（基于全局q块ID）
    q_start_in_seq = global_pid_q * BLOCK_SIZE_Q + kernel_size - 1

    # 跳过超出实际序列长度的q块
    if q_start_in_seq >= q_len:
        return

    # 初始化q/k/v的块指针（基于全局q块ID）
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )

    # 加载当前q块数据
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    # 初始化统计量
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq  # 当前q块内的序列索引
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1  # k的偏移
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_D), 0.0, dtype=tl.float32)

    # 注意力计算（核心逻辑不变）
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        i = tl.multiple_of(i, BLOCK_SIZE_K)
        # 加载k
        k = tl.load(k_ptrs, boundary_check=(1, 0), padding_option="zero")
        # 计算qk相似度
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(
            off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf")
        )
        qk += tl.dot(q, k) * qk_scale
        # 计算max和log-sum-exp
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # 更新累加器
        acc_o_scale = tl.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # 加载v并更新输出
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # 更新统计量
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # 移动k/v指针
        k_ptrs = tl.advance(k_ptrs, (0, BLOCK_SIZE_K))
        v_ptrs = tl.advance(v_ptrs, (BLOCK_SIZE_K, 0))

    # 最终缩放
    acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]

    # 保存输出
    Q = q_start_in_seq + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    D = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask = (Q < q_len) & (D < HEAD_DIM)
    ptr = (
        o_ptr + q_start * stride_on + pid_h * stride_oh + Q * stride_on + D * stride_od
    )
    tl.store(ptr, acc_o.to(o_ptr.dtype.element_ty), mask=mask)

    # 保存lse
    l_mask = off_q < q_len
    l_ptrs = lse_ptr + q_start * stride_ln + pid_h * stride_lh + off_q * stride_ln
    tl.store(l_ptrs, lse_i, mask=l_mask)


@triton.jit
def backward_sum_o_do(
    o_ptr,  # O: n x h x d
    do_ptr,  # dO: n x h x d
    delta_ptr,  # D: h x n
    o_len,
    HEAD_DIM,
    stride_on,
    stride_oh,
    stride_od,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dh,
    stride_dn,
    o_block_offset: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_n = tl.program_id(0) + o_block_offset
    pid_h = tl.program_id(1)
    off_n = pid_n * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr
        + off_n[:, None] * stride_on
        + pid_h * stride_oh
        + off_d[None, :] * stride_od,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    do = tl.load(
        do_ptr
        + off_n[:, None] * stride_don
        + pid_h * stride_doh
        + off_d[None, :] * stride_dod,
        mask=(off_n[:, None] < o_len) & (off_d[None, :] < HEAD_DIM),
        other=0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(
        delta_ptr + pid_h * stride_dh + off_n * stride_dn, delta, mask=off_n < o_len
    )


@triton.jit
def backward_dkdv(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dk_ptr,  # DK: sh x n x kh x d
    dv_ptr,  # DV: sh x n x kh x d
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dks,
    stride_dkn,
    stride_dkh,
    stride_dkd,
    stride_dvs,
    stride_dvn,
    stride_dvh,
    stride_dvd,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    pid_sh = pid_h % NUM_SHARE_Q_HEADS
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if BLOCK_SIZE_K * pid_k >= k_len:
        return
    # init pointers
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # dk_ptrs = tl.make_block_ptr(
    #     base=dk_ptr + k_start * stride_dkn + pid_kh * stride_dkh + pid_sh * stride_dks,
    #     shape=(k_len, HEAD_DIM),
    #     strides=(stride_dkn, stride_dkd),
    #     offsets=(pid_k * BLOCK_SIZE_K, 0),
    #     block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
    #     order=(1, 0),
    # )
    Q = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    D = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask = (Q < k_len) & (D < HEAD_DIM)
    ptr = (
        dk_ptr
        + k_start * stride_dkn
        + pid_kh * stride_dkh
        + pid_sh * stride_dks
        + Q * stride_dkn
        + D * stride_dkd
    )

    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(pid_k * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # dv_ptrs = tl.make_block_ptr(
    #     base=dv_ptr + k_start * stride_dvn + pid_kh * stride_dvh + pid_sh * stride_dvs,
    #     shape=(k_len, HEAD_DIM),
    #     strides=(stride_dvn, stride_dvd),
    #     offsets=(pid_k * BLOCK_SIZE_K, 0),
    #     block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
    #     order=(1, 0),
    # )
    Q1 = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    D1 = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask1 = (Q1 < k_len) & (D1 < HEAD_DIM)
    ptr1 = (
        dv_ptr
        + k_start * stride_dvn
        + pid_kh * stride_dvh
        + pid_sh * stride_dvs
        + Q1 * stride_dvn
        + D1 * stride_dvd
    )

    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q)
    off_k = (
        pid_k * BLOCK_SIZE_K * kernel_stride
        + tl.arange(0, BLOCK_SIZE_K) * kernel_stride
        + kernel_size
        - 1
    )
    # load k v and keep in SRAM
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dk dv
    dk = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_D), dtype=tl.float32)
    q_lo = pid_k * BLOCK_SIZE_K * kernel_stride + kernel_size - 1
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(HEAD_DIM, q_len),
        strides=(stride_qd, stride_qn),
        offsets=(0, q_lo),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_Q),
        order=(0, 1),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(HEAD_DIM, q_len),
        strides=(stride_dod, stride_don),
        offsets=(0, q_lo),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_Q),
        order=(0, 1),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        # shape=(1, q_len),
        # strides=(1, stride_dn),
        # offsets=(0, q_lo),
        # block_shape=(1, BLOCK_SIZE_Q),
        # order=(1, 0),
        shape=(q_len,),
        strides=(stride_dn,),
        offsets=(q_lo,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        # shape=(1, q_len),
        # strides=(1, stride_ln),
        # offsets=(0, q_lo),
        # block_shape=(1, BLOCK_SIZE_Q),
        # order=(0, 1),
        shape=(q_len,),
        strides=(stride_ln,),
        offsets=(q_lo,),
        block_shape=(BLOCK_SIZE_Q,),
        order=(0,),
    )
    # loop for q blocks
    for i in range(q_lo, q_len, BLOCK_SIZE_Q):
        # load
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
        # lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        # d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0,), padding_option="zero")[None, :]
        d = tl.load(d_ptrs, boundary_check=(0,), padding_option="zero")[None, :]
        # compute qk
        # [BLOCK_SIZE_K, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        qk = tl.where(off_k[:, None] <= (off_q + i)[None, :], float(0.0), float("-inf"))
        qk += tl.dot(k, q) * qk_scale
        # compute p, ds
        # [BLOCK_SIZE_K, BLOCK_SIE_Q] - [1, BLOCK_SIZE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        p = tl.exp2(qk - lse)
        # [BLOCK_SIZE_K, HEAD_DIM] @ [HEAD_DIM, BLOCK_SIE_Q] -> [BLOCK_SIZE_K, BLOCK_SIE_Q]
        dp = tl.dot(v, do)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        p = p.to(do.dtype)
        ds = ds.to(q.dtype)
        # update dk and dv
        # [BLOCK_SIZE_K, BLOCK_SIE_Q] @ [BLOCK_SIE_Q, HEAD_DIM] -> [BLOCK_SIZE_K, HEAD_DIM]
        dk += tl.dot(ds, tl.trans(q))
        dv += tl.dot(p, tl.trans(do))
        # increment pointers
        q_ptrs = tl.advance(q_ptrs, (0, BLOCK_SIZE_Q))
        do_ptrs = tl.advance(do_ptrs, (0, BLOCK_SIZE_Q))
        # lse_ptrs = tl.advance(lse_ptrs, (0, BLOCK_SIZE_Q))
        # d_ptrs = tl.advance(d_ptrs, (0, BLOCK_SIZE_Q))
        lse_ptrs = tl.advance(lse_ptrs, (BLOCK_SIZE_Q,))
        d_ptrs = tl.advance(d_ptrs, (BLOCK_SIZE_Q,))
    # save dk dv
    # tl.store(dk_ptrs, dk.to(dk_ptr.dtype.element_ty), boundary_check=(0, 1))
    # tl.store(dv_ptrs, dv.to(dv_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(ptr, dk.to(dk_ptr.dtype.element_ty), mask=mask)
    tl.store(ptr1, dv.to(dv_ptr.dtype.element_ty), mask=mask1)


@triton.jit
def backward_dq(
    q_ptr,  # Q: n x qh x d
    k_ptr,  # K: n x kh x d
    v_ptr,  # V: n x kh x d
    lse_ptr,  # LSE: qh x n
    d_ptr,  # Delta: qh x n
    do_ptr,
    dq_ptr,
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_lh,
    stride_ln,
    stride_dh,
    stride_dn,
    stride_don,
    stride_doh,
    stride_dod,
    stride_dqn,
    stride_dqh,
    stride_dqd,
    q_block_offset: tl.constexpr,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2) + q_block_offset
    pid_kh = pid_h // NUM_SHARE_Q_HEADS
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    # skip first kernel_size query block, because they do no attend to any keys
    q_start_in_seq = pid_q * BLOCK_SIZE_Q + kernel_size - 1
    if q_start_in_seq >= q_len:
        return
    # init pointers
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    # dq_ptrs = tl.make_block_ptr(
    #     base=dq_ptr + q_start * stride_dqn + pid_h * stride_dqh,
    #     shape=(q_len, HEAD_DIM),
    #     strides=(stride_dqn, stride_dqd),
    #     offsets=(q_start_in_seq, 0),
    #     block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
    #     order=(1, 0),
    # )
    Q1 = q_start_in_seq + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    D1 = tl.arange(0, BLOCK_SIZE_D)[None, :]
    mask1 = (Q1 < q_len) & (D1 < HEAD_DIM)
    ptr1 = (
        dq_ptr
        + q_start * stride_dqn
        + pid_h * stride_dqh
        + Q1 * stride_dqn
        + D1 * stride_dqd
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(k_len, HEAD_DIM),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
        order=(1, 0),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + k_start * stride_vn + pid_kh * stride_vh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_vd, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    do_ptrs = tl.make_block_ptr(
        base=do_ptr + q_start * stride_don + pid_h * stride_doh,
        shape=(q_len, HEAD_DIM),
        strides=(stride_don, stride_dod),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
        order=(1, 0),
    )
    d_ptrs = tl.make_block_ptr(
        base=d_ptr + q_start * stride_dn + pid_h * stride_dh,
        shape=(q_len, 1),
        strides=(stride_dn, stride_dh),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
        shape=(q_len, 1),
        strides=(stride_ln, stride_lh),
        offsets=(q_start_in_seq, 0),
        block_shape=(BLOCK_SIZE_Q, 1),
        order=(0, 1),
    )
    # offsets
    off_q = tl.arange(0, BLOCK_SIZE_Q) + q_start_in_seq
    off_k = tl.arange(0, BLOCK_SIZE_K) * kernel_stride + kernel_size - 1
    # load q, do, lse, delta, and keep in SRAM
    q = tl.load(q_ptrs, boundary_check=(1, 0), padding_option="zero")
    do = tl.load(do_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
    d = tl.load(d_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init dq
    dq = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_D), dtype=tl.float32)
    lo = 0
    hi = min(k_len, (q_start_in_seq + BLOCK_SIZE_Q - kernel_size) // kernel_stride + 1)
    for i in range(lo, hi, BLOCK_SIZE_K):
        # load
        k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(v_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where(
            off_q[:, None] >= (i * kernel_stride + off_k)[None, :], 0, float("-inf")
        )
        qk += tl.dot(q, tl.trans(k)) * qk_scale
        # compute p, ds
        p = tl.exp2(qk - lse)
        dp = tl.dot(do, v)
        ds = sm_scale * p * (dp - d)
        # cast dtype
        ds = ds.to(q.dtype)
        # update dq
        dq += tl.dot(ds, k)
        # increment pointers
        k_ptrs = tl.advance(k_ptrs, (BLOCK_SIZE_K, 0))
        v_ptrs = tl.advance(v_ptrs, (0, BLOCK_SIZE_K))
    # save dq
    # tl.store(dq_ptrs, dq.to(dq_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(ptr1, dq.to(dq_ptr.dtype.element_ty), mask=mask1)


def _compressed_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    # 原有检查和初始化逻辑
    assert k.dtype == q.dtype and v.dtype == q.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert k_len == v_len and q_len > k_len
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads

    o = torch.zeros_like(q)
    lse = torch.full(
        (num_q_heads, q_len),
        fill_value=-torch.inf,
        dtype=torch.float32,
        device=q.device,
    )

    # 配置参数
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)

    # 计算总q块数量和单次最大允许的q块数量
    num_q_blocks = triton.cdiv(max_seqlen_q, BLOCK_SIZE_Q)  # 总q块数
    total_grid_size = batch_size * num_q_heads * num_q_blocks  # 总grid尺寸

    if total_grid_size <= utils.MAX_GRID_DIM:
        # 无需切割，单次启动kernel
        grid = (batch_size, num_q_heads, num_q_blocks)
        forward_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            lse.stride(0),
            lse.stride(1),
            q_block_start=0,  # 起始q块偏移为0
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # 计算单次最大可处理的q块数量（确保batch_size * num_q_heads * curr_blocks <= 65536）
        MAX_GRID_Q_BLOCKS_PER_LAUNCH = utils.MAX_GRID_DIM // (batch_size * num_q_heads)
        if MAX_GRID_Q_BLOCKS_PER_LAUNCH <= 0:
            raise ValueError(
                f"batch_size ({batch_size}) 和 num_q_heads ({num_q_heads}) 过大，无法满足grid限制"
            )

        # 按q块分批次处理
        for start_block in range(0, num_q_blocks, MAX_GRID_Q_BLOCKS_PER_LAUNCH):
            # 当前批次处理的q块数量（最后一批可能不足）
            curr_blocks = min(MAX_GRID_Q_BLOCKS_PER_LAUNCH, num_q_blocks - start_block)
            grid = (batch_size, num_q_heads, curr_blocks)  # 本次grid尺寸

            # 启动kernel处理当前批次的q块
            forward_kernel[grid](
                q,
                k,
                v,
                o,
                lse,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                num_k_heads,
                num_share_q_heads,
                head_dim,
                sm_scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                lse.stride(0),
                lse.stride(1),
                q_block_start=start_block,  # 传递当前批次的起始q块偏移
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return o, lse


def _compressed_attention_bwd(
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
):
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    v_len, num_v_heads, head_dim = v.shape
    o_len, num_o_heads, head_dim = o.shape
    num_share_q_heads = num_q_heads // num_k_heads
    # compute D
    print(f"o.shape is {o.shape}")
    delta = torch.zeros([num_o_heads, o_len], device=o.device, dtype=torch.float32)
    BLOCK_SIZE_O = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_O, IS_HOPPER_GPU)
    o_blocks = triton.cdiv(o_len, BLOCK_SIZE_O)
    total_grid_size = num_o_heads * o_blocks
    if total_grid_size <= utils.MAX_GRID_DIM:
        grid = lambda META: (o_blocks, num_o_heads)
        backward_sum_o_do[grid](
            o,
            do,
            delta,
            o_len,
            head_dim,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            delta.stride(0),
            delta.stride(1),
            o_block_offset=0,
            BLOCK_SIZE_O=BLOCK_SIZE_O,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        # 计算单次最大可处理的q块数量（确保不超过grid限制）
        # 单次grid尺寸 = batch_size * num_k_heads * curr_q_blocks <= 65535
        max_curr_o_blocks = utils.MAX_GRID_DIM // num_o_heads

        # 按q块分批次处理（切割max_seqlen_q）
        for o_block_start in range(0, o_blocks, max_curr_o_blocks):
            # 当前批次处理的q块数量（最后一批可能不足）
            curr_o_blocks = min(max_curr_o_blocks, o_blocks - o_block_start)
            # 当前批次的grid尺寸
            grid = (num_o_heads, curr_o_blocks)
            backward_sum_o_do[grid](
                o,
                do,
                delta,
                o_len,
                head_dim,
                o.stride(0),
                o.stride(1),
                o.stride(2),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                delta.stride(0),
                delta.stride(1),
                o_block_offset=o_block_start,
                BLOCK_SIZE_O=BLOCK_SIZE_O,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    # compute dk dv
    dk = torch.zeros(
        num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype
    )
    dv = torch.zeros(
        num_share_q_heads, k_len, num_k_heads, head_dim, device=k.device, dtype=k.dtype
    )
    batch_size = cu_seqlens_q.shape[0] - 1
    print(f"compressed max_seqlen_k {max_seqlen_k},max_seqlen_q {max_seqlen_q}")
    grid = lambda META: (
        batch_size,
        num_q_heads,
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_K = 32
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_K, IS_HOPPER_GPU)
    backward_dkdv[grid](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dk,
        dv,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    dk = dk.sum(0)
    dv = dv.sum(0)
    # compute dq
    dq = torch.zeros_like(q)
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_K = 32
    num_warps, num_stages = get_num_warps_stages(head_dim, BLOCK_SIZE_Q, IS_HOPPER_GPU)

    q_blocks = triton.cdiv(max_seqlen_q, BLOCK_SIZE_Q)
    total_grid_size = batch_size * num_q_heads * q_blocks
    if total_grid_size <= utils.MAX_GRID_DIM:
        grid = lambda META: (batch_size, num_q_heads, q_blocks)
        backward_dq[grid](
            q,
            k,
            v,
            lse,
            delta,
            do,
            dq,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            lse.stride(0),
            lse.stride(1),
            delta.stride(0),
            delta.stride(1),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            q_block_offset=0,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        max_curr_q_blocks = utils.MAX_GRID_DIM // (batch_size * num_q_heads)
        # 按q块分批次处理（切割max_seqlen_q）
        for q_block_start in range(0, q_blocks, max_curr_q_blocks):
            # 当前批次处理的q块数量（最后一批可能不足）
            curr_q_blocks = min(max_curr_q_blocks, q_blocks - q_block_start)
            # 当前批次的grid尺寸
            grid = (batch_size, num_k_heads, curr_q_blocks)
            backward_dq[grid](
                q,
                k,
                v,
                lse,
                delta,
                do,
                dq,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                num_k_heads,
                num_share_q_heads,
                head_dim,
                sm_scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                lse.stride(0),
                lse.stride(1),
                delta.stride(0),
                delta.stride(1),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                q_block_offset=q_block_start, 
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    return dq, dk, dv


class CompressedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kernel_size: int,
        kernel_stride: int,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_scale=None,
    ):
        # dtype check
        assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
        assert q.dtype == k.dtype and k.dtype == v.dtype
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
        # softmax scale
        if sm_scale is None:
            sm_scale = 1 / math.sqrt(q.shape[-1])
        o, lse = _compressed_attention_fwd(
            q,
            k,
            v,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )
        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.sm_scale = sm_scale
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.kernel_size = kernel_size
        ctx.kernel_stride = kernel_stride
        return o, lse

    @staticmethod
    def backward(ctx, do: torch.Tensor, *args) -> Any:
        q, k, v, o, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        max_seqlen_q = ctx.max_seqlen_q
        max_seqlen_k = ctx.max_seqlen_k
        sm_scale = ctx.sm_scale
        kernel_size = ctx.kernel_size
        kernel_stride = ctx.kernel_stride
        dq, dk, dv = _compressed_attention_bwd(
            o,
            do,
            lse,
            q,
            k,
            v,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            sm_scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


@triton.jit
def score_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    s_ptr,
    kernel_size,
    kernel_stride,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_ln,
    stride_sh,
    stride_sq,
    stride_sk,
    # 新增：当前批次的q块起始偏移
    q_block_offset: tl.constexpr,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_bkh = tl.program_id(0)
    pid_b = pid_bkh // NUM_KV_HEADS
    pid_kh = pid_bkh % NUM_KV_HEADS
    # 当前批次内的局部q块ID -> 转换为全局q块ID（加上偏移）
    pid_q_local = tl.program_id(1)
    pid_q = pid_q_local + q_block_offset  # 全局q块ID
    pid_k = tl.program_id(2)

    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start

    # 跳过超出实际序列长度的块
    if pid_q * BLOCK_SIZE_Q >= q_len or pid_k * BLOCK_SIZE_K >= k_len:
        return

    # init k pointer and load k
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")

    # offsets：基于全局q块ID计算实际序列索引
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q
    off_k = tl.arange(0, BLOCK_SIZE_K) + pid_k * BLOCK_SIZE_K
    causal_mask = off_q[:, None] >= (off_k * kernel_stride + kernel_size - 1)[None, :]

    # init score
    s = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)

    # loop over gqa heads
    for h in range(NUM_SHARE_Q_HEADS):
        pid_h = pid_kh * NUM_SHARE_Q_HEADS + h
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, HEAD_DIM),
            strides=(stride_qn, stride_qd),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),  # 使用全局q块ID
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
            shape=(q_len, 1),
            strides=(stride_ln, stride_lh),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),  # 使用全局q块ID
            block_shape=(BLOCK_SIZE_Q, 1),
            order=(0, 1),
        )
        # load q and lse
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.dot(q, k) * qk_scale
        # compute score
        s += tl.where(causal_mask, tl.exp2(qk - lse), 0)

    # save output
    Q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    D = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
    mask = (Q < q_len) & (D < k_len)
    ptr = (
        s_ptr + pid_kh * stride_sh + q_start * stride_sq + Q * stride_sq + D * stride_sk
    )
    tl.store(ptr, s.to(s_ptr.dtype.element_ty), mask=mask)


def _get_attention_score(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    kernel_size: int,
    kernel_stride: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
) -> torch.Tensor:
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert lse.dtype == torch.float32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len > k_len
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # init score
    score = torch.zeros(
        num_k_heads, q_len, max_seqlen_k, dtype=torch.float32, device=q.device
    )

    # 配置参数
    BLOCK_SIZE_Q = 64
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    # 计算总q块和k块数量
    q_blocks = triton.cdiv(max_seqlen_q, BLOCK_SIZE_Q)
    k_blocks = triton.cdiv(max_seqlen_k, BLOCK_SIZE_K)
    # 计算总grid尺寸
    total_grid_size = batch_size * num_k_heads * q_blocks * k_blocks
    if total_grid_size <= utils.MAX_GRID_DIM:
        # 无需切割，直接运行
        grid = (batch_size * num_k_heads, q_blocks, k_blocks)
        score_kernel[grid](
            q,
            k,
            lse,
            score,
            kernel_size,
            kernel_stride,
            cu_seqlens_q,
            cu_seqlens_k,
            num_k_heads,
            num_share_q_heads,
            head_dim,
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            lse.stride(0),
            lse.stride(1),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            q_block_offset=0,  # 起始q块偏移为0
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            num_warps=8,
            num_stages=3,
        )
    else:
        # 计算单次最大可处理的q块数量（确保不超过grid限制）
        # 单次grid尺寸 = batch_size * num_k_heads * curr_q_blocks * k_blocks <= 65535
        max_curr_q_blocks = utils.MAX_GRID_DIM // (batch_size * num_k_heads * k_blocks)
        if max_curr_q_blocks <= 0:
            raise ValueError(
                f"无法满足grid限制,batch_size={batch_size}, num_k_heads={num_k_heads}, k_blocks={k_blocks}"
            )

        # 按q块分批次处理（切割max_seqlen_q）
        for q_block_start in range(0, q_blocks, max_curr_q_blocks):
            # 当前批次处理的q块数量（最后一批可能不足）
            curr_q_blocks = min(max_curr_q_blocks, q_blocks - q_block_start)
            # 当前批次的grid尺寸
            grid = (batch_size * num_k_heads, curr_q_blocks, k_blocks)

            # 启动kernel处理当前批次
            score_kernel[grid](
                q,
                k,
                lse,
                score,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                num_k_heads,
                num_share_q_heads,
                head_dim,
                sm_scale,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                lse.stride(0),
                lse.stride(1),
                score.stride(0),
                score.stride(1),
                score.stride(2),
                q_block_offset=q_block_start,  # 传递当前批次的q块起始偏移
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_D=BLOCK_SIZE_D,
                num_warps=8,
                num_stages=3,
            )

    return score


@triton.jit
def _transform_score_kernel(
    s_ptr,  # score, shape: [num_heads, q_len, k_len]
    bs_ptr,  # block wise score: [num_heads, q_len, num_k_block]
    offs,
    cu_seqlens_q,
    # shape
    num_heads,
    num_offs,
    max_k_len,
    max_blocks,
    pad_len,
    # kernel & block size
    block_size,
    block_stride,  # block_size // kernel_stride
    init_blocks,
    local_blocks,
    # stride
    stride_sh,
    stride_sq,
    stride_sk,
    stride_bsh,
    stride_bsq,
    stride_bsk,
    # 新增：当前批次的q块起始偏移
    q_block_offset: tl.constexpr,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_O: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    # 当前批次内的局部q块ID -> 转换为全局q块ID（加上偏移）
    pid_q_local = tl.program_id(1)
    pid_q = pid_q_local + q_block_offset  # 全局q块ID
    pid_k = tl.program_id(2)

    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = pid_k * BLOCK_SIZE_K

    # 跳过超出实际序列长度的块（使用全局q块ID判断）
    if pid_q * BLOCK_SIZE_Q >= q_len:
        return

    # load weight
    off_o = tl.arange(0, BLOCK_SIZE_O)
    w = tl.load(offs + off_o, mask=off_o < num_offs, other=0)

    # load score：基于全局q块ID计算实际序列索引
    off_q = pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)  # 全局q索引
    off_k = (k_start + tl.arange(0, BLOCK_SIZE_K)) * block_stride - pad_len
    off_k= tl.maximum(0, off_k)
    off_k = off_k[None, :] + off_o[:, None]

    s_ptrs = (
        s_ptr
        + q_start * stride_sq
        + pid_h * stride_sh
        + off_q[:, None, None] * stride_sq
        + off_k[None, :, :] * stride_sk
    )

    # weighted sum, [BQ, BO, BK] * [1, BO, 1] -> [BQ, BO, BK] -> [BQ, BK]
    s = tl.load(
        s_ptrs,
        mask=(off_q < q_len)[:, None, None] & (off_k >= 0) & (off_k < max_k_len),
        other=0,
    )
    s = s * w[None, :, None]
    s = tl.sum(s, axis=1)

    # init mask and local mask
    off_bq = off_q // block_size
    off_bk = k_start + tl.arange(0, BLOCK_SIZE_K)
    s = tl.where(
        (
            (off_bq[:, None] >= off_bk[None, :])  # causal mask
            & (off_bq[:, None] < off_bk[None, :] + local_blocks)  # local window
        )
        | (off_bk[None, :] < init_blocks),  # init window
        float("inf"),
        s,
    )

    # store block wise score
    bs_ptrs = (
        bs_ptr
        + q_start * stride_bsq
        + pid_h * stride_bsh
        + off_q[:, None] * stride_bsq
        + off_bk[None, :] * stride_bsk
    )
    tl.store(
        bs_ptrs,
        s,
        mask=(off_q < q_len)[:, None] & (off_bk < max_blocks)[None, :],
    )


def transform_score(
    score: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
) -> torch.Tensor:
    num_k_heads, total_query_len, max_key_len = score.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    pad_len = kernel_size // kernel_stride - 1
    max_blocks = math.ceil(max_seqlen_q / block_size)
    block_score = torch.zeros(
        num_k_heads,
        total_query_len,
        max_blocks,
        dtype=torch.float32,
        device=score.device,
    )
    offs = (
        torch.arange(kernel_size // kernel_stride, device=score.device)[:, None]
        + torch.arange(block_size // kernel_stride, device=score.device)[None, :]
    ).view(-1)
    offs = torch.histc(offs, bins=offs.max() + 1, min=0, max=offs.max())
    num_offs = int(offs.shape[0])
    BLOCK_SIZE_K = min(128, triton.next_power_of_2(max_blocks))
    BLOCK_SIZE_O = triton.next_power_of_2(num_offs)
    BLOCK_SIZE_Q = 8

    # 计算总q块和k块数量
    q_blocks = triton.cdiv(total_query_len, BLOCK_SIZE_Q)
    k_blocks = triton.cdiv(max_blocks, BLOCK_SIZE_K)
    # 计算总grid尺寸
    total_grid_size = num_k_heads * batch_size * q_blocks * k_blocks
    if total_grid_size <= utils.MAX_GRID_DIM:
        print(
            f"transform_score: num_k_heads {num_k_heads},batch_size {batch_size},q_blocks {q_blocks},k_blocks {k_blocks}"
        )
        print(
            f"transform_score: score {score},score shape {score.shape},cu_seqlens_q {cu_seqlens_q},cu_seqlens_q shape  {cu_seqlens_q.shape},num_offs {num_offs}"
        )
        # 无需切割，直接运行完整grid
        grid = (num_k_heads * batch_size, q_blocks, k_blocks)
        _transform_score_kernel[grid](
            score,
            block_score,
            offs,
            cu_seqlens_q,
            num_k_heads,
            num_offs,
            max_key_len,
            max_blocks,
            pad_len,
            block_size,
            block_size // kernel_stride,
            init_blocks,
            local_blocks,
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_score.stride(0),
            block_score.stride(1),
            block_score.stride(2),
            q_block_offset=0,  # 起始q块偏移为0
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_O=BLOCK_SIZE_O,
            num_warps=8,
            num_stages=3,
        )
    else:
        print(f"enter transform_score split grid")
        print(f"BLOCK_SIZE_Q {BLOCK_SIZE_Q},BLOCK_SIZE_K {BLOCK_SIZE_K} triton.next_power_of_2(max_blocks) {triton.next_power_of_2(max_blocks)},BLOCK_SIZE_O {BLOCK_SIZE_O}")
        print(
            f"transform_score split grid: num_k_heads {num_k_heads},batch_size {batch_size},q_blocks {q_blocks},k_blocks {k_blocks}"
        )
        print(
            f"transform_score split grid: score {score},score shape {score.shape},cu_seqlens_q {cu_seqlens_q},cu_seqlens_q shape  {cu_seqlens_q.shape},num_offs {num_offs}"
        )
        
        # 计算单次最大可处理的q块数量（确保不超过grid限制）
        # 单次grid尺寸 = num_k_heads * batch_size * curr_q_blocks * k_blocks <= 65535
        max_curr_q_blocks = utils.MAX_GRID_DIM // (num_k_heads * batch_size * k_blocks)
        if max_curr_q_blocks <= 0:
            raise ValueError(
                f"无法满足grid限制,num_k_heads={num_k_heads}, batch_size={batch_size}, k_blocks={k_blocks}"
            )
        # 按q块分批次处理（切割total_query_len）
        for q_block_start in range(0, q_blocks, max_curr_q_blocks):
            print(f"q_block_start {q_block_start}")
            # 当前批次处理的q块数量（最后一批可能不足）
            curr_q_blocks = min(max_curr_q_blocks, q_blocks - q_block_start)
            # 当前批次的grid尺寸
            grid = (num_k_heads * batch_size, curr_q_blocks, k_blocks)
            print(
                f"max_curr_q_blocks {max_curr_q_blocks},q_block_start {q_block_start},q_blocks {q_blocks}"
            )
            # 启动kernel处理当前批次
            _transform_score_kernel[grid](
                score,
                block_score,
                offs,
                cu_seqlens_q,
                num_k_heads,
                num_offs,
                max_key_len,
                max_blocks,
                pad_len,
                block_size,
                block_size // kernel_stride,
                init_blocks,
                local_blocks,
                score.stride(0),
                score.stride(1),
                score.stride(2),
                block_score.stride(0),
                block_score.stride(1),
                block_score.stride(2),
                q_block_offset=q_block_start,  # 传递当前批次的q块起始偏移
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                BLOCK_SIZE_O=BLOCK_SIZE_O,
                num_warps=8,
                num_stages=3,
            )

    return block_score


def compressed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int = None,
    max_seqlen_k: int = None,
    sm_scale: float = None,
    init_blocks: int = 1,
    local_blocks: int = 2,
    parallel_topk_compute: Union[str, bool] = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention between query and compressed key and value. Compute attention output and topk block idx used in topk_sparse_attention.

    Args:
        q (torch.Tensor): shape [total_q_len, num_q_heads, head_dim]
        k (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        v (torch.Tensor): shape [total_kv_len, num_kv_heads, head_dim]
        kernel_size (int): kernel size in compress_key_value
        kernel_stride (int): stride of compress_key_value
        block_size (int): key value block size for topk sparse attention.
        topk (int): number of blocks for each query.
        cu_seqlens_q (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_q in flash_attn_func_varlen.
        cu_seqlens_k (torch.Tensor): shape [batch_size + 1], similar to cu_seqlens_k in flash_attn_func_varlen.
        max_seqlen_q (int): max q len of the batch. Defaults to None, means (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item().
        max_seqlen_k (int): max k len of the batch. Defaults to None, means (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item().
        sm_scale (float, optional): softmax scale. Defaults to None, means 1/sqrt(head_dim).
        init_blocks (int, optional): Number of init blocks for each query. Defaults to 1.
        local_blocks (int, optional): Number of local blocks for each query. Defaults to 2.
        parallel_topk_compute (str, optional): Only set it to False when the sequence length is too long. This can avoid a current bug.
            We'll fix this issue later. Defaults to auto, it will be set to False when the sequence length is greater than 32k and True otherwise.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention output and topk_idx used in topk_sparse_attention
    """
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
    attn_output, lse = CompressedAttention.apply(
        q,
        k,
        v,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        sm_scale,
    )

    # do not select topk index
    if topk <= 0:
        warnings.warn("topk <= 0, returned topk_idx will be None")
        return attn_output, None

    assert topk >= init_blocks + local_blocks
    with torch.no_grad():
        num_k_heads, num_q_heads = k.shape[1], q.shape[1]
        num_shared_q_heads = num_q_heads // num_k_heads
        batch_size = cu_seqlens_q.shape[0] - 1
        q_idx = torch.cat(
            [
                torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                for i in range(batch_size)
            ],
            dim=0,
        )
        q_idx = q_idx // block_size
        # whether to use parallel version
        if parallel_topk_compute == "auto":
            parallel_topk_compute = cu_seqlens_q[-1] <= 32768
        # parallel version
        if parallel_topk_compute:
            # recompute score
            score = _get_attention_score(
                q,
                k,
                lse,
                kernel_size,
                kernel_stride,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                sm_scale,
            )
            # transform score to block-wise score
            score = transform_score(
                score,
                kernel_size,
                kernel_stride,
                block_size,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                init_blocks,
                local_blocks,
            )
            # get topk
            topk = min(topk, score.shape[-1])
            topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
            topk_idx[topk_idx > q_idx[None, :, None]] = -1
            topk_idx = topk_idx.to(torch.int32)
        # non parallel version, avoid some current bugs when sequence length is too long
        # FIXME: need to fix later
        else:
            topk_idx_list = []
            for h in range(num_k_heads):
                # recompute score
                score = _get_attention_score(
                    q[:, h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    k[:, h : h + 1],
                    lse[h * num_shared_q_heads : (h + 1) * num_shared_q_heads],
                    kernel_size,
                    kernel_stride,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    sm_scale,
                )
                # transform score to block-wise score
                score = transform_score(
                    score,
                    kernel_size,
                    kernel_stride,
                    block_size,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    init_blocks,
                    local_blocks,
                )
                # get topk
                topk = min(topk, score.shape[-1])
                topk_idx = score.topk(topk, dim=-1).indices.sort(-1).values
                topk_idx[topk_idx > q_idx[None, :, None]] = -1
                topk_idx = topk_idx.to(torch.int32)
                topk_idx_list.append(topk_idx)
            topk_idx = torch.cat(topk_idx_list, dim=0)
    return attn_output, topk_idx
