# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start nodes.
# > bazel run -c opt //examples/python/utils:nodectl -- up
#
# Run this example script.
# > bazel run -c opt //examples/python/ml/flax_mlp:flax_mlp

import argparse
import json
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import lax

import spu
import spu.utils.distributed as ppd
# import spu.spu_pb2 as spu_pb2

import pdb



parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="/home/admin/dev/examples/python/conf/3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)

ppd.init(conf["nodes"], conf["devices"])

copts = spu.libspu.CompilerOptions()
copts.enable_pretty_print = False
# copts.disable_partial_sort_optimization = True

# copts.xla_pp_kind = spu.libspu.RuntimeConfig.SortMethod.SORT_QUICK
# copts.xla_pp_kind = 2
copts.xla_pp_kind = spu.libspu.XLAPrettyPrintKind.HTML
#  pdb.set_trace()

# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True


# def compute_sim(q, k):
#     return q @ jnp.swapaxes(k, -1, -2)


def idx_to_onehot(x):
    x = nn.one_hot(x, num_classes=k*T, axis=-1)
    x = x.reshape(-1, k*T)
    return x
def idx_to_onehotE(x):
    x = nn.one_hot(x, num_classes=E, axis=-1)
    x = x.reshape(-1, E)
    return x

def max_one(x):
    res = jnp.max(x, axis=-2)
    return res


def max_two(x, y):
    max_val = jnp.maximum(x, y)
    return max_val


def top2(x):
    topk_values, top_index = lax.top_k(x, k=2)
    return topk_values, top_index

def top2_loop(x):
    k = 2  # 取 top2
    top_values = jnp.zeros((T, k), dtype=x.dtype)
    top_indices = jnp.zeros((T, k), dtype=jnp.int32)

    for i in range(T):
        row = x[i]
        row_top_values, row_top_indices = lax.top_k(row, k=k)
        top_values = top_values.at[i].set(row_top_values)  # 更新值
        top_indices = top_indices.at[i].set(row_top_indices)  # 更新索引

    return top_values, top_indices


def topk(x):
    x = jnp.transpose(x)
    topk_values, top_index = lax.top_k(x, k=int(k*T/E))
    x = jnp.transpose(x)
    return topk_values, top_index

def transpose(x):
    x = jnp.transpose(x)
    return x

def sort(x):
    sorted_values = lax.sort(x, dimension=-1)
    return sorted_values


def softmax(x):
    # # exp first
    # x_sum = jnp.sum(x, axis=-1, keepdims=True)
    # # return x / x_sum
    
    # x_sum_recip = 1 / x_sum
    # return x * x_sum_recip
    return nn.softmax(x)

def ewmm(q, bmax):
    return q * bmax

def gemv(a, b):
    return a @ b

# def attention(q, k):
#     aw = q @ jnp.swapaxes(k, -1, -2)
#     aw = nn.softmax(aw)
#     output = aw @ k
#     return output


def nonzero(x):
    return  jnp.nonzero(x, size=x.size, fill_value=-1)
    # return jax.jit(jnp.nonzero, static_argnames='size')

def where(x):
    return jnp.argwhere(x > 0.1, size=x.size, fill_value=-1)

def unique(x):
    return jnp.unique(x, return_counts=True, size=x.size, fill_value=-1)
    
def tile(x):
    return jnp.tile(x, reps=(k, 1))

def flatten(x):
    return x.T.reshape(-1, 1)

def equal(x, y):
    return jnp.equal(x, y)

def argsort(x):
    x = jnp.transpose(x)
    x = jnp.argsort(x)
    x = jnp.transpose(x)
    return x

def choose(x):
    return x[0:int(k*T/E)]

def gelu(x):
    return nn.gelu(x)


def gate_routing(input, gate_linear):
    print("----------------Start Gate Routing------------------")
    # gate_out = input @ gate_linear
    gate_out = lax.dot(input, gate_linear)
    print("Gate network lax.dot", input.shape, gate_linear.shape, "->", gate_out.shape)

    top_value, top_index = lax.top_k(gate_out, k=k)
    # top_value, top_index = top2_loop(gate_out)


    print("top_expert", gate_out.shape, "->", top_value.shape)
    print("top_index", gate_out.shape, "->", top_index.shape)

    softmax_value = softmax(top_value)
    print("softmax", softmax_value.shape, "->", top_value.shape)
    
    fla_top_value = flatten(softmax_value)
    fla_top_index = flatten(top_index)
    print("fla_top_value", top_value.shape, "->", fla_top_value.shape)
    print("fla_top_index", top_index.shape, "->", fla_top_index.shape)

    print("-----------------End Gate Routing-------------------")
    return fla_top_value, fla_top_index

# TODO:implement batch optimization
def swiglu(tokens, expert_upproj, expert_gateproj, expert_downproj):
    # tokens (E, T', d)
    expert_med = lax.dot_general(tokens, expert_upproj,
                                 dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    print("upproj", tokens.shape, expert_upproj.shape, "->", expert_med.shape)

    expert_gate = lax.dot_general(tokens, expert_gateproj,
                                  dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    print("gateproj", tokens.shape, expert_gateproj.shape, "->", expert_gate.shape)
    # return expert_med, expert_gate
    silu_gate = jax.nn.gelu(expert_gate)
    # silu_gate = jax.nn.gelu(tokens)
    # generate a zero array with shape (E,T,de)
    # silu_gate = np.zeros((E,T,de))
    # num_part = int(E / 8)
    # for i in range(8):
    #     silu_gate[int(i*E/8):int((i+1)*E/8)] = jax.nn.silu(expert_gate[int(i*E/8):int((i+1)*E/8)])
    # silu_gate = jax.nn.silu(expert_gate)
    # return silu_gate, None
    print("silu", expert_gate.shape, "->", silu_gate.shape)
    expert_med2 = silu_gate * expert_med
    print("silu", expert_gate.shape, "->", expert_med2.shape)
    # return expert_med2, expert_gate
    expert_out = lax.dot_general(expert_med2, expert_downproj,
                                 dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    print("downproj", expert_med2.shape, expert_downproj.shape, "->", expert_out.shape)
    return expert_out

def swiglu_batched(tokens, expert_upproj, expert_gateproj, expert_downproj):
    # tokens (E, T', d)
    tokens = tokens.reshape(-1, tokens.shape[-1])
    expert_med = tokens @ expert_upproj[0]
    print("upproj", tokens.shape, expert_upproj[0].shape, "->", expert_med.shape)

    expert_gate = tokens @ expert_gateproj[0]
    print("gateproj", tokens.shape, expert_gateproj[0].shape, "->", expert_gate.shape)

    # expert_med2 = jax.nn.silu(expert_gate) * expert_med
    expert_med2 = jax.nn.gelu(expert_gate) * expert_med
    print("silu", expert_gate.shape, "->", expert_med2.shape)

    expert_out = expert_med2 @ expert_downproj[0]
    print("downproj", expert_med2.shape, expert_downproj[0].shape, "->", expert_out.shape)
    expert_out = expert_out.reshape(E, -1, d)
    return expert_out

def attention(input, qkv_linear, w_o):
    # RMS Norm
    input = input / jnp.sqrt(jnp.mean(input**2, axis=-1, keepdims=True))
    qkv = input @ qkv_linear
    q, k, v = jnp.split(qkv, 3, axis=-1)
    print("qkv", input.shape, qkv_linear.shape, "->", qkv.shape)
    print("q,k,v", q.shape, k.shape, v.shape)
    q_h = q.reshape(T, num_head, -1).transpose(1,0,2)
    k_h = k.reshape(T, num_head, -1).transpose(1,0,2)
    v_h = v.reshape(T, num_head, -1).transpose(1,0,2)
    print("divide head", q.shape, "->", q_h.shape)
    k_t = k_h.transpose(0,2,1)
    print("k_t", k_h.shape, "->", k_t.shape)
    attn = q_h @ k_t
    print("attn", q_h.shape, k_t.shape, "->", attn.shape)
    attn = jax.nn.softmax(attn, axis=-1)
    print("softmax", attn.shape, "->", attn.shape)
    out = attn @ v_h
    print("out", attn.shape, v_h.shape, "->", out.shape)
    out = out.transpose(1,0,2).reshape(T, -1)
    print("out", out.shape, "->", out.shape)
    out = out @ w_o
    print("out", out.shape, "->", out.shape)
    # RMS Norm
    out = out / jnp.sqrt(jnp.mean(out**2, axis=-1, keepdims=True))
    return out

def insecure_baseline(input, gate_linear, expert_upproj, expert_gateproj, expert_downproj):
    # print("torch.max", np.max(input))
    fla_top_value, fla_top_index = gate_routing(input, gate_linear)
    # return fla_top_value, fla_top_index
    print("--------------------Start Token Dispatch---------------------")
    choose_token = jnp.tile(input, reps=(E, 1, 1))
    choose_token = choose_token[:,:num_per_expert,:]
    print("choose_token", input.shape, "->", choose_token.shape)
    print("--------------------Start Expert Computation---------------------")
    # expert_out = swiglu_batched(choose_token, expert_upproj, expert_gateproj, expert_downproj)
    expert_out = swiglu(choose_token, expert_upproj, expert_gateproj, expert_downproj)
    print("--------------------End Expert Computation--------------------")
    ### only weighted_expert_out is needed in dense baseline
    choose_masked_score = np.random.randn(E, num_per_expert, 1)
    weighted_expert_out = ewmm(choose_masked_score, expert_out)
    trans_onehot = jnp.ones((E, T, num_per_expert))
    combine_out = lax.dot_general(trans_onehot, weighted_expert_out,
                                  dimension_numbers=(((2, 0), (1, 0)), ((), ())))
    
    return combine_out, fla_top_index, fla_top_value
    
def pro3_addmask(input, gate_linear, expert_upproj, expert_gateproj, expert_downproj):
    # print("torch.max", np.max(input))
    fla_top_value, fla_top_index = gate_routing(input, gate_linear)
    # return fla_top_value, fla_top_index
    print("--------------------Start Token Dispatch---------------------")
    zero_mask = equal(fla_top_index, expert_index)
    zero_mask = jnp.transpose(zero_mask)
    print("zero_mask", fla_top_index.shape, expert_index.shape, "->", zero_mask.shape)

    mask = zero_mask*10 + fla_top_value.reshape(-1)
    masked_score = zero_mask * fla_top_value.reshape(-1)
    print("add mask", zero_mask.shape, fla_top_value.shape, "->", mask.shape)
    # return masked_score
    _, choose_index = lax.top_k(mask, k=num_per_expert)
    choose_index1 = jnp.floor(choose_index / k)
    print("topk choose index", mask.shape, "->", choose_index.shape)
    
    print("--------------------New Way to Dispatch---------------------")
    choose_onehot = nn.one_hot(choose_index, num_classes=k*T, axis=-1).reshape(E, -1, int(k*T))
    print("onehot", choose_index.shape, "->", choose_onehot.shape)
    choose_onehot1 = nn.one_hot(choose_index1, num_classes=T, axis=-1).reshape(E, -1, T)
    print("onehot1", choose_index1.shape, "->", choose_onehot1.shape)

    choose_token = choose_onehot1 @ input
    print("matmul choose_token", choose_onehot1.shape, input.shape, "->", choose_token.shape)
    choose_masked_score = lax.dot_general(choose_onehot, masked_score,
                                          dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    
    choose_masked_score = choose_masked_score.reshape(E, -1, 1)
    print("matmul choose_masked_score", choose_onehot.shape, masked_score.shape, "->", choose_masked_score.shape)
    ### if use batch optization:
    # choose_onehot = choose_onehot.reshape(-1, choose_onehot.shape[-1])
    # masked_score = masked_score[0]
    # choose_masked_score = choose_onehot @ masked_score
    # choose_masked_score = choose_masked_score.reshape(E, -1, 1)
    # print("matmul choose_masked_score", choose_onehot.shape, masked_score.shape, "->", choose_masked_score.shape)
    print("--------------------Start Expert Computation---------------------")
    # expert_med = lax.dot_general(choose_token, expert_upproj,
    #                              dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    # print("upproj", choose_token.shape, expert_upproj.shape, "->", expert_med.shape)

    # expert_med2 = gelu(expert_med)
    # print("gelu", expert_med.shape, "->", expert_med2.shape)

    # expert_out = lax.dot_general(expert_med2, expert_downproj,
    #                              dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    # print("downproj", expert_med2.shape, expert_downproj.shape, "->", expert_out.shape)
    # expert_out = swiglu(choose_token, expert_upproj, expert_gateproj, expert_downproj)
    expert_out = swiglu_batched(choose_token, expert_upproj, expert_gateproj, expert_downproj)
    print("--------------------End Expert Computation--------------------")

    weighted_expert_out = ewmm(choose_masked_score, expert_out)
    print("element_mul", choose_masked_score.shape, expert_out.shape, "->", weighted_expert_out.shape)
    trans_onehot = choose_onehot1.swapaxes(1, 2)
    print("swapaxes", choose_onehot1.shape, "->", trans_onehot.shape)

    combine_out = lax.dot_general(trans_onehot, weighted_expert_out,
                                  dimension_numbers=(((2, 0), (1, 0)), ((), ())))
    print("before_combine", trans_onehot.shape, expert_out.shape, "->", combine_out.shape)

    # return choose_index1
    return combine_out

def spu_baseline1():
    input_enc = ppd.device("P2")(lambda x: x)(input)
    # plaintext @ cyphertext
    stime = time.time()
    gate_out = ppd.device("SPU")(gemv)(input_enc, gate_linear)
    print("plain-cypher matmul", input.shape, gate_linear.shape, "->",gate_out.shape)
    print(f'  time: {time.time() - stime} s')

    # cyphertext @ cyphertext
    gate_enc = ppd.device("P2")(lambda x: x)(gate_linear)
    stime = time.time()
    gate_out_enc = ppd.device("SPU")(gemv)(input_enc, gate_enc)
    print("cypher-cypher matmul", input.shape, gate_linear.shape, "->",gate_out.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    top_value, top_index = ppd.device("SPU")(top2)(gate_out_enc)
    print("top_expert", gate_out_enc.shape, "->", top_value.shape)
    print("top_index", gate_out_enc.shape, "->", top_index.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    softmax_value = ppd.device("SPU")(softmax)(top_value)
    print("softmax", softmax_value.shape, "->", top_value.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    tile_input_enc = ppd.device("SPU")(tile)(input_enc)
    print("tile_input_enc", input.shape, "->", tile_input_enc.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    fla_top_value = ppd.device("SPU")(flatten)(top_value)
    print("fla_top_value", top_value.shape, "->", fla_top_value.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    fla_top_index = ppd.device("SPU")(flatten)(top_index)
    print("fla_top_index", top_index.shape, "->", fla_top_index.shape)
    print(f'  time: {time.time() - stime} s')

    print("-----------------------------------------------------")
    stime = time.time()
    token_expert_index = ppd.device("SPU")(lambda x: x[0])(top_index)
    token_expert_value = ppd.device("SPU")(lambda x: x[0])(top_value)
    print("choose_token_index", top_index.shape, "->", token_expert_index.shape)
    print("choose_token_value", top_value.shape, "->", token_expert_value.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    expert_onehot = ppd.device("SPU")(idx_to_onehotE)(token_expert_index)
    print("expert_onehot", token_expert_index.shape, "->", expert_onehot.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    token_expert_upproj = ppd.device("SPU")(gemv)(expert_onehot, all_expert_upproj)
    token_expert_downproj =  ppd.device("SPU")(gemv)(expert_onehot, all_expert_downproj)
    print("token_expert_upproj", all_expert_upproj.shape, "->", token_expert_upproj.shape)
    print("token_expert_downproj", all_expert_downproj.shape, "->", token_expert_downproj.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    token_expert_upproj1 = ppd.device("SPU")(lambda x: x.swapaxes(0, 1))(token_expert_upproj)
    token_expert_downproj1 = ppd.device("SPU")(lambda x: x.swapaxes(0, 1))(token_expert_downproj)
    print("swapaxes", token_expert_upproj.shape, "->", token_expert_upproj1.shape)
    print("swapaxes", token_expert_downproj.shape, "->", token_expert_downproj1.shape)
    print(f'  time: {time.time() - stime} s')

    print("-----------------------------------------------------")
    
    stime = time.time()
    expert_med = ppd.device("SPU")(gemv)(input[0], token_expert_upproj1)
    print("upproj", input[0].shape, token_expert_upproj1.shape, "->", expert_med.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    expert_med2 = ppd.device("SPU")(gelu)(expert_med)
    print("gelu", expert_med.shape, "->", expert_med2.shape)
    print(f'  time: {time.time() - stime} s')

    stime = time.time()
    expert_out = ppd.device("SPU")(gemv)(expert_med2, token_expert_downproj1)
    print("downproj", expert_med2.shape, token_expert_downproj.shape, "->", expert_out.shape)
    print(f'  time: {time.time() - stime} s')

def baseline2(input, gate_linear, expert_upproj, expert_gateproj, expert_downproj):
    fla_top_value, fla_top_index = gate_routing(input, gate_linear)

    print("--------------------Start Expert Computation---------------------")
    tile_input = jnp.tile(input, reps=(E, 1, 1))
    print("tile_input", input.shape, "->", tile_input.shape)
    # return tile_input

    # expert_med = lax.dot_general(tile_input, expert_upproj,
    #                              dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    # print("upproj", tile_input.shape, expert_upproj.shape, "->", expert_med.shape)

    # expert_med2 = gelu(expert_med)
    # print("gelu", expert_med.shape, "->", expert_med2.shape)

    # expert_out = lax.dot_general(expert_med2, expert_downproj,
    #                              dimension_numbers=(((2,), (1,)), ((0,), (0,))))
    # print("downproj", expert_med2.shape, expert_downproj.shape, "->", expert_out.shape)
    expert_out = swiglu(tile_input, expert_upproj, expert_gateproj, expert_downproj)
    # expert_out = swiglu_batched(tile_input, expert_upproj, expert_gateproj, expert_downproj)
    print("--------------------End Expert Computation--------------------")
    # return expert_out,fla_top_index,fla_top_value

    expert_out1 = expert_out.swapaxes(0, 1)
    print("swapaxes", expert_out.shape, "->", expert_out1.shape)

    top_index = fla_top_index.reshape(T, k)
    all_onehot = nn.one_hot(top_index, num_classes=E, axis=-1).reshape(T, k, E)
    print("onehot", top_index.shape, "->", all_onehot.shape)
    
    top_value1 = fla_top_value.reshape(T, k)
    onehot_weight = lax.dot_general(top_value1, all_onehot, 
                                    dimension_numbers=(((1,), (1,)), ((0,), (0,))))
    print("onehot dot_general", top_value1.shape, all_onehot.shape, "->", onehot_weight.shape)

    # combine_out = gemv(onehot_weight, expert_out1)
    combine_out = lax.dot_general(onehot_weight, expert_out1, 
                                    dimension_numbers=(((1,), (1,)), ((0,), (0,))))
    print("combine dot_general", onehot_weight.shape, expert_out1.shape, "->", combine_out.shape)

    return combine_out


def spu_baseline2():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    gate_linear_enc = ppd.device("P2")(lambda x: x)(gate_linear)
    expert_upproj_enc = ppd.device("P2")(lambda x: x)(all_expert_upproj)
    expert_gateproj_enc = ppd.device("P2")(lambda x: x)(all_expert_gateproj)
    expert_downproj_enc = ppd.device("P2")(lambda x: x)(all_expert_downproj)

    stime = time.time()
    # out = ppd.device("SPU")(baseline2)(input_enc, gate_linear, all_expert_upproj, all_expert_downproj)
    out = ppd.device("SPU")(baseline2, copts=copts)(input_enc, gate_linear_enc, expert_upproj_enc, expert_gateproj_enc, expert_downproj_enc)
    print(f'  time: {time.time() - stime} s')
    out = ppd.get(out)
    print(out[0][0:32])


def spu_pro3_1():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    gate_linear_enc = ppd.device("P2")(lambda x: x)(gate_linear)
    expert_upproj_enc = ppd.device("P2")(lambda x: x)(all_expert_upproj)
    expert_gateproj_enc = ppd.device("P2")(lambda x: x)(all_expert_gateproj)
    expert_downproj_enc = ppd.device("P2")(lambda x: x)(all_expert_downproj)

    stime = time.time()
    # out = ppd.device("SPU")(pro3)(input_enc, gate_linear_enc, expert_upproj_enc, expert_downproj_enc)
    out = ppd.device("SPU")(pro3_addmask, copts=copts)(input_enc, gate_linear_enc, expert_upproj_enc, expert_gateproj_enc, expert_downproj_enc)
    print(f'  time: {time.time() - stime} s')
    out = ppd.get(out)
    print(out[0][0:32])
    
def spu_inseure_baseline():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    gate_linear_enc = ppd.device("P2")(lambda x: x)(gate_linear)
    expert_upproj_enc = ppd.device("P2")(lambda x: x)(all_expert_upproj)
    expert_gateproj_enc = ppd.device("P2")(lambda x: x)(all_expert_gateproj)
    expert_downproj_enc = ppd.device("P2")(lambda x: x)(all_expert_downproj)

    stime = time.time()
    # out = ppd.device("SPU")(pro3)(input_enc, gate_linear_enc, expert_upproj_enc, expert_downproj_enc)
    out = ppd.device("SPU")(insecure_baseline, copts=copts)(input_enc, gate_linear_enc, expert_upproj_enc, expert_gateproj_enc, expert_downproj_enc)
    print(f'  time: {time.time() - stime} s')
    out = ppd.get(out)
    print(out[0][0:32])

def spu_attention():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    qkv_linear_enc = ppd.device("P2")(lambda x: x)(qkv_linear)
    w_o_enc = ppd.device("P2")(lambda x: x)(w_o)

    stime = time.time()
    out = ppd.device("SPU")(attention, copts=copts)(input_enc, qkv_linear_enc, w_o_enc)
    print(f'  time: {time.time() - stime} s')
    out = ppd.get(out)
    print(out[0][0:32])
    

def gate_routing_test():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    gate_linear_enc = ppd.device("P2")(lambda x: x)(gate_linear)

    stime = time.time()
    out = ppd.device("SPU")(gate_routing, copts=copts)(input_enc, gate_linear_enc)
    print(f'  time: {time.time() - stime} s')


def gelu_test():
    input_enc = ppd.device("P2")(lambda x: x)(test_input)
    stime = time.time()
    gelu_out = ppd.device("SPU")(gelu)(input_enc)
    print("gelu", test_input.shape, "->", gelu_out.shape)
    print(f'  time: {time.time() - stime} s')
    
def swiglu_test():
    input_enc = ppd.device("P1")(lambda x: x)(input)
    expert_upproj_enc = ppd.device("P2")(lambda x: x)(all_expert_upproj)
    expert_gateproj_enc = ppd.device("P2")(lambda x: x)(all_expert_gateproj)
    expert_downproj_enc = ppd.device("P2")(lambda x: x)(all_expert_downproj)
    def func(input, expert_upproj, expert_gateproj, expert_downproj):
        tokens = jnp.tile(input, reps=(E, 1, 1))
        output, _ = swiglu(tokens, expert_upproj, expert_gateproj, expert_downproj)
        return output
    stime = time.time()
    gemm_out = ppd.device("SPU")(func)(input_enc, expert_upproj_enc, expert_gateproj_enc, expert_downproj_enc)
    print("gemm", input.shape, expert_upproj.shape, "->", gemm_out.shape)
    print(f'  time: {time.time() - stime} s')

def test_test():
    spu = ppd.device("SPU")
    pdb.set_trace()

# ours: 1, 1.5, 2
# ablation: 256/512, 1->1.5, 1.5->2
if __name__ == '__main__':
    np.random.seed(135)
    use_batch_optim = True
    T = 128
    num_head = 16
    # deepseekmoe
    # d = 2048
    # E = 64
    # k = 6
    # de = 1408
    

    # qwen1.5-moe
    d = 2048
    E = 60
    k = 4
    de = 1408
    
    # olmoe
    # d = 2048
    # E = 64
    # k = 8
    # de = 1024
    
    # test dimension
    # d = 768
    # E = 8
    # k = 2
    # de = 384
    num_per_expert = int(1*k*T/E)
    # print(dir(spu.libspu.XLAPrettyPrintKind))
    # print(copts.xla_pp_kind.value)

    input = np.random.randn(T, d)
    qkv_linear = np.random.randn(d, 3*d)
    w_o = np.random.randn(d, d)
    gate_linear = np.random.randn(d, E)

    expert_index = np.arange(E)

    expert_upproj = np.random.randn(d, de)
    expert_downproj = np.random.randn(de, d)

    all_expert_upproj = np.random.randn(E, d, de)
    all_expert_gateproj = np.random.randn(E, d, de)
    all_expert_downproj = np.random.randn(E, de, d)

    test_input = np.random.randn(E, T, de)

    print('\n------\nRun on CPU protocol3')
    # out = pro3(input, gate_linear, all_expert_upproj, all_expert_gateproj, all_expert_downproj)
    # print(out[0][0:32])
    
    # out = pro3_addmask(input, gate_linear, all_expert_upproj, all_expert_gateproj, all_expert_downproj)
    # print(out[0][0:32])
    # exit(0)

    # test_test()

    print('\n------\nRun on SPU GateRouting')
    # gate_routing_test()

    print('\n------\nRun on SPU baseline1')
    # spu_baseline1()

    print('\n------\nRun on SPU protocol3')

    # spu_pro3_1()
    spu_inseure_baseline()
    # attention(input, qkv_linear, w_o)
    # spu_attention()
    
    print('\n------\nRun on SPU baseline2')
    # out = baseline2(input, gate_linear, all_expert_upproj, all_expert_gateproj, all_expert_downproj)
    # print(out[0][0:32])
    # spu_baseline2()
    # swiglu_test()

    # gelu_test()