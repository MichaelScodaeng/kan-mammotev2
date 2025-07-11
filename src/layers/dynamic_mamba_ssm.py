# kan_mamote/src/layers/dynamic_mamba_ssm.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from einops import rearrange, repeat

# --- Assume these are importable from your mamba_ssm setup ---
# (Keeping dummy functions for robustness, but you need actual ones for performance)
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    print("Warning: causal_conv1d not found, using dummy functions for conv operations.")
    causal_conv1d_fn = None # No dummy for this path, will fallback to non-kernel path
    causal_conv1d_update = None
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    print("Warning: mamba_ssm.ops.triton.selective_state_update not found, using dummy for selective state update.")
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm import Mamba2
from huggingface_hub import PyTorchModelHubMixin

class DynamicMambaSSM(Mamba2):
    """
    A Mamba2 block adapted to incorporate dynamic parameters influenced by
    the delta_t_embedding from K-MOTE.

    This implementation modulates the 'dt_bias' *internally within the DynamicMambaSSM*
    before it's used in the final `dt` calculation that's fed into the Triton kernels.
    The Triton kernels themselves still receive a (nheads,) dt_bias and a (B,L,nheads) dt.
    """
    def __init__(self,
                 d_model: int,
                 k_mote_delta_t_embedding_dim: int, # Dimension of the delta_t_embedding from K-MOTE (your D_time)
                 **kwargs):
        super().__init__(d_model, **kwargs)
        self.k_mote_delta_t_embedding_dim = k_mote_delta_t_embedding_dim

        # Add a projection layer to derive dt_modulation from delta_t_embedding.
        # The output dimension should match self.nheads, as dt_bias has this dimension.
        self.dt_modulation_proj = nn.Linear(k_mote_delta_t_embedding_dim, self.nheads, bias=True)

        # Initialize this projection to zeros. This ensures that at the start of training,
        # the dynamic modulation is zero, and the model behaves like a standard Mamba2.
        nn.init.zeros_(self.dt_modulation_proj.weight)
        nn.init.zeros_(self.dt_modulation_proj.bias)

    def forward(self, u: torch.Tensor, delta_t_embedding: torch.Tensor,
                seqlen: Optional[int] = None, seq_idx: Optional[torch.Tensor] = None,
                cu_seqlens: Optional[torch.Tensor] = None, inference_params: Optional[dict] = None):
        """
        Forward pass for the DynamicMambaSSM.

        Args:
            u: (batch, seqlen, hidden_dim) - Input features for the current token/timestep.
            delta_t_embedding: (batch, seqlen, k_mote_delta_t_embedding_dim)
                               - Time difference embedding from K-MOTE for each sequence element.
            seqlen, seq_idx, cu_seqlens, inference_params: Same as Mamba2's forward.
        """
        #print(f"DynamicMambaSSM forward called with input shape: {u.shape}, delta_t_embedding shape: {delta_t_embedding.shape}")
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        # --- Calculate dynamic dt_bias based on delta_t_embedding ---
        # dt_modulation_proj expects (N, k_mote_dim) where N is batch*seqlen.
        # Reshape delta_t_embedding to (batch*seqlen, k_mote_dim)
        dt_modulation_input = rearrange(delta_t_embedding, "b l d -> (b l) d")

        # Compute the modulation: (batch*seqlen, nheads)
        dt_modulation = self.dt_modulation_proj(dt_modulation_input)
        # --- NEW FIX: Reshape dt_modulation to match zxbcdt_modified's dimension ---
        dt_modulation_reshaped = rearrange(dt_modulation, "(b l) h -> b l h", b=batch, l=seqlen)
        # --- Mamba2 forward pass, with `dt_bias` (nheads) and `dt` (B,L,nheads) ---
        # The original Mamba2 forward expects self.dt_bias (nheads) and `dt` split from zxbcdt.
        # We need to apply our `dt_modulation` to the `dt` values *before* they enter the kernel.

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                # Call original step method, passing our modulation
                out, _, _ = self.step(u, conv_state, ssm_state, dt_modulation_step=dt_modulation)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # Split zxbcdt to get individual components
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        
        # Determine the split points for zxbcdt. dt_raw_from_proj will be the last part.
        # It has shape (batch, seqlen, nheads) or (batch*seqlen, nheads) if flattened.
        split_sizes = [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads]
        
        # --- Path 1: use_mem_eff_path (fused kernel) ---
        if self.use_mem_eff_path and inference_params is None:
            # We need to modify the `zxbcdt` tensor *before* it's passed to the kernel
            # to inject our `dt_modulation`.
            # Locate the `dt` part within `zxbcdt` and add `dt_modulation` to it.

            # Create a copy to modify if zxbcdt is not contiguous or to be safe
            
            

            # The dt part is at the end of the zxbcdt tensor.
            # Calculate the starting index of the dt part.
            dt_start_idx = sum(split_sizes[:-1])

            
            # Add dt_modulation to the dt slice of zxbcdt_modified
            # dt_modulation is (batch*seqlen, nheads), zxbcdt_modified[:, dt_start_idx:] is also (batch*seqlen, nheads)
            #print(f"Shape of self.dt_bias BEFORE KERNEL CALL: {self.dt_bias.shape}") # THIS SHOULD BE (4,)
            #print(f"Shape of dt_modulation BEFORE KERNEL CALL: {dt_modulation.shape}") # THIS SHOULD BE (batch*seqlen, nheads)
        
            #print(f"dt_start_idx: {dt_start_idx}, split_sizes: {split_sizes}")

            dt_modulation_reshaped = rearrange(dt_modulation, "(b l) h -> b l h", b=batch, l=seqlen)
            #print(f"Shape of dt_modulation_reshaped: {dt_modulation_reshaped.shape}") # THIS SHOULD BE (batch, seqlen, nheads)
            
            # Create a properly aligned tensor
            zxbcdt_modified = torch.empty_like(zxbcdt)
            zxbcdt_modified.copy_(zxbcdt)

            # Modify the dt part
            zxbcdt_modified[:, :, dt_start_idx:] += dt_modulation_reshaped

            # Ensure 8-byte alignment by padding the last dim to be divisible by 8
            hidden_dim = zxbcdt_modified.shape[-1]
            pad_size = (8 - (hidden_dim % 8)) % 8
            if pad_size > 0:
                zxbcdt_modified = F.pad(zxbcdt_modified, (0, pad_size))  # Pad last dim only

            # Ensure the tensor is contiguous
            zxbcdt_modified = zxbcdt_modified

            # For debugging - check strides before passing to kernel
            #print(f"Strides after making contiguous: {zxbcdt_modified.stride()}")
            #zxbcdt_modified = zxbcdt_modified
            # Now we can pass the modified zxbcdt to the kernel
            
            out = mamba_split_conv1d_scan_combined(
                zxbcdt_modified, # Use the modified zxbcdt
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias, # Pass the original self.dt_bias (nheads,)
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        
        # --- Path 2: Unfused path or inference (not use_mem_eff_path) ---
        else:
            # Inside DynamicMambaSSM.forward, else branch (Path 2)
            z0, x0, z, xBC, dt_raw_from_proj = torch.split(zxbcdt, split_sizes, dim=-1)
            #print(f"Shape of xBC after split: {xBC.shape}")
            #print(f"Strides of xBC after split (before contiguous): {xBC.stride()}")
            xBC = xBC
            #print(f"Strides of xBC AFTER contiguous(): {xBC.stride()}")

            xBC_transposed = xBC.transpose(1, 2)
            #print(f"Shape of xBC_transposed: {xBC_transposed.shape}")
            #print(f"Strides of xBC_transposed: {xBC_transposed.stride()}")
            # Add dt_modulation to dt_raw_from_proj
            # dt_raw_from_proj is (batch, seqlen, nheads) or (batch*seqlen, nheads)
            # dt_modulation is (batch*seqlen, nheads)
            # Need to align shapes. If dt_raw_from_proj is (B,L,H), dt_modulation is (BL,H).
            # We must rearrange dt_raw_from_proj to (BL,H) or dt_modulation to (B,L,H)
            '''if dt_raw_from_proj.dim() == 3 and dt_modulation.dim() == 2:
                dt_raw_from_proj_flat = rearrange(dt_raw_from_proj, "b l h -> (b l) h")
                dt_modified = dt_raw_from_proj_flat + dt_modulation_reshaped # (batch*seqlen, nheads)
                dt_modified = rearrange(dt_modified, "(b l) h -> b l h", b=batch, l=seqlen) # Back to (B,L,H)
            else: # Already flat or already 3D for both (should be latter if seqlen_og is None)
                dt_modified = dt_raw_from_proj + dt_modulation_reshaped'''
            dt_modified = dt_raw_from_proj + dt_modulation_reshaped


            if conv_state is not None:
                if cu_seqlens is None:
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )
            else:
                xBC_transposed_contiguous = xBC.transpose(1, 2) # <--- ADD  HERE
                xBC = causal_conv1d_fn(
                    xBC_transposed_contiguous,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt_modified, # Use the modified dt values
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias = self.dt_bias, # Keep passing the original self.dt_bias (nheads,)
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor,
             dt_modulation_step: Optional[torch.Tensor] = None): # Renamed arg for clarity
        """
        Step function for inference (single token decoding).

        Args:
            hidden_states: (batch, 1, hidden_dim) - Input features for the current token.
            conv_state: (batch, d_conv_channels, d_conv)
            ssm_state: (batch, nheads, headdim, d_state)
            dt_modulation_step: (batch, nheads) - Time difference modulation for this single step.
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B, d_in_proj)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt_raw_from_proj = torch.split( # dt_raw_from_proj (B, nheads)
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # --- Apply dt_modulation_step to dt_raw_from_proj ---
        dt_modified = dt_raw_from_proj
        if dt_modulation_step is not None:
            # dt_raw_from_proj is (B, nheads), dt_modulation_step is (B, nheads)
            dt_modified = dt_raw_from_proj + dt_modulation_step


        # Conv step (unchanged)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = xBC
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step (modified to use dt_modified)
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            # Use dt_modified here for the dt calculation, then add original self.dt_bias
            dt_final = F.softplus(dt_modified + self.dt_bias.to(dtype=dt_modified.dtype))  # (batch, nheads)
            dA = torch.exp(dt_final * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt_final, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            # The dt_final for kernel is (batch, nheads, headdim)
            # Calculate dt_final by adding original dt_bias to dt_modified (per-timestep) and softplussing.
            dt_for_kernel = F.softplus(dt_modified + self.dt_bias.to(dtype=dt_modified.dtype)) # (B, nheads)
            dt_for_kernel = repeat(dt_for_kernel, "b h -> b h p", p=self.headdim) # (batch, nheads, headdim)

            # dt_bias argument for the kernel expects (nheads, headdim) (original bias, not dynamic)
            dt_bias_for_kernel = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            kernel_dt_bias_for_kernel = repeat(self.dt_bias.view(self.nheads), "h -> h p", p=self.headdim) # (nheads, headdim)


            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt_for_kernel, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=kernel_dt_bias_for_kernel, # Pass the explicitly shaped dt_bias to the kernel
                dt_softplus=False
            )

            '''D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt_for_kernel, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias_for_kernel, # Pass the original dt_bias to the kernel
                dt_softplus=False # dt_for_kernel is already softplussed
            )'''
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state