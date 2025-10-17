# file: models/time_encoders/controllable_mamba2.py (Corrected with FiLM-affine transformation)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Import from the installed mamba_ssm library
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_update

class ControllableMamba2(Mamba2):
    """
    An extension of the Mamba2 model that allows its internal `dt` parameter
    to be externally modulated by a FiLM-style affine transformation.
    """

    def forward(self, u, temporal_modulators, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        Overrides the standard forward pass to incorporate the temporal modulators.

        Args:
            u (torch.Tensor): The primary input sequence (e.g., absolute time embeddings).
            temporal_modulators (tuple): A tuple containing (gamma, beta) tensors for FiLM.
                                         Both have shape: (B, S, nheads).
        """
        if inference_params is not None and inference_params.seqlen_offset > 0:
            return self.step_with_gate(u, temporal_modulators, inference_params)

        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        zxbcdt = self.in_proj(u)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        split_dims = [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads]
        z0, x0, z, xBC, dt_content = torch.split(zxbcdt, split_dims, dim=-1)
        
        # --- START OF FiLM MODIFICATION ---
        # Unpack the gamma and beta modulators
        gamma, beta = temporal_modulators
        #print("shape of gamma, beta, dt_content:", gamma.shape, beta.shape, dt_content.shape)
        # MEMORY FIX: Ensure tensors are contiguous and properly shaped
        if gamma.shape != dt_content.shape:
            # Handle dimension mismatch gracefully
            if gamma.dim() == 3 and dt_content.dim() == 3:
                if gamma.shape[-1] != dt_content.shape[-1]:
                    # Expand or reduce gamma/beta to match dt_content
                    target_dim = dt_content.shape[-1]
                    if gamma.shape[-1] < target_dim:
                        gamma = gamma.repeat(1, 1, target_dim // gamma.shape[-1])
                        beta = beta.repeat(1, 1, target_dim // beta.shape[-1])
                    else:
                        gamma = gamma[..., :target_dim]
                        beta = beta[..., :target_dim]
        
        # Apply the affine transformation: y = gamma * x + beta
        dt_fused = gamma * dt_content + beta
        # --- END OF FiLM MODIFICATION ---
        
        
        #zxbcdt_modified = torch.cat([z0, x0, z, xBC, dt_fused], dim=-1).contiguous()
        zxbcdt_modified = torch.cat([z0, x0, z, xBC, dt_fused], dim=-1)

        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt_modified,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, self.dt_bias, A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size, seq_idx=seq_idx, activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups, norm_before_gate=self.norm_before_gate, **dt_limit_kwargs
            )
        else:
            raise NotImplementedError("Non-memory-efficient path is not supported.")

        if seqlen_og is not None:
            out = rearrange(out, "b l d -> (b l) d")
            
        if self.process_group is not None:
            from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)
            
        return out

    def step_with_gate(self, hidden_states, temporal_modulators, inference_params):
        """ Wrapper for the step method. """
        conv_state, ssm_state = self._get_states_from_cache(inference_params, batch_size=hidden_states.shape[0])
        return self.step(hidden_states, conv_state, ssm_state, temporal_modulators)

    def step(self, hidden_states, conv_state, ssm_state, temporal_modulators):
        """ Overrides the step function for autoregressive inference. """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        #zxbcdt = self.in_proj(hidden_states.squeeze(1)).contiguous()
        zxbcdt = self.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt_content = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # --- START OF FiLM MODIFICATION (for step) ---
        gamma, beta = temporal_modulators
        # Squeeze the sequence length dim (which is 1) from the modulators
        print("shape of gamma, beta, dt_content:", gamma.shape, beta.shape, dt_content.shape)
        dt_fused = gamma.squeeze(1) * dt_content + beta.squeeze(1)
        # --- END OF FiLM MODIFICATION (for step) ---

        '''xBC = causal_conv1d_update(
            xBC, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias, self.activation,
        ).contiguous()'''
        xBC = causal_conv1d_update(
            xBC, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"),
            self.conv1d.bias, self.activation,
        )
        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())
        A_reshaped = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt_reshaped = repeat(dt_fused, "b h -> b h p", p=self.headdim)
        dt_bias_reshaped = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        D_reshaped = repeat(self.D, "h -> h p", p=self.headdim)
        B_reshaped = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C_reshaped = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        z_reshaped = None if self.rmsnorm else rearrange(z, "b (h p) -> b h p", p=self.headdim)
        y = selective_state_update(
            ssm_state, x_reshaped, dt_reshaped, A_reshaped, B_reshaped, C_reshaped, D_reshaped,
            z=z_reshaped, dt_bias=dt_bias_reshaped, dt_softplus=True
        )
        y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state