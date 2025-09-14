# file: models/time_encoders/controllable_mamba2.py (Corrected Imports)

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# --- START OF CORRECTION ---
# We now import directly from the globally installed libraries,
# NOT from the local 'imported_lib' folder.
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from causal_conv1d import causal_conv1d_update
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# --- END OF CORRECTION ---


class ControllableMamba2(Mamba2):
    """
    An extension of the Mamba2 model that allows its internal `dt` parameter
    (the content-aware selective gate) to be externally modulated.

    This class overrides the `forward` and `step` methods to accept an additional
    `temporal_gate` tensor. This gate is multiplicatively fused with the model's
    internally computed `dt_content`, enabling dynamic, time-aware control over
    the SSM's selection mechanism.

    This is designed to be a drop-in replacement for a standard Mamba2 block
    within a larger framework like KAN-MAMMOTE.
    """

    def forward(self, u, temporal_gate, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        Overrides the standard forward pass to incorporate the temporal gate.

        Args:
            u (torch.Tensor): The primary input sequence (e.g., absolute time embeddings).
            temporal_gate (torch.Tensor): The learned gate from the fusion MLP.
                                          Shape: (B, S, nheads).
            All other args are passed from the original Mamba2 forward signature.
        """
        # We handle inference redirection here. If `step` is being used, `forward` just calls it.
        if inference_params is not None and inference_params.seqlen_offset > 0:
            return self.step_with_gate(u, temporal_gate, inference_params)

        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        # --- Start of Modified Logic ---

        # 1. Standard Mamba2 projection to get all internal parameters from the input `u`
        zxbcdt = self.in_proj(u)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)

        # 2. Deconstruct the projected tensor to isolate the content-based dt
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm
                 - 2 * self.ngroups * self.d_state
                 - self.nheads) // 2

        z0, x0, z, xBC, dt_content = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # 3. Apply the external temporal gate via multiplicative fusion
        dt_fused = dt_content * temporal_gate

        # 4. Reconstruct the tensor with our new fused gate
        zxbcdt_modified = torch.cat([z0, x0, z, xBC, dt_fused], dim=-1)

        # --- End of Modified Logic ---


        # The rest of the forward pass uses the original Mamba2's optimized path,
        # but with our `zxbcdt_modified` tensor.
        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # We assume the memory-efficient path is used for training
        if self.use_mem_eff_path:
            out = mamba_split_conv1d_scan_combined(
                zxbcdt_modified, # <-- The only change is using the modified tensor here
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
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
        else:
            # Fallback to the less-efficient path if needed (logic copied from original)
            # We must use zxbcdt_modified here as well
            z0_mod, x0_mod, z_mod, xBC_mod, dt_fused_mod = torch.split(
                zxbcdt_modified,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            # The rest of this non-mem-eff path would need careful adaptation,
            # but since use_mem_eff_path=True is the default, we focus on that.
            raise NotImplementedError("Non-memory-efficient path for ControllableMamba2 not fully implemented.")

        if seqlen_og is not None:
            out = rearrange(out, "b l d -> (b l) d")
            
        # Handle sequence parallelism if used
        if self.process_group is not None:
            reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
            out = reduce_fn(out, self.process_group)
            
        return out


    def step_with_gate(self, hidden_states, temporal_gate, inference_params):
        """
        A wrapper for the original `step` method to handle inference-time gating.
        This method retrieves the states and then calls the main step logic.
        """
        conv_state, ssm_state = self._get_states_from_cache(inference_params, batch_size=hidden_states.shape[0])
        return self.step(hidden_states, conv_state, ssm_state, temporal_gate)


    def step(self, hidden_states, conv_state, ssm_state, temporal_gate):
        """
        Overrides the standard step function for autoregressive inference.
        This method is a near-direct copy of the original, with `dt` being modulated.

        Args:
            hidden_states (torch.Tensor): The input for the current step. Shape (B, 1, D).
            conv_state, ssm_state: The recurrent states from the previous step.
            temporal_gate (torch.Tensor): The learned gate for THIS step. Shape (B, 1, nheads).
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time"

        # --- Start of Modified Logic ---

        # 1. Project input
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B, d_in_proj)

        # 2. Deconstruct
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt_content = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # 3. Apply the external temporal gate (the only modified part)
        # Squeeze the sequence length dim (which is 1)
        dt_fused = dt_content * temporal_gate.squeeze(1)

        # --- End of Modified Logic ---

        # The rest of this function is copied VERBATIM from `mamba2.py`,
        # ensuring that `dt_fused` is used in place of the original `dt`.

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias, self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())

        # SSM step
        # This is where dt_fused is used
        if selective_state_update is None:
            # Fallback path (from original code)
            assert self.ngroups == 1
            dt = F.softplus(dt_fused + self.dt_bias.to(dtype=dt_fused.dtype)) # Use dt_fused
            dA = torch.exp(dt * A)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)
        else:
            # Main path using Triton kernel (from original code)
            A_reshaped = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt_reshaped = repeat(dt_fused, "b h -> b h p", p=self.headdim) # Use dt_fused
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

        # Final processing (from original code)
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state