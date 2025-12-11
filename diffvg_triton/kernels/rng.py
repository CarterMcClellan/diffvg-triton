"""
Random number generation using PCG32.

Provides deterministic pseudo-random numbers for sample jittering
and other stochastic rendering operations.

Ported from diffvg/pcg.h
"""

import torch
import triton
import triton.language as tl


# PCG32 constants
PCG_DEFAULT_STATE = 0x853c49e6748fea9b
PCG_DEFAULT_STREAM = 0xda3e39cb94b95bdb
PCG_MULTIPLIER = 6364136223846793005


@triton.jit
def pcg32_init(seed: tl.uint64, stream: tl.uint64):
    """
    Initialize PCG32 state from seed and stream.

    Returns: (state, inc) tuple
    """
    # inc must be odd
    inc = (stream << 1) | 1

    # Initialize state
    state = tl.zeros_like(seed, dtype=tl.uint64)

    # First step
    state = state * PCG_MULTIPLIER + inc
    state = state + seed

    # Second step
    state = state * PCG_MULTIPLIER + inc

    return state, inc


@triton.jit
def pcg32_next(state: tl.uint64, inc: tl.uint64):
    """
    Generate next random uint32 and update state.

    Returns: (random_value, new_state)
    """
    # Update state
    old_state = state
    new_state = old_state * PCG_MULTIPLIER + inc

    # Generate output (XSH-RR output function)
    # xorshifted = ((old_state >> 18) ^ old_state) >> 27
    # rot = old_state >> 59
    # output = (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    xorshifted = ((old_state >> 18) ^ old_state) >> 27
    rot = (old_state >> 59).to(tl.uint32)

    # Right rotation
    xorshifted_32 = xorshifted.to(tl.uint32)
    output = (xorshifted_32 >> rot) | (xorshifted_32 << ((32 - rot) & 31))

    return output, new_state


@triton.jit
def pcg32_uniform(state: tl.uint64, inc: tl.uint64):
    """
    Generate uniform random float in [0, 1) and update state.

    Returns: (random_float, new_state)
    """
    rand_val, new_state = pcg32_next(state, inc)

    # Convert to float in [0, 1)
    # 2^32 - 1 = 4294967295
    uniform = rand_val.to(tl.float32) / 4294967296.0

    return uniform, new_state


@triton.jit
def pcg32_init_kernel(
    seed_ptr,        # [N] per-element seeds
    stream_ptr,      # [N] per-element streams (or single value)
    state_ptr,       # [N] output states
    inc_ptr,         # [N] output increments
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    use_single_stream: tl.constexpr,
):
    """
    Initialize PCG32 states for N elements.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load seed
    seed = tl.load(seed_ptr + offs, mask=mask, other=0).to(tl.uint64)

    # Load stream
    if use_single_stream:
        stream = tl.load(stream_ptr).to(tl.uint64)
    else:
        stream = tl.load(stream_ptr + offs, mask=mask, other=0).to(tl.uint64)

    # Initialize
    state, inc = pcg32_init(seed, stream)

    # Store
    tl.store(state_ptr + offs, state, mask=mask)
    tl.store(inc_ptr + offs, inc, mask=mask)


@triton.jit
def pcg32_generate_kernel(
    state_ptr,       # [N] states (in/out)
    inc_ptr,         # [N] increments
    output_ptr,      # [N] output random floats
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Generate N uniform random floats in [0, 1).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load state
    state = tl.load(state_ptr + offs, mask=mask, other=0).to(tl.uint64)
    inc = tl.load(inc_ptr + offs, mask=mask, other=1).to(tl.uint64)

    # Generate
    uniform, new_state = pcg32_uniform(state, inc)

    # Store
    tl.store(state_ptr + offs, new_state, mask=mask)
    tl.store(output_ptr + offs, uniform, mask=mask)


# Python reference implementation
class PCG32:
    """Python reference implementation of PCG32 random number generator."""

    def __init__(self, seed=42, stream=1):
        """Initialize PCG32 with seed and stream."""
        self.state = 0
        self.inc = (stream << 1) | 1

        # Warm up
        self.state = self.state * PCG_MULTIPLIER + self.inc
        self.state = (self.state + seed) & 0xFFFFFFFFFFFFFFFF
        self.state = (self.state * PCG_MULTIPLIER + self.inc) & 0xFFFFFFFFFFFFFFFF

    def next_uint32(self):
        """Generate next random uint32."""
        old_state = self.state

        # Update state
        self.state = (old_state * PCG_MULTIPLIER + self.inc) & 0xFFFFFFFFFFFFFFFF

        # Generate output
        xorshifted = (((old_state >> 18) ^ old_state) >> 27) & 0xFFFFFFFF
        rot = (old_state >> 59) & 0x1F

        output = ((xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))) & 0xFFFFFFFF
        return output

    def uniform(self):
        """Generate uniform random float in [0, 1)."""
        return self.next_uint32() / 4294967296.0

    def uniform_range(self, low, high):
        """Generate uniform random float in [low, high)."""
        return low + self.uniform() * (high - low)


# PyTorch wrapper
class TorchPCG32:
    """
    PyTorch wrapper for PCG32 random number generation.

    Supports batched generation on GPU via Triton kernels.
    """

    def __init__(self, num_elements: int, seed: int = 42, device=None):
        """
        Initialize PCG32 states for num_elements.

        Args:
            num_elements: Number of independent RNG streams
            seed: Base seed (will be combined with element index)
            device: PyTorch device
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        self.num_elements = num_elements

        # Create unique seeds per element
        seeds = torch.arange(num_elements, dtype=torch.int64, device=device) + seed
        streams = torch.arange(num_elements, dtype=torch.int64, device=device)

        # Initialize states
        self.state = torch.zeros(num_elements, dtype=torch.int64, device=device)
        self.inc = torch.zeros(num_elements, dtype=torch.int64, device=device)

        # Run initialization on CPU for now (Triton uint64 support varies)
        for i in range(num_elements):
            rng = PCG32(seed=seeds[i].item(), stream=streams[i].item())
            self.state[i] = rng.state
            self.inc[i] = rng.inc

    def generate(self) -> torch.Tensor:
        """
        Generate num_elements uniform random floats in [0, 1).

        Returns:
            [num_elements] tensor of random floats
        """
        output = torch.zeros(self.num_elements, dtype=torch.float32, device=self.device)

        # CPU fallback for now
        for i in range(self.num_elements):
            state = self.state[i].item()
            inc = self.inc[i].item()

            # Generate
            old_state = state
            new_state = (old_state * PCG_MULTIPLIER + inc) & 0xFFFFFFFFFFFFFFFF

            xorshifted = (((old_state >> 18) ^ old_state) >> 27) & 0xFFFFFFFF
            rot = (old_state >> 59) & 0x1F
            rand_val = ((xorshifted >> rot) | (xorshifted << ((32 - rot) & 31))) & 0xFFFFFFFF

            output[i] = rand_val / 4294967296.0
            self.state[i] = new_state

        return output

    def generate_2d(self) -> torch.Tensor:
        """
        Generate num_elements pairs of random floats.

        Returns:
            [num_elements, 2] tensor of random float pairs
        """
        x = self.generate()
        y = self.generate()
        return torch.stack([x, y], dim=1)


def generate_sample_offsets(
    num_pixels: int,
    num_samples_x: int,
    num_samples_y: int,
    seed: int = 42,
    device=None,
) -> torch.Tensor:
    """
    Generate sub-pixel sample offsets for anti-aliasing.

    Uses stratified sampling with jitter within each stratum.

    Args:
        num_pixels: Number of pixels
        num_samples_x: Number of samples in x per pixel
        num_samples_y: Number of samples in y per pixel
        seed: Random seed
        device: PyTorch device

    Returns:
        [num_pixels, num_samples_x * num_samples_y, 2] tensor of offsets in [0, 1)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_samples = num_samples_x * num_samples_y
    total_samples = num_pixels * num_samples

    # Initialize RNG
    rng = TorchPCG32(total_samples * 2, seed=seed, device=device)

    # Generate jitter
    jitter_x = rng.generate().view(num_pixels, num_samples)
    jitter_y = rng.generate().view(num_pixels, num_samples)

    # Create stratified grid
    offsets = torch.zeros(num_pixels, num_samples, 2, device=device)

    for sy in range(num_samples_y):
        for sx in range(num_samples_x):
            s_idx = sy * num_samples_x + sx

            # Stratum center + jitter
            base_x = (sx + 0.5) / num_samples_x
            base_y = (sy + 0.5) / num_samples_y

            # Jitter within stratum (scale down to stratum size)
            jx = (jitter_x[:, s_idx] - 0.5) / num_samples_x
            jy = (jitter_y[:, s_idx] - 0.5) / num_samples_y

            offsets[:, s_idx, 0] = base_x + jx
            offsets[:, s_idx, 1] = base_y + jy

    return offsets
