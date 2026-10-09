"""Official workload RNG policy, imported without loading Torch.

Adapted from Ascend/AscendC-KernelGen-data, Apache-2.0.
Upstream commit: 0607a29d231cde384c9dfdb8a39f54f91837c327
Source: npu-kernelbench/src/npu_kernelbench/evaluator/generator.py
The random tensor body follows that revision with lazy Torch loading and optional dtype lookup.
"""

import random

OFFICIAL_GENERATOR_COMMIT = "0607a29d231cde384c9dfdb8a39f54f91837c327"


def _rand_tensor(shape, dtype, device, range_spec=None, distribution=None, **dist_kwargs):
    """Generate a random tensor with dtype-aware distribution.

    The default distribution for floating-point types is ``mixed``
    (50% uniform[-5,5] + 50% normal with mu~U[-5,5], sigma~U[0.1,2]).
    Override per-tensor by setting ``distribution`` in the workload JSON.

    Args:
        shape: Tensor dimensions.
        dtype: Target torch dtype.
        device: Target device.
        range_spec: Optional [low, high] for integer types (inclusive).
        distribution: ``"uniform"``, ``"randn"``, ``"mixed"``,
            ``"normal"``, ``"zeros"``, ``"ones"``, ``"uniform_signed"``,
            or ``None`` (defaults to ``"mixed"``).
            * "uniform" — uniform in [low, high] (``dist_kwargs``) or [-5,5].
            * "randn"   — standard normal ``N(0,1)`` (floats only).
            * "normal"  — ``N(mu, sigma)`` with random mu∈[-5,5],
               sigma∈[0.1,2] (floats only).
            * "mixed"   — 50% uniform[-5,5], 50% normal with random
               mu∈[-5,5], sigma∈[0.1,2] (floats only).
            * "zeros"   — all zeros tensor.
            * "ones"    — all ones tensor.
            * "uniform_signed" — uniform(low,high) * random sign (±1).
            * None      — same as ``"mixed"``.
        **dist_kwargs: Extra keyword args for distribution tuning, e.g.
            ``low``, ``high``, ``mu_low``, ``mu_high``, ``sigma_low``,
            ``sigma_high``.

    Returns:
        Random tensor of requested shape and dtype.

    Raises:
        ValueError: If dtype is not supported for random generation.
    """
    import torch

    fill_value = dist_kwargs.get("fill")
    if fill_value is not None:
        fill_map = {
            "inf": float("inf"),
            "-inf": float("-inf"),
            "nan": float("nan"),
            "zero": 0.0,
        }
        if fill_value not in fill_map:
            raise ValueError(
                f"Unsupported fill value '{fill_value}'. Supported: {list(fill_map.keys())}"
            )
        return torch.full(shape, fill_map[fill_value], dtype=dtype, device=device)

    is_float = dtype in (torch.float32, torch.float16, torch.bfloat16)

    if is_float:
        # Default: mixed (50% uniform[-5,5] + 50% normal)
        if distribution is None:
            distribution = "mixed"

        if distribution == "uniform":
            low = float(dist_kwargs.get("low", -5.0))
            high = float(dist_kwargs.get("high", 5.0))
            return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
        if distribution == "zeros":
            return torch.zeros(shape, dtype=dtype, device=device)
        if distribution == "ones":
            return torch.ones(shape, dtype=dtype, device=device)
        if distribution == "uniform_signed":
            low = float(dist_kwargs.get("low", 0.3))
            high = float(dist_kwargs.get("high", 3.0))
            g = torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
            sign = torch.randint(0, 2, shape, device=device).to(dtype) * 2 - 1
            return g * sign

        if distribution in ("normal", "mixed"):

            def _rand_norm():
                mu_low = float(dist_kwargs.get("mu_low", -5.0))
                mu_high = float(dist_kwargs.get("mu_high", 5.0))
                sigma_low = float(dist_kwargs.get("sigma_low", 0.1))
                sigma_high = float(dist_kwargs.get("sigma_high", 2.0))
                mu = float(torch.empty(1, device=device).uniform_(mu_low, mu_high).item())
                sigma = float(torch.empty(1, device=device).uniform_(sigma_low, sigma_high).item())
                # NPU does not support torch.normal for bfloat16, so generate in
                # float32 and cast down to the target dtype.
                if dtype is torch.bfloat16:
                    return torch.normal(mu, sigma, shape, dtype=torch.float32, device=device).to(
                        dtype
                    )
                return torch.normal(mu, sigma, shape, dtype=dtype, device=device)

            if distribution == "mixed":
                # 50% uniform / 50% normal, decided per tensor (statistical 50/50).
                if random.random() < 0.5:
                    return _rand_norm()
                low = float(dist_kwargs.get("low", -5.0))
                high = float(dist_kwargs.get("high", 5.0))
                return torch.empty(shape, dtype=dtype, device=device).uniform_(low, high)
            return _rand_norm()

        # randn 和任何未识别的 distribution 默认走标准正态
        return torch.randn(shape, dtype=dtype, device=device)

    # Low-precision floats (FP8, FP4) — generate in f32 then clamp and cast
    if dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float4_e2m1fn_x2", None),
    ):
        t = torch.randn(shape, dtype=torch.float32, device=device).clamp_(-2.0, 2.0)
        return t.to(dtype)

    # Boolean
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)

    # Integers — range_spec is [low, high) exclusive (matches torch.randint convention)
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        if range_spec is not None:
            low, high = range_spec
        else:
            ranges: dict[torch.dtype, tuple[int, int]] = {
                torch.int8: (-128, 128),
                torch.int16: (-1024, 1024),
                torch.int32: (-1024, 1024),
                torch.int64: (-1024, 1024),
            }
            low, high = ranges[dtype]
        return torch.randint(low, high, shape, device=device, dtype=dtype)

    raise ValueError(f"Unsupported random dtype: {dtype}")


def _seed_rng(torch, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.manual_seed_all(seed)


def generate_inputs(
    case, seed=200, device="npu", custom_inputs_code=None, custom_inputs_entrypoint=None
):
    """Materialize one normalized workload inside the assigned device worker.

    ``device`` is an internal test seam; the public adapter always uses NPU.
    Definition order is carried separately so canonical JSON cannot reorder it.
    """
    import torch

    callback = None
    if any(spec["type"] == "custom" for spec in case["inputs"].values()):
        if custom_inputs_code is None:
            raise ValueError("custom inputs require supplied callback source")
        _seed_rng(torch, seed)
        callback_namespace = {"torch": torch, "__name__": "_operator_custom_inputs"}
        exec(custom_inputs_code, callback_namespace)
        callback_name = (custom_inputs_entrypoint or "get_inputs").split("::")[-1]
        callback = callback_namespace.get(callback_name)
        if not callable(callback):
            raise ValueError("custom input source must define its declared input callback")
    _seed_rng(torch, seed)
    family = "normal" if random.random() < 0.5 else "uniform"
    dev = torch.device(device)
    custom = callback(dict(case["axes"]), dev) if callback is not None else None
    if callback is not None and not isinstance(custom, dict):
        raise ValueError("custom input callback must return a dictionary")
    result = []
    for name in case["input_order"]:
        spec = case["inputs"][name]
        kind = spec["type"]
        if kind == "scalar":
            result.append(spec["value"])
        elif kind == "custom":
            if custom is None or name not in custom:
                raise RuntimeError(
                    f"CustomInput for '{name}' requires a matching get_inputs result"
                )
            value = custom[name]
            result.append(
                value.to(device=dev, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
        elif kind == "tensor_list":
            dtype = getattr(torch, spec["dtype"])
            result.append(
                [
                    _rand_tensor(list(shape), dtype, dev, distribution=family)
                    for shape in spec["shapes"]
                ]
            )
        elif kind == "random":
            dtype = getattr(torch, spec["dtype"])
            distribution = spec.get("distribution")
            if distribution is None and dtype in (torch.float32, torch.float16, torch.bfloat16):
                distribution = family
            kwargs = {
                key: float(spec[key])
                for key in ("low", "high", "mu_low", "mu_high", "sigma_low", "sigma_high")
                if spec.get(key) is not None
            }
            if spec.get("fill") is not None:
                kwargs["fill"] = spec["fill"]
            shape = spec["shape"]
            value = _rand_tensor(
                () if shape is None else list(shape),
                dtype,
                dev,
                spec.get("range"),
                distribution,
                **kwargs,
            )
            result.append(value.item() if shape is None else value)
        else:
            raise ValueError(f"Unsupported normalized input type: {kind}")
    return result
