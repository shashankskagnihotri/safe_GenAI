"""Fail-closed, offline LPIPS contract for segmented-video evaluation.

The benchmark must not let ``lpips`` or ``torchvision`` resolve model weights
from a network cache.  This module therefore authenticates the installed
runtime and the preregistered read-only artifacts before constructing LPIPS
v0.1/Alex.  Both the ImageNet trunk and learned LPIPS calibration are loaded
with ``weights_only=True`` and complete state coverage.

Importing this module is intentionally cheap: PyTorch, torchvision, SciPy, and
LPIPS are imported only by :meth:`FixedLpipsMetric.from_preregistered_artifacts`.
The dispatcher can consequently run the artifact/dependency preflight without
allocating a model or initializing CUDA.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from importlib import import_module, metadata
import io
import json
import math
from pathlib import Path
import stat
import struct
import sys
from types import MappingProxyType
from typing import Any
from zipfile import BadZipFile, ZipFile


SCHEMA_VERSION = 2
CONTRACT_STATUS = "passed"
DEPENDENCY_ROOT_RELATIVE = Path("debugging/dependencies/finer_detailing_20260720")

# This is a prospective schema-2 numeric admission, not a tolerance check.  A
# generation attempt is admissible only when its target Python environment
# authenticates the exact fixed-LPIPS CPU implementation/runtime bytes below
# and evaluates this sentinel bit-for-bit before any generation model is
# allocated.  ``float.hex`` describes the Python value obtained from the
# published float32 tensor; the byte digest is over exactly one little-endian
# IEEE-754 binary32 value.
TEMPORAL_METRIC_RUNTIME_PREFLIGHT_SCHEMA_VERSION = 2
TEMPORAL_METRIC_SENTINEL_VALUE = 0.11892151087522507
TEMPORAL_METRIC_SENTINEL_HEX = "0x1.e71a3e0000000p-4"
TEMPORAL_METRIC_SENTINEL_FLOAT32_LE_SHA256 = (
    "fb8e52f059ea96bb05b1706d7b333ef26fb1e358a540c499763f748457f18362"
)

_RUNTIME_ARTIFACT_SPEC_KEYS = {
    "artifact_id",
    "path_kind",
    "path",
    "role",
    "size_bytes",
    "sha256",
    "must_be_mapped",
}
_MKL_CPU_DISPATCH_ARTIFACT_IDS = frozenset(
    {
        "mkl_avx512",
        "mkl_vml_avx512",
        "mkl_def",
        "mkl_vml_def",
    }
)
_OPTIONAL_MKL_RUNTIME_ARTIFACT_IDS = frozenset(
    {
        *_MKL_CPU_DISPATCH_ARTIFACT_IDS,
        "mkl_gf_lp64",
        "mkl_intel_thread",
        "mkl_rt",
    }
)
_MKL_INVARIANT_MUST_MAP_ARTIFACT_IDS = frozenset(
    {
        "mkl_core",
        "mkl_gnu_thread",
        "mkl_intel_lp64",
    }
)
# MKL chooses the AVX-512 pair on the local Intel CPU path and the ``def`` pair
# on the AMD EPYC 9474F H100 host. Diagnostic array 291962 established that
# both authenticated paths produce the same schema-2 sentinel bit-for-bit.
# Treating the four files as independently optional would admit partial or
# mixed arithmetic paths, so selection is an exactly-one complete-bundle rule.
_MKL_CPU_DISPATCH_POLICY = MappingProxyType(
    {
        "policy_id": "mkl_cpu_dispatch_exactly_one_complete_bundle_v1",
        "selection_rule": "exactly_one_complete_bundle",
        "bundles": (
            MappingProxyType(
                {
                    "bundle_id": "avx512",
                    "artifact_ids": ("mkl_avx512", "mkl_vml_avx512"),
                }
            ),
            MappingProxyType(
                {
                    "bundle_id": "generic_def",
                    "artifact_ids": ("mkl_def", "mkl_vml_def"),
                }
            ),
        ),
    }
)


def _runtime_artifact_spec(
    artifact_id: str,
    path: str,
    size_bytes: int,
    sha256: str,
    *,
    role: str,
    path_kind: str = "prefix_relative",
    must_be_mapped: bool = False,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "artifact_id": artifact_id,
            "path_kind": path_kind,
            "path": path,
            "role": role,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "must_be_mapped": must_be_mapped,
        }
    )


_COMMON_FRAMEWORK_RUNTIME_ARTIFACTS = (
    _runtime_artifact_spec(
        "torch_source_init",
        "lib/python3.10/site-packages/torch/__init__.py",
        95_190,
        "e1b5bf4151bd246a21e3ac5318f6edfcf2e6ef0fd4f233bfb77a264698ec1319",
        role="framework_source",
    ),
    _runtime_artifact_spec(
        "torch_extension",
        "lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so",
        33_848,
        "faf6b29b25725553025e08908fc3780e8a6f7a2813cbb8029f6525214d5bb9e8",
        role="framework_extension",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_libc10",
        "lib/python3.10/site-packages/torch/lib/libc10.so",
        1_450_705,
        "6f087734a2d8a0bb3b8da9ab588d52c40a877f4e547255eefd5ca1e47d2864ca",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_libtorch",
        "lib/python3.10/site-packages/torch/lib/libtorch.so",
        166_736,
        "206866941065fd7d6edf4d7b4dc099c8a4b8c187c76b3d0ddbde87e31ceaf4dc",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_libtorch_cpu",
        "lib/python3.10/site-packages/torch/lib/libtorch_cpu.so",
        314_174_561,
        "c7b3eec12a7a7a40b25338c263f17a5e1fef9069f9a02c132cb4e872be288bbb",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_global_deps",
        "lib/python3.10/site-packages/torch/lib/libtorch_global_deps.so",
        15_696,
        "2e1633883d7314c01d7aca877d3690f6ddb661c2eab27669c5473bf7f4399355",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_python_runtime",
        "lib/python3.10/site-packages/torch/lib/libtorch_python.so",
        26_660_624,
        "a187b5617a6d5355d14193cdef7c5c065fdf7d717f26186d77427a315339fa95",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    _runtime_artifact_spec(
        "torch_shared_memory_runtime",
        "lib/python3.10/site-packages/torch/lib/libshm.so",
        44_488,
        "c8bb7d7e565e31dda5c34821b46533f581189fe7c762befcc872579d1f875184",
        role="framework_cpu_runtime",
        must_be_mapped=True,
    ),
    *(
        _runtime_artifact_spec(artifact_id, path, size_bytes, sha256, role="framework_source")
        for artifact_id, path, size_bytes, sha256 in (
            (
                "torch_nn_functional_source",
                "lib/python3.10/site-packages/torch/nn/functional.py",
                235_999,
                "e91cd1c772ef9302727169f89bf8375befd3fc64a8d445df6084958708a95d53",
            ),
            (
                "torch_nn_module_source",
                "lib/python3.10/site-packages/torch/nn/modules/module.py",
                124_492,
                "d125e6e19b4495e65758dc4c4d8fbdc3e30758c27eded646427971f0098eb117",
            ),
            (
                "torch_nn_convolution_source",
                "lib/python3.10/site-packages/torch/nn/modules/conv.py",
                75_723,
                "a9d8f85848988ebed74853a7dc8172310c7002f3bb4948189477e8555b70d418",
            ),
            (
                "torch_nn_linear_source",
                "lib/python3.10/site-packages/torch/nn/modules/linear.py",
                10_722,
                "d7f5cadf773deb5ab4bcc9f57414e8239b2859c70d2aa6f141576de1ec918823",
            ),
            (
                "torch_nn_dropout_source",
                "lib/python3.10/site-packages/torch/nn/modules/dropout.py",
                11_187,
                "1fc9974c3cf2078dd4bc257f73afb6a6714adfec2e9d099d47cf923e766d3490",
            ),
            (
                "torch_nn_pooling_source",
                "lib/python3.10/site-packages/torch/nn/modules/pooling.py",
                58_817,
                "bd14b2627aca9358d85e76353a408d8df0b61abe977bf41a4f0cc6e219f30337",
            ),
            (
                "torch_nn_activation_source",
                "lib/python3.10/site-packages/torch/nn/modules/activation.py",
                57_657,
                "1e748e0df0d70491fe82f0178f32126ee6ee138a53de7fc76463af10c75f5477",
            ),
            (
                "torch_nn_container_source",
                "lib/python3.10/site-packages/torch/nn/modules/container.py",
                35_029,
                "33af8d1fe4fb542cbba5542c5079b1bd8ef8a6493abbef329411f71b2071d24f",
            ),
            (
                "torchvision_source_init",
                "lib/python3.10/site-packages/torchvision/__init__.py",
                3_534,
                "ee2c9f4110cf1203db48c42601607329ac1f19709fa91c152f8d95eb53437a73",
            ),
            (
                "torchvision_alexnet_source",
                "lib/python3.10/site-packages/torchvision/models/alexnet.py",
                4_488,
                "76f0592d51fad133931c234d9168344c52edb89e46861ff15cdef7af7572ab9f",
            ),
            (
                "torchvision_model_utils_source",
                "lib/python3.10/site-packages/torchvision/models/_utils.py",
                10_893,
                "4bcb830fb99a35e7f2f9f116ea6a73f1d154ebc69c2b51f137492dd6aa64903a",
            ),
        )
    ),
    *(
        _runtime_artifact_spec(
            artifact_id,
            path,
            size_bytes,
            sha256,
            role="system_cpu_runtime",
            path_kind="absolute",
            must_be_mapped=False,
        )
        for artifact_id, path, size_bytes, sha256 in (
            (
                "system_dynamic_loader",
                "/usr/lib64/ld-linux-x86-64.so.2",
                938_792,
                "98661496afebb21b2d91a6acf906736ce346fc82c6d20fb3d9d8a6c4869591d2",
            ),
            (
                "system_libc",
                "/usr/lib64/libc.so.6",
                2_549_360,
                "84c10445194fbc691849d57b7e215ae57fc70da5799845155c798c6402414044",
            ),
            (
                "system_libdl",
                "/usr/lib64/libdl.so.2",
                15_464,
                "5036302e04157d0b5dfef8f72cf8616fd719b64b73d28b5e21112c047673138f",
            ),
            (
                "system_libm",
                "/usr/lib64/libm.so.6",
                912_968,
                "bd928aa20b54cc9bb4c209a9d7c33f25e98ec4be64f2386ecbda6a49338d29c6",
            ),
            (
                "system_libmvec",
                "/usr/lib64/libmvec.so.1",
                176_824,
                "35faf231dc6abfb805c530f9c7e596415cfd3830f34b6ae5345e1b0dbf638a9f",
            ),
            (
                "system_libpthread",
                "/usr/lib64/libpthread.so.0",
                15_480,
                "d0ea6a7f4e085ed41320be9193f2de528693b3ea885996b38a15c8e33d759455",
            ),
            (
                "system_librt",
                "/usr/lib64/librt.so.1",
                15_640,
                "5969e001e12ac221896ca5bccaacc1ddf1df8678ad41b41a0e27f997eae4c792",
            ),
            (
                "system_libutil",
                "/usr/lib64/libutil.so.1",
                15_464,
                "ec91e0bce7804df2e925d5541652897372761c3f47a182d43dceb1f63de38355",
            ),
        )
    ),
)


def _environment_runtime_artifacts(
    *,
    python_sha256: str,
    torch_conda_meta_size: int,
    torch_conda_meta_sha256: str,
    torchvision_conda_meta_size: int,
    torchvision_conda_meta_sha256: str,
    torchvision_extension_sha256: str,
    torchvision_image_sha256: str,
    torchvision_video_reader_sha256: str,
    optional_cpu_math_specs: tuple[tuple[str, str, int, str], ...],
    mkl_specs: tuple[tuple[str, int, str], ...],
) -> tuple[Mapping[str, Any], ...]:
    return (
        _runtime_artifact_spec(
            "python_executable",
            "bin/python3.10",
            17_331_920,
            python_sha256,
            role="python_runtime",
        ),
        _runtime_artifact_spec(
            "torch_conda_package_metadata",
            "conda-meta/pytorch-2.5.1-py3.10_cuda12.4_cudnn9.1.0_0.json",
            torch_conda_meta_size,
            torch_conda_meta_sha256,
            role="framework_package_metadata",
        ),
        _runtime_artifact_spec(
            "torchvision_conda_package_metadata",
            "conda-meta/torchvision-0.20.1-py310_cu124.json",
            torchvision_conda_meta_size,
            torchvision_conda_meta_sha256,
            role="framework_package_metadata",
        ),
        *_COMMON_FRAMEWORK_RUNTIME_ARTIFACTS,
        _runtime_artifact_spec(
            "torchvision_extension",
            "lib/python3.10/site-packages/torchvision/_C.so",
            7_808_128,
            torchvision_extension_sha256,
            role="framework_extension",
            must_be_mapped=True,
        ),
        _runtime_artifact_spec(
            "torchvision_image_extension",
            "lib/python3.10/site-packages/torchvision/image.so",
            445_912,
            torchvision_image_sha256,
            role="framework_extension",
            must_be_mapped=True,
        ),
        _runtime_artifact_spec(
            "torchvision_video_reader_extension",
            "lib/python3.10/site-packages/torchvision/video_reader.so",
            674_440,
            torchvision_video_reader_sha256,
            role="framework_extension",
            must_be_mapped=True,
        ),
        *(
            _runtime_artifact_spec(
                artifact_id,
                path,
                size_bytes,
                sha256,
                role=role,
                must_be_mapped=must_be_mapped,
            )
            for artifact_id, path, size_bytes, sha256, role, must_be_mapped in (
                (
                    "compiler_libgcc",
                    "lib/libgcc_s.so.1",
                    902_640,
                    "e1e904051f77f9569c2ea53c83bb4083c26575e0fbd4010e46f1cb8b21037ad1",
                    "compiler_cpu_runtime",
                    True,
                ),
                (
                    "openmp_runtime",
                    "lib/libgomp.so.1.0.0",
                    1_586_592,
                    "23edb748d6c9584e337fc00cb22eba3bf535478204d6e5eddbb637b7f417fd25",
                    "cpu_math_runtime",
                    True,
                ),
                (
                    "compiler_libstdcxx",
                    "lib/libstdc++.so.6.0.34",
                    21_295_144,
                    "9581ad615b7c073423f57b69a3b148a89f8ea76fc909124211f9007909b807a6",
                    "compiler_cpu_runtime",
                    True,
                ),
            )
        ),
        *(
            _runtime_artifact_spec(
                artifact_id,
                path,
                size_bytes,
                sha256,
                role="cpu_math_runtime",
                must_be_mapped=False,
            )
            for artifact_id, path, size_bytes, sha256 in optional_cpu_math_specs
        ),
        *(
            _runtime_artifact_spec(
                artifact_id,
                path,
                size_bytes,
                sha256,
                role="cpu_math_runtime",
                must_be_mapped=artifact_id not in _OPTIONAL_MKL_RUNTIME_ARTIFACT_IDS,
            )
            for artifact_id, path, size_bytes, sha256 in (
                (
                    f"mkl_{Path(path).name.removeprefix('libmkl_').split('.')[0]}",
                    path,
                    size_bytes,
                    sha256,
                )
                for path, size_bytes, sha256 in mkl_specs
            )
        ),
    )


_FRAMEWORK_PACKAGE_IDENTITIES = MappingProxyType(
    {
        "torch": MappingProxyType(
            {
                "conda_name": "pytorch",
                "version": "2.5.1",
                "build": "py3.10_cuda12.4_cudnn9.1.0_0",
                "build_number": 0,
                "subdir": "linux-64",
                "archive_sha256": (
                    "416e11c6e34a22e7f6bbc524acddb71281d400882e8d13e10407e822dd959593"
                ),
                "metadata_artifact_id": "torch_conda_package_metadata",
            }
        ),
        "torchvision": MappingProxyType(
            {
                "conda_name": "torchvision",
                "version": "0.20.1",
                "build": "py310_cu124",
                "build_number": 0,
                "subdir": "linux-64",
                "archive_sha256": (
                    "0e4319d753e0d30bafc0409249ea3db12b7e630283de285126aad1ec0603327d"
                ),
                "metadata_artifact_id": "torchvision_conda_package_metadata",
            }
        ),
    }
)


TEMPORAL_METRIC_RUNTIME_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "safe_genai_conceptsteer": MappingProxyType(
            {
                "schema_version": 2,
                "environment_name": "safe_genai_conceptsteer",
                "environment_prefix": ("/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer"),
                "framework_packages": _FRAMEWORK_PACKAGE_IDENTITIES,
                "cpu_dispatch_policy": _MKL_CPU_DISPATCH_POLICY,
                "artifacts": _environment_runtime_artifacts(
                    python_sha256=(
                        "1690e9aa10589652f1759b52f7db78158be57cc14820789ba4718cb491968f49"
                    ),
                    torch_conda_meta_size=5_763_049,
                    torch_conda_meta_sha256=(
                        "1413398f650ce931eab381bff5238a8c86f03ce03deb08a66c6dc5587689daef"
                    ),
                    torchvision_conda_meta_size=170_371,
                    torchvision_conda_meta_sha256=(
                        "453830c3b3ec12bb1e57c5fdfe1492ba31034dcd7b7e81fdaa443973e57b3064"
                    ),
                    torchvision_extension_sha256=(
                        "2043ebbb95a8a3062c54d057556222690e657fc55137ce66c54b70ceda84a61d"
                    ),
                    torchvision_image_sha256=(
                        "0904d68002185e158eb2600bc4e79afa65a063e3a3be678cfecf5b262c12e546"
                    ),
                    torchvision_video_reader_sha256=(
                        "33ad4d71e996eee9855835a668147ea6cbc2d3ab30ef6152c830767a2bb1850f"
                    ),
                    optional_cpu_math_specs=(
                        (
                            "blas_runtime",
                            "lib/libblas.so.3.9.0",
                            476_536,
                            "3bcff391b3b77991ff436bddd5d2bef9d66a2b38889f2b72a4cbb1c61177dd61",
                        ),
                        (
                            "lapack_runtime",
                            "lib/liblapack.so.3.9.0",
                            7_890_728,
                            "185986f1366b05ae29ebaeb5c61aa2273aa72e7b0ffc5d89b35b8f1d1db9b881",
                        ),
                    ),
                    mkl_specs=(
                        (
                            "lib/libmkl_avx512.so.2",
                            67_317_209,
                            "7c3d5a1bd3cf76695ccc50d4a89c7b9ef7e881fb0cf87fa9c91a6750cea7f973",
                        ),
                        (
                            "lib/libmkl_def.so.2",
                            42_978_729,
                            "0b3cf1a5e4337773a40e30061cf6c34eb75d4a67e57875808b4fc3f4a8161330",
                        ),
                        (
                            "lib/libmkl_core.so.2",
                            75_223_945,
                            "19edadb815b79d8f646438a84b29e5a3e1abafac48ec6344e44e2d422dff94f7",
                        ),
                        (
                            "lib/libmkl_gnu_thread.so.2",
                            31_960_209,
                            "5d4e01395b75733e2233105efa95ca8039c3eae40cf10423e89c9487bef60ed0",
                        ),
                        (
                            "lib/libmkl_intel_lp64.so.2",
                            22_027_945,
                            "f0039b0dfdd5a70840f6287e3d77282c640d4a11a8212d10e5e8837794d1d074",
                        ),
                        (
                            "lib/libmkl_vml_avx512.so.2",
                            14_527_137,
                            "e5e0dff27c80dfe18f2eb400a7bef590f520f6488908dc6a5c8f2b4be51b7162",
                        ),
                        (
                            "lib/libmkl_vml_def.so.2",
                            8_881_249,
                            "b4b8d9fb3392667af4c5cbfce64b1f0c886fd5385999dcbc734ddb63f5a5ac88",
                        ),
                        (
                            "lib/libmkl_intel_thread.so.2",
                            65_963_657,
                            "67a7403a52ed53c852682f81a40c615503a09d1c32206d7265ff256f4a730c8c",
                        ),
                        (
                            "lib/libmkl_gf_lp64.so.2",
                            18_461_113,
                            "af2aaf48e18405dda1f392cbb0aa8cb8fe017e69d679caa9d9047d9aca0d0b3b",
                        ),
                        (
                            "lib/libmkl_rt.so.2",
                            15_822_785,
                            "75173baf355b7493185b2268dc65617e6e061b18fb10cff224171db108ca7600",
                        ),
                    ),
                ),
            }
        ),
        "safe_genai_ltx23": MappingProxyType(
            {
                "schema_version": 2,
                "environment_name": "safe_genai_ltx23",
                "environment_prefix": "/ceph/sagnihot/miniconda3/envs/safe_genai_ltx23",
                "framework_packages": _FRAMEWORK_PACKAGE_IDENTITIES,
                "cpu_dispatch_policy": _MKL_CPU_DISPATCH_POLICY,
                "artifacts": _environment_runtime_artifacts(
                    python_sha256=(
                        "309d8936dd1d4f2258a89f253bdd45da2462b1ca3d249d49168d3a2d4bc5aabb"
                    ),
                    torch_conda_meta_size=5_763_094,
                    torch_conda_meta_sha256=(
                        "af67d0db0054dd9ad223cf88ca8312b4d3b19599a48358fcc730a232614f6b63"
                    ),
                    torchvision_conda_meta_size=170_416,
                    torchvision_conda_meta_sha256=(
                        "47666e9ec7191c600b52ea7767cc4f0fac004f5833b1533b59f08da8c9e3d3eb"
                    ),
                    torchvision_extension_sha256=(
                        "5e6af42282499bc52e705dab3a5e61d6baf84314645006f36a071595d56d92f6"
                    ),
                    torchvision_image_sha256=(
                        "86b3b7e3536aee331adc1951bd62299f1f5aed3d4d43c1905d3e85577db4f763"
                    ),
                    torchvision_video_reader_sha256=(
                        "0df72047b5e9ee3e8a838cf31a929fa6b82c06976cfc33ae7d21333f69502d89"
                    ),
                    # This environment's BLAS/LAPACK SONAMEs resolve to the
                    # already-pinned libmkl_rt binary and no independent
                    # BLAS/LAPACK file was mapped in H100 diagnostic 291962_1.
                    # An independently mapped future libblas/liblapack path is
                    # nevertheless relevant and therefore rejected as unknown.
                    optional_cpu_math_specs=(),
                    mkl_specs=(
                        (
                            "lib/libmkl_avx512.so.2",
                            67_212_984,
                            "5dbc11d0fac83ce35e5a7d9e66c99de4ea5b88a4ca0b32e8f490fa791d3d815a",
                        ),
                        (
                            "lib/libmkl_def.so.2",
                            42_899_160,
                            "a10a0386e95b3dcc3bc1843df8268629c88ee1abbf189e2b4fb94c83c4ad3033",
                        ),
                        (
                            "lib/libmkl_core.so.2",
                            75_119_752,
                            "3d6f3d5e6ace46852e1d22f47ce32c5baebca80b99c721a56ec9e3351c1d21ae",
                        ),
                        (
                            "lib/libmkl_gnu_thread.so.2",
                            31_910_672,
                            "44c26166da452ab302c931c2de17681cf01e92c5b1c2d856011825f4ababc5a0",
                        ),
                        (
                            "lib/libmkl_intel_lp64.so.2",
                            17_695_648,
                            "bd259733d7a3044c0656f32325df145e63ca52aba71560db0db2e991a138ee16",
                        ),
                        (
                            "lib/libmkl_vml_avx512.so.2",
                            14_507_944,
                            "a91658c37ca33fae183a22cd99a980d6f24158d1bbb55a5e31c09b676ebc0b50",
                        ),
                        (
                            "lib/libmkl_vml_def.so.2",
                            8_881_208,
                            "38a93d521ac6d5aa274041965b1442cb9e974d62fc00c25439a0070308735463",
                        ),
                        (
                            "lib/libmkl_intel_thread.so.2",
                            66_016_520,
                            "2a568b1acf5d528539bbff53be43384bba822ea1c78669240c06a6bf76fed3ce",
                        ),
                        (
                            "lib/libmkl_gf_lp64.so.2",
                            17_683_304,
                            "a83b1fa88e1fa88fd02c1cf9aff40835456d6a12811b82864b735965c570afa5",
                        ),
                        (
                            "lib/libmkl_rt.so.2",
                            12_201_528,
                            "eba5c737cbd0653fd887cb7f8cdac1c728280e9a95324536ed3a515f002cf07c",
                        ),
                    ),
                ),
            }
        ),
    }
)

TEMPORAL_METRIC_DEPENDENCY_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "lpips": "0.1.4",
        "scipy": "1.15.3",
        "wcwidth": "0.2.13",
    }
)
TEMPORAL_METRIC_DISTRIBUTION_RECORD_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "lpips": "6ac488fe9f8278bb0afc2971595f4cbbe8868702da1b726a8999427820a0ef17",
        "scipy": "c82f29fed713623c4774afcec7b0b71043e7715efe9621de48ef4efb933bef20",
        "wcwidth": "0222d77c6d7f6e4125d9a6b921a6f66f03d338933a30364816fddf421e660661",
    }
)
TEMPORAL_METRIC_FRAMEWORK_VERSIONS: Mapping[str, str] = MappingProxyType(
    {
        "torch": "2.5.1",
        "torchvision": "0.20.1",
    }
)

_ARTIFACT_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "alexnet_backbone": MappingProxyType(
            {
                "filename": "alexnet-owt-7be5be79.pth",
                "size_bytes": 244_408_911,
                "sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
            }
        ),
        "lpips_wheel": MappingProxyType(
            {
                "filename": "lpips-0.1.4-py3-none-any.whl",
                "size_bytes": 53_763,
                "sha256": "fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e",
            }
        ),
        "scipy_wheel": MappingProxyType(
            {
                "filename": (
                    "scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
                ),
                "size_bytes": 37_662_964,
                "sha256": "9e2abc762b0811e09a0d3258abee2d98e0c703eee49464ce0069590846f31d40",
            }
        ),
        "wcwidth_wheel": MappingProxyType(
            {
                "filename": "wcwidth-0.2.13-py2.py3-none-any.whl",
                "size_bytes": 34_166,
                "sha256": "3da69048e4540d84af32131829ff948f1e022c1c6bdb8d6102117aac784f6859",
            }
        ),
    }
)

LPIPS_CALIBRATION_MEMBER = "lpips/weights/v0.1/alex.pth"
LPIPS_CALIBRATION_SIZE_BYTES = 6_009
LPIPS_CALIBRATION_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
_LPIPS_SOURCE_MEMBERS: Mapping[str, str] = MappingProxyType(
    {
        "lpips/__init__.py": ("36ee004a45e2cc2c5ab47fa3955f00c0e41f6a151ef0249fb6f362533ed58b22"),
        "lpips/lpips.py": ("780d09b907cb9b661e0ae28b2d163ddfba92f9e870d7feba34d4790cc6590658"),
        "lpips/pretrained_networks.py": (
            "6a27f714c51796db466e86bebba6a617c1bfc4d566f3a1756497629c1248686e"
        ),
    }
)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SEGMENTED_METRIC_PARAMETERS: Mapping[str, Any] = _deep_freeze(
    {
        "schema_version": 2,
        "metric_id": "lpips_v0.1_alex_preregistered_fixed_weights_canonical_cpu_fp64",
        "implementation": {
            "package": "lpips",
            "package_version": "0.1.4",
            "network": "alex",
            "lpips_version": "0.1",
            "spatial": False,
            "learned_calibration": True,
            "dropout_module_present": True,
            "evaluation_mode": True,
            "parameters_frozen": True,
            "weight_loading": "torch.load(weights_only=True, map_location='cpu')",
            "network_downloads": "forbidden",
            "canonical_execution_device": "cpu",
            "canonical_intraop_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
        },
        "normalization": {
            "input_layout": "BCHW",
            "input_color": "RGB",
            "input_dtype": "floating",
            "input_range_inclusive": [0.0, 1.0],
            "input_canonicalization_dtype": "float32",
            "model_compute_dtype": "float64",
            "published_output_dtype": "float32",
            "lpips_normalize_argument": True,
            "lpips_model_range": [-1.0, 1.0],
            "minimum_height_width": 64,
        },
        "reduction": {
            "per_layer": "channel-calibrated squared normalized-feature difference",
            "spatial": "mean",
            "layers": "sum_of_five_alex_feature_slices",
            "output": (
                "one_finite_nonnegative_float32_value_per_sample_cast_from_canonical_cpu_float64"
            ),
        },
        "weights": {
            "alexnet_backbone_sha256": _ARTIFACT_SPECS["alexnet_backbone"]["sha256"],
            "lpips_calibration_sha256": LPIPS_CALIBRATION_SHA256,
            "lpips_calibration_archive_member": LPIPS_CALIBRATION_MEMBER,
        },
        "dependencies": dict(TEMPORAL_METRIC_DEPENDENCY_VERSIONS),
        "installed_distribution_record_sha256": dict(TEMPORAL_METRIC_DISTRIBUTION_RECORD_SHA256),
        "frameworks": dict(TEMPORAL_METRIC_FRAMEWORK_VERSIONS),
    }
)
SEGMENTED_METRIC_PARAMETERS_SHA256 = _canonical_sha256(SEGMENTED_METRIC_PARAMETERS)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _below(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} resolves outside {parent}: {resolved}.") from exc
    return resolved


def _validate_distribution(
    package_name: str,
    expected_version: str,
    *,
    prefix: Path,
    expected_record_sha256: str | None = None,
    distribution: metadata.Distribution | None = None,
) -> tuple[dict[str, Any], metadata.Distribution]:
    try:
        distribution = distribution or metadata.distribution(package_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Temporal metric runtime lacks {package_name}=={expected_version}."
        ) from exc
    if distribution.version != expected_version:
        raise RuntimeError(
            f"Temporal metric runtime has {package_name}=={distribution.version}; "
            f"expected exactly {expected_version}."
        )
    distribution_path = _below(
        Path(distribution.locate_file("")),
        prefix,
        label=f"{package_name} distribution",
    )
    record: dict[str, str | int] | None = None
    if expected_record_sha256 is not None:
        files = distribution.files
        if files is None:
            raise RuntimeError(f"Installed {package_name} distribution has no file inventory.")
        candidates = [
            item
            for item in files
            if Path(str(item)).name == "RECORD"
            and Path(str(item)).parent.name.endswith(".dist-info")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Installed {package_name} distribution must expose exactly one "
                f"dist-info RECORD; found {len(candidates)}."
            )
        raw_record_path = Path(distribution.locate_file(candidates[0]))
        if raw_record_path.is_symlink():
            raise RuntimeError(
                f"Installed {package_name} dist-info RECORD must not be a symbolic link."
            )
        record_path = _below(
            raw_record_path,
            prefix,
            label=f"installed {package_name} dist-info RECORD",
        )
        try:
            record_status = record_path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Installed {package_name} dist-info RECORD is absent: {record_path}."
            ) from exc
        if not stat.S_ISREG(record_status.st_mode):
            raise RuntimeError(f"Installed {package_name} dist-info RECORD is not a regular file.")
        record_digest = _sha256_file(record_path)
        if record_digest != expected_record_sha256:
            raise RuntimeError(
                f"Installed {package_name} dist-info RECORD SHA-256 drifted: expected "
                f"{expected_record_sha256}, got {record_digest}."
            )
        record = {
            "path": str(record_path),
            "size_bytes": int(record_status.st_size),
            "sha256": record_digest,
        }
    result: dict[str, Any] = {
        "package_name": package_name,
        "version": distribution.version,
        "distribution_path": str(distribution_path),
    }
    if record is not None:
        result["dist_info_record"] = record
    return result, distribution


def _validate_read_only_artifact(
    dependency_root: Path,
    *,
    label: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    path = dependency_root / str(spec["filename"])
    if path.is_symlink():
        raise RuntimeError(f"Preregistered {label} must not be a symbolic link: {path}.")
    path = _below(path, dependency_root, label=f"preregistered {label}")
    try:
        status = path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Preregistered {label} is absent: {path}.") from exc
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"Preregistered {label} is not a regular file: {path}.")
    if status.st_mode & 0o222:
        raise RuntimeError(
            f"Preregistered {label} is writable (mode {stat.filemode(status.st_mode)}): {path}."
        )
    size = int(status.st_size)
    if size != int(spec["size_bytes"]):
        raise RuntimeError(
            f"Preregistered {label} size drifted: expected {spec['size_bytes']}, got {size}."
        )
    digest = _sha256_file(path)
    if digest != spec["sha256"]:
        raise RuntimeError(
            f"Preregistered {label} SHA-256 drifted: expected {spec['sha256']}, got {digest}."
        )
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256": digest,
        "read_only": True,
    }


def _wheel_member_bytes(
    wheel_path: Path,
    member: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str,
) -> bytes:
    try:
        with ZipFile(wheel_path) as archive:
            if archive.namelist().count(member) != 1:
                raise RuntimeError(
                    f"Authenticated LPIPS wheel must contain exactly one {member!r}."
                )
            info = archive.getinfo(member)
            if expected_size is not None and info.file_size != expected_size:
                raise RuntimeError(
                    f"LPIPS wheel member {member!r} size drifted: expected "
                    f"{expected_size}, got {info.file_size}."
                )
            payload = archive.read(info)
    except (BadZipFile, KeyError) as exc:
        raise RuntimeError(f"Authenticated LPIPS wheel is malformed: {wheel_path}.") from exc
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(
            f"LPIPS wheel member {member!r} SHA-256 drifted: "
            f"expected {expected_sha256}, got {digest}."
        )
    return payload


def _validate_lpips_install_sources(
    distribution: metadata.Distribution,
    *,
    prefix: Path,
    wheel_path: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for member, expected_digest in _LPIPS_SOURCE_MEMBERS.items():
        wheel_payload = _wheel_member_bytes(
            wheel_path,
            member,
            expected_sha256=expected_digest,
        )
        installed_path = _below(
            Path(distribution.locate_file(member)),
            prefix,
            label=f"installed {member}",
        )
        try:
            installed_payload = installed_path.read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Installed LPIPS source is absent: {installed_path}.") from exc
        installed_digest = _sha256_bytes(installed_payload)
        if installed_payload != wheel_payload or installed_digest != expected_digest:
            raise RuntimeError(
                f"Installed LPIPS source differs from the authenticated wheel: {member}."
            )
        records[member] = {
            "path": str(installed_path),
            "size_bytes": len(installed_payload),
            "sha256": installed_digest,
        }
    return records


def validate_temporal_metric_contract(
    *,
    project_root: Path,
    prefix: Path | None = None,
) -> dict[str, Any]:
    """Authenticate dependencies and artifacts without loading neural weights."""

    root = Path(project_root).resolve()
    resolved_prefix = Path(prefix or sys.prefix).resolve()
    if not resolved_prefix.is_dir():
        raise RuntimeError(f"Temporal metric environment prefix is absent: {resolved_prefix}.")

    dependencies: dict[str, dict[str, str]] = {}
    distributions: dict[str, metadata.Distribution] = {}
    for package_name, expected_version in TEMPORAL_METRIC_DEPENDENCY_VERSIONS.items():
        record, distribution = _validate_distribution(
            package_name,
            expected_version,
            prefix=resolved_prefix,
            expected_record_sha256=TEMPORAL_METRIC_DISTRIBUTION_RECORD_SHA256[package_name],
        )
        dependencies[package_name] = record
        distributions[package_name] = distribution

    frameworks: dict[str, dict[str, str]] = {}
    for package_name, expected_version in TEMPORAL_METRIC_FRAMEWORK_VERSIONS.items():
        record, _ = _validate_distribution(
            package_name,
            expected_version,
            prefix=resolved_prefix,
        )
        frameworks[package_name] = record

    dependency_root = (root / DEPENDENCY_ROOT_RELATIVE).resolve()
    if dependency_root.parent != (root / DEPENDENCY_ROOT_RELATIVE.parent).resolve():
        raise RuntimeError("Temporal metric dependency root resolves outside the project.")
    if not dependency_root.is_dir():
        raise RuntimeError(f"Temporal metric dependency root is absent: {dependency_root}.")
    artifacts = {
        label: _validate_read_only_artifact(dependency_root, label=label, spec=spec)
        for label, spec in _ARTIFACT_SPECS.items()
    }
    lpips_wheel = Path(artifacts["lpips_wheel"]["path"])
    calibration_payload = _wheel_member_bytes(
        lpips_wheel,
        LPIPS_CALIBRATION_MEMBER,
        expected_size=LPIPS_CALIBRATION_SIZE_BYTES,
        expected_sha256=LPIPS_CALIBRATION_SHA256,
    )
    artifacts["lpips_calibration"] = {
        "archive_path": str(lpips_wheel),
        "archive_member": LPIPS_CALIBRATION_MEMBER,
        "size_bytes": len(calibration_payload),
        "sha256": _sha256_bytes(calibration_payload),
        "read_only_archive": True,
    }
    lpips_sources = _validate_lpips_install_sources(
        distributions["lpips"],
        prefix=resolved_prefix,
        wheel_path=lpips_wheel,
    )

    source_path = Path(__file__).resolve()
    source_digest = _sha256_file(source_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": CONTRACT_STATUS,
        "environment_prefix": str(resolved_prefix),
        "dependency_distributions": dependencies,
        "framework_distributions": frameworks,
        "artifacts": artifacts,
        "lpips_installed_sources": lpips_sources,
        "implementation": {
            "source_path": str(source_path),
            "source_sha256": source_digest,
        },
        "parameters": _plain(SEGMENTED_METRIC_PARAMETERS),
        "parameters_sha256": SEGMENTED_METRIC_PARAMETERS_SHA256,
    }


def _require_state_dict(value: Any, *, label: str, torch_module: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"{label} is not a non-empty state dictionary.")
    state = dict(value)
    if any(not isinstance(key, str) for key in state):
        raise RuntimeError(f"{label} contains a non-string key.")
    if any(not torch_module.is_tensor(tensor) for tensor in state.values()):
        raise RuntimeError(f"{label} contains a non-tensor value.")
    return state


def _assert_complete_load(result: Any, *, label: str) -> None:
    missing = tuple(getattr(result, "missing_keys", ()))
    unexpected = tuple(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"{label} state coverage is incomplete: missing={missing}, unexpected={unexpected}."
        )


def _assert_authenticated_module(module: Any, *, expected_root: Path, name: str) -> None:
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        raise RuntimeError(f"Imported {name} module has no source path.")
    actual = Path(raw_path).resolve()
    _below(actual, expected_root, label=f"imported {name} module")


class FixedLpipsMetric:
    """Callable fixed LPIPS-v0.1/Alex metric for normalized RGB tensors."""

    def __init__(self, model: Any, *, torch_module: Any, device: Any, evidence: Mapping[str, Any]):
        self._model = model
        self._torch = torch_module
        self._device = device
        self.evidence = dict(evidence)

    @classmethod
    def from_preregistered_artifacts(
        cls,
        project_root: Path,
        device: str | Any = "cpu",
    ) -> FixedLpipsMetric:
        """Build the metric only from authenticated local artifacts; never download."""

        evidence = validate_temporal_metric_contract(
            project_root=Path(project_root),
            prefix=Path(sys.prefix),
        )
        torch = import_module("torch")
        torchvision = import_module("torchvision")
        scipy = import_module("scipy")
        wcwidth = import_module("wcwidth")
        lpips = import_module("lpips")
        for name, module in (
            ("torch", torch),
            ("torchvision", torchvision),
            ("scipy", scipy),
            ("wcwidth", wcwidth),
            ("lpips", lpips),
        ):
            group = (
                evidence["dependency_distributions"]
                if name in TEMPORAL_METRIC_DEPENDENCY_VERSIONS
                else evidence["framework_distributions"]
            )
            _assert_authenticated_module(
                module,
                expected_root=Path(group[name]["distribution_path"]),
                name=name,
            )

        resolved_device = torch.device(device)
        if resolved_device.type != "cpu":
            raise RuntimeError(
                "The schema-2 fixed LPIPS contract requires canonical CPU execution; "
                f"got device={resolved_device}."
            )

        # Float32 AlexNet convolution produced adjacent (one-ULP) results across
        # otherwise authenticated CPU runtimes.  The scientific metric therefore
        # uses an explicit, prospectively versioned arithmetic contract: inputs
        # are first canonicalized to float32, the fixed network is evaluated in
        # float64 on one CPU thread with deterministic algorithms and MKLDNN
        # disabled, and only the published scalar is cast back to float32.  This
        # is slower than the upstream default, but it is reproducible across the
        # two production environments and their distinct MKL installations.
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        if (
            torch.get_num_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()
            or torch.backends.mkldnn.enabled
        ):
            raise RuntimeError("Canonical fixed-LPIPS CPU execution controls did not apply.")

        alexnet_path = Path(evidence["artifacts"]["alexnet_backbone"]["path"])
        lpips_wheel_path = Path(evidence["artifacts"]["lpips_wheel"]["path"])
        calibration_payload = _wheel_member_bytes(
            lpips_wheel_path,
            LPIPS_CALIBRATION_MEMBER,
            expected_size=LPIPS_CALIBRATION_SIZE_BYTES,
            expected_sha256=LPIPS_CALIBRATION_SHA256,
        )

        # Model constructors initialize random tensors even when weights=None.
        # fork_rng keeps metric construction from perturbing experiment RNG state.
        with torch.random.fork_rng(devices=[]):
            backbone = torchvision.models.alexnet(weights=None, progress=False)
            backbone_state = _require_state_dict(
                torch.load(
                    alexnet_path,
                    map_location="cpu",
                    weights_only=True,
                ),
                label="AlexNet backbone artifact",
                torch_module=torch,
            )
            _assert_complete_load(
                backbone.load_state_dict(backbone_state, strict=True),
                label="AlexNet backbone",
            )

            # LPIPS pnet_rand=True is the explicit no-download construction path.
            # Every random trunk/linear tensor is replaced below under strict coverage.
            model = lpips.LPIPS(
                pretrained=False,
                net="alex",
                version="0.1",
                lpips=True,
                spatial=False,
                pnet_rand=True,
                pnet_tune=False,
                use_dropout=True,
                model_path=None,
                eval_mode=True,
                verbose=False,
            )

        feature_to_slice = {
            "0": "slice1.0",
            "3": "slice2.3",
            "6": "slice3.6",
            "8": "slice4.8",
            "10": "slice5.10",
        }
        mapped_features: dict[str, Any] = {}
        for key, tensor in backbone.features.state_dict().items():
            feature_index, separator, suffix = key.partition(".")
            if not separator or feature_index not in feature_to_slice:
                raise RuntimeError(f"Unexpected fixed AlexNet feature state key: {key!r}.")
            mapped_features[f"{feature_to_slice[feature_index]}.{suffix}"] = tensor
        _assert_complete_load(
            model.net.load_state_dict(mapped_features, strict=True),
            label="LPIPS AlexNet trunk",
        )

        calibration_state = _require_state_dict(
            torch.load(
                io.BytesIO(calibration_payload),
                map_location="cpu",
                weights_only=True,
            ),
            label="LPIPS v0.1 Alex calibration",
            torch_module=torch,
        )
        expected_calibration_keys = {f"lin{index}.model.1.weight" for index in range(5)}
        if set(calibration_state) != expected_calibration_keys:
            raise RuntimeError(
                "LPIPS calibration state coverage drifted: "
                f"expected={sorted(expected_calibration_keys)}, "
                f"actual={sorted(calibration_state)}."
            )
        for index, linear in enumerate(model.lins):
            source_key = f"lin{index}.model.1.weight"
            _assert_complete_load(
                linear.load_state_dict(
                    {"model.1.weight": calibration_state[source_key]},
                    strict=True,
                ),
                label=f"LPIPS calibration layer {index}",
            )

        model.requires_grad_(False)
        model.eval()
        model.to(device=resolved_device, dtype=torch.float64)
        if model.training or any(module.training for module in model.modules()):
            raise RuntimeError("Fixed LPIPS model is not completely in evaluation mode.")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("Fixed LPIPS model contains trainable parameters.")
        if any(
            not bool(torch.isfinite(value).all().item())
            for value in model.state_dict().values()
            if torch.is_tensor(value)
        ):
            raise RuntimeError("Fixed LPIPS model contains a non-finite state value.")
        evidence["canonical_execution"] = {
            "device": str(resolved_device),
            "input_canonicalization_dtype": "float32",
            "model_compute_dtype": "float64",
            "published_output_dtype": "float32",
            "intraop_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
        }
        return cls(
            model,
            torch_module=torch,
            device=resolved_device,
            evidence=evidence,
        )

    @property
    def device(self) -> str:
        return str(self._device)

    def __call__(self, left: Any, right: Any) -> Any:
        """Return one finite, nonnegative float32 LPIPS distance per sample."""

        torch = self._torch
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise TypeError("Fixed LPIPS inputs must both be torch tensors.")
        if left.shape != right.shape:
            raise ValueError(
                f"Fixed LPIPS input shapes differ: {tuple(left.shape)} != {tuple(right.shape)}."
            )
        if left.ndim != 4 or left.shape[0] < 1 or left.shape[1] != 3:
            raise ValueError(
                f"Fixed LPIPS inputs must be non-empty RGB BCHW tensors; got {tuple(left.shape)}."
            )
        if left.shape[2] < 64 or left.shape[3] < 64:
            raise ValueError("Fixed LPIPS input height and width must both be at least 64.")
        if not left.dtype.is_floating_point or not right.dtype.is_floating_point:
            raise TypeError("Fixed LPIPS inputs must use a floating dtype.")
        for label, value in (("left", left), ("right", right)):
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Fixed LPIPS {label} input contains a non-finite value.")
            minimum = float(value.min().item())
            maximum = float(value.max().item())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    f"Fixed LPIPS {label} input is outside [0, 1]: "
                    f"minimum={minimum}, maximum={maximum}."
                )
        # Preserve the historical input contract independently of the caller's
        # floating dtype, then lift those exact float32 values into the canonical
        # float64 evaluation path.
        left_float = left.to(device=self._device, dtype=torch.float32).to(dtype=torch.float64)
        right_float = right.to(device=self._device, dtype=torch.float32).to(dtype=torch.float64)
        with torch.inference_mode():
            values = self._model(left_float, right_float, normalize=True).reshape(left.shape[0])
        values = values.to(dtype=torch.float32)
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError("Fixed LPIPS produced a non-finite distance.")
        if bool((values < 0.0).any().item()):
            minimum = float(values.min().item())
            if not math.isfinite(minimum):
                raise RuntimeError("Fixed LPIPS produced a non-finite minimum distance.")
            raise RuntimeError(f"Fixed LPIPS produced a negative distance: {minimum}.")
        return values


_GPU_RUNTIME_EXCLUSION_POLICY = MappingProxyType(
    {
        "policy_id": "canonical_cpu_fixed_lpips_transitive_gpu_libraries_v1",
        "justification": (
            "The authenticated PyTorch packages are CUDA-enabled and load GPU libraries "
            "transitively at import. The schema-2 metric rejects every non-CPU device, "
            "constructs and evaluates the fixed network only on CPU, uses no CUDA RNG "
            "device, tensor, context, or kernel, and verifies CPU float64 output by an "
            "exact numeric sentinel; these mapped GPU bytes cannot participate in the "
            "admitted arithmetic path."
        ),
    }
)

_RUNTIME_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "environment",
    "runtime_profile",
    "framework_packages",
    "framework_runtime_artifacts",
    "framework_runtime_artifacts_sha256",
    "cpu_dispatch_bundle",
    "excluded_loaded_runtime",
    "sentinel",
    "canonical_execution",
    "metric_parameters_sha256",
    "metric_implementation_source_sha256",
}
_RUNTIME_ARTIFACT_RECEIPT_KEYS = {
    "artifact_id",
    "path",
    "role",
    "size_bytes",
    "sha256",
    "mapped_during_sentinel",
}
_RUNTIME_PROFILE_KEYS = {
    "schema_version",
    "environment_name",
    "environment_prefix",
    "framework_packages",
    "cpu_dispatch_policy",
    "artifacts",
}
_CPU_DISPATCH_POLICY_KEYS = {"policy_id", "selection_rule", "bundles"}
_CPU_DISPATCH_BUNDLE_KEYS = {"bundle_id", "artifact_ids"}
_CPU_DISPATCH_RECEIPT_KEYS = {
    "policy_id",
    "selection_rule",
    "selected_bundle_id",
    "selected_artifact_ids",
}
_CANONICAL_EXECUTION_RECEIPT = {
    "device": "cpu",
    "input_canonicalization_dtype": "float32",
    "model_compute_dtype": "float64",
    "published_output_dtype": "float32",
    "intraop_threads": 1,
    "deterministic_algorithms": True,
    "mkldnn_enabled": False,
}


def _validate_runtime_profile_structure(
    profile: Mapping[str, Any],
    *,
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Validate the immutable profile topology before trusting any profile field."""

    def fail(message: str) -> None:
        raise error_type(message)

    if not isinstance(profile, Mapping) or set(profile) != _RUNTIME_PROFILE_KEYS:
        fail("Temporal metric runtime profile field coverage is invalid.")
    schema_version = profile.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 2:
        fail("Temporal metric runtime profile schema drifted.")
    for key in ("environment_name", "environment_prefix"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            fail(f"Temporal metric runtime profile {key} is invalid.")
    if not isinstance(profile.get("framework_packages"), Mapping):
        fail("Temporal metric runtime profile framework packages are malformed.")

    raw_specs = profile.get("artifacts")
    if not isinstance(raw_specs, tuple) or not raw_specs:
        fail("Temporal metric runtime profile has no immutable artifact tuple.")
    ids: set[str] = set()
    raw_paths: set[tuple[str, str]] = set()
    specs_by_id: dict[str, Mapping[str, Any]] = {}
    for position, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != _RUNTIME_ARTIFACT_SPEC_KEYS:
            fail(f"Temporal metric runtime artifact profile {position} is malformed.")
        artifact_id = raw_spec.get("artifact_id")
        path_kind = raw_spec.get("path_kind")
        raw_path = raw_spec.get("path")
        role = raw_spec.get("role")
        size_bytes = raw_spec.get("size_bytes")
        sha256 = raw_spec.get("sha256")
        must_be_mapped = raw_spec.get("must_be_mapped")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in ids:
            fail("Temporal metric runtime profile artifact IDs are invalid.")
        if path_kind not in {"prefix_relative", "absolute"}:
            fail(f"Temporal metric runtime artifact {artifact_id} path kind is invalid.")
        if not isinstance(raw_path, str) or not raw_path:
            fail(f"Temporal metric runtime artifact {artifact_id} path is invalid.")
        raw_path_key = (str(path_kind), raw_path)
        if raw_path_key in raw_paths:
            fail("Temporal metric runtime profile contains duplicate raw artifact paths.")
        if not isinstance(role, str) or not role:
            fail(f"Temporal metric runtime artifact {artifact_id} role is invalid.")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            fail(f"Temporal metric runtime artifact {artifact_id} size is invalid.")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            fail(f"Temporal metric runtime artifact {artifact_id} SHA-256 is invalid.")
        if not isinstance(must_be_mapped, bool):
            fail(f"Temporal metric runtime artifact {artifact_id} mapping policy is invalid.")
        ids.add(artifact_id)
        raw_paths.add(raw_path_key)
        specs_by_id[artifact_id] = raw_spec

    policy = profile.get("cpu_dispatch_policy")
    if not isinstance(policy, Mapping) or set(policy) != _CPU_DISPATCH_POLICY_KEYS:
        fail("Temporal metric CPU-dispatch policy shape is invalid.")
    bundles = policy.get("bundles")
    if not isinstance(bundles, tuple) or len(bundles) != 2:
        fail("Temporal metric CPU-dispatch bundle profile is malformed.")
    for bundle in bundles:
        if not isinstance(bundle, Mapping) or set(bundle) != _CPU_DISPATCH_BUNDLE_KEYS:
            fail("Temporal metric CPU-dispatch bundle profile is malformed.")
        artifact_ids = bundle.get("artifact_ids")
        if (
            not isinstance(bundle.get("bundle_id"), str)
            or not bundle["bundle_id"]
            or not isinstance(artifact_ids, tuple)
            or len(artifact_ids) != 2
            or any(not isinstance(artifact_id, str) for artifact_id in artifact_ids)
        ):
            fail("Temporal metric CPU-dispatch bundle profile is malformed.")
    if _plain(policy) != _plain(_MKL_CPU_DISPATCH_POLICY):
        fail("Temporal metric CPU-dispatch policy semantics drifted.")

    if not _MKL_CPU_DISPATCH_ARTIFACT_IDS.issubset(ids):
        fail("Temporal metric CPU-dispatch artifacts are incomplete.")
    for artifact_id in _MKL_CPU_DISPATCH_ARTIFACT_IDS:
        spec = specs_by_id[artifact_id]
        if spec["role"] != "cpu_math_runtime" or spec["must_be_mapped"] is not False:
            fail("Temporal metric CPU-dispatch artifact policy drifted.")
    if not _MKL_INVARIANT_MUST_MAP_ARTIFACT_IDS.issubset(ids):
        fail("Temporal metric invariant MKL artifacts are incomplete.")
    if any(
        specs_by_id[artifact_id]["must_be_mapped"] is not True
        for artifact_id in _MKL_INVARIANT_MUST_MAP_ARTIFACT_IDS
    ):
        fail("Temporal metric invariant MKL mapping policy drifted.")
    for artifact_id in {"blas_runtime", "lapack_runtime"} & ids:
        spec = specs_by_id[artifact_id]
        if spec["role"] != "cpu_math_runtime" or spec["must_be_mapped"] is not False:
            fail("Temporal metric optional BLAS/LAPACK mapping policy drifted.")


def _cpu_dispatch_bundle_receipt(
    profile: Mapping[str, Any],
    mapped_by_artifact_id: Mapping[str, bool],
    *,
    error_type: type[Exception] = RuntimeError,
) -> dict[str, Any]:
    """Require and describe exactly one complete, unmixed MKL dispatch bundle."""

    _validate_runtime_profile_structure(profile, error_type=error_type)
    policy = profile["cpu_dispatch_policy"]
    mapped_dispatch = {
        artifact_id
        for artifact_id in _MKL_CPU_DISPATCH_ARTIFACT_IDS
        if mapped_by_artifact_id.get(artifact_id) is True
    }
    complete = [
        bundle
        for bundle in policy["bundles"]
        if all(
            mapped_by_artifact_id.get(artifact_id) is True for artifact_id in bundle["artifact_ids"]
        )
    ]
    selected_ids = set(complete[0]["artifact_ids"]) if len(complete) == 1 else set()
    if len(complete) != 1 or mapped_dispatch != selected_ids:
        raise error_type(
            "Temporal metric CPU-dispatch bundle mapping is invalid: exactly one complete "
            "approved bundle must be mapped without partial, mixed, or additional bundle "
            f"members; mapped={sorted(mapped_dispatch)}, "
            f"complete={[bundle['bundle_id'] for bundle in complete]}."
        )
    selected = complete[0]
    return {
        "policy_id": policy["policy_id"],
        "selection_rule": policy["selection_rule"],
        "selected_bundle_id": selected["bundle_id"],
        "selected_artifact_ids": list(selected["artifact_ids"]),
    }


def _runtime_profile(
    *,
    environment_name: str,
    prefix: Path,
) -> Mapping[str, Any]:
    if not isinstance(environment_name, str) or not environment_name:
        raise RuntimeError("Temporal metric runtime environment name is absent.")
    try:
        profile = TEMPORAL_METRIC_RUNTIME_PROFILES[environment_name]
    except KeyError as exc:
        raise RuntimeError(
            f"No schema-2 temporal metric runtime profile exists for {environment_name!r}."
        ) from exc
    _validate_runtime_profile_structure(profile)
    resolved_prefix = prefix.resolve()
    expected_prefix = Path(str(profile["environment_prefix"])).resolve()
    if resolved_prefix != expected_prefix:
        raise RuntimeError(
            "Temporal metric runtime prefix differs from its exact production profile: "
            f"expected={expected_prefix}, actual={resolved_prefix}."
        )
    if profile.get("schema_version") != TEMPORAL_METRIC_RUNTIME_PREFLIGHT_SCHEMA_VERSION:
        raise RuntimeError("Temporal metric runtime profile schema drifted.")
    if profile.get("environment_name") != environment_name:
        raise RuntimeError("Temporal metric runtime profile environment identity drifted.")
    return profile


def _runtime_artifact_path(spec: Mapping[str, Any], *, prefix: Path) -> Path:
    path_kind = spec.get("path_kind")
    raw_path = spec.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Temporal metric runtime profile contains an invalid artifact path.")
    if path_kind == "prefix_relative":
        raw = prefix / raw_path
        return _below(raw, prefix, label=f"runtime artifact {spec.get('artifact_id')}")
    if path_kind == "absolute":
        raw = Path(raw_path)
        if not raw.is_absolute():
            raise RuntimeError("Absolute temporal runtime artifact path is not absolute.")
        return raw.resolve()
    raise RuntimeError(f"Temporal metric runtime artifact has unsupported path kind {path_kind!r}.")


def _read_mapped_file_paths() -> set[Path]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("Cannot inspect the schema-2 process runtime mapping.") from exc
    paths: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5].removesuffix(" (deleted)")
        path = Path(raw)
        if not path.exists():
            raise RuntimeError(f"A mapped temporal runtime file is absent: {path}.")
        paths.add(path.resolve())
    return paths


def _is_excluded_gpu_runtime(path: Path) -> bool:
    name = path.name.lower()
    return any(token in name for token in ("cuda", "cudnn", "cupti", "cusparse"))


def _is_relevant_runtime_mapping(path: Path, *, prefix: Path) -> bool:
    try:
        relative = path.relative_to(prefix)
    except ValueError:
        return path.name in {
            "ld-linux-x86-64.so.2",
            "libc.so.6",
            "libcuda.so.595.80",
            "libdl.so.2",
            "libm.so.6",
            "libmvec.so.1",
            "libpthread.so.0",
            "librt.so.1",
            "libutil.so.1",
        }
    relative_text = relative.as_posix()
    if relative_text.startswith("lib/python3.10/site-packages/torch/") or relative_text.startswith(
        "lib/python3.10/site-packages/torchvision/"
    ):
        return path.suffix == ".so" or ".so." in path.name
    return relative.parent == Path("lib") and (
        relative.name.startswith("libmkl_")
        or relative.name.startswith("libblas.so")
        or relative.name.startswith("liblapack.so")
        or relative.name in {"libgcc_s.so.1", "libgomp.so.1.0.0", "libstdc++.so.6.0.34"}
    )


def _classify_runtime_mappings(
    *,
    mapped_paths: set[Path],
    known_paths: set[Path],
    prefix: Path,
) -> list[str]:
    """Return excluded GPU mappings and reject every unknown relevant CPU mapping."""

    relevant = {path for path in mapped_paths if _is_relevant_runtime_mapping(path, prefix=prefix)}
    excluded = sorted(
        str(path) for path in relevant - known_paths if _is_excluded_gpu_runtime(path)
    )
    unknown = sorted(
        str(path) for path in relevant - known_paths if not _is_excluded_gpu_runtime(path)
    )
    if unknown:
        raise RuntimeError(
            f"Unpinned CPU/framework files were mapped during the temporal sentinel: {unknown}."
        )
    return excluded


def _authenticate_runtime_artifacts(
    profile: Mapping[str, Any],
    *,
    prefix: Path,
    mapped_paths: set[Path],
    enforce_mapped: bool = True,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    _validate_runtime_profile_structure(profile)
    raw_specs = profile["artifacts"]
    records: list[dict[str, Any]] = []
    known_paths: set[Path] = set()
    ids: set[str] = set()
    mapped_by_artifact_id: dict[str, bool] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, Mapping):
            raise RuntimeError("Temporal metric runtime profile artifact is not an object.")
        spec = raw_spec
        artifact_id = spec.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in ids:
            raise RuntimeError("Temporal metric runtime profile artifact IDs are invalid.")
        ids.add(artifact_id)
        path = _runtime_artifact_path(spec, prefix=prefix)
        try:
            status = path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Temporal metric runtime artifact is absent: {path}.") from exc
        if not stat.S_ISREG(status.st_mode):
            raise RuntimeError(f"Temporal metric runtime artifact is not regular: {path}.")
        size_bytes = int(status.st_size)
        expected_size = spec.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise RuntimeError("Temporal metric runtime artifact size contract is invalid.")
        if size_bytes != expected_size:
            raise RuntimeError(
                f"Temporal metric runtime artifact size drifted for {artifact_id}: "
                f"expected={expected_size}, actual={size_bytes}."
            )
        sha256 = _sha256_file(path)
        if sha256 != spec.get("sha256"):
            raise RuntimeError(
                f"Temporal metric runtime artifact SHA-256 drifted for {artifact_id}: "
                f"expected={spec.get('sha256')}, actual={sha256}."
            )
        mapped = path.resolve() in mapped_paths
        if enforce_mapped and spec.get("must_be_mapped") is True and not mapped:
            raise RuntimeError(
                f"Required temporal metric runtime artifact was not mapped: {artifact_id}."
            )
        known_paths.add(path.resolve())
        mapped_by_artifact_id[artifact_id] = mapped
        record_path = (
            str(prefix / str(spec["path"]))
            if spec["path_kind"] == "prefix_relative"
            else str(spec["path"])
        )
        records.append(
            {
                "artifact_id": artifact_id,
                "path": record_path,
                "role": spec["role"],
                "size_bytes": size_bytes,
                "sha256": sha256,
                "mapped_during_sentinel": mapped,
            }
        )

    excluded = _classify_runtime_mappings(
        mapped_paths=mapped_paths,
        known_paths=known_paths,
        prefix=prefix,
    )
    dispatch_bundle = (
        _cpu_dispatch_bundle_receipt(profile, mapped_by_artifact_id) if enforce_mapped else None
    )
    return records, excluded, dispatch_bundle


def _authenticate_framework_package_metadata(
    profile: Mapping[str, Any],
    *,
    prefix: Path,
) -> dict[str, Any]:
    specs_by_id = {str(spec["artifact_id"]): spec for spec in profile["artifacts"]}
    result: dict[str, Any] = {}
    packages = profile.get("framework_packages")
    if not isinstance(packages, Mapping) or set(packages) != {"torch", "torchvision"}:
        raise RuntimeError("Temporal metric framework package profile is incomplete.")
    for package_name, raw_identity in packages.items():
        if not isinstance(raw_identity, Mapping):
            raise RuntimeError("Temporal metric framework package identity is malformed.")
        identity = dict(raw_identity)
        metadata_id = identity.pop("metadata_artifact_id", None)
        if not isinstance(metadata_id, str) or metadata_id not in specs_by_id:
            raise RuntimeError("Temporal metric framework metadata artifact is not pinned.")
        metadata_path = _runtime_artifact_path(specs_by_id[metadata_id], prefix=prefix)
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot read authenticated framework package metadata: {metadata_path}."
            ) from exc
        expected_fields = {
            "name": identity["conda_name"],
            "version": identity["version"],
            "build": identity["build"],
            "build_number": identity["build_number"],
            "subdir": identity["subdir"],
            "sha256": identity["archive_sha256"],
        }
        drift = {
            key: {"expected": value, "actual": payload.get(key)}
            for key, value in expected_fields.items()
            if payload.get(key) != value
            or (key == "build_number" and isinstance(payload.get(key), bool))
        }
        if drift:
            raise RuntimeError(
                f"Authenticated {package_name} conda package identity drifted: {drift}."
            )
        result[str(package_name)] = _plain(raw_identity)
    return result


def validate_temporal_metric_runtime_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_environment_name: str | None = None,
    expected_prefix: Path | None = None,
) -> None:
    """Strictly validate a persisted schema-2 numeric/byte preflight receipt."""

    if not isinstance(receipt, Mapping) or set(receipt) != _RUNTIME_RECEIPT_KEYS:
        actual = set(receipt) if isinstance(receipt, Mapping) else set()
        raise ValueError(
            "Temporal metric runtime receipt field coverage is invalid: "
            f"missing={sorted(_RUNTIME_RECEIPT_KEYS - actual)}, "
            f"unknown={sorted(actual - _RUNTIME_RECEIPT_KEYS)}."
        )
    schema_version = receipt.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 2:
        raise ValueError("Temporal metric runtime receipt schema_version must be integer 2.")
    if receipt.get("status") != "passed_before_generation":
        raise ValueError("Temporal metric runtime receipt status is not passed_before_generation.")
    environment = receipt.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != {"name", "prefix", "python"}:
        raise ValueError("Temporal metric runtime receipt environment identity is malformed.")
    name = environment.get("name")
    prefix_value = environment.get("prefix")
    python_value = environment.get("python")
    if not all(isinstance(value, str) and value for value in (name, prefix_value, python_value)):
        raise ValueError("Temporal metric runtime receipt environment fields must be non-empty.")
    if expected_environment_name is not None and name != expected_environment_name:
        raise ValueError("Temporal metric runtime receipt environment name drifted.")
    if (
        expected_prefix is not None
        and Path(str(prefix_value)).resolve() != expected_prefix.resolve()
    ):
        raise ValueError("Temporal metric runtime receipt environment prefix drifted.")
    try:
        profile = TEMPORAL_METRIC_RUNTIME_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            "Temporal metric runtime receipt uses an unknown environment profile."
        ) from exc
    _validate_runtime_profile_structure(profile, error_type=ValueError)
    prefix = Path(str(prefix_value)).resolve()
    if prefix != Path(str(profile["environment_prefix"])).resolve():
        raise ValueError(
            "Temporal metric runtime receipt prefix is not the pinned production prefix."
        )
    if Path(str(python_value)) != prefix / "bin/python3.10":
        raise ValueError("Temporal metric runtime receipt Python executable binding drifted.")

    runtime_profile = receipt.get("runtime_profile")
    expected_profile = {
        "schema_version": 2,
        "environment_name": name,
        "profile_sha256": _canonical_sha256(profile),
    }
    if runtime_profile != expected_profile:
        raise ValueError("Temporal metric runtime profile receipt drifted.")
    if receipt.get("framework_packages") != _plain(profile["framework_packages"]):
        raise ValueError("Temporal metric framework package identities drifted.")

    raw_artifacts = receipt.get("framework_runtime_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(profile["artifacts"]):
        raise ValueError("Temporal metric runtime artifact receipt count drifted.")
    mapped_by_artifact_id: dict[str, bool] = {}
    for position, (record, spec) in enumerate(
        zip(raw_artifacts, profile["artifacts"], strict=True)
    ):
        if not isinstance(record, Mapping) or set(record) != _RUNTIME_ARTIFACT_RECEIPT_KEYS:
            raise ValueError(f"Temporal metric runtime artifact receipt {position} is malformed.")
        expected_path = (
            str(prefix / str(spec["path"]))
            if spec["path_kind"] == "prefix_relative"
            else str(spec["path"])
        )
        expected = {
            "artifact_id": spec["artifact_id"],
            "path": expected_path,
            "role": spec["role"],
            "size_bytes": spec["size_bytes"],
            "sha256": spec["sha256"],
        }
        for key, value in expected.items():
            if record.get(key) != value or (
                key == "size_bytes" and isinstance(record.get(key), bool)
            ):
                raise ValueError(
                    f"Temporal metric runtime artifact receipt {position} drifted at {key}."
                )
        mapped = record.get("mapped_during_sentinel")
        if not isinstance(mapped, bool):
            raise ValueError("Temporal metric runtime artifact mapped flag must be boolean.")
        if spec["must_be_mapped"] is True and mapped is not True:
            raise ValueError("A required temporal metric runtime artifact was not mapped.")
        mapped_by_artifact_id[str(spec["artifact_id"])] = mapped
    expected_artifact_digest = _canonical_sha256(raw_artifacts)
    if receipt.get("framework_runtime_artifacts_sha256") != expected_artifact_digest:
        raise ValueError("Temporal metric runtime artifact receipt digest drifted.")
    expected_dispatch_bundle = _cpu_dispatch_bundle_receipt(
        profile,
        mapped_by_artifact_id,
        error_type=ValueError,
    )
    dispatch_bundle = receipt.get("cpu_dispatch_bundle")
    if (
        not isinstance(dispatch_bundle, Mapping)
        or set(dispatch_bundle) != _CPU_DISPATCH_RECEIPT_KEYS
        or dispatch_bundle != expected_dispatch_bundle
    ):
        raise ValueError("Temporal metric CPU-dispatch bundle receipt drifted.")

    excluded = receipt.get("excluded_loaded_runtime")
    if not isinstance(excluded, Mapping) or set(excluded) != {
        "policy_id",
        "justification",
        "paths",
    }:
        raise ValueError("Temporal metric excluded-runtime receipt is malformed.")
    if (
        excluded.get("policy_id") != _GPU_RUNTIME_EXCLUSION_POLICY["policy_id"]
        or excluded.get("justification") != _GPU_RUNTIME_EXCLUSION_POLICY["justification"]
    ):
        raise ValueError("Temporal metric excluded-runtime justification drifted.")
    excluded_paths = excluded.get("paths")
    if (
        not isinstance(excluded_paths, list)
        or excluded_paths != sorted(set(excluded_paths))
        or any(
            not isinstance(path, str) or not _is_excluded_gpu_runtime(Path(path))
            for path in excluded_paths
        )
    ):
        raise ValueError("Temporal metric excluded-runtime paths are invalid.")

    sentinel = receipt.get("sentinel")
    expected_sentinel = {
        "definition": "linspace_rgb_64x64_vs_roll_height_7_width_minus_11_v1",
        "value": TEMPORAL_METRIC_SENTINEL_VALUE,
        "float_hex": TEMPORAL_METRIC_SENTINEL_HEX,
        "float32_little_endian_sha256": TEMPORAL_METRIC_SENTINEL_FLOAT32_LE_SHA256,
        "same_image_value": 0.0,
        "symmetry_exact": True,
        "repeat_exact": True,
    }
    if sentinel != expected_sentinel:
        raise ValueError("Temporal metric numeric sentinel receipt drifted.")
    if not isinstance(sentinel.get("value"), float) or not isinstance(
        sentinel.get("same_image_value"), float
    ):
        raise ValueError(
            "Temporal metric numeric sentinel values must be JSON floats, not booleans."
        )
    if receipt.get("canonical_execution") != _CANONICAL_EXECUTION_RECEIPT:
        raise ValueError("Temporal metric canonical execution receipt drifted.")
    if receipt.get("metric_parameters_sha256") != SEGMENTED_METRIC_PARAMETERS_SHA256:
        raise ValueError("Temporal metric parameter digest drifted in the runtime receipt.")
    source_sha256 = receipt.get("metric_implementation_source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("Temporal metric implementation source digest is invalid.")


def validate_temporal_metric_runtime_preflight(
    *,
    project_root: Path,
    environment_name: str,
    prefix: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the live CPU runtime and execute the exact schema-2 sentinel."""

    resolved_prefix = Path(prefix or sys.prefix).resolve()
    if resolved_prefix != Path(sys.prefix).resolve():
        raise RuntimeError(
            "The temporal metric numeric preflight must execute inside the prefix it binds."
        )
    profile = _runtime_profile(
        environment_name=environment_name,
        prefix=resolved_prefix,
    )
    expected_python = resolved_prefix / "bin/python3.10"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(
            "The temporal metric numeric preflight Python executable is not profile-bound."
        )

    # Authenticate the full byte contract before model construction.  Mapping
    # flags are filled after the sentinel so they describe the arithmetic run,
    # while every file is already fixed before any neural weight is loaded.
    _authenticate_runtime_artifacts(
        profile,
        prefix=resolved_prefix,
        mapped_paths=set(),
        enforce_mapped=False,
    )
    framework_packages = _authenticate_framework_package_metadata(
        profile,
        prefix=resolved_prefix,
    )
    metric = FixedLpipsMetric.from_preregistered_artifacts(
        Path(project_root),
        device="cpu",
    )
    torch = metric._torch
    left = torch.linspace(0.0, 1.0, 3 * 64 * 64, dtype=torch.float32).reshape(1, 3, 64, 64)
    right = torch.roll(left, shifts=(7, -11), dims=(2, 3))
    same = metric(left, left)
    forward = metric(left, right)
    backward = metric(right, left)
    repeated = metric(left, right)
    if not torch.equal(same, torch.zeros_like(same)):
        raise RuntimeError("Temporal metric sentinel same-image result is not exactly zero.")
    if not torch.equal(forward, backward):
        raise RuntimeError("Temporal metric sentinel is not exactly symmetric.")
    if not torch.equal(forward, repeated):
        raise RuntimeError("Temporal metric sentinel is not exactly repeatable.")
    if tuple(forward.shape) != (1,) or forward.dtype != torch.float32:
        raise RuntimeError("Temporal metric sentinel output shape or dtype drifted.")
    value = float(forward.item())
    float32_digest = _sha256_bytes(struct.pack("<f", value))
    if (
        value != TEMPORAL_METRIC_SENTINEL_VALUE
        or value.hex() != TEMPORAL_METRIC_SENTINEL_HEX
        or float32_digest != TEMPORAL_METRIC_SENTINEL_FLOAT32_LE_SHA256
    ):
        raise RuntimeError(
            "Temporal metric schema-2 numeric sentinel drifted: "
            f"value={value!r}, hex={value.hex()}, float32_le_sha256={float32_digest}."
        )
    canonical_execution = metric.evidence.get("canonical_execution")
    if canonical_execution != _CANONICAL_EXECUTION_RECEIPT:
        raise RuntimeError("Temporal metric canonical execution controls drifted.")
    mapped_paths = _read_mapped_file_paths()
    artifacts, excluded_paths, dispatch_bundle = _authenticate_runtime_artifacts(
        profile,
        prefix=resolved_prefix,
        mapped_paths=mapped_paths,
    )
    if dispatch_bundle is None:
        raise RuntimeError("Temporal metric CPU-dispatch admission was not recorded.")
    receipt = {
        "schema_version": TEMPORAL_METRIC_RUNTIME_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed_before_generation",
        "environment": {
            "name": environment_name,
            "prefix": str(resolved_prefix),
            "python": str(expected_python),
        },
        "runtime_profile": {
            "schema_version": 2,
            "environment_name": environment_name,
            "profile_sha256": _canonical_sha256(profile),
        },
        "framework_packages": framework_packages,
        "framework_runtime_artifacts": artifacts,
        "framework_runtime_artifacts_sha256": _canonical_sha256(artifacts),
        "cpu_dispatch_bundle": dispatch_bundle,
        "excluded_loaded_runtime": {
            **_plain(_GPU_RUNTIME_EXCLUSION_POLICY),
            "paths": excluded_paths,
        },
        "sentinel": {
            "definition": "linspace_rgb_64x64_vs_roll_height_7_width_minus_11_v1",
            "value": value,
            "float_hex": value.hex(),
            "float32_little_endian_sha256": float32_digest,
            "same_image_value": float(same.item()),
            "symmetry_exact": True,
            "repeat_exact": True,
        },
        "canonical_execution": dict(canonical_execution),
        "metric_parameters_sha256": metric.evidence["parameters_sha256"],
        "metric_implementation_source_sha256": metric.evidence["implementation"]["source_sha256"],
    }
    validate_temporal_metric_runtime_receipt(
        receipt,
        expected_environment_name=environment_name,
        expected_prefix=resolved_prefix,
    )
    return receipt


__all__ = [
    "FixedLpipsMetric",
    "SEGMENTED_METRIC_PARAMETERS",
    "SEGMENTED_METRIC_PARAMETERS_SHA256",
    "TEMPORAL_METRIC_DEPENDENCY_VERSIONS",
    "TEMPORAL_METRIC_DISTRIBUTION_RECORD_SHA256",
    "TEMPORAL_METRIC_FRAMEWORK_VERSIONS",
    "TEMPORAL_METRIC_RUNTIME_PREFLIGHT_SCHEMA_VERSION",
    "TEMPORAL_METRIC_RUNTIME_PROFILES",
    "TEMPORAL_METRIC_SENTINEL_FLOAT32_LE_SHA256",
    "TEMPORAL_METRIC_SENTINEL_HEX",
    "TEMPORAL_METRIC_SENTINEL_VALUE",
    "validate_temporal_metric_contract",
    "validate_temporal_metric_runtime_preflight",
    "validate_temporal_metric_runtime_receipt",
]
