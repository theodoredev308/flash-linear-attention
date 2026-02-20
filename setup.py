import ast
import os
import re
from pathlib import Path

from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()

# Optional: build CUDA extension from existing .cu (see guide_cu.md). No .cu is generated.
def _get_ext_modules():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        if not torch.cuda.is_available():
            return [], None
        csrc = Path(__file__).resolve().parent / 'fla' / 'ops' / 'quasar' / 'csrc'
        return [
            CUDAExtension(
                name='quasar_forward_substitution_cuda',
                sources=[
                    str(csrc / 'quasar_forward_substitution.cpp'),
                    str(csrc / 'quasar_forward_substitution.cu'),
                ],
                extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
            ),
        ], BuildExtension
    except Exception:
        return [], None


_ext_modules, _build_ext_class = _get_ext_modules()
_cmdclass = {'build_ext': _build_ext_class} if _build_ext_class else {}


def get_package_version():
    init_file = Path(os.path.dirname(os.path.abspath(__file__))) / 'fla' / '__init__.py'
    with open(init_file) as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"Could not find `__version__` in the file {init_file}")
    return ast.literal_eval(version_match.group(1))


setup(
    name='flash-linear-attention',
    version=get_package_version(),
    description='Fast Triton-based implementations of causal linear attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Songlin Yang, Yu Zhang',
    author_email='yangsl66@mit.edu, yzhang.cs@outlook.com',
    url='https://github.com/fla-org/flash-linear-attention',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'transformers',
        'einops',
    ],
    extras_require={
        'conv1d': ['causal-conv1d>=1.4.0'],
        'benchmark': ['matplotlib', 'datasets>=3.3.0'],
        'test': ['pytest'],
    },
    ext_modules=_ext_modules,
    cmdclass=_cmdclass,
)
