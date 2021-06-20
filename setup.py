try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmcubes/mcubes.pyx',
        'im2mesh/utils/libmcubes/pywrapper.cpp',
        'im2mesh/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
