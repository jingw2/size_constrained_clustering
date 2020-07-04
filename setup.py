from setuptools import setup, find_packages, Extension, dist

import os
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

dist.Distribution().fetch_build_eggs(["cython>=0.29", "numpy>=1.13"])


try:
    from numpy import get_include
except:
    def get_include():
        # Defer import to later
        from numpy import get_include
        return get_include()

try:
    from Cython.Build import cythonize
except ImportError:
    print("! Could not import Cython !")
    cythonize = None


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

extensions = [
    Extension("src.k_means_constrained.mincostflow_vectorized_", ["src/k_means_constrained/mincostflow_vectorized_.pyx"],
              include_dirs=[get_include()]),
    Extension("src.sklearn_import.cluster._k_means", ["src/sklearn_import/cluster/_k_means.pyx"],
              include_dirs=[get_include()]),
    Extension("src.sklearn_import.metrics.pairwise_fast", ["src/sklearn_import/metrics/pairwise_fast.pyx"],
                  include_dirs=[get_include()]),
    Extension("src.sklearn_import.utils.sparsefuncs_fast", ["src/sklearn_import/utils/sparsefuncs_fast.pyx"],
                      include_dirs=[get_include()]),
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 1))) and cythonize is not None

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
      ext_modules=extensions,
      install_requires=install_requires,
      name='size_constrained_clustering',
      description='Size Constrained Clustering solver',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/jingw2/size_constrained_clustering',
      author='Jing Wang',
      author_email='jingw2@foxmail.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3.6')
