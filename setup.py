import io
import os
import re
import setuptools

here = os.path.realpath(os.path.dirname(__file__))


name = 'torchcde'

# for simplicity we actually store the version in the __version__ attribute in the source
with io.open(os.path.join(here, name, '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

author = 'Patrick Kidger'

author_email = 'contact@kidger.site'

description = "Differentiable controlled differential equation solvers for PyTorch with GPU support and " \
              "memory-efficient adjoint backpropagation."

with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

url = "https://github.com/patrick-kidger/torchcde"

license = "Apache-2.0"

classifiers = ["Development Status :: 4 - Beta",
               "Intended Audience :: Developers",
               "Intended Audience :: Financial and Insurance Industry",
               "Intended Audience :: Information Technology",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: Apache Software License",
               "Natural Language :: English",
               "Operating System :: MacOS :: MacOS X",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Artificial Intelligence",
               "Topic :: Scientific/Engineering :: Information Analysis",
               "Topic :: Scientific/Engineering :: Mathematics"]

python_requires = "~=3.6"

install_requires = ['torch>=1.7.0', 'torchdiffeq>=0.2.0', 'torchsde>=0.2.5']

setuptools.setup(name=name,
                 version=version,
                 author=author,
                 author_email=author_email,
                 maintainer=author,
                 maintainer_email=author_email,
                 description=description,
                 long_description=readme,
                 long_description_content_type="text/markdown",
                 url=url,
                 license=license,
                 classifiers=classifiers,
                 zip_safe=False,
                 python_requires=python_requires,
                 install_requires=install_requires,
                 packages=[name])
