
:github_url: https://github.com/facebookresearch/DRTK

Installation
===================================

Currently, we do not provide pre-compiled binaries for DRTK.
You will need to build the package from source. The current version of DRTK is |version|.

Prerequisites
^^^^^^^^^^^^^

Before installing DRTK, ensure you have the following prerequisites installed:

* PyTorch >= 2.1.0
* CUDA Toolkit

Additionally, we recommend installing the following packages to run tests and examples:

* torchvision
* opencv_python

Installing DRTK from GitHub using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install DRTK is by using pip with the GitHub repository directly:

.. code-block:: shell

    # To install latest
    pip install git+https://github.com/facebookresearch/DRTK.git

.. code-block:: shell

    # To install stable
    pip install git+https://github.com/facebookresearch/DRTK.git@stable

.. warning::

    It may take significant amount of time to compile. The time could be 30 minutes or more.

This should be enough in most cases, given that PyTorch, CUDA Toolkit, and Build Essentials for your platform
are installed and the environment is correctly configured.


.. _Specifying Architectures:

Specifying Architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you know the CUDA architecture of the device where the code will run, then it would be better to specify it directly, e.g.:

.. code-block:: shell

    # TORCH_CUDA_ARCH_LIST can use "named" architecture, see table below
    TORCH_CUDA_ARCH_LIST="Ampere" install git+https://github.com/facebookresearch/DRTK.git

or specify numerical values for architectures explicitly:

.. code-block:: shell

    # TORCH_CUDA_ARCH_LIST can combine several architectures separated with semicolon or space.
    # Add `+PTX` if you want also to save intermediate byte code for better compatibility.
    TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" install git+https://github.com/facebookresearch/DRTK.git

If ``TORCH_CUDA_ARCH_LIST`` is not specified, DRTK will build for the following architectures by default: 7.2, 7.5, 8.0, 8.6, 9.0.

``TORCH_CUDA_ARCH_LIST`` can take one or more values from the supported named or numerical architectures list.
When combining values, use a semicolon `;` or space to combine numerical values and the `+` symbol to combine named values.

List of numerical architectures values supported by PyTorch: ``'3.5', '3.7', '5.0', '5.2', '5.3', '6.0', '6.1', '6.2', '7.0', '7.2', '7.5', '8.0', '8.6', '8.7', '8.9', '9.0', '9.0a'``.

The "named" architectures supported by PyTorch are listed in the table below:

.. list-table:: Named architectures
   :header-rows: 1

   * - Name
     - Arch
   * - Kepler+Tesla
     - 3.7
   * - Kepler
     - 3.5+PTX
   * - Maxwell+Tegra
     - 5.3
   * - Maxwell
     - 5.0;5.2+PTX
   * - Pascal
     - 6.0;6.1+PTX
   * - Volta+Tegra
     - 7.2
   * - Volta
     - 7.0+PTX
   * - Turing
     - 7.5+PTX
   * - Ampere+Tegra
     - 8.7
   * - Ampere
     - 8.0;8.6+PTX
   * - Ada
     - 8.9+PTX
   * - Hopper
     - 9.0+PTX

.. note::

    We do not test the DRTK package on architectures before Volta.

For more information about ``TORCH_CUDA_ARCH_LIST``, refer to the `PyTorch documentation <https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.CUDAExtension>`_ or view the `source code on GitHub <https://github.com/pytorch/pytorch/blob/c9653bf2ca6dd88b991d71abf836bd9a7a1d9dc3/torch/utils/cpp_extension.py#L1980>`_.

Installing from a Cloned Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, you can install the package from a local repository clone.
It could be helpful if you need to modify the package code.

Clone the repository and ``cd`` into it:

.. code-block:: shell

    git clone https://github.com/facebookresearch/DRTK
    cd DRTK

Then, install the package from the path using ``pip``. Note the ``--no-build-isolation`` flag, it is needed for modern build
system to disable building in a separate, clean Python environment.
The reason is that it will install a default ``torch`` version from pip, which likely will not match the one already installed in the system (due to usage of ``--index-url``).

.. code-block:: shell

    pip install . --no-build-isolation


Building and installing a wheel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build a wheel run:

.. code-block:: shell

    # You might need to install `build` first
    # pip install --upgrade build
    python -m build --wheel --no-isolation

Alternatively, you can use the deprecated CLI of ``setuptools``:

.. code-block:: shell

    # You might need to install `wheel` first, though newer versions of setuptools do not require it anymore.
    # pip install --upgrade wheel
    python setup.py bdist_wheel

Then, you will find a wheel in the ``dist/`` folder. You can install this wheel by running:

.. code-block:: shell

    pip install dist/drtk-<tags>.whl

where ``<tags>`` are compatibility tags. You can figure them out by listing the ``dist/`` directory. E.g.:

.. code-block:: shell

    pip install dist/drtk-0.1.0-cp310-cp310-linux_x86_64.whl

Reinstalling the Wheel
^^^^^^^^^^^^^^^^^^^^^^

If you have already installed the package using ``pip``, it will not reinstall the package unless the version number has been incremented.
This behavior can be problematic if you are modifying the package locally and need to reinstall it.

To force a reinstall, add the following arguments: ``--upgrade --force-reinstall --no-deps``. For example:

.. code-block:: shell

    pip install --upgrade --force-reinstall --no-deps .


In place build
^^^^^^^^^^^^^^^^^^^

For package development, it can be beneficial to do an inplace build with:

.. code-block:: shell

    # There can be issues with concurrent build jobs, it is safer to specify `-inplace -j 1`
    python setup.py build_ext --inplace -j 1

Then you can use the root of the cloned repository as a working directory, and you should be able to do ``import drtk`` and run tests and examples.


Troubleshooting
================

1. CUDA Error: No Kernel Image Available
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example Error Message:**

    RuntimeError: CUDA error: no kernel image is available for execution on the device


**Cause:** This error occurs when the CUDA code was not built for the architecture of the device on which the code is running.

**Solution:** Specify the correct architecture using the ``TORCH_CUDA_ARCH_LIST`` environment variable when building the package. Refer to the examples in the :ref:`Specifying Architectures` section above.

2. Import Error: ``*.so`` Not Found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example Error Message:**

    ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory

**Cause:** This issue is likely due to build isolation. Since DRTK currently does not distribute pre-compiled binaries, it is hard to get version mismatch otherwise.

**Solution:** Ensure you include the ``--no-build-isolation`` flag when installing from a local clone to use the correct CUDA and PyTorch libraries from your current environment:

.. code-block:: shell

    python -m build --wheel --no-isolation

and

.. code-block:: shell

    pip install . --no-build-isolation


3. Compilation Errors in CUDA or PyTorch headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example Error Message:**

    error: no suitable conversion function from "const __half" to "unsigned short" exists

**Cause:** This error typically indicates a compiler version mismatch. It is likely that your C++ or CUDA compiler version is too old to support some of the features.

**Solution:** Please consult PyTorch and CUDA documentation to figure out what CUDA version is supported by your PyTorch version and what C++ compiler version is needed.


4. C++ Compilation Errors in PyTorch header aten/src/ATen/core/boxing/impl/boxing.h
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Example Error Message:**

    aten/src/ATen/core/boxing/impl/boxing.h:41:105: error: expected primary-expression before ‘>’ token


**Cause:** This issue is related to problematic SFINAE logic in template code. It has been observed in some recent versions of PyTorch.

**Solution:** One recommended solution is to add ``-std=c++20`` to the **nvcc** arguments, as suggested in `this GitHub issue <https://github.com/pytorch/pytorch/issues/122169>`_.
This line has already been added to ``setup.py```:

.. code-block:: python

    nvcc_args.append("-std=c++20")


.. warning::

    If you are using an older version of the CUDA Toolkit, adding this flag might result in an error:

    .. code-block:: shell

        unrecognized command line option '-std=c++20'


    In this case, you may need to remove the ``-std=c++20`` flag from ``setup.py``, or update your CUDA Toolkit to a version that supports C++20.

.. note::

    According to the comments in the `GitHub issue <https://github.com/pytorch/pytorch/issues/122169>`_, this was fixed in PyTorch v2.4.0 release.
