import numpy as np
import pytest


@pytest.fixture(scope="module", params=["numpy", "jax"])
def xp(request):
    """array namespaces to test"""
    impl = request.param
    if impl == "numpy":
        try:
            import numpy as np
            import scipy as scipy
        except ModuleNotFoundError:
            pytest.skip("requires numpy, scipy")
        else:
            return np
    if impl == "jax":
        try:
            import jax
            import jax.numpy as jnp
        except ModuleNotFoundError:
            pytest.skip("requires jax")
        else:
            jax.config.update("jax_enable_x64", True)
            return jnp


@pytest.fixture(params=[1, 2, 5, 10, 11, 100, 101, 1000, 1001])
def n(request):
    n = request.param
    if "closed" in request.fixturenames:
        closed = request.getfixturevalue("closed")
        if closed and n == 1:
            pytest.skip("no closed transform of length 1")
    return n


@pytest.fixture(params=[False, True])
def closed(request):
    return request.param


@pytest.fixture
def rng():
    return np.random.default_rng(42)
