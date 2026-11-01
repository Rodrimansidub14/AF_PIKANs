import numpy as np

from src.sciml.sampling import pools


def test_allen_cahn_sobol_pools_are_reproducible_independent_and_in_domain():
    first = pools("allen_cahn", seed=101, n=256, heldout=512)
    repeated = pools("allen_cahn", seed=101, n=256, heldout=512)
    other_seed = pools("allen_cahn", seed=102, n=256, heldout=512)

    assert set(first) == {"interior", "diagnostic", "heldout", "boundary_probe", "initial_probe"}
    for name in first:
        np.testing.assert_array_equal(first[name], repeated[name])
    assert not np.array_equal(first["interior"], first["heldout"][: len(first["interior"])])
    assert not np.array_equal(first["interior"], other_seed["interior"])
    assert first["interior"].shape == (256, 2)
    assert first["diagnostic"].shape == (256, 2)
    assert first["heldout"].shape == (512, 2)
    for name in ("interior", "diagnostic", "heldout"):
        points = first[name]
        assert np.all((-1.0 <= points[:, 0]) & (points[:, 0] <= 1.0))
        assert np.all((0.0 <= points[:, 1]) & (points[:, 1] <= 1.0))
    np.testing.assert_array_equal(np.unique(first["boundary_probe"][:, 0]), [-1.0, 1.0])
    np.testing.assert_array_equal(first["initial_probe"][:, 1], 0.0)

