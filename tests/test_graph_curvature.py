import numpy as np
import networkx as nx
from pygraph_curvature import forman_curvature, forman_aug_curvature, lly_curvature


def test_forman_curvature():

    G = nx.complete_graph(0)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (0, 0))

    G = nx.complete_graph(1)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (1, 1))
    assert (np.isnan(G_fcurv) == [[True]])

    G = nx.complete_graph(2)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (2, 2))
    assert (np.all(np.isnan(G_fcurv) == np.eye(2)))
    assert (np.all(G_fcurv[np.isfinite(G_fcurv)] == np.array([2., 2.])))

    G = nx.complete_graph(3)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv) == np.eye(3)))
    assert (np.all(G_fcurv[np.isfinite(G_fcurv)] == [0] * 6))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv) == np.array([[True, True, False],
                                                 [True, True, False],
                                                 [False, False, True]])))
    assert (np.all(G_fcurv[np.isfinite(G_fcurv)] == np.array([1, 1, 1, 1])))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    G.remove_edge(1, 2)
    G_fcurv, _ = forman_curvature(G)
    assert (G_fcurv.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv) == np.array([[True, True, False],
                                                 [True, True, True],
                                                 [False, True, True]])))
    assert (np.all(G_fcurv[np.isfinite(G_fcurv)] == [2] * 2))


test_forman_curvature()


def test_forman_aug_curvature():

    G = nx.complete_graph(0)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (0, 0))

    G = nx.complete_graph(1)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (1, 1))
    assert (np.isnan(G_fcurv_aug) == [[True]])

    G = nx.complete_graph(2)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (2, 2))
    assert (np.all(np.isnan(G_fcurv_aug) == np.eye(2)))
    assert (np.all(G_fcurv_aug[np.isfinite(G_fcurv_aug)] == np.array([2., 2.])))

    G = nx.complete_graph(3)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv_aug) == np.eye(3)))
    assert (np.all(G_fcurv_aug[np.isfinite(G_fcurv_aug)] == [3] * 6))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv_aug) == np.array([[True, True, False],
                                                     [True, True, False],
                                                     [False, False, True]])))
    assert (np.all(G_fcurv_aug[np.isfinite(G_fcurv_aug)] == np.array([1, 1, 1, 1])))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    G.remove_edge(1, 2)
    G_fcurv_aug, _ = forman_aug_curvature(G)
    assert (G_fcurv_aug.shape == (3, 3))
    assert (np.all(np.isnan(G_fcurv_aug) == np.array([[True, True, False],
                                                     [True, True, True],
                                                     [False, True, True]])))
    assert (np.all(G_fcurv_aug[np.isfinite(G_fcurv_aug)] == [2] * 2))


test_forman_aug_curvature()


def test_lly_curvature():

    G = nx.complete_graph(0)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (0, 0))

    G = nx.complete_graph(1)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (1, 1))
    assert (np.isnan(LLY_curvatures) == [[True]])

    G = nx.complete_graph(2)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (2, 2))
    assert (np.all(np.isnan(LLY_curvatures) == np.eye(2)))
    assert (np.all(LLY_curvatures[np.isfinite(LLY_curvatures)] == np.array([2., 2.])))

    G = nx.complete_graph(3)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (3, 3))
    assert (np.all(np.isnan(LLY_curvatures) == np.eye(3)))
    assert (np.allclose(LLY_curvatures[np.isfinite(LLY_curvatures)], [1.5] * 6))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (3, 3))
    assert (np.all(np.isnan(LLY_curvatures) == np.array([[True, True, False],
                                                        [True, True, False],
                                                        [False, False, True]])))
    assert (np.allclose(LLY_curvatures[np.isfinite(LLY_curvatures)], [1., 1., 1., 1.]))

    G = nx.complete_graph(3)
    G.remove_edge(0, 1)
    G.remove_edge(1, 2)
    LLY_curvatures, _ = lly_curvature(G)
    assert (LLY_curvatures.shape == (3, 3))
    assert (np.all(np.isnan(LLY_curvatures) == np.array([[True, True, False],
                                                        [True, True, True],
                                                        [False, True, True]])))
    assert (np.all(LLY_curvatures[np.isfinite(LLY_curvatures)] == [2] * 2))


test_lly_curvature()
