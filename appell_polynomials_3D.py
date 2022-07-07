from scipy.special import gamma, factorial, comb
import numpy as np
from enum import Enum
import torch
from itertools import product


class Appell_Type(Enum):
    U = 1
    V = 0


class Appell_polynomial_weights(Enum):
    NO_WEIGHT = 0
    WEIGHTED = 1
    RANK_NORM = 2
    WEIGHTED_3 = 3
    WEIGHTED_4 = 4
    WEIGHTED_5 = 5


#          w=3 U*sqrt((2*(m+n+o)+3)*m!*n!*o!/4/pi)/(m+n+o)!^(9/32);
#              V*sqrt((2*(m+n+o)+3)*m!*n!*o!/4/pi)/(m+n+o)!^(23/32);
#          w=4 polynomials are normalized by ((m+n+o)!)^(5/32)
#          w=5 U*sqrt((2(m+n+o)+3)/(((m+n+o)/3)!)^(3/2)/((m+n+o)!)^(1/2)/4/pi)/((m+n+o)!)^(9/32);
#              V*sqrt((2(m+n+o)+3)/(((m+n+o)/3)!)^(3/2)/((m+n+o)!)^(1/2)/4/pi)/((m+n+o)!)^(23/32);


def pochham(x: np.ndarray, k: int):
    x = np.array(x, dtype=np.float64)
    return np.prod(x[..., np.newaxis] + np.arange(0, k), axis=-1)


def appell_weighting(UV, x, y, z, m, n, o, s, weight, appell_type):
    if weight == Appell_polynomial_weights.NO_WEIGHT:
        return UV
    elif weight == Appell_polynomial_weights.WEIGHTED:
        UV *= (
            np.sqrt((m + n + o + s + 1 / 2) * factorial(m) * factorial(n) * factorial(o) *
                    gamma(s + 3 / 2) / (np.pi ** (3 / 2) * gamma(s) * (s + 1 / 2) * pochham(2 * s - 1, m + n + o))))
        return UV * (1 - x ** 2 - y ** 2 - z ** 2) ** ((s - 1) / 2)
    elif weight == Appell_polynomial_weights.RANK_NORM:
        return UV / factorial(m) / factorial(n) / factorial(o)
    elif weight == Appell_polynomial_weights.WEIGHTED_3:
        if appell_type == Appell_Type.U:
            UV *= (
                    np.sqrt((2 * (m + n + o) + 3) * factorial(m) * factorial(n) * factorial(o) / 4 / np.pi) /
                    factorial(m + n + o) ** (9 / 32)
            )
        else:
            UV *= (
                    np.sqrt((2 * (m + n + o) + 3) * factorial(m) * factorial(n) * factorial(o) / 4 / np.pi) /
                    factorial(m + n + o) ** (23 / 32)
            )
        return UV * (1 - x ** 2 - y ** 2 - z ** 2) ** ((s - 1) / 2)
    elif weight == Appell_polynomial_weights.WEIGHTED_4:
        if appell_type == Appell_Type.U:
            UV *= factorial(m + n + o) ** (5 / 32)
        else:
            UV /= factorial(m + n + o) ** (5 / 32)
        return UV
    elif weight == Appell_polynomial_weights.WEIGHTED_5:
        if appell_type == Appell_Type.U:
            UV *= (np.sqrt((2 * (m + n + o) + 3) /
                           gamma((m + n + o) / 3 + 1) ** (3 / 2) /
                           gamma(m + n + o + 1) ** (1 / 2) / 4 / np.pi) /
                   factorial(m + n + o) ** (9 / 32))
        else:
            UV *= (np.sqrt((2 * (m + n + o) + 3) /
                           gamma((m + n + o) / 3 + 1) ** (3 / 2) /
                           gamma(m + n + o + 1) ** (1 / 2) / 4 / np.pi) /
                   factorial(m + n + o) ** (23 / 32))
        return UV
    else:
        raise NotImplementedError()


def get_all_appell_polynomials_3d(m, n, o,
                                  x, y, z,
                                  appell_type: Appell_Type, s, weight: Appell_polynomial_weights):
    vl = np.min([x.size, y.size, z.size])
    UV = np.zeros((m + 1, n + 1, o + 1, vl))
    for p, q, r in product(range(m + 1), range(n + 1), range(o + 1)):
        UV[p, q, r] = np.squeeze(appell_polynomials_3d(p, q, r, x, y, z, appell_type, s, weight))
    return UV


def appell_polynomials_3d(m, n, o,
                          x, y, z,
                          appell_type: Appell_Type, s, weight: Appell_polynomial_weights):
    UV = np.zeros_like(x)

    if appell_type == Appell_Type.U:
        for i in range(0, int(np.floor(m / 2)) + 1):
            for j in range(0, int(np.floor(n / 2)) + 1):
                for k in range(0, int(np.floor(o / 2)) + 1):
                    UV += ((-1) ** (i + j + k) * pochham(-m, 2 * i) * pochham(-n, 2 * j) * pochham(-o, 2 * k) /
                           (4 ** (i + j + k) * factorial(i) * factorial(j) * factorial(k) * pochham(s, i + j + k)) *
                           x ** (m - 2 * i) * y ** (n - 2 * j) * z ** (o - 2 * k) * (1 - x ** 2 - y ** 2 - z ** 2) ** (
                                   i + j + k))
        UV *= pochham(2 * s - 1, m + n + o)
    elif appell_type == Appell_Type.V:
        for i in range(0, int(np.floor(m / 2)) + 1):
            for j in range(0, int(np.floor(n / 2)) + 1):
                for k in range(0, int(np.floor(o / 2)) + 1):
                    #UV += (
                    #        pochham(s + 1 / 2, m + n + o - i - j - k) * pochham(i - m, i) * pochham(j - n, j) * pochham(
                    #    k - o, k) /
                    #        (factorial(i) * factorial(j) * factorial(k) * factorial(m - i) * factorial(
                    #            n - j) * factorial(o - k) * 4 ** (i + j + k)) *
                    #        x ** (m - 2 * i) * y ** (n - 2 * j) * z ** (o - 2 * k)
                    #)
                    UV +=(
                            pochham(s + 1 / 2, m + n + o - i - j - k) * pochham(i - m, i) * pochham(j - n, j) *
                            pochham(k - o, k) *
                            comb(m, i) * comb(n, j) * comb(o, k) / 4 ** (i+j+k) *
                            x ** (m-2*i) * y ** (n-2*j) * z ** (o-2*k)
                    )
        UV *= 2 ** (m + n + o)
    else:
        raise NotImplementedError(f"Unknown type {appell_type}")

    return appell_weighting(UV, x=x, y=y, z=z,
                            m=m, n=n, o=o, s=s,
                            weight=weight, appell_type=appell_type)


def appell_polynomials_recursive_3d(m, n, o,
                                    x: np.ndarray, y: np.ndarray, z: np.ndarray,
                                    appell_type: Appell_Type, s, weight: Appell_polynomial_weights):
    vl = np.min([x.size, y.size, z.size])
    xd = np.zeros((1, 1, 1, vl))
    xd[0, 0, 0, ...] = x
    yd = np.zeros((1, 1, 1, vl))
    yd[0, 0, 0, ...] = y
    zd = np.zeros((1, 1, 1, vl))
    zd[0, 0, 0, ...] = z
    UV = np.zeros((m + 1, n + 1, o + 1, vl))

    UV[0, 0, 0, ...] = np.ones((1, 1, 1, vl,))
    if appell_type == Appell_Type.U:
        UV[1, 0, 0, ...] = (2 * s - 1) * xd
        UV[0, 1, 0, ...] = (2 * s - 1) * yd
        UV[0, 0, 1, ...] = (2 * s - 1) * zd
        UV[2, 0, 0, ...] = (2 * s - 1) * ((2 * s + 1) * xd ** 2 + yd ** 2 + zd ** 2 - 1)
        UV[0, 2, 0, ...] = (2 * s - 1) * (xd ** 2 + (2 * s + 1) * yd ** 2 + zd ** 2 - 1)
        UV[0, 0, 2, ...] = (2 * s - 1) * (xd ** 2 + yd ** 2 + (2 * s + 1) * zd ** 2 - 1)
        UV[1, 1, 0, ...] = (2 * s - 1) * 2 * s * xd * yd
        UV[1, 0, 1, ...] = (2 * s - 1) * 2 * s * xd * zd
        UV[0, 1, 1, ...] = (2 * s - 1) * 2 * s * yd * zd
    else:
        UV[1, 0, 0, ...] = (2 * s + 1) * xd
        UV[0, 1, 0, ...] = (2 * s + 1) * yd
        UV[0, 0, 1, ...] = (2 * s + 1) * zd
        UV[2, 0, 0, ...] = (2 * s + 1) * ((2 * s + 3) * xd ** 2 - 1)
        UV[0, 2, 0, ...] = (2 * s + 1) * ((2 * s + 3) * yd ** 2 - 1)
        UV[0, 0, 2, ...] = (2 * s + 1) * ((2 * s + 3) * zd ** 2 - 1)
        UV[1, 1, 0, ...] = (2 * s + 1) * (2 * s + 3) * xd * yd
        UV[1, 0, 1, ...] = (2 * s + 1) * (2 * s + 3) * xd * zd
        UV[0, 1, 1, ...] = (2 * s + 1) * (2 * s + 3) * yd * zd
    mr = np.max([m, n, o])
    for orr in range(2, mr):
        for m in range(0, orr + 1):
            for n in range(0, orr - m + 1):
                o = orr - m - n
                uv111 = UV[m, n, o, ...]
                uv011 = UV[m - 1, n, o, ...] if m > 0 else 0
                uv101 = UV[m, n - 1, o, ...] if n > 0 else 0
                uv110 = UV[m, n, o - 1, ...] if o > 0 else 0
                if appell_type == Appell_Type.U:
                    uv100 = UV[m, n - 1, o - 1, ...] if (n > 0) and (o > 0) else 0
                    uv010 = UV[m - 1, n, o - 1, ...] if (m > 0) and (o > 0) else 0
                    uv001 = UV[m - 1, n - 1, o, ...] if (m > 0) and (n > 0) else 0
                    uv000 = UV[m - 1, n - 1, o - 1, ...] if (m > 0) and (n > 0) and (o > 0) else 0
                    UV[m + 1, n, o, ...] = (
                            (2 * m + n + o + 1) * xd * uv111 + m * o * xd * zd * uv110 + m * n * xd * yd *
                            uv101 + 2 * m * n * o * xd * yd * zd * uv100 +
                            m * ((yd ** 2 + zd ** 2 - 1) * m + (yd ** 2 + 2 * zd ** 2 - 1) * o + (
                            2 * yd ** 2 + zd ** 2 - 1) * n) * uv011 +
                            m * o * zd * ((yd ** 2 - 1) * (m + o - 1) + (3 * yd ** 2 - 1) * n) * uv010 +
                            m * n * yd * ((3 * zd ** 2 - 1) * o + (zd ** 2 - 1) * (
                            m + n - 1)) * uv001 - 2 * m * n * o * yd * zd * (m + n + o - 2) * uv000)
                    UV[m, n + 1, o, ...] = ((
                                                    m + 2 * n + o + 1) * yd * uv111 + n * o * yd * zd * uv110 + m * n * xd * yd * uv011 + 2 * m * n * o * xd * yd * zd * uv010 +
                                            n * ((xd ** 2 + zd ** 2 - 1) * n + (xd ** 2 + 2 * zd ** 2 - 1) * o +
                                                 (2 * xd ** 2 + zd ** 2 - 1) * m) * uv101 +
                                            n * o * zd * ((xd ** 2 - 1) * (n + o - 1) + (
                                    3 * xd ** 2 - 1) * m) * uv100 + m * n * xd *
                                            ((3 * zd ** 2 - 1) * o + (zd ** 2 - 1) * (
                                                    m + n - 1)) * uv001 - 2 * m * n * o * xd * zd * (
                                                    m + n + o - 2) * uv000)
                    UV[m, n, o + 1, ...] = ((
                                                    m + n + 2 * o + 1) * zd * uv111 + m * o * xd * zd * uv011 + n * o * yd * zd * uv101 + 2 * m * n * o * xd * yd * zd * uv001
                                            + o * ((xd ** 2 + yd ** 2 - 1) * o + (2 * xd ** 2 + yd ** 2 - 1) * m + (
                                    xd ** 2 + 2 * yd ** 2 - 1) * n) * uv110 + m * o * xd *
                                            ((yd ** 2 - 1) * (m + o - 1) + (
                                                    3 * yd ** 2 - 1) * n) * uv010 + n * o * yd * (
                                                    (xd ** 2 - 1) * (n + o - 1) + (3 * xd ** 2 - 1) * m)
                                            * uv100 - 2 * m * n * o * xd * yd * (m + n + o - 2) * uv000)
                else:
                    uv2_11 = UV[m + 1, n - 2, o, ...] if n > 1 else 0
                    uv21_1 = UV[m + 1, n, o - 2, ...] if o > 1 else 0
                    uv_121 = UV[m - 2, n + 1, o, ...] if m > 1 else 0
                    uv12_1 = UV[m, n + 1, o - 2, ...] if o > 1 else 0
                    uv_112 = UV[m - 2, n, o + 1, ...] if m > 1 else 0
                    uv1_12 = UV[m, n - 2, o + 1, ...] if n > 1 else 0
                    UV[m + 1, n, o, ...] = (2 * (m + n + o + 1) + s) * xd * uv111 + n * (n - 1) * uv2_11 + o * (
                            o - 1) * uv21_1 - m * (m + 2 * n + 2 * o + s + 1) * uv011
                    UV[m, n + 1, o, ...] = (2 * (m + n + o + 1) + s) * yd * uv111 + m * (m - 1) * uv_121 + o * (
                            o - 1) * uv12_1 - n * (2 * m + n + 2 * o + s + 1) * uv101
                    UV[m, n, o + 1, ...] = (2 * (m + n + o + 1) + s) * zd * uv111 + m * (m - 1) * uv_112 + n * (
                            n - 1) * uv1_12 - o * (2 * m + 2 * n + o + s + 1) * uv110

    # Weighting
    for orr in range(0, mr + 1):
        for m in range(0, orr + 1):
            for n in range(0, orr - m + 1):
                o = orr - m - n
                UV[m, n, o, ...] = appell_weighting(UV[m, n, o, ...],
                                                    x=xd, y=yd, z=zd,
                                                    m=m, n=n, o=o, s=s,
                                                    weight=weight,
                                                    appell_type=appell_type)
    return UV


@torch.jit.script
def appell_moments_3d_predef(imgs: torch.Tensor, polynomial: torch.Tensor, min_img_size: int):
    sc = ((min_img_size - 1) / 2) ** 3
    assert imgs.ndim == 4, "Images expected in format (Batch size, x, y, z)"
    img_length = imgs.shape[1] * imgs.shape[2] * imgs.shape[3]
    return torch.sum(polynomial * imgs.reshape((imgs.shape[0], 1, 1, 1, img_length)), dim=-1) / sc


#@torch.jit.script
def moments_volume_normalization(moments: torch.Tensor, types, device=torch.device('cpu')):
    # Volume normalization constant
    v = (types % 2) + 2
    m000 = moments[..., 0:1, 0:1, 0:1]
    index_sum = np.sum(np.indices(moments.shape[-3:]), axis=0)
    index_sum = torch.from_numpy(index_sum).to(device)
    # Normalization to scaling
    moments /= m000 ** ((index_sum + v) / v)
    if types == 1:
        moments *= np.pi ** (index_sum / 6) * (index_sum + 3) / 1.5 ** (index_sum / 3 + 1)
    elif types == 2:
        moments *= np.pi ** (index_sum / 4) * 2 ** (index_sum / 2)
    return moments


def cafmi3d(moments: torch.Tensor, invariant_ind, invariant_coef, invariant_sizes,device=torch.device("cpu")):
    """
    Computes values of the invariants in the array f
    from the values of the moments in the 3D array m (mainly for 3D moments)
    :return:
    """
    assert isinstance(moments, torch.Tensor)

    invariants = torch.zeros((len(invariant_ind)), dtype=torch.double).to(device)
    for idx, (inv_idx, coef) in enumerate(zip(invariant_ind, invariant_coef)):
        assert inv_idx.shape[-1] % 3 == 0
        prod_num = int(inv_idx.shape[-1] / 3.0)
        inv = 1 * coef
        for prod_idx in range(prod_num):
            test_idx = inv_idx[:, 3 * prod_idx:3 * prod_idx + 3]
            inv *= moments[test_idx[:, 0], test_idx[:, 1], test_idx[:, 2]]

        invariants[idx] = torch.sum(inv)

    # magnitude normalization to degree
    invariants = torch.sign(invariants) * torch.abs(invariants) ** (1.0 / invariant_sizes)
    return invariants


#@torch.jit.script
def cafmi3d_torch(moments: torch.Tensor,
                  invariant_ind,
                  invariant_coef,
                  invariant_sizes,
                  out: torch.Tensor):
    assert invariant_ind[0].ndim == 3
    assert out.shape[0] == moments.shape[0]

    for idx, (inv_idx, inv_coef) in enumerate(zip(invariant_ind, invariant_coef)):
        products = moments[:,
                           inv_idx[..., 0],
                           inv_idx[..., 1],
                           inv_idx[..., 2]]
        out[:, idx] = torch.sum(torch.prod(products, dim=-2) * inv_coef, dim=-1)
    # Normalization
    out = torch.sign(out) * (torch.abs(out) ** (1.0 / invariant_sizes))
    return out

