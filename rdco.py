#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="", usage="./rdco.py [-i k.dat -s spectrum -o rdco.dat]"
    )
    parser.add_argument(
        "-i", "--input", type=str, default="k.dat", help="input file, default = k.dat"
    )
    parser.add_argument(
        "-s",
        "--spectrum",
        type=str,
        default="spectrum",
        help="spectrum file, default = spectrum",
    )
    parser.add_argument(
        "-u",
        "--unit-cell-scale",
        type=float,
        default=1,
        help="unit cell scaling factor, default = 1",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="rdco.dat",
        help="output file, default = rdco.dat",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    input_fn = args.input
    spectrum = args.spectrum
    output_fn = args.output
    shift = args.unit_cell_scale

    sk, si = np.loadtxt(spectrum, skiprows=1, unpack=True)

    sk *= shift

    k_peak, r_peak = np.loadtxt(input_fn, unpack=True)
    d_peak = 1e10 / r_peak

    ds = sorted(d_peak)

    lims = []
    n = len(ds)
    nps = 12

    for i in range(nps):
        lims.append(np.array((ds[i * n // nps], ds[(i + 1) * n // nps - 1])))

    lims = np.array(lims)

    intensity = []
    for lim in lims:
        print(lim)

        k1 = k_peak[np.where((d_peak < lim[1]) & (d_peak > lim[0]))]
        try:
            k10 = sorted(k1)[len(k1) * 10 // 100 + 1]
            intensity.append(
                np.interp(
                    [
                        k10,
                    ],
                    sk,
                    si,
                    left=si[0],
                )
            )
        except:
            intensity.append(
                np.array(
                    [
                        1,
                    ]
                )
            )

    intensity = np.array(intensity)[:, 0]

    intensity[np.where(intensity < 0.2)] = 0.2
    np.savetxt(output_fn, list(zip(np.mean(lims, axis=1)[:], intensity)))

    plt.ylabel("Spectral intensity cut-off")
    plt.xlabel("Resolution, 1/A")
    plt.plot(1 / np.mean(lims, axis=1), intensity)
    plt.tight_layout()
    plt.show()
