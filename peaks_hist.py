#!/usr/bin/env python3


import sys
import time
from multiprocessing import Pool, Manager
import numpy as np

from cfelpyutils.crystfel_utils import load_crystfel_geometry
import matplotlib.pyplot as plt
import argparse


def process_chunk(chunk, q, detector):
    reading_reflections = False
    reading_peaks = False
    indexed = False
    peaks = []
    reflections = []
    k_peak = []
    r_peak = []

    to_ev = 1239.84193e-9

    for line in chunk.split("\n"):

        if line.startswith("  fs/px   ss/px"):
            reading_peaks = True

        elif line.startswith("End of peak list"):
            reading_peaks = False

        elif reading_peaks:
            s = line.split()
            fs, ss, res, intensity = (float(i) for i in s[:4])
            peaks.append([fs, ss, s[-1]])

        elif line.startswith("   h    k    l "):
            reading_reflections = True

        elif line.startswith("End of reflections"):
            reading_reflections = False

        elif reading_reflections:
            s = line.split()
            reflections.append(
                [
                    float(s[-3]),
                    float(s[-2]),
                    int(s[0]),
                    int(s[1]),
                    int(s[2]),
                    va,
                    vb,
                    vc,
                    s[-1],
                ]
            )

        elif line.startswith("astar"):
            indexed = True
            s = line.split()
            va = np.array((float(s[2]), float(s[3]), float(s[4]))) * 1e9

        elif line.startswith("bstar"):
            s = line.split()
            vb = np.array((float(s[2]), float(s[3]), float(s[4]))) * 1e9

        elif line.startswith("cstar"):
            s = line.split()
            vc = np.array((float(s[2]), float(s[3]), float(s[4]))) * 1e9

    if not indexed:
        return

    for p in peaks:
        d = 100500.0
        nr = None
        for r in reflections:
            if (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2 < d:
                nr = r
                d = (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2

        if nr and np.sqrt(d) < 3:

            h = np.linalg.norm(nr[2] * nr[5] + nr[3] * nr[6] + nr[4] * nr[7])

            panel = detector["panels"][p[-1]]
            xp = (
                (p[0] - panel["min_fs"]) * panel["fsx"]
                + (p[1] - panel["min_ss"]) * panel["ssx"]
                + panel["cnx"]
            ) / panel["res"]
            yp = (
                (p[0] - panel["min_fs"]) * panel["fsy"]
                + (p[1] - panel["min_ss"]) * panel["ssy"]
                + panel["cny"]
            ) / panel["res"]
            dp = panel["clen"]

            k_peak.append(
                h / np.sqrt(2 * (1 - dp / np.sqrt(dp ** 2 + xp ** 2 + yp ** 2))) * to_ev
            )
            r_peak.append(h)

    q.put((k_peak, r_peak))


from scipy.optimize import curve_fit, leastsq


def parse_args():
    parser = argparse.ArgumentParser(
        description="", usage="./peaks_hist.py stream geometry spectrum [options]"
    )
    parser.add_argument("stream", type=str, help="stream file")
    parser.add_argument("geometry", type=str, help="geometry file")
    parser.add_argument("spectrum", type=str, help="spectrum file")

    parser.add_argument(
        "-n",
        "--nproc",
        metavar="N",
        type=int,
        default=1,
        help="run N processes in parallel, default = 1",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    stream = args.stream
    geometry = args.geometry
    spectrum = args.spectrum
    nproc = args.nproc

    manager = Manager()
    q = manager.Queue()

    pool = Pool(processes=nproc)
    reading_chunk = False

    detector = load_crystfel_geometry(geometry)

    k_peak = []
    r_peak = []
    i = 0
    with open(stream) as f:
        for line in f:
            if line.startswith("----- Begin chunk -----"):
                reading_chunk = True
                chunk = line

            elif line.startswith("----- End chunk -----"):
                reading_chunk = False
                chunk += line
                pool.apply_async(process_chunk, (chunk, q, detector))
                i += 1
                print("\rChunk %d" % i, end="", flush=True)
                while pool._taskqueue.qsize() > nproc:
                    while not q.empty():
                        k, r = q.get()
                        k_peak.extend(k)
                        r_peak.extend(r)
                    time.sleep(1)

            elif reading_chunk:
                chunk += line

    pool.close()
    pool.join()

    while not q.empty():
        k, r = q.get()
        k_peak.extend(k)
        r_peak.extend(r)

    np.savetxt("k.dat", list(zip(k_peak, r_peak)))

    sk, si = np.loadtxt(spectrum, skiprows=1, unpack=True)

    d_peak = 1e10 / np.array(r_peak)
    k_peak = np.array(k_peak)

    ax2 = plt.gca()
    ax1 = ax2.twinx()

    ax1.plot(sk, si, label="Spectrum", color="C0")
    ax1.set_xlabel("k, eV")
    ax1.set_ylabel("Normalized intensity", color="C0")
    for tl in ax1.get_yticklabels():
        tl.set_color("C0")

    ax1.set_ylim(0, 1.05)

    fitfunc = lambda x, p0, p1: p0 * np.interp(p1 * np.asarray(x), sk, si, right=0)

    y, bins, patches = ax2.hist(
        k_peak[np.where(d_peak < 30)], 100, alpha=0.6, color="C2", label="Found peaks"
    )

    ax2.set_ylabel("N peaks", color="C2")
    for tl in ax2.get_yticklabels():
        tl.set_color("C2")

    x = [(bins[i] + bins[i + 1]) / 2.0 for i in range(y.shape[0])]

    p = curve_fit(fitfunc, x, y, p0=((y.max() / 1.0, 1)))
    sf = 1 / p[0][1]

    ax1.plot(sk * sf, si, "C0:", label="Fitted spectrum, \n scale factor = %.4f" % sf)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0, prop={"size": 10})

    ax2.set_xlabel("k, eV")
    plt.tight_layout()

    plt.show()
