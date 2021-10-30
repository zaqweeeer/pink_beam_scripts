#!/usr/bin/env python3


import sys
import numpy as np
from multiprocessing import Pool, Manager
import time
import argparse

from scipy import interpolate


def process_chunk(chunk, q, ks, ws, ico):
    maxw = max(ws)
    m_to_ev = 1.23984197e-6

    reading_reflections = False
    reading_peaks = False
    reading_chunk = False
    reading_cell = False
    for line in chunk.split("\n"):
        if line.startswith("----- Begin chunk -----"):
            reading_chunk = True
            indexed = False
            r = 0.01e9

        elif line.startswith("----- End chunk -----"):
            reading_chunk = False

        elif reading_chunk:

            if not reading_reflections:
                if line.startswith("   h    k    l "):
                    reading_reflections = True
                    nr = 0
                    nt = 0

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

                elif line.startswith("profile_radius"):
                    try:
                        r = float(line.split()[-2]) * 1e9
                    except:
                        pass

                elif line.startswith("num_reflections"):
                    num_refl_line = line

            elif reading_reflections:
                if line.startswith("End of reflections"):
                    reading_reflections = False
                    chunk = chunk.replace(
                        num_refl_line, "num_reflections = %d" % (nt - nr)
                    )
                    # print(nr, nt, r)
                else:
                    nt += 1

                    # This bit calculates central x-ray energy (k) contributing to the reflection:
                    s = [int(i) for i in line.split()[:3]]
                    h = s[0] * va + s[1] * vb + s[2] * vc
                    kmin = (
                        -np.linalg.norm(h) ** 2 / 2 / (h[2] - r) * m_to_ev - r * m_to_ev
                    )
                    kmax = (
                        -np.linalg.norm(h) ** 2 / 2 / (h[2] + r) * m_to_ev + r * m_to_ev
                    )
                    w = 0
                    n = 0
                    k = 0
                    for i in range(len(ks)):
                        if kmin < ks[i] <= kmax:
                            k += ks[i]
                            n += 1
                        elif i != len(ks) - 1 and ks[i] < kmin < kmax < ks[i + 1]:
                            k = (kmin + kmax) / 2
                            n = 1
                    try:
                        k /= n
                    except ZeroDivisionError:
                        k = 0

                    # And correspondong spectral weight (w):
                    for i in range(len(ks) - 1):
                        if ks[i] < k <= ks[i + 1]:
                            w = ((ks[i + 1] - k) * ws[i] + (k - ks[i]) * ws[i + 1]) / (
                                ks[i + 1] - ks[i]
                            )

                    res = np.linalg.norm(h) * 1e-10
                    minw = ico(1 / res)

                    # Reflection is saved in the output stream file if spectral weight of the
                    # contributing X-ray energy w > minw, integrated intensity is scaled by w:
                    if w / maxw >= minw and line.split()[3] not in ("nan", "-nan"):
                        spl = line.split()
                        spl[3] = str(float(spl[3]) / w * maxw)
                        chunk = chunk.replace(
                            line,
                            "%4d %4d %4d %10.2f %10.2f %10.2f %10.2f %6.1f %6.1f %s"
                            % tuple((float(i) if i[0] != "p" else i for i in spl)),
                        )
                    else:
                        nr += 1
                        chunk = chunk.replace(line + "\n", "")
    # print(chunk)
    q.put(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="", usage="./scale.py stream spectrum [options]"
    )
    parser.add_argument("stream", type=str, help="stream file")
    parser.add_argument("spectrum", type=str, help="spectrum file")

    parser.add_argument(
        "-n",
        "--nproc",
        metavar="N",
        type=int,
        default=1,
        help="run N threads in parallel, default = 1",
    )

    parser.add_argument(
        "-u",
        "--unit-cell-scale",
        metavar=1.0,
        type=float,
        default=1.0,
        help="unit cell scaling factor, default = 1.",
    )
    parser.add_argument(
        "-m",
        "--min-spectral-weight",
        metavar=0.2,
        type=float,
        default=None,
        help="relative spectral weight cut-off, default = 0.2",
    )
    parser.add_argument(
        "-r",
        "--rdco",
        metavar="rdco.dat",
        type=str,
        default=None,
        help="resolution dependant spectral weight cut-off",
    )

    args = parser.parse_args()

    if args.min_spectral_weight and args.rdco:
        parser.error("please use either -r or -m but not both")

    return args


if __name__ == "__main__":

    args = parse_args()
    stream_fn = args.stream
    spectrum = args.spectrum
    shift = args.unit_cell_scale
    nproc = args.nproc

    if args.rdco:
        r, i = np.loadtxt(args.rdco, unpack=True)
        ico = interpolate.interp1d(r, i, fill_value="extrapolate")
    else:
        minw = args.min_spectral_weight
        ico = interpolate.interp1d([0, 1], [minw, minw], fill_value="extrapolate")

    out = open(stream_fn[:-7] + "-%s.stream" % "s", "w")

    stream = open(stream_fn)
    ks, ws = np.loadtxt(spectrum, unpack=True, skiprows=1)

    ks *= shift
    if ks[1] > ks[2]:
        ks = ks[::-1]
        ws = ws[::-1]

    manager = Manager()
    q = manager.Queue()

    pool = Pool(processes=nproc)
    reading_chunk = False
    i = 0
    for line in stream:
        if line.startswith("----- Begin chunk -----"):
            reading_chunk = True
            chunk = line

        elif line.startswith("----- End chunk -----"):
            reading_chunk = False
            chunk += line
            pool.apply_async(process_chunk, (chunk, q, ks, ws, ico))
            i += 1
            print("\rChunk %d" % i, end="", flush=True)
            while pool._taskqueue.qsize() > nproc:
                while not q.empty():
                    out.write(q.get())
                time.sleep(1)

        elif reading_chunk:
            chunk += line

        else:
            out.write(line)

    pool.close()
    pool.join()

    while not q.empty():
        out.write(q.get())

    print()

    stream.close()
    out.close()
