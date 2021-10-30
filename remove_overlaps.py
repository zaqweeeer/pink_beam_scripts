#!/usr/bin/env python3


import sys
import time
from multiprocessing import Pool, Manager

import argparse


def process_chunk(chunk, q):
    reading_reflections = False
    reflections = set()
    for line in chunk.split("\n"):
        if reading_reflections:
            if line.startswith("End of reflections"):
                reading_reflections = False
            else:
                x, y = line.split()[7:9]
                reflections.add((line, float(x), float(y)))
        elif line.startswith("   h    k    l "):
            reading_reflections = True
    refl_list = list(reflections)
    # print(reflections)
    while reflections:
        line, x, y = reflections.pop()
        for line1, x1, y1 in refl_list:
            if (x1 - x) ** 2 + (y1 - y) ** 2 <= mind ** 2 and line != line1:
                reflections.discard((line1, x1, y1))
                # print(line)
                chunk = chunk.replace(line + "\n", "")
                chunk = chunk.replace(line1 + "\n", "")
                break
    q.put(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="", usage="./remove_overlaps.py stream min_distance [options]"
    )
    parser.add_argument("stream", type=str, help="stream file")
    parser.add_argument("mind", type=int, help="minimum distance between reflections")

    parser.add_argument(
        "-n",
        "--nproc",
        metavar="N",
        type=int,
        default=1,
        help="run N threads in parallel, default = 1",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    stream_fn = args.stream
    mind = args.mind
    out = open(stream_fn[:-7] + "-ol.stream", "w")
    nproc = args.nproc
    stream = open(stream_fn, "r")

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
            pool.apply_async(process_chunk, (chunk, q))
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
