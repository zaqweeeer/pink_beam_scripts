#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='', usage='./lorentz.py in.hkl uc.cell spectrum profile_radius')
    parser.add_argument('input', type=str, help=".hkl file")
    parser.add_argument('cell', type=str, help=".cell file")
    parser.add_argument('spectrum', type=str, help="spectrum file")
    parser.add_argument('radius', type=float, help="profile radius in meters")

    args = parser.parse_args()

    return args

def rec_cell(cell_fn):
    p = {}
    with open(cell_fn) as f:
        for line in f:
            s = [i.strip() for i in line.split('=')]
            if s[0] in ('a', 'b', 'c'):
                p[s[0]] = float(s[1].split()[0])
            elif s[0] in ('al', 'be', 'ga'):
                p[s[0]] = float(s[1].split()[0]) / 180 * np.pi
    
    cell = np.zeros((3,3))
    cell[0] = (p['a'], 0, 0)
    cell[1] = (p['b'] * np.cos(p['ga']), p['b'] * np.sin(p['ga']), 0)
    cell[2] = (p['c'] * np.cos(p['be']),
               p['c'] * (np.cos(p['al']) - np.cos(p['be']) * np.cos(p['ga']))/np.sin(p['ga']),
               p['c'] * np.sqrt(1 - (np.cos(p['be'])) ** 2 - ((np.cos(p['al']) - np.cos(p['be']) * np.cos(p['ga']))/np.sin(p['ga'])) ** 2))

    return((1e10 * np.linalg.inv(cell)).T)

if __name__ == '__main__':

    args = parse_args()

    astar, bstar, cstar = rec_cell(args.cell)

    sk, si = np.loadtxt(args.spectrum, skiprows=1, unpack=True)

    kmin = min((sk[0], sk[-1]))
    kmax = max((sk[0], sk[-1]))

    dk = kmax - kmin

    m_to_ev = 1.23984197e-6

    r = args.radius * m_to_ev

    out = open(args.input[:-4] + '-l.hkl', 'w')

    hs = []
    lfs = []

    reading_reflections = False
    with open(args.input) as f:
        for line in f:
            if line.startswith('End of reflections'):
                out.write(line)
                reading_reflections = False
            
            elif reading_reflections:
                h, k, l, i, p, s, n = line.split()

                rv = np.linalg.norm(int(h) * astar + int(k) * bstar + int(l) * cstar) * m_to_ev
                sthetamax = rv/2/kmax
                rmax = np.array((rv * sthetamax, rv * np.sqrt(1 - sthetamax ** 2)))

                sthetamin = rv/2/kmin
                rmin = np.array((rv * sthetamin, rv * np.sqrt(1 - sthetamin ** 2)))

                dr = np.linalg.norm((rmin - rmax))

                if 2*r > dr:
                    lf = 1
                else:
                    lf = dr / 2 / r

                hs.append(rv)
                lfs.append(lf)
                
                out.write('%4d %4d %4d %7.2f        - %7.2f %.7d\n'%(int(h), int(k), int(l), float(i)*lf / 100, float(s)*lf / 100, int(n)))

            elif line.startswith('   h    k    l'):
                reading_reflections = True
                out.write(line)

            else:
                out.write(line)


    out.close()
