#!/usr/bin/env python3

""" MA3.py

Student: Julia Rask Lind
Mail: julia.rask-lind.0494@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
import numpy as np
import functools
import multiprocessing as mp
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    #n = {1000, 10000, 100000}
    x_in_circle = []
    y_in_circle= []
    x_not_circle = []
    y_not_circle= []


    for i in range(n): 
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        if x**2 + y**2 <= 1:
            x_in_circle.append(x)
            y_in_circle.append(y)

        else:
            x_not_circle.append(x)
            y_not_circle.append(y)

    pi_estimate = 4 * len(x_in_circle) / n


    plt.figure(figsize=(6, 6))
    plt.scatter(x_in_circle, y_in_circle, color = 'red', s = 1)
    plt.scatter(x_not_circle, y_not_circle, color = 'blue', s = 1)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"Approximation of π with n = {n}, π = {4 * len(x_in_circle) / n} ")

    plt.savefig("pi_approximation_{n}.png")
    plt.show()

    return pi_estimate

# Exc2, approximation
#för n punkter, d dimensioner och radie r mha byggare, lambda, map, functools, filter, zip
def sphere_volume(n, d): 
    # n is the number of points

    x_d = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    coord_sum = list(map(lambda point: sum(x**2 for x in point), x_d))

    in_circle = list(filter(lambda sum: sum <= 1, coord_sum))

    return (len(in_circle) / n) * (2**d)


#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    return (m.pi**(d/2))/(m.gamma(d/2 + 1))


#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    in_circle = 0
    for _ in range(n):
        squared_sum = 0
        for _ in range(d):
            x = np.random.uniform(-1, 1)
            squared_sum += x**2
        if squared_sum <= 1:
            in_circle += 1

    return  in_circle/n * 2**d
          

#Exc4: parallel code - parallelize actual computations by splitting data
#help function:

def _point_counter(n, d):
    xd = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]

    sums = map(lambda p: sum(x**2 for x in p), xd)

    inside_circle = len(list(filter(lambda s: s <= 1, sums)))

    return inside_circle

def sphere_volume_parallel(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    s = n // np

    with future.ProcessPoolExecutor(max_workers=np) as ex:
        process = [ex.submit(_point_counter, s, d) for _ in range(np)]
        total_inside_circle = sum(p.result() for p in process)

    return (total_inside_circle / (s * np)) * 2**d

def sphere_volume_parallel_n(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    s = n // np

    with future.ProcessPoolExecutor(max_workers=np) as ex:
        process = [ex.submit(sphere_volume_numba, s, d) for _ in range(np)]
        total_inside_circle = [p.result() for p in process]

    return mean(total_inside_circle)

def main():
    #Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume_parallel(n, d)
    stop = pc()
    print(f"Exc4: parallell time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_parallel_n(n, d)
    stop = pc()
    print(f"Exc4: parallell numba time of {d} and {n}: {stop-start}")
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Exc4: numba time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

if __name__ == '__main__':
	main()
