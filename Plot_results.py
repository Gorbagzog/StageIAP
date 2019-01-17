#!/usr/bin/env python3
# -*-coding:Utf-8 -*

"""Load and plot best fit parameters estimated from MCMC output"""

import numpy as np
import matplotlib.pyplot as plt


def load_results(directory):
    results = np.loadtxt(directory + '/Results.txt', skiprows=2).astype('float')
    results = results[results[:, 0].argsort()]  # sort the array by redshift
    return results


def plot_one(directory, results, idx_result, result_label):
    redshiftsbinTrue = np.array([0.37, 0.668, 0.938, 1.286, 1.735, 2.220, 2.683, 3.271, 3.926, 4.803])
    errm = results[:, idx_result+1] - results[:, idx_result+2]
    errp = results[:, idx_result+3] - results[:, idx_result+1]
    plt.figure()
    plt.errorbar(redshiftsbinTrue[:], results[:, idx_result+1], yerr=[errm, errp])
    plt.ylabel(result_label, size=20)
    plt.xlabel('Redshift')
    plt.savefig(directory + "/Plots/Result_" + result_label + '.pdf')
    plt.close()


def plot_all(directory):
    labels = ['$M_{1}$', '$M_{*,0}$', '$\\beta$', '$\delta$', '$\gamma$', r'$\xi$']
    results = load_results(directory)
    for i in range(6):
        plot_one(directory, results, 3*i, labels[i])
