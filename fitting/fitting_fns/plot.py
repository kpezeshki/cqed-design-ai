import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import time

# from git import Repo
import csv
import os
import uncertainties

import fit as fit
import cavity_functions as ff

# import fit as fit
# import cavity_functions as ff
fontsize = 12
textsize = 10
params = {
    "legend.fontsize": fontsize,
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.size": fontsize,
}

plt.rcParams.update(params)


def name_plot(filename, strmethod, output_path, format=".pdf"):
    if filename.endswith(".csv"):
        filename = filename[:-4]
    filename = filename.replace(".", "p")
    return output_path + strmethod + "_" + filename + format


def plot(x, y, name, output_path, x_c=None, y_c=None, r=None, p_x=None, p_y=None):
    # plot any given set of x and y data
    # fig = plt.figure("raw_data", figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 4)
    ax = plt.subplot(gs[0:2, 0:2])  ## plot
    # plot axes
    ax.plot(x, y, "bo", label="raw data", markersize=3)
    # plot guide circle if it applies
    if x_c is not None and y_c is not None and r is not None:
        circle = Circle(
            (x_c, y_c),
            r,
            facecolor="none",
            edgecolor=(0, 0.8, 0.8),
            linewidth=3,
            alpha=0.5,
        )
        ax.add_patch(circle)
    # plot a red point to represent something if it applies (resonance or off resonance for example)
    if p_x is not None and p_y is not None:
        ax.plot(p_x, p_y, "*", color="red", markersize=5)
    ax.set_aspect("equal", adjustable="box")
    # fig.savefig(output_path + name + ".pdf", format="pdf")


def plot2(x, y, x2, y2, name, output_path):
    # plot any given set of x and y data
    fig = plt.figure("raw_data", figsize=(10, 10))
    gs = GridSpec(2, 2)
    ax = plt.subplot(gs[0:2, 0:2])  ## plot
    ax.plot(x, y, "bo", label="raw data", markersize=3)
    ax.plot(x2, y2, "bo", label="raw data", markersize=3, color="red")
    fig.savefig(output_path + name + ".pdf", format="pdf")


def name_folder(dir, strmethod):
    result = time.localtime(time.time())
    output = strmethod + "_" + str(result.tm_year)
    if len(str(result.tm_mon)) < 2:
        output = output + "0" + str(result.tm_mon)
    else:
        output = output + str(result.tm_mon)
    if len(str(result.tm_mday)):
        output = (
            output
            + "0"
            + str(result.tm_mday)
            + "_"
            + str(result.tm_hour)
            + "_"
            + str(result.tm_min)
            + "_"
            + str(result.tm_sec)
        )
    else:
        output = (
            output
            + str(result.tm_mday)
            + "_"
            + str(result.tm_hour)
            + "_"
            + str(result.tm_min)
            + "_"
            + str(result.tm_sec)
        )
    if dir is not None:
        output_path = dir + "/" + output + "/"
    else:
        output_path = output + "/"
    count = 2
    path = output_path
    while os.path.isdir(output_path):
        output_path = path[0:-1] + "_" + str(count) + "/"
        count = count + 1
    return output_path


def create_metadata(Method, output_path):
    x = 10
    # repo = Repo(fit.ROOT_DIR)
    # sha = repo.head.object.hexsha
    # # write input parameters to metadata file
    # with open(output_path + "metadata.csv", "w", newline='') as file:
    #     writer = csv.writer(file)
    #     fields = ['Method', 'MC_iteration', 'MC_rounds',
    #               'MC_weight', 'MC_weightvalue', 'MC_fix',
    #               'MC_step_const', 'manual_init',
    #               'preprocess_method', 'Current Git Commit']
    #     vals = [Method.method, Method.MC_iteration,
    #             Method.MC_rounds,Method.MC_weight,
    #             Method.MC_weightvalue, Method.MC_fix,
    #             Method.MC_step_const, Method.manual_init,
    #             Method.preprocess_method, sha]
    #     writer.writerow(fields)
    #     writer.writerow(vals)
    #     file.close()


def PlotFit(
    x,
    y,
    x_initial,
    y_initial,
    slope,
    intercept,
    slope2,
    intercept2,
    params,
    Method,
    error,
    figurename,
    x_c,
    y_c,
    radius,
    output_path,
    conf_array,
    xlim,
    ylim,
    extract_factor=None,
    title="Fit",
    manual_params=None,
    dfac: int = 1,
    msizes: list = [4, 4],
    xstr: str = r"$(f-f_c)$ [kHz]",
    fscale: float = 1e3,
    fsize: float = fontsize,
):
    """
    Plots data and outputs fit parameters to a file

    Args:
        x: cropped frequency data
        y: cropped S21 data after normalization
        x_initial: original frequency data before normalization
        y_initial: original S21 data before normalization
        slope: slope of the normalization line for magnitude
        intercept: intercept of normalization line for magnitude
        slope2: slope of the normalization line for phase
        intercept2: intercept of normalization line for phase
        params: parameters generated from fit function
        Method: Method class instance
        func: function used to generate data for plotting
              (DCM/DCM REFLECTION/INV/PHI/CPZM)
        error: error from Monte Carlo Fit function
        figurename: name the plot should have
        x_c: center x coordinate for fit circle
        y_c: center y coordinate for fit circle
        radius: radius of fit circle
        output__path: file path where the plots will be saved
        conf_array: array with 95% confidence interval values
        extract_factor: contains starting and ending frequency values
        title: title the plot will have
        manual_params: user input manual initial guess parameters for fit
        dfac : decimation factor on the plotting of points in the resonance
               circle Im S21 vs. Re S21
        xstr : x-axis string label
        fsize : fontsize for axes numbers and labels

    Returns:
        plot output to file
    """
    # close plot if still open
    plt.close(figurename)

    # func = Method.method_class.func
    func = Method.func
    # generate an even distribution of 5000 frequency points between
    # the min and max of x for graphing purposes
    if extract_factor is not None and isinstance(extract_factor, list):
        x_fit = np.linspace(extract_factor[0], extract_factor[1], 5000)
    else:
        x_fit = np.linspace(x.min(), x.max(), 5000)
    # plug in the 5000 x points to respective fit function to create set of
    # S21 data for graphing
    y_fit = func(x_fit, *params)

    fig = plt.figure(figurename, figsize=(10, 8))
    gs = GridSpec(11, 10)
    ax0 = plt.subplot(gs[1:10, 0:6])
    ax1 = plt.subplot(gs[0:4, 7:10])
    ax2 = plt.subplot(gs[4:8, 7:10])
    # fig.set_tight_layout(True)

    # Marker sizes
    msize1, msize2 = msizes

    # # Add title to figure
    # plt.gcf().text(0.05, 0.92, title, fontsize=30)

    ## Rounding to n significant figures
    round_sigfig = lambda x, n: round(x, n - int(np.floor(np.log10(abs(x)))) - 1)

    # manual parameters
    textstr = ""
    fcscale = 1.0 if params[2] < 1e9 else 1e9
    if manual_params != None:
        if func == ff.cavity_inverse:
            textstr = (
                r"Manually input parameters:"
                + "\n"
                + r"$Q_c$ = "
                + "%s" % float("{0:.5g}".format(manual_params[1]))
                + "\n"
                + r"$Q_i$ = "
                + "%s" % float("{0:.5g}".format(manual_params[0]))
                + "\n"
                + r"$f_c$ = "
                + "%s" % float("{0:.8g}".format(manual_params[2]))
                + " GHz"
                "\n"
                + r"$\phi$ = "
                + "%s" % float("{0:.5g}".format(manual_params[3]))
                + " radians"
            )
        elif func == ff.cavity_CPZM:
            textstr = (
                r"Manually input parameters:"
                + "\n"
                + r"$Q_c$ = "
                + "%s"
                % float("{0:.5g}".format(manual_params[0] * manual_params[1] ** -1))
                + "\n"
                + r"$Q_i$ = "
                + "%s" % float("{0:.5g}".format(manual_params[0]))
                + "\n"
                + r"$Q_a$ = "
                + "%s"
                % float("{0:.5g}".format(manual_params[0] * manual_params[3] ** -1))
                + "\n"
                + r"$f_c$ = "
                + "%s" % float("{0:.8g}".format(manual_params[2]))
                + " GHz"
            )
        else:
            Qc = manual_params[1] / np.exp(1j * manual_params[3])
            Qi = (manual_params[0] ** -1 - abs(np.real(Qc**-1))) ** -1
            textstr = (
                r"Manually input parameters:"
                + "\n"
                + "Q = "
                + "%s" % float("{0:.5g}".format(manual_params[0]))
                + "\n"
                + r"1/Re[1/$Q_c$] = "
                + "%s" % float("{0:.5g}".format(1 / np.real(1 / Qc)))
                + "\n"
                + r"$Q_c$ = "
                + "%s" % float("{0:.5g}".format(manual_params[1]))
                + "\n"
                + r"$Q_i$ = "
                + "%s" % float("{0:.5g}".format(Qi))
                + "\n"
                + r"$>>f_c$ = "
                + "%s" % float("{0:.8g}".format(manual_params[2]) / fcscale)
                + " GHz"
                + "\n"
                + r"$\phi$ = "
                + "%s" % float("{0:.5g}".format(manual_params[3]))
                + " radians"
            )
        plt.gcf().text(0.1, 0.7, textstr, fontsize=15)
    else:
        pass

    if isinstance(extract_factor, list) == True:
        x_fit_full = np.linspace(x.min(), x.max(), 5000)
        y_fit_full = func(x_fit_full, *params)

        x_fit = np.copy(x_fit_full)
        y_fit = np.copy(y_fit_full)

    for i in range(len(x_initial)):
        x_initial[i] = round(x_initial[i], 8)

    # Decimate the x and y-data
    x = x[0::dfac]
    y = y[0::dfac]

    if func == ff.cavity_inverse:
        ax1.set_ylabel("Mag[1/S21]", fontsize=fsize)
        ax2.set_ylabel("Ang[1/S21]", fontsize=fsize)
    else:
        ax1.set_ylabel("Mag[S21]", fontsize=fsize)
        ax2.set_ylabel("Ang[S21]", fontsize=fsize)

    # Plot the resonance circle
    phi = params[3]
    ph = 1.0  # np.exp(1j*phi)
    ax0.plot(
        np.real(y * ph),
        np.imag(y * ph),
        "o",
        label="Normalized data",
        markersize=msize1,
        color="#8c000f",
    )
    ax0.plot(
        np.real(y_fit * ph), np.imag(y_fit * ph), "cadetblue", label="Fit function"
    )

    ax0.axhline(y=0.0, color="k")
    ax0.axvline(x=1.0, color="k")
    # ax0.axvline(x=0.0, color="k")

    if func == ff.cavity_inverse:
        ax0.set_ylabel("Im[$S_{21}^{-1}$]")
        ax0.set_xlabel("Re[$S_{21}^{-1}$]")
    else:
        ax0.set_ylabel(r"Im[$S_{21}$]")
        ax0.set_xlabel(r"Re[$S_{21}$]")
    leg = ax0.legend(loc="upper left", fancybox=True, fontsize=fontsize)
    # ax0.set_xlim([None, 1.1])
    # ax0.set_ylim([-1, 1])
    ax0.set_aspect(1.0)

    # Subtract the resonance to label as f-f0
    ax1.plot(
        (x - params[2]) / fscale,
        np.log10(np.abs(y)) * 20,
        "o",
        label="Normalized data",
        markersize=msize1,
        color="#8c000f",
    )
    ax1.plot(
        (x_fit - params[2]) / fscale,
        np.log10(np.abs(y_fit)) * 20,
        color="cadetblue",
        label="Fit function",
    )
    ax1.set_xlim(left=(x[0] - params[2]) / fscale, right=(x[-1] - params[2]) / fscale)
    ax1.set_xlabel(xstr)

    #    for tick in ax1.xaxis.get_major_ticks():
    #        tick.set_fontsize(fsize)

    ax2.plot(
        (x - params[2]) / fscale,
        np.angle(y),
        "o",
        label="normalized data",
        markersize=msize1,
        color="#8c000f",
    )
    ax2.plot(
        (x_fit - params[2]) / fscale,
        np.angle(y_fit),
        color="cadetblue",
        label="fit function",
    )
    ax2.set_xlim(left=(x[0] - params[2]) / fscale, right=(x[-1] - params[2]) / fscale)
    ax2.set_xlabel(xstr)

    # plot resonance points
    if func == ff.cavity_inverse:
        resonance = 1 + params[0] / params[1] * np.exp(1j * params[3]) / (
            1 + 1j * 2 * params[0] * (params[2] - params[2]) / params[2]
        )
    elif func == ff.cavity_DCM:
        resonance = 1 - params[0] / params[1] * np.exp(1j * params[3])
    elif func == ff.cavity_DCM_REFLECTION:
        resonance = 1 - 2 * params[0] / params[1] * np.exp(1j * params[3]) / (
            1 + 1j * (params[2] - params[2]) / params[2] * 2 * params[0]
        )
    elif func == ff.cavity_CPZM:
        resonance = 1 / (1 + params[1] + 1j * params[3])
    else:
        resonance = 1 + 1j * 0
        # markerfacecolor='none',
    ax0.plot(
        np.real(resonance),
        np.imag(resonance),
        "o",
        color="#fac205",
        label="resonance",
        markersize=2 * msize2,
    )
    ax1.plot(
        0,
        np.log10(np.abs(resonance)) * 20,
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )
    ax2.plot(
        0,
        np.angle(resonance),
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )

    ax0.plot(
        np.real(ylim[0]),
        np.imag(ylim[0]),
        "o",
        color="#fac205",
        label="resonance",
        markersize=2 * msize2,
    )
    ax1.plot(
        (xlim[0] - params[2]) / fscale,
        np.log10(np.abs(ylim[0])) * 20,
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )
    ax2.plot(
        (xlim[0] - params[2]) / fscale,
        np.angle(ylim[0]),
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )

    ax0.plot(
        np.real(ylim[-1]),
        np.imag(ylim[-1]),
        "o",
        color="#fac205",
        label="resonance",
        markersize=2 * msize2,
    )
    ax1.plot(
        (xlim[-1] - params[2]) / fscale,
        np.log10(np.abs(ylim[-1])) * 20,
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )
    ax2.plot(
        (xlim[-1] - params[2]) / fscale,
        np.angle(ylim[-1]),
        "o",
        color="#fac205",
        label="resonance",
        markersize=msize2,
    )

    #    for tick in ax2.xaxis.get_major_ticks():
    #        tick.label.set_fontsize(fsize)

    # get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(6)

    try:
        if params:
            if func == ff.cavity_inverse:
                if params[0] < 0:
                    print(
                        "Qi is less than zero. Please make sure data is of"
                        " correct format: decibals (log10*20 version), and "
                        "radians. Otherwise, it is quite likely that the "
                        "resonator being fit is not a Notch type resonator. "
                        "For reflection type geometry, "
                        "please use DCM REFLECTION."
                    )
                if conf_array[0] > 10**-10 and conf_array[0] != float("inf"):
                    Qi = params[0] - params[0] % (
                        10 ** int(np.log10(conf_array[0]) - 1)
                    )
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float("inf"):
                    Qc = params[1] - params[1] % (
                        10 ** int(np.log10(conf_array[1]) - 1)
                    )
                else:
                    Qc = params[1]
                if conf_array[2] > 10**-10 and conf_array[2] != float("inf"):
                    phi = params[3] - params[3] % (
                        10 ** int(np.log10(conf_array[2]) - 1)
                    )
                else:
                    phi = params[3]
                if conf_array[3] > 10**-10 and conf_array[3] != float("inf"):
                    f_c = params[2] - params[2] % (
                        10 ** int(np.log10(conf_array[3]) - 1)
                    )
                else:
                    f_c = params[2]
                textstr = (
                    r"$Q_i$ = "
                    + "%s" % float("{0:.10g}".format(Qi))
                    + r"$\pm$"
                    + "%s" % float("{0:.1g}".format(conf_array[0]))
                    + "\n"
                    + r"$Q_c^*$ = "
                    + "%s" % float("{0:.10g}".format(Qc))
                    + r"$\pm$"
                    + "%s" % float("{0:.1g}".format(conf_array[1]))
                    + "\n"
                    + r"$\phi$ = "
                    + "%s" % float("{0:.10g}".format(phi))
                    + r"$\pm$"
                    + "%s" % float("{0:.1g}".format(conf_array[2]))
                    + " radians"
                    + "\n"
                    + r"$f_c$ = "
                    + "%s" % float("{0:.10g}".format(f_c / fcscale))
                    + r"$\pm$"
                    + "%s" % float("{0:.1g}".format(conf_array[3]))
                    + " GHz"
                )
                plt.gcf().text(0.7, 0.11, textstr, fontsize=textsize)
            elif func == ff.cavity_CPZM:
                if params[0] < 0:
                    print(
                        "Qi is less than zero. Please make sure data is "
                        "of correct format: decibals (log10*20 version), "
                        "and radians. Otherwise, it is quite likely that the "
                        "resonator being fit is not a Notch type resonator. "
                        "For reflection type geometry, "
                        "please use DCM REFLECTION."
                    )
                if conf_array[0] > 10**-10 and conf_array[0] != float("inf"):
                    Qi = params[0] - params[0] % (
                        10 ** int(np.log10(conf_array[0]) - 1)
                    )
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float("inf"):
                    Qc = (params[0] * params[1] ** -1) - (
                        params[0] * params[1] ** -1
                    ) % (10 ** int(np.log10(conf_array[1]) - 1))
                else:
                    Qc = params[0] * params[1] ** -1
                if conf_array[2] > 10**-10 and conf_array[2] != float("inf"):
                    Qa = (params[0] * params[3] ** -1) - (
                        params[0] * params[3] ** -1
                    ) % (10 ** int(np.log10(conf_array[2]) - 1))
                else:
                    Qa = params[0] * params[3] ** -1
                if conf_array[3] > 10**-10 and conf_array[3] != float("inf"):
                    f_c = params[2] - params[2] % (
                        10 ** int(np.log10(conf_array[3]) - 1)
                    )
                else:
                    f_c = params[2]
                reports = ["$Q_i$", "$Q_c$", "$Q_a$", "$f_c$"]
                p_ref = [Qi, Qc, Qa, f_c / 1e6]
                textstr = ""

                for val in reports:
                    textstr += (
                        val
                        + f" = {p_ref[reports.index(val)]:.5f}"
                        + r"$\pm$"
                        + f"{conf_array[reports.index(val)]:.1f}"
                        + "\n"
                    )

                plt.gcf().text(0.7, 0.11, textstr, fontsize=textsize)

            else:
                Qc = params[1] / np.exp(1j * params[3])
                if Method.method == "PHI":
                    Qi = (params[0] ** -1 - np.abs(Qc**-1)) ** -1
                else:
                    Qi = (params[0] ** -1 - np.real(Qc**-1)) ** -1
                    # print(Qi)

                    # Qi2 = (params[0] ** -1 - np.abs(Qc ** -1)) ** -1
                    # print(Qi2)

                if Qi < 0 and Method.method != "DCM REFLECTION":
                    print(
                        "Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. Other types of resonators will not work with this code."
                    )
                if 1 / np.real(1 / Qc) < 0:
                    print("Warning: 1/Real[1/Qc] is less than 0. Calculating Qi anyway")
                    if conf_array[0] > 10**-10 and conf_array[0] != float("inf"):
                        Q = params[0] - params[0] % (
                            10 ** int(np.log10(conf_array[0]) - 1)
                        )
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float("inf"):
                        Qi = Qi - Qi % (10 ** int(np.log10(conf_array[1]) - 1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float("inf"):
                        Qc = params[1] - params[1] % (
                            10 ** int(np.log10(conf_array[2]) - 1)
                        )
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float("inf"):
                        Qc_Re = (
                            1 / np.real(1 / (params[1] / np.exp(1j * params[3])))
                        ) - (1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) % (
                            10 ** int(np.log10(conf_array[3]) - 1)
                        )
                    else:
                        Qc_Re = 1 / np.real(1 / (params[1] / np.exp(1j * params[3])))
                    if conf_array[4] > 10**-10 and conf_array[4] != float("inf"):
                        phi = params[3] - params[3] % (
                            10 ** int(np.log10(conf_array[4]) - 1)
                        )
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float("inf"):
                        f_c = params[2] - params[2] % (
                            10 ** int(np.log10(conf_array[5]) - 1)
                        )
                    else:
                        f_c = params[2]
                    textstr = (
                        "Q = "
                        + "%s" % float("{0:.10g}".format(Q))
                        + r"$\pm$"
                        + "%s" % float("{0:.1g}".format(conf_array[0]))
                        + "\n"
                        + r"$Q_i$ = "
                        + "%s" % float("{0:.10g}".format(Qi))
                        + r"$\pm$"
                        + "%s" % float("{0:.1g}".format(conf_array[1]))
                        + "\n"
                        + r"$Q_c$ = "
                        + "%s" % float("{0:.10g}".format(Qc))
                        + r"$\pm$"
                        + "%s" % float("{0:.1g}".format(conf_array[2]))
                        + "\n"
                        + r"$\phi$ = "
                        + "%s" % float("{0:.10g}".format(phi))
                        + r"$\pm$"
                        + "%s" % float("{0:.2g}".format(conf_array[4]))
                        + " radians"
                        + "\n"
                        + r"$qqqf_c$ = "
                        + "%s" % float("{0:.7g}".format(f_c / fcscale))
                        + r"$\pm$"
                        + "%s" % float("{0:.1g}".format(conf_array[5]))
                        + " GHz"
                    )
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=textsize)
                    Qc_str = (
                        r"1/Re[1/$Q_c$] = "
                        + "%s" % float("{0:.10g}".format(Qc_Re))
                        + r"$\pm$"
                        + "%s" % float("{0:.1g}".format(conf_array[3]))
                    )
                    plt.gcf().text(0.7, 0.245, Qc_str, fontsize=textsize, color="red")

                else:
                    if conf_array[0] > 10**-10 and conf_array[0] != float("inf"):
                        Q = params[0] - params[0] % (
                            10 ** int(np.log10(conf_array[0]) - 1)
                        )
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float("inf"):
                        Qi = Qi - Qi % (10 ** int(np.log10(conf_array[1]) - 1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float("inf"):
                        Qc = params[1] - params[1] % (
                            10 ** int(np.log10(conf_array[2]) - 1)
                        )
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float("inf"):
                        Qc_Re = (
                            1 / np.real(1 / (params[1] / np.exp(1j * params[3])))
                        ) - (1 / np.real(1 / (params[1] / np.exp(1j * params[3])))) % (
                            10 ** int(np.log10(conf_array[3]) - 1)
                        )
                    else:
                        Qc_Re = 1 / np.real(1 / (params[1] / np.exp(1j * params[3])))
                    if conf_array[4] > 10**-10 and conf_array[4] != float("inf"):
                        phi = params[3] - params[3] % (
                            10 ** int(np.log10(conf_array[4]) - 1)
                        )
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float("inf"):
                        f_c = params[2] - params[2] % (
                            10 ** int(np.log10(conf_array[5]) - 1)
                        )
                    else:
                        f_c = params[2]

                    reports = [
                        "Q",
                        r"$Q_i$",
                        r"$Q_c$",
                        r"$Re(Q_c)$",
                        r"$\phi$",
                        r"$f_c$",
                    ]
                    p_ref = [Q, Qi, Qc, Qc_Re, phi, f_c]

                    textstr = ""
                    for val in reports:
                        if val == r"$f_c$":
                            vscale = fcscale
                        else:
                            vscale = 1.0

                        # Round to one significant figure
                        err = round_sigfig(conf_array[reports.index(val)] / vscale, 1)
                        v = p_ref[reports.index(val)] / vscale

                        # Report the uncertainties using the LaTeX format
                        un = uncertainties.ufloat(v, err)
                        if val == r"$f_c$":
                            latexstr = f"{un:.08g}"
                        else:
                            latexstr = f"{un:L}"
                        textstr += r"%s: $%s$" % (val, latexstr)

                        # if conf_array[reports.index(val)] > 0:
                        #     textstr += r'$\pm$' + f'{conf_array[reports.index(val)] / vscale:.1g}'
                        if val == r"$\phi$":
                            textstr += " radians"
                        elif val == r"$f_c$":
                            textstr += " GHz"
                        textstr += "\n"
                    plt.gcf().text(0.63, 0.05, textstr, fontsize=fontsize)

            plt.show()
            # write to output csv file
            with open(output_path + "fit_params.csv", "w", newline="") as file:
                writer = csv.writer(file)
                if func == ff.cavity_inverse:
                    fields = ["Qi", "Qc*", "phi", "fc"]
                    vals = [
                        [
                            float("{0:.10g}".format(Qi)),
                            float("{0:.10g}".format(Qc)),
                            float("{0:.10g}".format(phi)),
                            float("{0:.10g}".format(f_c)),
                        ],
                        [
                            float("{0:.1g}".format(conf_array[0])),
                            float("{0:.1g}".format(conf_array[1])),
                            float("{0:.1g}".format(conf_array[2])),
                            float("{0:.1g}".format(conf_array[3])),
                        ],
                    ]

                elif func == ff.cavity_CPZM:
                    fields = ["Qi", "Qc", "Qa", "fc"]
                    vals = [
                        [
                            float("{0:.10g}".format(Qi)),
                            float("{0:.10g}".format(Qc)),
                            float("{0:.10g}".format(Qa)),
                            float("{0:.10g}".format(f_c)),
                        ],
                        [
                            float("{0:.1g}".format(conf_array[0])),
                            float("{0:.1g}".format(conf_array[1])),
                            float("{0:.1g}".format(conf_array[2])),
                            float("{0:.1g}".format(conf_array[3])),
                        ],
                    ]
                else:

                    fields = ["Q", "Qi", "Qc", "1/Re[1/Qc]", "phi", "fc"]
                    vals = [
                        [
                            float("{0:.10g}".format(Q)),
                            float("{0:.10g}".format(Qi)),
                            float("{0:.10g}".format(Qc)),
                            float("{0:.10g}".format(Qc_Re)),
                            float("{0:.10g}".format(phi)),
                            float("{0:.10g}".format(f_c)),
                        ],
                        [
                            float("{0:.1g}".format(conf_array[0])),
                            float("{0:.1g}".format(conf_array[1])),
                            float("{0:.1g}".format(conf_array[2])),
                            float("{0:.1g}".format(conf_array[3])),
                            float("{0:.1g}".format(conf_array[4])),
                            float("{0:.1g}".format(conf_array[5])),
                        ],
                    ]
                writer.writerow(fields)
                writer.writerows(vals)
                file.close()
    except Exception as e:
        print(">Error when trying to write parameters on plot")
        print(f">{e}")
    #    quit()
    # Create plot metadata output file
    try:
        create_metadata(Method, output_path)
    except Exception as e:
        print(">Error when trying to create metadata file")
        print(f">{e}")
        # quit()
    return fig
