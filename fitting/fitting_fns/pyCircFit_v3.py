# 10/13/2017 [Chan U Lei], modified by Chris Wang
# Circle fitting package

import numpy as np
import matplotlib.pyplot as pl
import lmfit


# define function
def S11refl_v1(a, alpha, tau, fr, Qtot, Qc, f):
    return (
        a
        * np.exp(1j * (alpha + 2 * np.pi * f * tau))
        * (1 - 2 * (Qtot / Qc) / (1 - 2j * Qtot * (f / fr - 1)))
    )


def S11refl_v2(a, alpha, tau, fr, Qi, Qc, f):
    return (
        a
        * np.exp(1j * (alpha + 2 * np.pi * f * tau))
        * (Qc - Qi + 2j * Qc * Qi * (f / fr - 1))
        / (Qc + Qi + 2j * Qc * Qi * (f / fr - 1))
    )


def S21trans_v1(a, alpha, tau, fr, Qtot, Qc, f):
    return (
        a
        * np.exp(1j * (alpha + 2 * np.pi * f * tau))
        * 2
        * (Qtot / Qc)
        / (2j * Qtot * (f / fr - 1) - 1)
    )


def S21trans_v1_asym(a, alpha, tau, fr, Qtot, Qc, phi, f):
    return S21trans_v1(a, alpha, tau, fr, Qtot, Qc * np.exp(1j * phi), f)


def S21trans_v1_asym_offset(a, alpha, tau, fr, Qtot, Qc, phi, xoff, yoff, f):
    return (
        S21trans_v1(a, alpha, tau, fr, Qtot, Qc * np.exp(1j * phi), f)
        + xoff
        + 1j * yoff
    )


def S11refl_v1_asym(a, alpha, tau, fr, Qtot, Qc, phi, f):
    return S11refl_v1(a, alpha, tau, fr, Qtot, Qc * np.exp(1j * phi), f)


def S21hanger_v1(a, alpha, tau, fr, Qtot, Qc, f):
    return (
        a
        * np.exp(1j * (alpha + 2 * np.pi * f * tau))
        * (1 - (Qtot / Qc) / (1 - 2j * Qtot * (f / fr - 1)))
    )


def S21hanger_v1_asym(a, alpha, tau, fr, Qtot, Qc, phi, f):
    return S21hanger_v1(a, alpha, tau, fr, Qtot, Qc * np.exp(1j * phi), f)


# nbar calculation
def nbar_refl(Qtot, Qc, fr, PdBm):
    Pin = 1e-3 * 10.0 ** (PdBm / 10.0)
    return 4 * Qtot * Qtot / Qc * Pin / 1.054e-34 / ((fr * 2 * np.pi) ** 2)


def nbar_hanger(Qtot, Qc, fr, PdBm):
    Pin = 1e-3 * 10.0 ** (PdBm / 10.0)
    return 2 * Qtot * Qtot / Qc * Pin / 1.054e-34 / ((fr * 2 * np.pi) ** 2)


def SfigGen(fig, fontsize=14, conf=1, gridalpha=0.3):

    if conf == 1:
        figsize = [12, 5]
        fig.set_size_inches(figsize)

        ax1 = pl.subplot(131)
        ax1.set_xlabel("X", fontsize=fontsize)
        ax1.set_ylabel("Y", fontsize=fontsize)
        ax1.axis("equal")
        ax1.grid(True, linestyle="solid", alpha=gridalpha)

        ax2 = pl.subplot(232)
        ax2.set_ylabel("X", fontsize=fontsize)
        ax2.grid(True, linestyle="solid", alpha=gridalpha)

        ax3 = pl.subplot(235)
        ax3.set_xlabel("$f$ (Hz)", fontsize=fontsize)
        ax3.set_ylabel("Y", fontsize=fontsize)
        ax3.grid(True, linestyle="solid", alpha=gridalpha)

        ax4 = pl.subplot(233)
        # ax4.set_xlabel('$f$ (Hz)', fontsize=frontsize)
        ax4.set_ylabel("amp (dB)", fontsize=fontsize)
        ax4.grid(True, linestyle="solid", alpha=gridalpha)
        # ax4.set_yscale('log')

        ax5 = pl.subplot(236)
        ax5.set_xlabel("$f$ (Hz)", fontsize=fontsize)
        ax5.set_ylabel("ph (deg)", fontsize=fontsize)
        ax5.grid(True, linestyle="solid", alpha=gridalpha)
        # ax5.axis('tight')

        pl.tight_layout()

    elif conf == 2:
        figsize = [10, 6]
        fig.set_size_inches(figsize)

        ax1 = pl.subplot(211)
        ax1.set_ylabel("amp (dB)", fontsize=fontsize)
        ax1.grid(True, linestyle="solid", alpha=gridalpha)
        # ax1.set_yscale('log')

        ax2 = pl.subplot(212)
        ax2.set_xlabel("$f$ (Hz)", fontsize=fontsize)
        ax2.set_ylabel("ph (deg)", fontsize=fontsize)
        ax2.grid(True, linestyle="solid", alpha=gridalpha)
        # ax5.axis('tight')
    elif conf == 3:
        figsize = [8.944, 4]
        fig.set_size_inches(figsize)

        ax1 = pl.subplot(111)
        ax1.set_xlabel("$f-f_r$ (Hz)", fontsize=fontsize)
        ax1.set_ylabel("|S11| (dB)", fontsize=fontsize)
        ax1.grid(True, linestyle="solid", alpha=gridalpha)
        # ax1.set_yscale('log')

    pl.tight_layout()


def Splot(fig, f, X, Y, style=".", conf=1, **kwargs):
    if conf == 1:
        amp = np.sqrt(X**2 + Y**2)
        ampdBm = 20.0 * np.log10(amp)
        ph = np.unwrap(2.0 * np.arctan(Y / X), np.pi) / 2.0 * 180.0 / np.pi
        # ph = np.unwrap(np.arctan(Y/X),np.pi)

        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
        ax5 = fig.axes[4]

        ax1.plot(X, Y, style, **kwargs)
        ax2.plot(f, X, style, **kwargs)
        ax3.plot(f, Y, style, **kwargs)
        # ax4.plot(f, amp, style, *kwarg)
        ax4.plot(f, ampdBm, style, **kwargs)
        ax5.plot(f, ph, style, **kwargs)

        ax1.legend(loc="best")
        pl.tight_layout()
        return fig
    elif conf == 2:
        amp = np.sqrt(X**2 + Y**2)
        ampdBm = 20.0 * np.log10(amp)
        ph = np.unwrap(2.0 * np.arctan(Y / X), np.pi) / 2.0 * 180.0 / np.pi
        # ph = np.unwrap(np.arctan(Y/X),np.pi)

        ax1 = fig.axes[0]
        ax2 = fig.axes[1]

        ax1.plot(f, ampdBm, style, **kwargs)
        ax2.plot(f, ph, style, **kwargs)
        pl.tight_layout()

        ax1.legend(loc="best")
        return fig
    elif conf == 3:
        amp = np.sqrt(X**2 + Y**2)
        ampdBm = 20.0 * np.log10(amp)
        ph = np.unwrap(2.0 * np.arctan(Y / X), np.pi) / 2.0 * 180.0 / np.pi
        # ph = np.unwrap(np.arctan(Y/X),np.pi)
        ax1 = fig.axes[0]
        ax1.plot(f, ampdBm, style, **kwargs)
        pl.tight_layout()
        ax1.legend(loc="best")
        return fig


def circlefit(
    freqs,
    Xn,
    Yn,
    fittype="hanger",
    plotlabel="",
    fitQscl=5.0,
    Qc0=0.0,
    Qc0_stderr=0.0,
    show_plots=True,
    print_results=True,
    fano=False,
):
    tempdict = {}

    if print_results:
        print("Using circle fit to get initial fit parameters.")
        print("Fitting type = " + fittype)
    # prepare figure for circle fit
    if show_plots:
        # fig = pl.figure(fittype+' circle fit '+plotlabel)
        fig = pl.figure()
        fig.set_size_inches([12, 8])
        # plot IQ circle
        ax01 = pl.subplot(221)
        ax01.set_title("IQ data")
        ax01.set_xlabel("I")
        ax01.set_ylabel("Q")
        ax01.axhline(y=0, color="k", linewidth=0.5)
        # ax01.axvline(x=0, color="k", linewidth=0.5)
        ax01.axis("equal")
        ax01.grid("on")
        # plot IQ circle residual
        ax03 = pl.subplot(223)
        ax03.set_title("circle fit residual")
        ax03.set_xlabel("freq (Hz)")
        ax03.set_ylabel("Delta r**2")
        # ax03.axhline(y=0, color='k', linewidth = 0.5)
        # ax03.axvline(x=0, color='k', linewidth = 0.5)
        # ax03.grid('on')
        # plot theta
        ax02 = pl.subplot(222)
        ax02.set_title("phase of shifted data")
        ax02.set_xlabel("freq")
        ax02.set_ylabel("phase (rad)")
        ax02.grid("on")
        # plot theta residual
        ax04 = pl.subplot(224)
        ax04.set_title("phase fit residual")
        ax04.set_xlabel("freq (Hz)")
        ax04.set_ylabel("Delta theta")
        # styling
        # pl.tight_layout()
        # plot raw data
        ax01.plot(Xn, Yn, "k.", linewidth=1.0, markersize=2.0, alpha=0.5, label="raw")

    def circle_fitresidual(params, Idata, Qdata):
        xc = params["xc"]
        yc = params["yc"]
        rc = params["rc"]
        return (Idata - xc) ** 2 + (Qdata - yc) ** 2 - rc**2

    # guess parameters
    Na_guess = np.where(Xn == max(Xn))
    za_guess = Xn[Na_guess] + 1j * Yn[Na_guess]
    Nb_guess = np.where(Xn == min(Xn))
    zb_guess = Xn[Nb_guess] + 1j * Yn[Nb_guess]
    zc_guess = za_guess + (zb_guess - za_guess) / 2
    if show_plots:
        ax01.plot(
            np.real(za_guess),
            np.imag(za_guess),
            "gs",
            linewidth=1.0,
            markersize=8.0,
            alpha=0.5,
            label="guess",
        )
        ax01.plot(
            np.real(zb_guess),
            np.imag(zb_guess),
            "gs",
            linewidth=1.0,
            markersize=8.0,
            alpha=0.5,
            label="guess",
        )
    rc_guess = np.abs(za_guess - zb_guess) / 2
    xc_guess = np.real(zc_guess)
    yc_guess = np.imag(zc_guess)
    # plot guess circle
    xx = np.linspace(-np.pi, np.pi, 1000)
    Xg = xc_guess + rc_guess * np.cos(xx)
    Yg = yc_guess + rc_guess * np.sin(xx)
    if show_plots:
        ax01.plot(Xg, Yg, "g-", linewidth=1.0, markersize=2.0, alpha=0.5, label="guess")

    # set fit parameters
    params = lmfit.Parameters()
    params.add("xc", xc_guess, vary=1)
    params.add("yc", yc_guess, vary=1)
    params.add("rc", rc_guess, min=0.0, vary=1)

    # perform fitting
    fitresults = lmfit.minimize(circle_fitresidual, params, args=(Xn, Yn))
    if print_results:
        print(fitresults)
    # print(lmfit.fit_report(fitresults))

    # get fit result
    xc_fit = fitresults.params.get("xc").value
    xc_fit_stderr = fitresults.params.get("xc").stderr
    yc_fit = fitresults.params.get("yc").value
    yc_fit_stderr = fitresults.params.get("yc").stderr
    rc_fit = fitresults.params.get("rc").value
    rc_fit_stderr = fitresults.params.get("rc").stderr

    # plot fit
    Xfit = xc_fit + rc_fit * np.cos(xx)
    Yfit = yc_fit + rc_fit * np.sin(xx)
    if show_plots:
        ax01.plot(
            Xfit, Yfit, "r-", linewidth=1.0, markersize=2.0, alpha=0.5, label="fit"
        )

        # plot residual
        ax03.plot(
            freqs,
            fitresults.residual,
            "-",
            linewidth=1.0,
            markersize=2.0,
            alpha=0.8,
            label="residual",
        )
        pl.tight_layout()

    # store them in dictionary
    tempdict["xc"] = xc_fit
    tempdict["xc_stderr"] = xc_fit_stderr
    tempdict["yc"] = yc_fit
    tempdict["yc_stderr"] = yc_fit_stderr
    tempdict["rc"] = rc_fit
    tempdict["rc_stderr"] = rc_fit_stderr

    # shifted data to get theta0
    Xs = Xn - xc_fit
    Ys = Yn - yc_fit
    theta = np.unwrap(np.angle(Xs + 1j * Ys))
    # if show_plots:
    #     ax01.plot(
    #         Xs, Ys, "b.", linewidth=1.0, markersize=2.0, alpha=0.5, label="shifted"
    #     )
    #     # plot the phase of the shifted data
    #     ax02.plot(freqs, theta, "b.", linewidth=1.0, markersize=2.0, alpha=0.5)

    # fit the phase of the shifted data to get theta0
    def phase_fitresidual(params, f, data=None, eps_data=None):
        theta0 = params["theta0"]
        Qtot = params["Qtot"]
        fr = params["fr"]
        model = theta0 - 2 * np.arctan(2 * Qtot * (1 - f / fr))
        if data is None:
            return model
        if eps_data is None:
            return data - model
        return (data - model) / eps_data

    # guess parameters
    theta0c_guess = np.mean(theta)
    Qtotc_guess = fitQscl * np.mean(freqs) / np.abs(freqs[-1] - freqs[0])
    frc_guess = np.mean(freqs)

    # set fit parameters
    phase_params = lmfit.Parameters()
    phase_params.add("theta0", theta0c_guess, vary=1)
    phase_params.add("Qtot", Qtotc_guess, vary=1)
    phase_params.add("fr", frc_guess, vary=1)

    # plot guess phase
    ff = np.linspace(freqs[0], freqs[-1], 1000)
    theta_guess = phase_fitresidual(phase_params, ff)
    if show_plots:
        ax02.plot(ff, theta_guess, "g-", linewidth=1.0, markersize=2.0, alpha=0.5)

    # perform fitting
    phase_fitresults = lmfit.minimize(
        phase_fitresidual, phase_params, args=(freqs, theta)
    )
    if print_results:
        print(lmfit.fit_report(phase_fitresults))

    # get fit result
    theta0c_fit = phase_fitresults.params.get("theta0").value
    theta0c_fit_stderr = phase_fitresults.params.get("theta0").stderr
    Qtotc_fit = phase_fitresults.params.get("Qtot").value
    Qtotc_fit_stderr = phase_fitresults.params.get("Qtot").stderr
    frc_fit = phase_fitresults.params.get("fr").value
    frc_fit_stderr = phase_fitresults.params.get("fr").stderr

    # plot fit
    theta_fit = phase_fitresidual(phase_fitresults.params, ff)
    if show_plots:
        ax02.plot(ff, theta_fit, "r-", linewidth=1.0, markersize=2.0, alpha=0.5)

        # plot residual
        ax04.plot(
            freqs,
            phase_fitresults.residual,
            "-",
            linewidth=1.0,
            markersize=2.0,
            alpha=0.8,
            label="residual",
        )

    # store them in dictionary
    tempdict["theta0"] = theta0c_fit
    tempdict["theta0_stderr"] = theta0c_fit_stderr
    tempdict["Qtot"] = Qtotc_fit
    tempdict["Qtot_stderr"] = Qtotc_fit_stderr
    tempdict["fr"] = frc_fit
    tempdict["fr_stderr"] = frc_fit_stderr

    # calculate and plot on & off resonance point in shifted data
    beta = theta0c_fit + np.pi
    beta_stderr = theta0c_fit_stderr
    xoffrs = rc_fit * np.cos(beta)
    xoffrs_stderr = np.sqrt(
        (np.cos(beta) * rc_fit_stderr) ** 2 + (rc_fit * np.sin(beta) * beta_stderr) ** 2
    )
    yoffrs = rc_fit * np.sin(beta)
    yoffrs_stderr = np.sqrt(
        (np.sin(beta) * rc_fit_stderr) ** 2 + (rc_fit * np.cos(beta) * beta_stderr) ** 2
    )
    xonrs = -rc_fit * np.cos(beta)
    # xonrs_stderr = np.sqrt((np.cos(beta)*rc_fit_stderr)**2 + (rc_fit*np.sin(beta)*beta_stderr)**2)
    yonrs = -rc_fit * np.sin(beta)
    # yonrs_stderr = np.sqrt((np.sin(beta)*rc_fit_stderr)**2 + (rc_fit*np.cos(beta)*beta_stderr)**2)

    # if show_plots:
    #     ax01.plot(xoffrs, yoffrs, "bx", linewidth=4.0, markersize=8.0, alpha=1.0)
    #     ax01.plot(xonrs, yonrs, "bo", linewidth=4.0, markersize=8.0, alpha=1.0)

    # calculate and plot on & off resonance point in raw data
    xoffr = xc_fit + xoffrs
    xoffr_stderr = np.sqrt((xc_fit_stderr) ** 2 + (xoffrs_stderr) ** 2)
    yoffr = yc_fit + yoffrs
    yoffr_stderr = np.sqrt((yc_fit_stderr) ** 2 + (yoffrs_stderr) ** 2)
    doffr = np.sqrt(xoffr**2 + yoffr**2)  # ***
    doffr_stderr = np.sqrt(
        (xoffr * xoffr_stderr / doffr) ** 2 + (yoffr * yoffr_stderr / doffr) ** 2
    )
    xonr = xc_fit + xonrs
    # xonr_stderr = np.sqrt((xc_fit_stderr)**2 + (xonrs_stderr)**2)
    yonr = yc_fit + yonrs
    # yonr_stderr = np.sqrt((yc_fit_stderr)**2 + (yonrs_stderr)**2)
    if show_plots:
        ax01.plot(
            xoffr,
            yoffr,
            "kx",
            linewidth=4.0,
            markersize=8.0,
            alpha=1.0,
            label="off resonance point",
        )
        ax01.plot(
            xonr,
            yonr,
            "ko",
            linewidth=4.0,
            markersize=8.0,
            alpha=1.0,
            label="on resonance point",
        )

    # rotate alpha
    alphac_fit = np.angle(xoffr + 1j * yoffr)
    alphac_fit_stderr = np.sqrt(
        (yoffr_stderr / xoffr) ** 2 + (yoffr * xoffr_stderr / xoffr / xoffr) ** 2
    ) / (1 + (yoffr / xoffr) ** 2)
    tempdict["alpha"] = alphac_fit
    tempdict["alpha_stderr"] = alphac_fit_stderr
    Xr1 = Xn * np.cos(alphac_fit) + Yn * np.sin(alphac_fit)
    Yr1 = Yn * np.cos(alphac_fit) - Xn * np.sin(alphac_fit)
    xoffr1 = xoffr * np.cos(alphac_fit) + yoffr * np.sin(alphac_fit)
    yoffr1 = yoffr * np.cos(alphac_fit) - xoffr * np.sin(alphac_fit)
    xonr1 = xonr * np.cos(alphac_fit) + yonr * np.sin(alphac_fit)
    yonr1 = yonr * np.cos(alphac_fit) - xonr * np.sin(alphac_fit)
    # if show_plots:
    #     ax01.plot(
    #         Xr1,
    #         Yr1,
    #         "m.",
    #         linewidth=1.0,
    #         markersize=2.0,
    #         alpha=0.5,
    #         label="rotate alpha",
    #     )
    #     ax01.plot(xoffr1, yoffr1, "mx", linewidth=4.0, markersize=8.0, alpha=1.0)
    #     ax01.plot(xonr1, yonr1, "mo", linewidth=4.0, markersize=8.0, alpha=1.0)
    #     ax01.grid("on")
    #     ax01.legend(loc="best")
    # calculate ac
    ac_fit = doffr
    ac_fit_stderr = doffr_stderr
    tempdict["a"] = ac_fit
    tempdict["a_stderr"] = ac_fit_stderr

    # shift to origin to get phi
    Xs1 = Xr1 - xoffr1
    Ys1 = Yr1 - yoffr1
    xoffr1s = 0.0
    yoffr1s = 0.0

    # xonr1s = xonr1 - xoffr1 #
    xonr1s = -2 * rc_fit * np.cos(beta - alphac_fit)
    xonr1s_stderr = 2.0 * np.sqrt(
        (np.cos(beta - alphac_fit) * rc_fit_stderr) ** 2
        + (rc_fit * np.sin(beta - alphac_fit) * beta_stderr) ** 2
        + (rc_fit * np.sin(beta - alphac_fit) * alphac_fit_stderr) ** 2
    )
    # yonr1s = yonr1 - yoffr1
    yonr1s = -2 * rc_fit * np.sin(beta - alphac_fit)
    yonr1s_stderr = 2.0 * np.sqrt(
        (np.sin(beta - alphac_fit) * rc_fit_stderr) ** 2
        + (rc_fit * np.cos(beta - alphac_fit) * beta_stderr) ** 2
        + (rc_fit * np.cos(beta - alphac_fit) * alphac_fit_stderr) ** 2
    )

    phic_fit = np.pi - np.angle(xonr1s + 1j * yonr1s)
    phic_fit_stderr = np.sqrt(
        (yonr1s_stderr / xonr1s) ** 2 + (yonr1s * xonr1s_stderr / xonr1s / xonr1s) ** 2
    ) / (1 + (yonr1s / xonr1s) ** 2)

    tempdict["phi"] = phic_fit
    tempdict["phi_stderr"] = phic_fit_stderr
    # if show_plots:
    #     ax01.plot(
    #         Xs1,
    #         Ys1,
    #         "c.",
    #         linewidth=1.0,
    #         markersize=2.0,
    #         alpha=0.5,
    #         label="rotate alpha shifted",
    #     )
    #     ax01.plot(xoffr1s, yoffr1s, "cx", linewidth=4.0, markersize=8.0, alpha=1.0)
    #     ax01.plot(xonr1s, yonr1s, "co", linewidth=4.0, markersize=8.0, alpha=1.0)
    #     ax01.grid("on")
    #     ax01.legend(loc="best")

    if fittype == "hanger":
        Qcc_fit = ac_fit / 2 / rc_fit * Qtotc_fit  # hanger
        Qcc_fit_stderr = np.sqrt(
            (Qcc_fit / ac_fit * ac_fit_stderr) ** 2
            + (Qcc_fit / rc_fit * rc_fit_stderr) ** 2
            + (Qcc_fit / Qtotc_fit * Qtotc_fit_stderr) ** 2
        )

    if fittype == "reflection":  # need to be checked
        Qcc_fit = ac_fit / rc_fit * Qtotc_fit  # hanger
        Qcc_fit_stderr = np.sqrt(
            (Qcc_fit / ac_fit * ac_fit_stderr) ** 2
            + (Qcc_fit / rc_fit * rc_fit_stderr) ** 2
            + (Qcc_fit / Qtotc_fit * Qtotc_fit_stderr) ** 2
        )

    if fittype == "transmission":
        Qcc_fit = ac_fit / rc_fit * Qtotc_fit
        Qcc_fit_stderr = np.sqrt(
            (Qcc_fit / ac_fit * ac_fit_stderr) ** 2
            + (Qcc_fit / rc_fit * rc_fit_stderr) ** 2
            + (Qcc_fit / Qtotc_fit * Qtotc_fit_stderr) ** 2
        )

    tempdict["Qc"] = Qcc_fit
    tempdict["Qc_stderr"] = Qcc_fit_stderr
    if print_results:
        print("Qc = " + str(Qcc_fit) + " +- " + str(Qcc_fit_stderr))

    Qcc2_fit = Qcc_fit / np.cos(phic_fit)
    Qcc2_fit_stderr = np.sqrt(
        (Qcc_fit_stderr / np.cos(phic_fit)) ** 2
        + (Qcc_fit * np.tan(phic_fit) / np.cos(phic_fit) * phic_fit_stderr) ** 2
    )
    tempdict["Qc2"] = Qcc2_fit
    tempdict["Qc2_stderr"] = Qcc2_fit_stderr

    def circle_offset_fitresidual(params, Idata, Qdata):
        xc = params["x_off"]
        yc = params["y_off"]
        return (Idata - xc) ** 2 + (Qdata - yc) ** 2

    # include extra rotation due to fano interference
    if fano:
        xoff_guess = 0.01
        yoff_guess = -0.01
        # set fit parameters
        offset_params = lmfit.Parameters()
        offset_params.add("x_off", xoff_guess, vary=1)
        offset_params.add("y_off", yoff_guess, vary=1)

        cfitdata = S21trans_v1_asym(
            tempdict["a"],
            tempdict["alpha"],
            0,
            tempdict["fr"],
            tempdict["Qtot"],
            tempdict["Qc"],
            tempdict["phi"],
            freqs,
        )

        # perform fitting
        offset_fitresults = lmfit.minimize(
            circle_offset_fitresidual,
            offset_params,
            args=(Xn - np.real(cfitdata), Yn - np.imag(cfitdata)),
        )
        if print_results:
            print(lmfit.fit_report(offset_fitresults))
        tempdict["x_off"] = offset_fitresults.params["x_off"]
        tempdict["y_off"] = offset_fitresults.params["y_off"]

    # this definition can lead to diverge result, overestimate the error, don't use it.
    # Qic_fit = Qcc2_fit*Qtotc_fit/(Qcc2_fit-Qtotc_fit)
    # Qic_fit_stderr = np.sqrt((Qcc2_fit_stderr*(Qic_fit/Qcc2_fit)**2)**2+(Qtotc_fit_stderr*(Qic_fit/Qtotc_fit)**2)**2)

    QicInv_fit = 1 / Qtotc_fit - np.cos(phic_fit) / Qcc_fit
    QicInv_fit_stderr = np.sqrt(
        (Qtotc_fit_stderr / Qtotc_fit / Qtotc_fit) ** 2
        + (phic_fit_stderr * np.sin(phic_fit) / Qcc_fit) ** 2
        + (Qcc_fit_stderr * np.cos(phic_fit) / Qcc_fit / Qcc_fit) ** 2
    )
    Qic_fit = 1 / QicInv_fit
    Qic_fit_stderr = QicInv_fit_stderr * Qic_fit**2

    tempdict["Qi"] = Qic_fit
    tempdict["Qi_stderr"] = Qic_fit_stderr
    ax01.legend(loc="best")
    if Qc0 > 0.0:  # use the given Qc0 to calculate Qint
        Qi0Inv_fit = 1 / Qtotc_fit - 1 / Qc0
        Qi0Inv_fit_stderr = np.sqrt(
            (Qtotc_fit_stderr / Qtotc_fit / Qtotc_fit) ** 2
            + (Qc0_stderr / Qc0 / Qc0) ** 2
        )
        Qi0_fit = 1 / Qi0Inv_fit
        Qi0_fit_stderr = Qi0Inv_fit_stderr * Qi0_fit**2
        tempdict["Qi0"] = Qi0_fit
        tempdict["Qi0_stderr"] = Qi0_fit_stderr

    # print phase fit results
    if print_results:
        print("\ncircle fit results:")
        print("xc = %6.3e +- %6.3e" % (xc_fit, xc_fit_stderr))
        print("yc = %6.3e +- %6.3e" % (yc_fit, yc_fit_stderr))
        print("rc = %6.3e +- %6.3e" % (rc_fit, rc_fit_stderr))
        print("\ntheta fit results:")
        print("theta0 = %6.3e +- %6.3e" % (theta0c_fit, theta0c_fit_stderr))
        print("Qtot   = %6.3e +- %6.3e" % (Qtotc_fit, Qtotc_fit_stderr))
        print("fr     = %6.9e +- %6.3e" % (frc_fit, frc_fit_stderr))
        print("\nderived results:")
        print("alpha = %6.3e +- %6.3e" % (alphac_fit, alphac_fit_stderr))
        print("    a = %6.3e +- %6.3e" % (ac_fit, ac_fit_stderr))
        print("   Qc = %6.3e +- %6.3e" % (Qcc_fit, Qcc_fit_stderr))
        print("  phi = %6.3e +- %6.3e" % (phic_fit, phic_fit_stderr))
        print("  Qc2 = %6.3e +- %6.3e" % (Qcc2_fit, Qcc2_fit_stderr))
        print("   Qi = %6.3e +- %6.3e" % (Qic_fit, Qic_fit_stderr))

    return tempdict
