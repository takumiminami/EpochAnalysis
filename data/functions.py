#!/usr/bin/env python3
# -*-coding:utf-8-*-

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from matplotlib.colors import LogNorm, Normalize
import sdf_helper as sh
import copy, os, gc, contextlib, sys, glob, re, warnings
from numba import njit, prange


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["legend.frameon"] = False
plt.rcParams["pcolor.shading"] = "auto"#"gouraud"





# ----- functions ----- #
# TODO: make a class to wrap all of these functions

def exclude_edge(_data, _particle, scaler_):
    #    exclude_width = 1e-6
    particle_x = eval("_data.Grid_Particles_{}.data[0]".format(_particle))
    particle_y = eval("_data.Grid_Particles_{}.data[1]".format(_particle))
    x_ = _data.Grid_Grid.data[0]
    y_ = _data.Grid_Grid.data[1]
    x_max = np.max(x_) * 0.95   # - exclude_width
    x_min = np.min(x_) * 0.95   # + exclude_width
    y_max = np.max(y_) * 0.95   # - exclude_width
    y_min = np.min(y_) * 0.95   # + exclude_width

    index_ = (particle_x > x_min) & (particle_x < x_max) & (particle_y > y_min) & (particle_y < y_max)
    if np.sum(index_) == 0:
        return 0
    else:
        return scaler_[index_]


def calc_direction(_data, _particle):
    px = eval("_data.Particles_Px_{}.data".format(_particle))
    py = eval("_data.Particles_Py_{}.data".format(_particle))

    angle = np.arctan2(py, px, dtype="float64")
    return angle


def calc_direction_woedge(_data, _particle):
    angle = calc_direction(_data, _particle)
    angle_woedge = exclude_edge(_data, _particle, angle)
    return angle_woedge


@njit('f8[:,:](f8[:], f8[:], i8, f8[:], i8, f8)', parallel=True)
def calc_ek_theta(_ek, angle__, nbin_, bins_theta_, rmin_, rmax_):
    _data = np.empty((nbin_, nbin_))
    for n in prange(nbin_):
        pos = (angle__ > bins_theta_[n]) & (angle__ < bins_theta_[n + 1])
        _data[:, n], bins_ = np.histogram(_ek[pos], bins=nbin_, range=(rmin_, rmax_))
    return _data


@njit('f8[:,:](f8[:], f8[:], i8, f8[:], f8, f8)', parallel=True)
def calc_eklog_theta(_ek, angle__, nbin_, bins_theta_, rminlog_, rmaxlog_):
    _data = np.empty((nbin_, nbin_))
    for n in prange(nbin_):
        pos = (angle__ > bins_theta_[n]) & (angle__ < bins_theta_[n + 1])
        _data[:, n], bins_ = np.histogram(log10(_ek[pos]), bins=nbin_, range=(rminlog_, rmaxlog_))
    return _data


def energy_direction(_ek, angle_, _save_name, _label):
    ek_theta = calc_ek_theta(_ek, np.rad2deg(angle_), nbin, bins_theta, rmin, rmax)
    np.save(dir_field + "/{}ekth_{}{}.npy".format(header_, _save_name, fname), ek_theta,)
    eklog_theta = calc_eklog_theta(_ek, np.rad2deg(angle_), nbin, bins_theta, rminlog, rmaxlog)
    np.save(dir_field + "/{}eklogth_{}{}.npy".format(header_, _save_name, fname), eklog_theta)

    # --- linear
    fig, ax = plt.subplots()
    if np.average(ek_theta) == 0:
        frame = ax.pcolormesh(ek_label, theta, ek_theta.T)
    else:
        frame = ax.pcolormesh(ek_label, theta, ek_theta.T, norm=LogNorm())
    cbar = fig.colorbar(frame)
    cbar.set_label("# of " + _label)
    ax.set_xlabel(r'$\varepsilon$ [eV]')
    ax.set_ylabel(r'$\theta$ [degrees]')
    ax.set_yticks([-180, -90, 0, 90, 180])

    ax.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
    fig.tight_layout()
    fig.savefig(dir_fig + '/{}ekth_{}{}.png'.format(header_, _save_name, fname))
    print("plotting ek-theta of {} has been done".format(_label))
    ax.cla()
    plt.close()

    # --- log
    fig, ax = plt.subplots()
    if np.average(eklog_theta) == 0:
        frame = ax.pcolormesh(ek_labellog, theta, eklog_theta.T)
    else:
        frame = ax.pcolormesh(ek_labellog, theta, eklog_theta.T, norm=LogNorm())
    cbar = fig.colorbar(frame)
    cbar.set_label("# of " + _label)
    ax.set_xlabel(r'$\varepsilon$ [eV]')
    ax.set_ylabel(r'$\theta$ [degrees]')
    ax.set_xscale("log")

    #    x_ticks = np.linspace(rminlog, rmaxlog - 1, 6)
    #    x_ticklabels = [r"$10^{}$".format(str(txt)[0]) for txt in x_ticks if txt < 10]
    #    x_ticklabels.append(r"$10^{0}$$^{1}$".format("1", "0"))
    #    ax.set_xticks(x_ticks)
    #    ax.set_xticklabels(x_ticklabels)
    ax.set_yticks([-180, -90, 0, 90, 180])

    ax.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
    fig.tight_layout()
    fig.savefig(dir_fig + '/{}eklogth_{}{}.png'.format(header_, _save_name, fname))
    print("plotting eklog-theta of {} has been done".format(_label))
    ax.cla()
    fig.clf()
    plt.close()


class PlotField:
    """
    plotting overlayed field variables (ion, electron densities)
    """
    def __init__(self, time_):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax.set_ylabel(r'$y\ [\mathrm{\mu m}]$')
        self.ax.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time_), xy=(0.05, 1.05), xycoords='axes fraction')

    def plot_field(self, _field, x_, y_, norm_, _label, color_):
        frame_ = self.ax.pcolormesh(x_, y_, _field.T, norm=norm_, cmap=color_, alpha=0.5)
        # cbar_ = self.fig.colorbar(frame_)
        # cbar_.set_label(_label)

    def save_image(self, _save_name, fname_):
        self.fig.tight_layout()
        self.fig.savefig(dir_fig + '/{}'.format(_save_name) + fname_ + '.png')
        print("plotting {} has been done".format(_save_name))

    def __del__(self):
        self.ax.cla()
        self.fig.clf()
        plt.close()


def plot_field(_field: np.ndarray, norm_: object, _label: str, _save_name: str, color_="viridis"):
    """
    plotting field variables (ex, ey, etc.)
    :param _field: data to plot
    :param norm_: color bar scale (min, max)
    :param _label: label of color bar
    :param _save_name: file name to save
    :param color_: (option, default is "viridis") color map of 2d plot
    :return:
    """
    fig_, ax_ = plt.subplots()
    frame_ = ax_.pcolormesh(x, y, _field.T, norm=norm_, cmap=color_)
    cbar_ = fig_.colorbar(frame_)
    cbar_.set_label(_label)
    ax_.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
    ax_.set_ylabel(r'$y\ [\mathrm{\mu m}]$')
    ax_.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
    fig_.tight_layout()
    if fig_flag:
        fig_.savefig(dir_fig + '/{}'.format(_save_name) + fname + '.png')
    if field_npy:
        np.save(dir_field + "/{}{}.npy".format(_save_name, fname), _field)
    print("plotting {} has been done".format(_save_name))
    ax_.cla()
    fig_.clf()
    plt.close()


def plot_field_onaxis(_field, _y=0, _ave_range=0, _label="field", _save_name="field", log_flag: bool=False):
    """
    Used for plotting grid variables on X-axis (e.g. Density, Electric field)
    """
    #field_on_axis = _field[:, x_axis_pos]
    field_on_axis = extract_x_axis_pos(_field, _y, _ave_range)
    np.savetxt(dir_field + "/on_axis_{}{}.txt".format(_save_name, fname), np.array((x, field_on_axis)).T, header="x [um]  {}".format(_label))

    fig_, ax_ = plt.subplots()
    ax_.plot(x, field_on_axis, lw=1)
    ax_.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
    ax_.set_ylabel(_label)
    ax_.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.15, 1.05), xycoords='axes fraction')
    if fig_flag:
        fig_.tight_layout()
        fig_.savefig(dir_fig + '/on_axis_{}'.format(_save_name) + fname + '.png')
        if log_flag:
            ax_.set_yscale("log")
            fig_.tight_layout()
            fig_.savefig(dir_fig + '/on_axis_log_{}'.format(_save_name) + fname + '.png')
    print("plotting {} ON AXIS has been done".format(_save_name))
    ax_.cla()
    fig_.clf()
    plt.close()


def extract_x_axis_pos(_field, _y=0, _ave_range: float=0):
    """
    function to extract field values on or around x-axis
    """
    if type(_y)==int:
        _result = _field[:, x_axis_pos]
    else:
        if _ave_range == 0:
            _x_axis_pos = np.argmin(np.abs(_y))
            _result = _field[:, _x_axis_pos]
        else:
            _index = np.abs(_y) < _ave_range
            _result = np.average(_field[:, _index], axis=1)
    return _result


def def_cbmax(_field):
    fmax = np.abs(_field.max())
    fmin = np.abs(_field.min())
    if fmax < fmin:
        return fmin
    else:
        return fmax


def define_mass(_particle):
    if re.findall("proton", _particle):
        mass_ = mass_u * 1.0073
    elif re.findall("carbon", _particle):
        mass_ = mass_u * 12.011
    elif re.findall("gold", _particle):
        mass_ = mass_u * 196.97
    elif re.findall("oxygen", _particle):
        mass_ = mass_u * 15.999
    elif re.findall("electron", _particle):
        mass_ = 9.1094e-31
    else:
        raise Exception("Undefined such particle: {}".format(_particle))

    return mass_


def calc_ek(_data, _particle: str) -> np.ndarray:
    """
    to obtain kinetic energies of particles
    :param _data:
    :param _particle:
    :return: particle energy [eV]
    """
    mass = define_mass(_particle)

    mc2 = mass * c ** 2
    px = eval("_data.Particles_Px_{}.data".format(_particle))
    py = eval("_data.Particles_Py_{}.data".format(_particle))
    pz = eval("_data.Particles_Pz_{}.data".format(_particle))
    pp2 = px ** 2 + py ** 2 + pz ** 2
    energy = np.sqrt(mc2 ** 2 + c ** 2 * pp2)

    return (energy - mc2) / q


def calc_ek_on_axis(_data, _particle):
    """
    to obtain kinetic energies of particles only around X-axis
    :param _data:
    :param _particle:
    :return:
    """
    _ek = calc_ek(_data, _particle)
    y_ = eval("_data.Grid_Particles_{}.data[1]".format(_particle))
    index_ = np.abs(y_) < dec_width
    if on_axis_flag:
        if np.sum(index_) == 0:
            return 0
        else:
            return _ek[index_]
    else:
        return _ek


def calc_ek_without_edge(_data, _particle):
    """
    to obtain kinetic energies of particles excluding around simulation boundaries
    :param _data:
    :param _particle:
    :return:
    """
    _ek = calc_ek(_data, _particle)
    x_ = eval("_data.Grid_Particles_{}.data[0]".format(_particle))
    y_ = eval("_data.Grid_Particles_{}.data[1]".format(_particle))

    x_grid = _data.Grid_Grid.data[0]
    y_grid = _data.Grid_Grid.data[1]
    x_grid_max = np.max(x_grid) * 0.95
    x_grid_mim = np.min(x_grid) * 0.95
    y_grid_max = np.max(y_grid) * 0.95
    y_grid_mim = np.min(y_grid) * 0.95

    x_index_ = (x_ > x_grid_mim) & (x_ < x_grid_max)
    y_index_ = (y_ > y_grid_mim) & (y_ < y_grid_max)
    index_ = x_index_ & y_index_

    if np.sum(index_) == 0:
        return 0
    else:
        return _ek[index_]


def calc_average_of_topXpercent(_ek: np.ndarray):
    """
    to obtain the average of kinetic energies of top X % ions
    :param _ek:
    :return:
    """
    length_ = int(len(_ek)*topXpercent/100)
    sort_ek_ = np.sort(_ek)
    topXp_ = sort_ek_[-length_:]
    return np.average(topXp_), np.std(topXp_)


class XpxOnAxisMulti:
    """
    plotting X-Px with Ex on X-axis
    """
    def __init__(self, _data, _ave_range):
        self.data = _data
        self.fig_ph, self.ax_ph = plt.subplots()
        self.fig_ph_ey, self.ax_ph_ey = plt.subplots()
        self.init_plot(_ave_range)

    def init_plot(self, _ave_range=0):
        x_grid_max = np.max(eval("self.data.Grid_Grid.data[0]")) * 1e6
        x_grid_min = np.min(eval("self.data.Grid_Grid.data[0]")) * 1e6

        _x_mid = eval("self.data.Grid_Grid_mid.data[0]")*1e6
        _y_mid = eval("self.data.Grid_Grid_mid.data[1]")*1e6

        #        x_axis_pos_ = np.argmin(np.abs(_y_mid))
        #        ex_center_ = eval("self.data.Electric_Field_Ex.data")[:, x_axis_pos_]
        #        ey_center_ = eval("self.data.Electric_Field_Ey.data")[:, x_axis_pos_]
        _ex = self.data.Electric_Field_Ex.data
        _ey = self.data.Electric_Field_Ey.data
        ex_center_ = extract_x_axis_pos(_ex, _y=_y_mid*1e-6, _ave_range=_ave_range)
        ey_center_ = extract_x_axis_pos(_ey, _y=_y_mid*1e-6, _ave_range=_ave_range)
        cb_exoa_ = def_cbmax(ex_center_) * 1.05
        cb_eyoa_ = def_cbmax(ey_center_) * 1.05
        #        cb_eoa = np.maximum(cb_exoa_, cb_eyoa_)

        self.ax_ph.set_xlim(x_grid_min, x_grid_max)
        self.ax_ph.set_ylim(-px_max, px_max)
        self.ax_ph.vlines(0, -px_max, px_max, linestyle=":", lw=1, color="black")
        self.ax_ph.hlines(0, x_grid_min, x_grid_max, linestyle=":", lw=1, color="black")
        self.ax_ph.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax_ph.set_ylabel(r"$p_x/mc$")
        self.ax_ph.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_ph2 = self.ax_ph.twinx()
        self.ax_ph2.plot(_x_mid, ex_center_, ls="-", lw=1, color="lightsteelblue", alpha=0.7)
        if cb_exoa_ > 0:
            self.ax_ph2.set_ylim(-cb_exoa_, cb_exoa_)
        self.ax_ph2.set_ylabel(r"$E_{x}$ [Vm$^{-1}$]")
        #        self.ax_ph2.legend()

        self.ax_ph_ey.set_xlim(x_grid_min, x_grid_max)
        self.ax_ph_ey.set_ylim(-px_max, px_max)
        self.ax_ph_ey.vlines(0, -px_max, px_max, linestyle=":", lw=1, color="black")
        self.ax_ph_ey.hlines(0, x_grid_min, x_grid_max, linestyle=":", lw=1, color="black")
        self.ax_ph_ey.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax_ph_ey.set_ylabel(r"$p_x/mc$")
        self.ax_ph_ey.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_ph2_ey = self.ax_ph_ey.twinx()
        self.ax_ph2_ey.plot(_x_mid, ey_center_, ls=":", lw=1, color="mediumseagreen", alpha=0.7)
        if cb_eyoa_ > 0:
            self.ax_ph2_ey.set_ylim(-cb_eyoa_, cb_eyoa_)
        self.ax_ph2_ey.set_ylabel(r"$E_{y}$ [Vm$^{-1}$]")
        #        self.ax_ph2_ey.legend()

        exy_save = np.zeros((len(_x_mid), 3))
        exy_save[:, 0] = _x_mid
        exy_save[:, 1] = ex_center_
        exy_save[:, 2] = ey_center_
        header__ = "x [um]  ex [V/m]  ey [V/m]"
        np.savetxt(dir_field + "/exy_center_{}.txt".format(fname), exy_save, header=header__)

    def calc_xpx_on_axis(self, _particle, order__):
        mass = define_mass(_particle)
        try:
            px_ = eval("self.data.Particles_Px_{}.data".format(_particle)) / mass / c
            x_ = eval("self.data.Grid_Particles_{}.data[0]".format(_particle)) * 1e6
            y_ = eval("self.data.Grid_Particles_{}.data[1]".format(_particle))
            index_ = np.abs(y_) < dec_width
            if np.sum(index_) == 0:
                dec_px = []
                dec_x = []
            else:
                dec_px = px_[index_]
                dec_x = x_[index_]
        except AttributeError:
            dec_px = []
            dec_x = []

        self.ax_ph.scatter(dec_x, dec_px, s=1, label=order__)
        self.ax_ph_ey.scatter(dec_x, dec_px, s=1, label=order__)

        header__ = "x [um]  px [mc]"
        np.savetxt(dir_phase + "/px_center_{}_{}.txt".format(_particle, fname), np.array((dec_x, dec_px)).T, header=header__, fmt="%.8e")

    def save_plot(self, _save_name):
        self.ax_ph.legend(fontsize=15)
        self.fig_ph.tight_layout()
        self.fig_ph.savefig(dir_fig + "/poa_ex_{}_{}.png".format(_save_name, fname))
        self.ax_ph.cla()
        self.fig_ph.clf()

        self.ax_ph_ey.legend(fontsize=15)
        self.fig_ph_ey.tight_layout()
        self.fig_ph_ey.savefig(dir_fig + "/poa_ey_{}_{}.png".format(_save_name, fname))
        self.ax_ph_ey.cla()
        self.fig_ph_ey.clf()

        self.ax_ph.cla()
        self.ax_ph_ey.cla()
        self.fig_ph.clf()
        self.fig_ph_ey.clf()
        plt.close()
        print("plotting px_{} has been done".format(_save_name))

class Spectra:
    """
    calculate distribution functions
    """
    def __init__(self, _particle):
        self.fig_fn, self.ax_fn = plt.subplots()
        self.particle = _particle

    def calc_hist(self, _ek, _save_name, _label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist_, bins_ = np.histogram(_ek, bins=nbin, range=(rmin, rmax))
        f_save[:, 1] = hist_ / dek
        np.savetxt(dir_fn + "/{}{}{}.txt".format(header_, _save_name, fname), f_save, header="energy [eV] fn [/eV]")
        self.ax_fn.plot(f_save[:, 0], f_save[:, 1], label=_label)

    def __del__(self):
        self.ax_fn.set_yscale('log')
        self.ax_fn.set_xlabel(r'$E\ [\mathrm{eV}]$')
        self.ax_fn.set_ylabel(r'$f\ [\mathrm{eV^{-1}}]$')
        self.ax_fn.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_fn.legend(frameon=False)
        self.fig_fn.tight_layout()
        self.fig_fn.savefig(dir_fig + '/{}fn_{}_'.format(header_, self.particle) + fname + '.png')
        self.ax_fn.cla()
        self.fig_fn.clf()
        plt.close()
        print("plotting fn {} of {} has been done".format(header_, self.particle))


class LogSpectra:
    """
    calculate distribution functions
    """
    def __init__(self, _particle):
        self.fig_fn, self.ax_fn = plt.subplots()
        self.particle = _particle

    def calc_hist(self, _ek, _save_name, _label):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist_, bins_ = np.histogram(log10(_ek), bins=nbin, range=(rminlog, rmaxlog))
        f_savelog[:, 1] = hist_ / deklog
        np.savetxt(dir_fn + "/{}{}eklog_{}.txt".format(header_, _save_name, fname), f_savelog, header="energy [eV] fn [/eV]")
        self.ax_fn.plot(f_savelog[:, 0], f_savelog[:, 1], label=_label)

    def __del__(self):
        self.ax_fn.set_xscale('log')
        self.ax_fn.set_yscale('log')
        self.ax_fn.set_xlabel(r'$E\ [\mathrm{eV}]$')
        self.ax_fn.set_ylabel(r'$f\ [\mathrm{eV^{-1}}]$')
        self.ax_fn.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_fn.legend(frameon=False)
        self.fig_fn.tight_layout()
        self.fig_fn.savefig(dir_fig + '/{}fn_{}_eklog_'.format(header_, self.particle) + fname + '.png')
        self.ax_fn.cla()
        self.fig_fn.clf()
        plt.close()
        print("plotting fn-eklog {} of {} has been done".format(header_, self.particle))
