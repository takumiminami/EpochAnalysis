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


# ----- parameters ----- #
##### please change name_e, name_p, and name_c as particle names for your use #####
name_e = "electron"
name_p = "proton"
name_c = "carbon"

den_min = 1e19  # minimum density to plot
dec_width = 0.5e-6
#order = ["1st", "2nd", "3rd"]
colors = ["Blues", "Reds", "Greens"]

topXpercent = 1  # percent

# ----- flags ----- #
output_electron = True
output_proton = True
output_carbon = True

# flag to save field data in .npy
field_npy = False
# saving figures
fig_flag = True
# magnetic field
mag_flag = True
# for charge density
charge_flag = False
# for averaged particle energy in each cell
ekbar_flag = False
# plot distribution functions of ions only on X-axis
on_axis_flag = False
# plot distribution functions of ions excluding around simulation boundaries
woe_flag = False

# ----- initialization ----- #
cur_dir = os.getcwd()

dir_fig = cur_dir + "/figures"
if not os.path.exists(dir_fig):
    os.makedirs(dir_fig)
dir_field = cur_dir + "/field"
if not os.path.exists(dir_field):
    os.makedirs(dir_field)
dir_fn = cur_dir + "/fn"
if not os.path.exists(dir_fn):
    os.makedirs(dir_fn)
dir_phase = cur_dir + "/phase"
if not os.path.exists(dir_phase):
    os.makedirs(dir_phase)

if on_axis_flag & woe_flag:  # woe_flag is prior than on_axis_flag
    on_axis_flag = False

if on_axis_flag:
    header_ = "on_axis_"
elif woe_flag:
    header_ = "woe_"
else:
    header_ = ""


file_list = glob.glob(cur_dir + '/*.sdf')
file_list.sort()

mass_u = 1.6605e-27
q = 1.6022e-19
c = 299792458
eps = 8.8542e-12

output_species_n = (output_electron + output_proton + output_carbon)

# ----- initializations for histograms ----- #
nbin = 400  # bin number
rmin = 1e3  # minimum energy [eV]
rmax = 2e9  # maximum energy [eV]
rminlog = log10(1)  # minimum energy [eV]
rmaxlog = log10(2e9)  # maximum energy [eV]


hist_ek, bins_ek = np.histogram(1, bins=nbin, range=(rmin, rmax))
nb_ek = len(bins_ek)
ek_label = (bins_ek[0:-1] + bins_ek[1:]) / 2
dek = (bins_ek[1:] - bins_ek[0:-1])
f_save = np.empty((hist_ek.__len__(), 2))
f_save[:, 0] = ek_label  # [eV]

hist_eklog, bins_eklog = np.histogram(1, bins=nbin, range=(rminlog, rmaxlog))
nb_eklog = len(bins_eklog)
ek_labellog = (10**bins_eklog[0:-1] + 10**bins_eklog[1:]) / 2
deklog = (10**bins_eklog[1:] - 10**bins_eklog[0:-1])
f_savelog = np.empty((hist_eklog.__len__(), 2))
f_savelog[:, 0] = ek_labellog  # [eV]

gamma_max = 1.3
px_max = np.sqrt(gamma_max**2 - 1)
hist_px_, bins_px_ = np.histogram(1, bins=nbin, range=(-px_max, px_max))
nb_px = len(bins_px_)
px_mid = (bins_px_[0:nb_px-1] + bins_px_[1:nb_px]) / 2
dpx = (bins_px_[1:nb_px] - bins_px_[0:nb_px-1])


rmin_theta = -180  # minimum angle
rmax_theta = 180  # maximum angle

hist_theta, bins_theta = np.histogram(1, bins=nbin, range=(rmin_theta, rmax_theta))
nb_theta = len(bins_theta)
theta = (bins_theta[0:nb_theta - 1] + bins_theta[1:nb_theta]) / 2
dtheta = (bins_theta[1:nb_theta] - bins_theta[0:nb_theta - 1])
theta_save = np.empty((len(hist_theta), 3))
theta_save[:, 0] = theta 
theta_save[:, 1] = ek_label
theta_save[:, 2] = ek_labellog
np.savetxt(dir_field + "/theta-ek-eklog.txt", theta_save, header="bins for ek-theta plot, theta [degree]  ek [eV]  ek-log [eV]", fmt="%.8e")


# ----- no stdout ----- #
# This is for displaying nothing when extracting .sdf files
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def no_stdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


# ----- functions ----- #

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
    fig.savefig(dir_fig + '/{}_eklogth_{}{}.png'.format(header_, _save_name, fname))
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


#file_list = ["{:04d}.sdf".format(n) for n in np.arange(33, 101)]
# file_list = ["0030.sdf"]


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- main loop ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("starting to plot for pre-ionized run")
    save_t = np.empty(len(file_list))
    t_ekmax = np.zeros((len(file_list), 1 + 3*output_species_n))
#    t_ekmax = np.zeros((len(file_list), 7))

    for m, file_name in enumerate(file_list):
        print("-----------------------------------")
        print("{}".format(file_name))

        # ----- initialization
        with no_stdout():
            data = sh.getdata(file_name)
        fname = os.path.splitext(os.path.basename(file_name))[0]
        x = data.Grid_Grid_mid.data[0] * 1e6
        y = data.Grid_Grid_mid.data[1] * 1e6
        x_axis_pos = np.argmin(np.abs(y))
        if field_npy:
            np.save(dir_field + "/x_{}.npy".format(fname), x)
            np.save(dir_field + "/y_{}.npy".format(fname), y)
        save_t[m] = data.Header['time']
        t_ekmax[m, 0] = save_t[m]
        time = save_t[m] * 1e15

        # ------ Ex
        ex = data.Electric_Field_Ex.data
        cb_ex = def_cbmax(ex)*1e-1
        field_shape = np.shape(ex)
        plot_field(ex, Normalize(vmin=-cb_ex, vmax=cb_ex), _label=r'$E_x\ [\mathrm{Vm}^{-1}]$', _save_name="ex_", color_="bwr")
        plot_field_onaxis(ex, _y=y*1e-6, _ave_range=dec_width, _label=r'$E_x\ [\mathrm{Vm}^{-1}]$', _save_name="ex_", log_flag=False)
        del ex, cb_ex

        # ----- Ey
        ey = data.Electric_Field_Ey.data
        cb_ey = def_cbmax(ey)*1e-1
        plot_field(ey, Normalize(vmin=-cb_ey, vmax=cb_ey), _label=r"$E_y\ [\mathrm{Vm}^{-1}]$", _save_name="ey_", color_="PuOr")
        plot_field_onaxis(ey, _y=y*1e-6, _ave_range=dec_width, _label=r'$E_y\ [\mathrm{Vm}^{-1}]$', _save_name="ey_", log_flag=False)
        del ey, cb_ey

        # ----- Ez
        ez = data.Electric_Field_Ez.data
        cb_ez = def_cbmax(ez)*1e-1
        plot_field(ez, Normalize(vmin=-cb_ez, vmax=cb_ez), _label=r"$E_z\ [\mathrm{Vm}^{-1}]$", _save_name="ez_", color_="PiYG")
        del ez, cb_ez

        # ----- magnetic field
        if mag_flag:
            # ----- Bx
            bx = data.Magnetic_Field_Bx.data
            cb_bx = def_cbmax(bx)*1e-1
            plot_field(bx, Normalize(vmin=-cb_bx, vmax=cb_bx), _label=r"$B_x\ [\mathrm{T}]$", _save_name="bx_", color_="bwr")
            del bx, cb_bx

            # ----- By
            by = data.Magnetic_Field_By.data
            cb_by = def_cbmax(by)*1e-1
            plot_field(by, Normalize(vmin=-cb_by, vmax=cb_by), _label=r"$B_y\ [\mathrm{T}]$", _save_name="by_", color_="PuOr")
            del by, cb_by

            # ----- Bz
            bz = data.Magnetic_Field_Bz.data
            cb_bz = def_cbmax(bz)*1e-1
            plot_field(bz, Normalize(vmin=-cb_bz, vmax=cb_bz), _label=r"$B_z\ [\mathrm{T}]$", _save_name="bz_", color_="PiYG")
            del bz, cb_bz

        # ----- charge density 
        if charge_flag:
            cd = data.Derived_Charge_Density.data
            cb_cd = def_cbmax(cd)
            plot_field(cd, Normalize(vmin=-cb_cd, vmax=cb_cd), _label=r"$\rho\ [\mathrm{Cm^{-3}}]$", _save_name="rho_", color_="bwr")
            del cd, cb_cd

        # ----- electron density
        if output_electron:
            try:
                den_e = eval("data.Derived_Number_Density_{}.data".format(name_e))
#            den_e = copy.deepcopy(data.Derived_Number_Density_electron.data)
            except AttributeError:
                den_e = np.zeros(field_shape)

            plot_field(den_e, LogNorm(vmin=den_min, vmax=1e31), _label=r"$N_e\ [\mathrm{m^{-3}}]$", _save_name="den_e_")
            plot_field(den_e, LogNorm(vmin=1e10, vmax=1e20), _label=r"$N_e\ [\mathrm{m^{-3}}]$", _save_name="den_e_low_")  # lower plot
            plot_field_onaxis(den_e * 1e-27, _y=y*1e-6, _ave_range=dec_width, _label=r"$N_e\ [10^{27}\mathrm{m^{-3}}]$", _save_name="den_e_", log_flag=True)  # density on X-axis
            del den_e


        plot_all_density = PlotField(time)
        # ----- proton density
        if output_proton:
            try:
                den_p = eval("data.Derived_Number_Density_{}.data".format(name_p))
            except AttributeError:
                den_p = np.zeros(field_shape)

            plot_field(den_p, LogNorm(vmin=den_min, vmax=1e29), _label=r"$N_{p}\ [\mathrm{m^{-3}}]$", _save_name="den_p_")
            plot_field(den_p, LogNorm(vmin=1e10, vmax=1e20), _label=r"$N_{p}\ [\mathrm{m^{-3}}]$", _save_name="den_p_low_")
            plot_field_onaxis(den_p * 1e-27, _y=y*1e-6, _ave_range=dec_width, _label=r"$N_{p}\ [10^{27}\mathrm{m^{-3}}]$", _save_name="den_p_", log_flag=True)
            plot_all_density.plot_field(den_p, x, y, LogNorm(vmin=den_min, vmax=1e29), _label=r"$N_{p}\ [\mathrm{m^{-3}}]$", color_=colors[0])

            del den_p

        # ----- den_c
        if output_carbon:
            den_c = np.zeros(field_shape)
            for n in range(1):   # This loop doesn't make any sense. (used for ionization run)
                text_den_c = "data.Derived_Number_Density_{}.data".format(name_c)
                try:
                    den_c += eval(text_den_c)
                except AttributeError:
                    pass

            plot_field(den_c, LogNorm(vmin=den_min, vmax=1e29), _label=r"$N_{C}\ [\mathrm{m^{-3}}]$", _save_name="den_c_")
            plot_field(den_c, LogNorm(vmin=1e10, vmax=1e20), _label=r"$N_{C}\ [\mathrm{m^{-3}}]$", _save_name="den_c_low_")
            plot_field_onaxis(den_c * 1e-27, _y=y*1e-6, _ave_range=dec_width, _label=r"$N_{C}\ [10^{27}\mathrm{m^{-3}}]$", _save_name="den_c_", log_flag=True)
            plot_all_density.plot_field(den_c, x, y, LogNorm(vmin=den_min, vmax=1e29), _label=r"$N_{C}\ [\mathrm{m^{-3}}]$", color_=colors[1])
            del den_c

            plot_all_density.save_image("den_all_", fname)
            del plot_all_density

        # ----- ekbar
        if ekbar_flag:
            # electron 
            if output_electron:
                ekebar = eval("data.Derived_Average_Particle_Energy_{}.data".format(name_e)) / q
                cb_ekebar = def_cbmax(ekebar)
                plot_field(ekebar, Normalize(cb_ekebar, 0), r"$<E_e>$ [eV]", "ekebar_")

            # proton 
            if output_proton:
                ekpbar = eval("data.Derived_Average_Particle_Energy_{}.data".format(name_p)) / q
                cb_ekpbar = def_cbmax(ekpbar)
                plot_field(ekpbar, Normalize(cb_ekpbar, 0), r"$<E_p>$ [eV]", "ekpbar_")

            # carbon 
            if output_carbon:
                ekcbar = eval("data.Derived_Average_Particle_Energy_{}.data".format(name_c)) / q
                cb_ekcbar = def_cbmax(ekcbar)
                plot_field(ekcbar, Normalize(cb_ekcbar, 0), r"$<E_C>$ [eV]", "ekcbar_")


        counter_tekmax = 0
        # ----- fn_e
        if output_electron:
            sp_e = Spectra(name_e)
            splog_e = LogSpectra(name_e)
            try:
                if header_ == "woe_":
                    ek_e = calc_ek_without_edge(data, name_e)
                    angle_e = calc_direction_woedge(data, name_e)
                elif header_ =="on_axis_":
                    ek_e = calc_ek_on_axis(data, name_e)
                    angle_e = calc_direction(data, name_e)
                else:
                    ek_e = calc_ek(data, name_e)
                    angle_e = calc_direction(data, name_e)
                t_ekmax[m, 3*counter_tekmax+1] = np.max(ek_e)
                eke_topXp = calc_average_of_topXpercent(ek_e)
                t_ekmax[m, 3*counter_tekmax+2] = eke_topXp[0]
                t_ekmax[m, 3*counter_tekmax+3] = eke_topXp[1]
                sp_e.calc_hist(ek_e, _save_name="fe_", _label=name_e)
                splog_e.calc_hist(ek_e, _save_name="fe_", _label=name_e)
                energy_direction(ek_e, angle_e, _save_name="e_", _label=name_e)
                del ek_e
            except AttributeError:
                print("no {}".format(name_e))
            del sp_e, splog_e
            counter_tekmax += 1

        # ----- fn_p
        if output_proton:
            sp_p = Spectra(name_p)
            splog_p = LogSpectra(name_p)
            try:
                if header_ == "woe_":
                    ek_p = calc_ek_without_edge(data, name_p)
                    angle_p = calc_direction_woedge(data, name_p)
                elif header_ == "on_axis_":
                    ek_p = calc_ek_on_axis(data, name_p)
                    angle_p = calc_direction(data, name_p)
                else:
                    ek_p = calc_ek(data, name_p)
                    angle_p = calc_direction(data, name_p)
                t_ekmax[m, 3*counter_tekmax+1] = np.max(ek_p)
                ekp_topXp = calc_average_of_topXpercent(ek_p)
                t_ekmax[m, 3*counter_tekmax+2] = ekp_topXp[0]
                t_ekmax[m, 3*counter_tekmax+3] = ekp_topXp[1]
                sp_p.calc_hist(ek_p, _save_name="fp_", _label=name_p)
                splog_p.calc_hist(ek_p, _save_name="fp_", _label=name_p)
                energy_direction(ek_p, angle_p, _save_name="p_", _label=name_p)
                del ek_p
            except AttributeError:
                print("no {}".format(name_p))
            del sp_p, splog_p
            counter_tekmax += 1

        # ----- fn_c
        if output_carbon:
            sp_c = Spectra(name_c)
            splog_c = LogSpectra(name_c)
            try:
                if header_ == "woe_":
                    ek_c = calc_ek_without_edge(data, name_c)
                    angle_c = calc_direction_woedge(data, name_c)
                elif header_ == "on_axis_":
                    ek_c = calc_ek_on_axis(data, name_c)
                    angle_c = calc_direction(data, name_c)
                else:
                    ek_c = calc_ek(data, name_c)
                    angle_c = calc_direction(data, name_c)
                t_ekmax[m, 3*counter_tekmax+1] = np.max(ek_c)
                ekc_topXp = calc_average_of_topXpercent(ek_c)
                t_ekmax[m, 3*counter_tekmax+2] = ekc_topXp[0]
                t_ekmax[m, 3*counter_tekmax+3] = ekc_topXp[1]
                sp_c.calc_hist(ek_c, _save_name="fc_", _label=name_c)
                splog_c.calc_hist(ek_c, _save_name="fc_", _label=name_c)
                energy_direction(ek_c, angle_c, _save_name="c_", _label=name_c)
                del ek_c
            except AttributeError:
                print("no {}".format(name_c))
            del sp_c, splog_c
            counter_tekmax += 1

        # ----- phase_p
        if output_proton:
            xpx_p = XpxOnAxisMulti(data, _ave_range=dec_width)
            xpx_p.calc_xpx_on_axis(name_p, "proton")
            xpx_p.save_plot("proton")
            del xpx_p

        # ----- phase_c
        if output_carbon:
            xpx_c = XpxOnAxisMulti(data, _ave_range=dec_width)
            xpx_c.calc_xpx_on_axis(name_c, "carbon")
            xpx_c.save_plot("carbon")
            del xpx_c

        del data
        gc.collect()


    header_e = "  eke_max [eV]  eke_max_stat [eV]  eke_stat_std [eV]"
    header_p = "  ekp_max [eV]  ekp_max_stat [eV]  ekp_stat_std [eV]"
    header_c = "  ekc_max [eV]  ekc_max_stat [eV]  ekc_stat_std [eV]"

    header = "time [s]"  # ekp_max_stat : average of top10%
    if output_electron:
        header = header + header_e
    if output_proton:
        header = header + header_p
    if output_carbon:
        header = header + header_c

    np.savetxt(dir_fn + "/{}t_ekmax.txt".format(header_), t_ekmax, header=header, fmt="%.8e")

    count_tekmax = 0
    if output_electron:
        fig_ekm, ax_ekm = plt.subplots()
        ax_ekm.set_xlabel(r"$t\ [\mathrm{ps}]$")
        ax_ekm.set_ylabel(r"$E_{e}\ [\mathrm{eV}]$")
        ax_ekm.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+1], label=r"$E_{e-max}$")
        ax_ekm.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2], label=r"$<E_{e-top"+"{:.0f}".format(topXpercent)+"\%}>$")
        ax_ekm.fill_between(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2] - t_ekmax[:, 3*count_tekmax+3], t_ekmax[:, 3*count_tekmax+2] + t_ekmax[:, 3*count_tekmax+3], facecolor="gray", alpha=0.5)
        fig_ekm.legend(bbox_to_anchor=(0.1, 1), loc='upper left', borderaxespad=3,  fontsize=15)
        fig_ekm.tight_layout()
        fig_ekm.savefig(dir_fig + "/{}t_ekmax_e.png".format(header_))
        print("Plotting Ekmax_e has been done.")
        ax_ekm.cla()
        fig_ekm.clf()
        count_tekmax += 1

    if output_proton:
        fig_ekm, ax_ekm = plt.subplots()
        ax_ekm.set_xlabel(r"$t\ [\mathrm{ps}]$")
        ax_ekm.set_ylabel(r"$E_{p}\ [\mathrm{eV}]$")
        ax_ekm.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+1], label=r"$E_{p-max}$")
        ax_ekm.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2], label=r"$<E_{p-top"+"{:.0f}".format(topXpercent)+"\%}>$")
        ax_ekm.fill_between(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2] - t_ekmax[:, 3*count_tekmax+3], t_ekmax[:, 3*count_tekmax+2] + t_ekmax[:, 3*count_tekmax+3], facecolor="gray", alpha=0.5)
        fig_ekm.legend(bbox_to_anchor=(0.1, 1), loc='upper left', borderaxespad=3,  fontsize=15)
        fig_ekm.tight_layout()
        fig_ekm.savefig(dir_fig + "/{}t_ekmax_p.png".format(header_))
        print("Plotting Ekmax_p has been done.")
        ax_ekm.cla()
        fig_ekm.clf()
        count_tekmax += 1

    if output_carbon:
        fig_ekm2, ax_ekm2 = plt.subplots()
        ax_ekm2.set_xlabel(r"$t\ [\mathrm{ps}]$")
        ax_ekm2.set_ylabel(r"$E_{C6}\ [\mathrm{eV}]$")
        ax_ekm2.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+1], label=r"$E_{C6-max}$")
        ax_ekm2.plot(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2], label=r"$<E_{C6-top"+"{:.0f}".format(topXpercent)+"\%}>$")
        ax_ekm2.fill_between(t_ekmax[:, 0] * 1e12, t_ekmax[:, 3*count_tekmax+2] - t_ekmax[:, 3*count_tekmax+3], t_ekmax[:, 3*count_tekmax+2] + t_ekmax[:, 3*count_tekmax+3], facecolor="gray", alpha=0.5)
        fig_ekm2.legend(bbox_to_anchor=(0.1, 1), loc='upper left', borderaxespad=3,  fontsize=15)
        fig_ekm2.tight_layout()
        fig_ekm2.savefig(dir_fig + "/{}t_ekmax_c.png".format(header_))
        print("Plotting Ekmax_c has been done.")

        ax_ekm2.cla()
        fig_ekm2.clf()
        count_tekmax += 1
        plt.close()

