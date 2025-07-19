#!/usr/bin/env python3
# -*-coding:utf-8-*-

#####################
###  Ver. 1.0.0  ####
#####################

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from matplotlib.colors import LogNorm, Normalize
import sdf_helper as sh
import copy, os, gc, contextlib, sys, glob, re, warnings
from numba import njit, prange


# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 15
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
plt.rcParams["xtick.major.size"] = 7
plt.rcParams["ytick.major.size"] = 7
plt.rcParams["legend.frameon"] = False
plt.rcParams["pcolor.shading"] = "auto"


# ----- parameters ----- #
##### please change name_p name_c as ion name for your use #####
name_p = "proton"
name_c = "carbon"

den_min = 1e19  # minimum density to plot
dec_width = 0.5e-6
#order = ["1st", "2nd", "3rd"]
colors = ["Blues", "Reds", "Greens"]


# ----- flags ----- #
# flag to save field data in .npy
field_npy = False
# saving figures
fig_flag = True
# magnetic field
mag_flag = True
# for charge density
charge_flag = False
# plot distribution functions of ions only on X-axis
on_axis_flag = True
# plot distribution functions of ions excluding around simulation boundaries
woe_flag = True


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


# ----- initializations for histograms ----- #
nbin = 400  # bin number
rmin = log10(1)  # minimum energy
rmax = log10(1e11)  # maximum energy

#hist, bins = np.histogram(1, bins=nbin, range=(rmin, rmax))
#nb = len(bins)
#ek = (bins[0:nb - 1] + bins[1:nb]) / 2
#dek = (bins[1:nb] - bins[0:nb - 1])
hist_ek, bins_ek = np.histogram(1, bins=nbin, range=(rmin, rmax))
nb_ek = len(bins_ek)
ek_label = (bins_ek[0:nb_ek - 1] + bins_ek[1:nb_ek]) / 2
dek = (bins_ek[1:nb_ek] - bins_ek[0:nb_ek - 1])
f_save = np.empty((hist_ek.__len__(), 2))
f_save[:, 0] = 10 ** ek_label  # [eV]

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
theta_save = np.empty((len(hist_theta), 2))
theta_save[:, 0] = theta  # [eV]


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

def exclude_edge(data_, particle_, scaler_):
#    exclude_width = 1e-6
    particle_x = eval("data_.Grid_Particles_{}.data[0]".format(particle_))
    particle_y = eval("data_.Grid_Particles_{}.data[1]".format(particle_))
    x_ = data_.Grid_Grid.data[0]
    y_ = data_.Grid_Grid.data[1]
    x_max = np.max(x_) * 0.95   # - exclude_width
    x_min = np.min(x_) * 0.95   # + exclude_width
    y_max = np.max(y_) * 0.95   # - exclude_width
    y_min = np.min(y_) * 0.95   # + exclude_width

    index_ = (particle_x > x_min) & (particle_x < x_max) & (particle_y > y_min) & (particle_y < y_max)
    if np.sum(index_) == 0:
        return 0
    else:
        return scaler_[index_]


def calc_direction(data_, particle_):
    px = eval("data_.Particles_Px_{}.data".format(particle_))
    py = eval("data_.Particles_Py_{}.data".format(particle_))

    angle = np.arctan2(py, px)
    return angle


def calc_direction_woedge(data_, particle_):
    angle = calc_direction(data_, particle_)
    angle_woedge = exclude_edge(data_, particle_, angle)
    return angle_woedge


@njit('f8[:,:](f8[:], f8[:], i8, f8[:], i8, i8)', parallel=True)
def calc_ek_theta(ek__, angle__, nbin_, bins_theta_, rmin_, rmax_):
    data_ = np.empty((nbin_, nbin_))
    for n in prange(nbin_):
        pos = (angle__ > bins_theta_[n]) & (angle__ < bins_theta_[n + 1])
        data_[:, n], bins_ = np.histogram(log10(ek__[pos]), bins=nbin_, range=(rmin_, rmax_))
    return data_


def energy_direction(ek_, angle_, save_name_, label_):
    ek_theta = calc_ek_theta(ek_, np.rad2deg(angle_), nbin, bins_theta, rmin, rmax)
    np.savetxt(dir_field + "/wo_edge_ekth_{}{}.txt".format(save_name_, fname), ek_theta)

    fig, ax = plt.subplots()
    if np.average(ek_theta) == 0:
        frame = ax.pcolormesh(ek_label, theta, ek_theta.T)
    else:
        frame = ax.pcolormesh(ek_label, theta, ek_theta.T, norm=LogNorm())
    cbar = fig.colorbar(frame)
    cbar.set_label("# of " + label_)
    ax.set_xlabel(r'$E_k\ [\mathrm{eV}]$')
    ax.set_ylabel(r'$\theta\ [\mathrm{degree}]$')
    x_ticks = np.linspace(rmin, rmax - 1, 6)
    x_ticklabels = [r"$10^{}$".format(str(txt)[0]) for txt in x_ticks if txt < 10]
    x_ticklabels.append(r"$10^{0}$$^{1}$".format("1", "0"))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticks([-180, -90, 0, 90, 180])

    ax.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
    fig.tight_layout()
    fig.savefig(dir_fig + '/wo_edge_ekth_{}{}.png'.format(save_name_, fname))
    print("plotting ek-theta of {} has been done".format(label_))
    ax.cla()
    fig.clf()
    plt.close()


class PlotField:
    """
    plotting field variables (ion, electron densities)
    """
    def __init__(self, time_):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax.set_ylabel(r'$y\ [\mathrm{\mu m}]$')
        self.ax.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time_), xy=(0.05, 1.05), xycoords='axes fraction')

    def plot_field(self, field_, x_, y_, norm_, label_, color_):
        frame_ = self.ax.pcolormesh(x_, y_, field_.T, norm=norm_, cmap=color_, alpha=0.5)
        # cbar_ = self.fig.colorbar(frame_)
        # cbar_.set_label(label_)

    def save_image(self, save_name_, fname_):
        self.fig.tight_layout()
        self.fig.savefig(dir_fig + '/{}'.format(save_name_) + fname_ + '.png')
        print("plotting {} has been done".format(save_name_))

    def __del__(self):
        self.ax.cla()
        self.fig.clf()
        plt.close()


def plot_field(field_: np.ndarray, norm_: object, label_: str, save_name_: str, color_="viridis"):
    """
    plotting field variables (ex, ey, etc.)
    :param field_: data to plot
    :param norm_: color bar scale (min, max)
    :param label_: label of color bar
    :param save_name_: file name to save
    :param color_: (option, default is "viridis") color map of 2d plot
    :return:
    """
    fig_, ax_ = plt.subplots()
    frame_ = ax_.pcolormesh(x, y, field_.T, norm=norm_, cmap=color_)
    cbar_ = fig_.colorbar(frame_)
    cbar_.set_label(label_)
    ax_.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
    ax_.set_ylabel(r'$y\ [\mathrm{\mu m}]$')
    ax_.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
    fig_.tight_layout()
    if fig_flag:
        fig_.savefig(dir_fig + '/{}'.format(save_name_) + fname + '.png')
    if field_npy:
        np.save(dir_field + "/{}{}.npy".format(save_name_, fname), field_)
    print("plotting {} has been done".format(save_name_))
    ax_.cla()
    fig_.clf()
    plt.close()


def plot_field_onaxis(field_, label_, save_name_, log_flag: bool):
    """
    Used for plotting grid variables on X-axis (e.g. Density, Electric field)
    """
    field_on_axis = field_[:, x_axis_pos]
    np.savetxt(dir_field + "/on_axis_{}.txt".format(save_name_), np.array((x, field_on_axis)).T, header="x [um]  {}".format(label_))

    fig_, ax_ = plt.subplots()
    ax_.plot(x, field_on_axis, lw=1)
    ax_.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
    ax_.set_ylabel(label_)
    ax_.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.15, 1.05), xycoords='axes fraction')
    if fig_flag:
        fig_.tight_layout()
        fig_.savefig(dir_fig + '/on_axis_{}'.format(save_name_) + fname + '.png')
    if log_flag:
        ax_.set_yscale("log")
    if fig_flag:
        fig_.tight_layout()
        fig_.savefig(dir_fig + '/on_axis_log_{}'.format(save_name_) + fname + '.png')
    print("plotting {} ON AXIS has been done".format(save_name_))
    ax_.cla()
    fig_.clf()
    plt.close()


def def_cbmax(field):
    fmax = np.abs(field.max())
    fmin = np.abs(field.min())
    if fmax < fmin:
        return fmin
    else:
        return fmax


def define_mass(particle___):
    if re.findall("proton", particle___):
        mass_ = mass_u * 1.0073
    elif re.findall("carbon", particle___):
        mass_ = mass_u * 12.011
    elif re.findall("gold", particle___):
        mass_ = mass_u * 196.97
    elif re.findall("oxygen", particle___):
        mass_ = mass_u * 15.999
    elif re.findall("electron", particle___):
        mass_ = 9.1094e-27
    else:
        raise Exception("Undefined such particle: {}".format(particle___))

    return mass_


def calc_ek(data___, particle__: str) -> np.ndarray:
    """
    to obtain kinetic energies of particles
    :param data___:
    :param particle__:
    :return:
    """
    mass = define_mass(particle__)

    mc2 = mass * c ** 2
    px = eval("data___.Particles_Px_{}.data".format(particle__))
    py = eval("data___.Particles_Py_{}.data".format(particle__))
    pz = eval("data___.Particles_Pz_{}.data".format(particle__))
    pp2 = px ** 2 + py ** 2 + pz ** 2
    energy = np.sqrt(mc2 ** 2 + c ** 2 * pp2)

    return (energy - mc2) / q


def calc_ek_on_axis(data_, particle_):
    """
    to obtain kinetic energies of particles only around X-axis
    :param data_:
    :param particle_:
    :return:
    """
    ek_ = calc_ek(data_, particle_)
    y_ = eval("data_.Grid_Particles_{}.data[1]".format(particle_))
    index_ = np.abs(y_) < dec_width
    if on_axis_flag:
        if np.sum(index_) == 0:
            return 0
        else:
            return ek_[index_]
    else:
        return ek_


def calc_ek_without_edge(data_, particle_):
    """
    to obtain kinetic energies of particles excluding around simulation boundaries
    :param data_:
    :param particle_:
    :return:
    """
    ek_ = calc_ek(data_, particle_)
    x_ = eval("data_.Grid_Particles_{}.data[0]".format(particle_))
    y_ = eval("data_.Grid_Particles_{}.data[1]".format(particle_))

    x_grid = data.Grid_Grid.data[0]
    y_grid = data.Grid_Grid.data[1]
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
        return ek_[index_]


def calc_average_of_top10percent(ek_: np.ndarray):
    """
    to obtain the average of kinetic energies of top 10 % ions
    :param ek_:
    :return:
    """
    length_ = int(len(ek_)*0.1)
    sort_ek_ = np.sort(ek_)
    top10_ = sort_ek_[-length_:]
    return np.average(top10_)


class XpxOnAxisMulti:
    """
    plotting X-Px with Ex on X-axis
    """
    def __init__(self, data_):
        self.data = data_
        self.fig_ph, self.ax_ph = plt.subplots()
        self.fig_ph_ey, self.ax_ph_ey = plt.subplots()
        self.init_plot()

    def init_plot(self):
        x_grid_max = np.max(eval("self.data.Grid_Grid.data[0]")) * 1e6
        x_grid_min = np.min(eval("self.data.Grid_Grid.data[0]")) * 1e6

        x_mid_ = eval("self.data.Grid_Grid_mid.data[0]")*1e6
        y_mid_ = eval("self.data.Grid_Grid_mid.data[1]")*1e6

        x_axis_pos_ = np.argmin(np.abs(y_mid_))
        ex_center_ = eval("self.data.Electric_Field_Ex.data")[:, x_axis_pos_]
        ey_center_ = eval("self.data.Electric_Field_Ey.data")[:, x_axis_pos_]
        cb_exoa_ = def_cbmax(ex_center_) * 1.05
        cb_eyoa_ = def_cbmax(ey_center_) * 1.05
#        cb_eoa = np.maximum(cb_exoa_, cb_eyoa_)

        self.ax_ph.set_xlim(x_grid_min, x_grid_max)
        self.ax_ph.set_ylim(-px_max, px_max)
        self.ax_ph.vlines(0, -px_max, px_max, linestyle=":", lw=1, color="black")
        self.ax_ph.hlines(0, x_grid_min, x_grid_max, linestyle=":", lw=1, color="black")
        self.ax_ph.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax_ph.set_ylabel(r"$p_x/mc\ []$")
        self.ax_ph.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_ph2 = self.ax_ph.twinx()
        self.ax_ph2.plot(x_mid_, ex_center_, ls="-", lw=1, color="lightsteelblue", alpha=0.7)
        self.ax_ph2.set_ylim(-cb_exoa_, cb_exoa_)
        self.ax_ph2.set_ylabel(r"$E_{x}$ [Vm$^{-1}$]")
#        self.ax_ph2.legend()

        self.ax_ph_ey.set_xlim(x_grid_min, x_grid_max)
        self.ax_ph_ey.set_ylim(-px_max, px_max)
        self.ax_ph_ey.vlines(0, -px_max, px_max, linestyle=":", lw=1, color="black")
        self.ax_ph_ey.hlines(0, x_grid_min, x_grid_max, linestyle=":", lw=1, color="black")
        self.ax_ph_ey.set_xlabel(r'$x\ [\mathrm{\mu m}]$')
        self.ax_ph_ey.set_ylabel(r"$p_x/mc\ []$")
        self.ax_ph_ey.annotate(r"$\tau=$" + "{:.2f} [fs]".format(time), xy=(0.05, 1.05), xycoords='axes fraction')
        self.ax_ph2_ey = self.ax_ph_ey.twinx()
        self.ax_ph2_ey.plot(x_mid_, ey_center_, ls=":", lw=1, color="mediumseagreen", alpha=0.7)
        self.ax_ph2_ey.set_ylim(-cb_eyoa_, cb_eyoa_)
        self.ax_ph2_ey.set_ylabel(r"$E_{y}$ [Vm$^{-1}$]")
#        self.ax_ph2_ey.legend()

        exy_save = np.zeros((len(x_mid_), 3))
        exy_save[:, 0] = x_mid_
        exy_save[:, 1] = ex_center_
        exy_save[:, 2] = ey_center_
        header__ = "x [um]  ex [V/m]  ey [V/m]"
        np.savetxt(dir_field + "/exy_center_{}.txt".format(fname), exy_save, header=header__)

    def calc_xpx_on_axis(self, particle_, order__):
        mass = define_mass(particle_)
        try:
            px_ = eval("self.data.Particles_Px_{}.data".format(particle_)) / mass / c
            x_ = eval("self.data.Grid_Particles_{}.data[0]".format(particle_)) * 1e6
            y_ = eval("self.data.Grid_Particles_{}.data[1]".format(particle_))
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
        np.savetxt(dir_phase + "/px_center_{}_{}.txt".format(particle_, fname), np.array((dec_x, dec_px)).T, header=header__)

    def save_plot(self, save_name_):
        self.ax_ph.legend(fontsize=15)
        self.fig_ph.tight_layout()
        self.fig_ph.savefig(dir_fig + "/poa_ex_{}_{}.png".format(save_name_, fname))
        self.ax_ph.cla()
        self.fig_ph.clf()

        self.ax_ph_ey.legend(fontsize=15)
        self.fig_ph_ey.tight_layout()
        self.fig_ph_ey.savefig(dir_fig + "/poa_ey_{}_{}.png".format(save_name_, fname))
        self.ax_ph_ey.cla()
        self.fig_ph_ey.clf()

        self.ax_ph.cla()
        self.ax_ph_ey.cla()
        self.fig_ph.clf()
        self.fig_ph_ey.clf()
        plt.close()
        print("plotting px_{} has been done".format(save_name_))


class Spectra:
    """
    calculate distribution functions
    """
    def __init__(self, particle_):
        self.fig_fn, self.ax_fn = plt.subplots()
        self.particle = particle_

    def calc_hist(self, ek_, save_name_, label_):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist_, bins_ = np.histogram(log10(ek_), bins=nbin, range=(rmin, rmax))
        f_save[:, 1] = hist_ / dek
        np.savetxt(dir_fn + "/{}{}{}.txt".format(header_, save_name_, fname), f_save, header="energy [eV] fn [/eV]")
        self.ax_fn.plot(f_save[:, 0], f_save[:, 1], label=label_)

    def __del__(self):
        self.ax_fn.set_xscale('log')
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


# file_list = ["{:04d}.sdf".format(n) for n in np.arange(49, 70)]
# file_list = ["0030.sdf"]


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- main loop ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("starting to plot for pre-ionized run")
    save_t = np.empty(len(file_list))
    t_ekmax = np.zeros((len(file_list), 5))

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

        # ----- phase_p
        xpx_p = XpxOnAxisMulti(data)
        xpx_p.calc_xpx_on_axis(name_p, "proton")
        xpx_p.save_plot("proton")
        del xpx_p

        # ----- phase_c
        xpx_c = XpxOnAxisMulti(data)
        xpx_c.calc_xpx_on_axis(name_c, "carbon")
        xpx_c.save_plot("carbon")
        del xpx_c

        del data
        gc.collect()


