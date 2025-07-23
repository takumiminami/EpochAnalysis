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
from functions import EpochProcessor, SpeciesConfig


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
dir_field = cur_dir + "/field"
dir_fn = cur_dir + "/fn"
dir_phase = cur_dir + "/phase"

dir_paths = {
    "fig": os.path.join(cur_dir, "figures"),
    "field": os.path.join(cur_dir, "field"),
    "fn": os.path.join(cur_dir, "fn"),
    "phase": os.path.join(cur_dir, "phase")
}

control_flags = {
    "npy": field_npy,
    "fig": fig_flag,
    "mag": mag_flag,
    "charge": charge_flag,
    "ekbar": ekbar_flag,
    "on_axis": on_axis_flag,
    "woe": woe_flag,
}

species_config = {
    "electron": SpeciesConfig(output_electron, name_e),
    "proton": SpeciesConfig(output_proton, name_p),
    "carbon": SpeciesConfig(output_carbon, name_c)
}

if on_axis_flag & woe_flag:  # woe_flag is prior than on_axis_flag
    on_axis_flag = False

if on_axis_flag:
    output_prefix = "on_axis_"
elif woe_flag:
    output_prefix = "woe_"
else:
    output_prefix = ""

from dataclasses import dataclass



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





# ----- functions ----- #


#file_list = ["{:04d}.sdf".format(n) for n in np.arange(33, 101)]
# file_list = ["0030.sdf"]


# -------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------- main loop ----------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# TODO: import a outer class defined in functions.py and apply it to all of the functions below. \
#      This will make the code more readable and easier to maintain.
#       Some functions can be extracted in a smaller script, such as plt_spectra.py, by importing the class.


if __name__ == "__main__":
    print("starting to plot for pre-ionized run")
    save_t = np.empty(len(file_list))
    t_ekmax = np.zeros((len(file_list), 1 + 3*output_species_n))
#    t_ekmax = np.zeros((len(file_list), 7))

    for m, file_name in enumerate(file_list):
        print("-----------------------------------")
        print("{}".format(file_name))

        # ----- initialization
        epoch_processor = EpochProcessor(file_name, dir_paths, output_prefix, control_flags, species_config)

        # with no_stdout():
            # data = sh.getdata(file_name)
        # fname = os.path.splitext(os.path.basename(file_name))[0]
        # x = data.Grid_Grid_mid.data[0] * 1e6
        # y = data.Grid_Grid_mid.data[1] * 1e6
        # x_axis_pos = np.argmin(np.abs(y))
        # if field_npy:
        #     np.save(dir_field + "/x_{}.npy".format(fname), x)
        #     np.save(dir_field + "/y_{}.npy".format(fname), y)
        # save_t[m] = data.Header['time']
        # t_ekmax[m, 0] = save_t[m]
        # time = save_t[m] * 1e15

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

