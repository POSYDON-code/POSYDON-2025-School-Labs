import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from posydon.binary_evol.flow_chart import STAR_STATES_CO

def plot_SN_evolution(df, df_oneline = None):
    """
    Plot the evolution of primary and secondary masses, He-core masses, and orbital period
    (with eccentricity on a twin y-axis).
    Only plot CC1 and CC2 if they exist. Warn if none exist.
    Shade the second subplot from the first time either orbital_period or eccentricity becomes invalid,
    or from the first disrupted/merged state, whichever occurs earlier.
    """

    if df_oneline is not None:
        f_fb = [df_oneline["S1_f_fb"].values, df_oneline["S2_f_fb"].values]
        v_kick = [df_oneline["S1_natal_kick_array_0"].values, df_oneline["S2_natal_kick_array_0"].values]
    else:
        f_fb = [None, None]
        v_kick = [None, None]
    
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9), sharex=True)

    # --- font sizes scale with figsize ---
    w, h = fig.get_size_inches()
    fs_base = min(w, h) * 2.5
    fs_label = fs_base * 0.8
    fs_xlabel = fs_base * 0.8
    fs_text = fs_base * 0.6
    fs_legend = fs_base * 0.5
    fs_ticks = fs_base * 1.0
    f_print = fs_base*0.07

    # --- Final time (end of binary evolution)
    mask_CC1_r = ((df['step_names'] == "step_SN") & (df['event'].shift(1) == "CC1")).fillna(False)
    mask_CC2_r = ((df['step_names'] == "step_SN") & (df['event'].shift(1) == "CC2")).fillna(False)
    mask_CC1 = ((df['step_names'].shift(-1) == "step_SN") & (df['event'] == "CC1")).fillna(False)
    mask_CC2 = ((df['step_names'].shift(-1) == "step_SN") & (df['event'] == "CC2")).fillna(False)

    t_end = (
        df.loc[mask_CC2, "time"].values[0] if mask_CC2.any()
        else df["time"].iloc[-1]
    )
    
    # --- Remaining time in Myr
    remaining_time = (t_end - df["time"]) / 1e6
    remaining_time = remaining_time.clip(lower=1e-2)  # avoid zeros for log plot
    df = df.copy()
    df["remaining_time"] = remaining_time

    # --- Exclude remnant states for plotting He-core mass
    exclude_states = ["WD", "BH", "NS", "massless_remnant"]
    mask1 = ~df["S1_state"].isin(exclude_states)
    mask2 = ~df["S2_state"].isin(exclude_states)

    if not mask_CC1.any() and not mask_CC2.any():
        print("This system might have failed, no CCSN found.")
        return  # stop function

    # --- Primary mass and He-core
    mask1_alive = ~df["S1_state"].isin(STAR_STATES_CO)
    mask1_dead  = df["S1_state"].isin(STAR_STATES_CO)

    axes[0].plot(df.loc[mask1_alive, "remaining_time"], df.loc[mask1_alive, "S1_mass"],
                 linewidth=1.5, color="blue", label="primary total mass")
    axes[0].plot(df.loc[mask1_dead, "remaining_time"], df.loc[mask1_dead, "S1_mass"],
                 linewidth=1.5, color="lightskyblue")
    axes[0].plot(df.loc[mask1_alive, "remaining_time"], df.loc[mask1_alive, "S1_he_core_mass"],
                 linewidth=2.5, linestyle="dotted", color="blue", label="primary He-core mass")
    
    axes[0].scatter(df.loc[mask1_alive, "remaining_time"], df.loc[mask1_alive, "S1_mass"],
                    color="blue", s=20, alpha=0.7, zorder=2)
    axes[0].scatter(df.loc[mask1_dead, "remaining_time"], df.loc[mask1_dead, "S1_mass"],
                    color="lightskyblue", s=20, alpha=0.7, zorder=2)
    axes[0].scatter(df.loc[mask1_alive, "remaining_time"], df.loc[mask1_alive, "S1_he_core_mass"],
                    color="blue", s=20, alpha=0.7, marker="o", zorder=2)

    # --- Secondary mass and He-core ---
    mask2_alive = ~df["S2_state"].isin(STAR_STATES_CO)
    mask2_dead  = df["S2_state"].isin(STAR_STATES_CO)

    axes[0].plot(df.loc[mask2_alive, "remaining_time"], df.loc[mask2_alive, "S2_mass"],
                 linewidth=1.5, color="green", label="secondary total mass")
    axes[0].plot(df.loc[mask2_dead, "remaining_time"], df.loc[mask2_dead, "S2_mass"],
                 linewidth=1.5, color="mediumspringgreen")
    axes[0].plot(df.loc[mask2_alive, "remaining_time"], df.loc[mask2_alive, "S2_he_core_mass"],
                 linewidth=2.5, linestyle="dotted", color="green", label="secondary He-core mass")
    
    axes[0].scatter(df.loc[mask2_alive, "remaining_time"], df.loc[mask2_alive, "S2_mass"],
                    color="green", s=20, alpha=0.7, zorder=2)
    axes[0].scatter(df.loc[mask2_dead, "remaining_time"], df.loc[mask2_dead, "S2_mass"],
                    color="green", s=20, alpha=0.7, zorder=2)
    axes[0].scatter(df.loc[mask2_alive, "remaining_time"], df.loc[mask2_alive, "S2_he_core_mass"],
                    color="green", s=20, alpha=0.7, marker="o", zorder=2)
        
    # --- Plot CC1 if exists
    if mask_CC1.any():
        time_CC1 = df.loc[mask_CC1, "remaining_time"].values[0]
        mass_CC1 = df.loc[mask_CC1, "S1_mass"].values[0]

        axes[0].scatter(time_CC1, mass_CC1,
                        facecolors='yellow', edgecolors='black', marker='*', s=400, label='CC1')
        axes[0].text(time_CC1, mass_CC1+1.5*f_print,
                     df.loc[mask_CC1_r, "S1_state"].values[0], color='black', fontsize=fs_text)
        axes[0].text(time_CC1, mass_CC1+0.5*f_print,
                     df.loc[mask_CC1_r, "state"].values[0], color='black', fontsize=fs_text)
        f_fb_val = f"{f_fb[0].item():.2f}"
        axes[0].text(time_CC1, mass_CC1-0.5*f_print,
                     "f_fb = "+f_fb_val, color='black', fontsize=fs_text)
        kick_val = f"{v_kick[0].item():.0f}"
        axes[0].text(time_CC1, mass_CC1-1.5*f_print,
                     "v_kick (km/s) = "+kick_val, color='black', fontsize=fs_text)
        '''
        idx_CC = mask_CC1[mask_CC1].index[0]
        idx_prev = df.index[df.index.get_loc(idx_CC) - 1]    
        Mprog = df.loc[idx_prev, "S1_mass"]
        Mej = f"{(Mprog - df.loc[mask_CC1_r, 'S1_mass'].values[0]):.1f}"
        axes[0].text(time_CC1, mass_CC1-2.5*f_print,
                     "M_ej = "+Mej, color='black', fontsize=fs_text)
        '''
        spin_val = f"{df.loc[mask_CC1_r, 'S1_spin'].values[0]:.2f}"
        axes[0].text(time_CC1, mass_CC1-2.5*f_print,
                     "spin = "+spin_val, color='black', fontsize=fs_text)


        op_CC1 = df.loc[mask_CC1, "orbital_period"].values[0]
        axes[1].scatter(time_CC1, op_CC1,
                        facecolors='yellow', edgecolors='black', marker='*', s=400, label='CC1')

    # --- Plot CC2 if exists
    if mask_CC2.any():
        time_CC2 = df.loc[mask_CC2, "remaining_time"].values[0]
        mass_CC2 = df.loc[mask_CC2, "S2_mass"].values[0]

        axes[0].scatter(time_CC2, mass_CC2,
                        facecolors='yellow', edgecolors='red', marker='*', s=400, label='CC2')
        axes[0].text(time_CC2-.1*f_print, mass_CC2+1.5*f_print,
                     df.loc[mask_CC2_r, "S2_state"].values[0], color='black', fontsize=fs_text)
        axes[0].text(time_CC2-.1*f_print, mass_CC2+0.5*f_print,
                     df.loc[mask_CC2_r, "state"].values[0], color='black', fontsize=fs_text)
        f_fb_val = f"{f_fb[1].item():.2f}"
        axes[0].text(time_CC2-.1*f_print, mass_CC2-0.5*f_print,
                     "f_fb = "+f_fb_val, color='black', fontsize=fs_text)
        kick_val = f"{v_kick[1].item():.0f}"
        axes[0].text(time_CC2-.1*f_print, mass_CC2-1.5*f_print,
                     "v_kick (km/s) = "+kick_val, color='black', fontsize=fs_text)
        spin_val = f"{df.loc[mask_CC2_r, 'S2_spin'].values[0]:.2f}"
        axes[0].text(time_CC2-.1*f_print, mass_CC2-2.5*f_print,
                     "spin = "+spin_val, color='black', fontsize=fs_text)

        op_CC2 = df.loc[mask_CC2, "orbital_period"].values[0]
        axes[1].scatter(time_CC2, op_CC2,
                        facecolors='yellow', edgecolors='red', marker='*', s=400, label='CC2')

    # --- Orbital period + eccentricity (twin y-axis)
    valid_op = df['orbital_period'] < 1e95
    valid_ecc = df['eccentricity'] < 1e95

    ax2 = axes[1].twinx()
    axes[1].plot(df.loc[valid_op, "remaining_time"], df.loc[valid_op, 'orbital_period'],
                 color="black", linestyle="-.", label="orbital period")
    ax2.plot(df.loc[valid_ecc, "remaining_time"], df.loc[valid_ecc, 'eccentricity'],
             color="red", linestyle="dashed", label="eccentricity")
    axes[1].scatter(df.loc[valid_op, "remaining_time"], df.loc[valid_op, 'orbital_period'],
                    color="black", s=20, alpha=0.7, zorder=2)
    ax2.scatter(df.loc[valid_ecc, "remaining_time"], df.loc[valid_ecc, 'eccentricity'],
                    color="red", s=20, alpha=0.7, zorder=2)

    axes[1].set_ylabel("Orbital period (days)", fontsize=fs_label, color="black")
    ax2.set_ylabel("Eccentricity", fontsize=fs_label, color="red")

    # --- Determine earliest shading start:
    # 1) first invalid in orbital_period or eccentricity
    t_start_invalid = None
    invalid_indices = []
    if (~valid_op).any():
        invalid_indices.append(np.where(~valid_op.values)[0][0])
    if (~valid_ecc).any():
        invalid_indices.append(np.where(~valid_ecc.values)[0][0])
    if invalid_indices:
        first_invalid_idx = min(invalid_indices)
        t_start_invalid = df["remaining_time"].iloc[first_invalid_idx]

    # 2) disrupted/merged state based on flags in 'state'
    no_orbit_states = ["disrupted", "merged", "initial_single_star"]
    mask_no_orbit = (df["orbital_period"] == 0) & (df["state"].isin(no_orbit_states))
    t_start_state = df.loc[mask_no_orbit, "remaining_time"].iloc[0] if mask_no_orbit.any() else None

    # choose the earliest (i.e., largest remaining_time value along the series order)
    t_candidates = [t for t in [t_start_invalid, t_start_state] if t is not None]
    if t_candidates:
        t_start_shade = max(t_candidates)  # earlier in time corresponds to larger remaining_time
        t_end_shade = df["remaining_time"].iloc[-1]
        #print(t_start_shade)
        if t_start_shade>t_end_shade:
            axes[1].axvspan(t_start_shade, t_end_shade, color="grey", alpha=0.2, zorder=0)
            # --- Add centered label inside the shaded region ---
            x_mid = 0.5 * (t_start_shade + t_end_shade)
            y_mid = 0.5 * (axes[1].get_ylim()[0] + axes[1].get_ylim()[1])
            axes[1].text(
                x_mid, y_mid,
                "no orbit",
                ha="center", va="center",
                fontsize=fs_text, color="black", alpha=0.7, zorder=1,
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.3")
            )

    # --- Vertical lines for step_name transitions (robust to pd.NA) ---
    _sentinel = "__NA__"
    s_steps = df["step_names"].astype("object").where(df["step_names"].notna(), _sentinel)
    prev_steps = s_steps.shift(1).fillna(_sentinel)
    exclude_steps = {"step_end", "step_dco", "initial_cond", "step_SN", _sentinel}
    mask_step_change = s_steps.ne(prev_steps) & ~s_steps.isin(exclude_steps)

    for _, row in df.loc[mask_step_change].iterrows():
        t_step = row["remaining_time"]
        step_label = row["step_names"]

        # mass panel
        axes[0].axvline(x=t_step, color="grey", linestyle="dotted", linewidth=1)
        axes[0].text(t_step, axes[0].get_ylim()[1],
                     str(step_label), rotation=90, va="bottom", ha="center",
                     fontsize=fs_text, color="black")

        # orbital panel
        axes[1].axvline(x=t_step, color="grey", linestyle="dotted", linewidth=1)

    
    # --- Axis settings
    axes[0].invert_xaxis()
    axes[0].set_ylabel("Mass $[M_{\\odot}]$", fontsize=fs_label)
    axes[0].legend(fontsize=fs_legend)

    axes[1].set_xlabel("Remaining time until end [Myr]", fontsize=fs_xlabel)

    from matplotlib.ticker import MultipleLocator
    
    # --- X-axis ticks: 10, 20, 30 ...
    axes[1].xaxis.set_major_locator(MultipleLocator(10))
    
    # Make xtick labels red (applies to both subplots since they sharex)
    for ax in axes:
        ax2.tick_params(axis="y", colors="red")

    
    # tick label sizes scale too
    for ax in [*axes, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
        ax.tick_params(axis='both', which='minor', labelsize=fs_ticks * 0.9)

    plt.tight_layout()
    plt.show()
