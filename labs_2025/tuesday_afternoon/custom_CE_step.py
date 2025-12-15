"""Common envelope evolution step.

Calculates the evolution of a star in a BinaryStar object. It uses
the alpha-perscription and caclulates how much the orbit has to shrink in
order to expel its envelope. If in that separation, one of the star fills its
RL, then the system is considered a merger. Else the common envelope is lost
and we are left with a binary system with the core of the donor star (which
initiated the unstable CEE) and the core of the copmanion. For now it works
with the donor star being either a type of H_giant (with a He core) or
He_giant (with a CO core) and the companion star being either a MS or compact
object star, not contribyting to the CE.

If profiles exist we are able to calculate on the spot the lambda parameter
for the donor star, else we use the default values

Parameters
----------
binary : BinaryStar (An object of class BinaryStar defined in POSYDON)
verbose : Boolean
    In case we want information about the CEE  (the default is False).

"""

import numpy as np
import pandas as pd
from posydon.utils import common_functions as cf
from posydon.utils import constants as const
from posydon.binary_evol.binarystar import BINARYPROPERTIES
from posydon.binary_evol.singlestar import STARPROPERTIES
from posydon.config import PATH_TO_POSYDON
from posydon.utils.common_functions import check_state_of_star
from posydon.utils.common_functions import calculate_lambda_from_profile, calculate_Mejected_for_integrated_binding_energy
from posydon.utils.posydonwarning import Pwarn
import unicodedata
import difflib

MODEL = {"prescription": 'alpha-lambda',
         "common_envelope_efficiency": 1.0,
         "common_envelope_lambda_default": 0.5,
         "common_envelope_option_for_lambda": 'lambda_from_grid_final_values',
         "common_envelope_option_for_HG_star": "optimistic",
         "common_envelope_alpha_thermal": 1.0,
         "core_definition_H_fraction": 0.3,     # with 0.01 no CE BBHs
         "core_definition_He_fraction": 0.1,
         "CEE_tolerance_err": 0.001,
         "verbose": False,
         "common_envelope_option_after_succ_CEE": 'two_phases_stableMT',
         "mass_loss_during_CEE_merged": False # If False, then no mass loss from this step for a merged star
                                              # If True, then we remove mass according to the alpha-lambda prescription
                                              # assuming a final separation where the inner core RLOF starts.
         # "one_phase_variable_core_definition" for core_definition_H_fraction=0.01

         }


# common_envelope_option_for_lambda:
# 1) 'default_lambda': using for lambda the constant value of
# common_envelope_lambda_default parameter
# 2) 'lambda_from_grid_final_values': using lambda parameter from MESA history
# which was calulated ni the same way as method (5) below
# 3) 'lambda_from_profile_gravitational': calculating the lambda parameter
# from the donor's profile by using the gravitational binding energy from the
# surface to the core (needing "mass", and "radius" as columns in the profile)
# 4) 'lambda_from_profile_gravitational_plus_internal': as above but taking
# into account a factor of common_envelope_alpha_thermal * internal energy too
# in the binding energy (needing also "energy" as column in the profile)
# 5) 'lambda_from_profile_gravitational_plus_internal_minus_recombination':
# as above but not taking into account the recombination energy in the internal
# energy (needing also "y_mass_fraction_He", "x_mass_fraction_H",
# "neutral_fraction_H", "neutral_fraction_He", and "avg_charge_He" as column
# in the profile)
# the mass fraction of an element which is used as threshold to define a core,


STAR_STATE_POST_MS = [
    "H-rich_Core_H_burning",
    "H-rich_Shell_H_burning",
    "H-rich_Core_He_burning",
    "H-rich_Central_He_depleted",
    "H-rich_Central_C_depletion",
    "H-rich_Core_He_depleted",
    "H-rich_non_burning",
    "accreted_He_non_burning"
]


STAR_STATE_POST_HeMS = [
    'accreted_He_Core_He_burning',
    'stripped_He_Core_He_burning',
    'stripped_He_Central_He_depleted',
    'stripped_He_Central_C_depletion',
    'stripped_He_non_burning'
]


class StepCEE(object):
    """Compute supernova final remnant mass, fallback fraction & stellar state.

    This consider the nearest neighboor of the He core mass of the star,
    previous to the collapse. Considering a set of data for which the He core
    mass of the compact object projenitos previous the collapse, the final
    remnant mass and final stellar state of the compact object is known.

    Parameters
    ----------
    verbose : bool
        If True, the messages will be prited in the console.

    Keyword Arguments
    -----------------
    prescription : str
        Prescription to use for computing the prediction of common enevelope
        evolution. Available options are:

        * 'alpha-lambda' : Considers the the alpha-lambda prescription
          described in [1]_ and [2]_ to predict the outcome of the common
          envelope evolution. If the profile of the donor star is available
          then it is used to compute the value of lambda.

    References
    ----------
    .. [1] Webbink, R. F. (1984). Double white dwarfs as progenitors of R
        Coronae Borealis stars and Type I supernovae. The Astrophysical
        Journal, 277, 355-360.

    .. [2] De Kool, M. (1990). Common envelope evolution and double cores of
        planetary nebulae. The Astrophysical Journal, 358, 189-195.
    """

    def __init__(
            self, prescription=MODEL['prescription'],
            common_envelope_efficiency=MODEL['common_envelope_efficiency'],
            common_envelope_lambda_default=MODEL[
                'common_envelope_lambda_default'],
            common_envelope_option_for_lambda=MODEL[
                'common_envelope_option_for_lambda'],
            common_envelope_option_for_HG_star=MODEL[
                'common_envelope_option_for_HG_star'],
            common_envelope_option_after_succ_CEE=MODEL[
                'common_envelope_option_after_succ_CEE'],
            common_envelope_alpha_thermal=MODEL[
                'common_envelope_alpha_thermal'],
            core_definition_H_fraction=MODEL[
                'core_definition_H_fraction'],
            core_definition_He_fraction=MODEL[
                'core_definition_He_fraction'],
            CEE_tolerance_err=MODEL['CEE_tolerance_err'],
            mass_loss_during_CEE_merged=MODEL['mass_loss_during_CEE_merged'],
            verbose=MODEL['verbose'],
            **kwargs):
        """Initialize a StepCEE instance."""
        # read kwargs to initialize the class
        if kwargs:
            for key in kwargs:
                if key not in MODEL:
                    raise ValueError(key + " is not a valid parameter name!")
            for varname in MODEL:
                default_value = MODEL[varname]
                setattr(self, varname, kwargs.get(varname, default_value))
        else:
            self.prescription = prescription
            self.common_envelope_efficiency = common_envelope_efficiency
            self.common_envelope_lambda_default = \
                common_envelope_lambda_default
            self.common_envelope_option_for_lambda = \
                common_envelope_option_for_lambda
            self.common_envelope_option_for_HG_star = \
                common_envelope_option_for_HG_star
            self.common_envelope_alpha_thermal = common_envelope_alpha_thermal
            self.core_definition_H_fraction = core_definition_H_fraction
            self.core_definition_He_fraction = core_definition_He_fraction
            self.CEE_tolerance_err = CEE_tolerance_err
            self.common_envelope_option_after_succ_CEE = \
                common_envelope_option_after_succ_CEE
            self.mass_loss_during_CEE_merged = mass_loss_during_CEE_merged

        self.verbose = verbose
        self.path_to_posydon = PATH_TO_POSYDON

    def __call__(self, binary):
        """Perform the CEE step for a BinaryStar object."""
        # Determine which star is the donor and which is the companion
        if binary.event in ["oCE1", "oDoubleCE1"]:
            donor_star = binary.star_1
            comp_star = binary.star_2
            star_to_merge = "1"
        elif binary.event in ["oCE2", "oDoubleCE2"]:
            donor_star = binary.star_2
            comp_star = binary.star_1
            star_to_merge = "2"
        else:
            raise ValueError("CEE does not apply if `event` is not "
                             "`oCE1`, `oDoubleCE1`, `oCE2`,or `oDoubleCE1`")
        # Check for double CE
        double_CE = binary.event in ["oDoubleCE1", "oDoubleCE2"]

        if self.verbose:
            print("binary.event : ", binary.event)
            print("common_envelope_efficiency : ",
                  self.common_envelope_efficiency)
            print("common_envelope_option_for_lambda : ",
                  self.common_envelope_option_for_lambda)

        # Check to make sure binary can go through a CE
        mergeable_donor = (donor_star.state in [
            'H-rich_Core_H_burning', 'stripped_He_Core_He_burning', 'accreted_He_Core_He_burning'])
        mergeable_HG_donor = (
            self.common_envelope_option_for_HG_star == "pessimistic"
            and donor_star.state in ['H-rich_Shell_H_burning'])
        if mergeable_donor or mergeable_HG_donor:
            # system merges
            binary.state = 'merged'
            binary.event = "oMerging" + star_to_merge
            return

        # Calculate binary's evolution
        if self.prescription == 'alpha-lambda':
            self.CEE_simple_alpha_prescription(
                binary, donor_star, comp_star,
                double_CE=double_CE, verbose=self.verbose,
                common_envelope_option_after_succ_CEE=(
                    self.common_envelope_option_after_succ_CEE),
                core_definition_H_fraction=self.core_definition_H_fraction,
                core_definition_He_fraction=self.core_definition_He_fraction,
                mass_loss_during_CEE_merged=self.mass_loss_during_CEE_merged)
        else:
            raise ValueError("Invalid common envelope prescription given.")


    def CEE_adjust_post_CE_core_masses(self, donor, mc1_i, rc1_i, donor_type,
            comp_star, mc2_i, rc2_i, comp_type, double_CE, verbose=False):
        """Calculate the post-common-envelope core masses and radii.

        It determines the post-CE parameters based on the core properties.
        Note that these parameters may be updated in a subsequent step 
        depending on assumptions about whether and how the final layers
        of the CE are removed from the cores.

        Parameters
        ----------
        donor : SingleStar object
            The donor star
        mc1_i : float
            Core mass of the donor after the fast CE phase (in Msun)
        rc1_i : float
            Core radius of the donor after the fast CE phase (in Rsun)
        donor_type : string
            String dictating whether the donor has a 'He_core' or 'CO_core'
        comp_star : SingleStar object
            The companion star
        mc2_i : float
            Core mass of the companion after the fast CE phase (in Msun)
        rc2_i : float
            Core radius of the companion after the fast CE phase (in Rsun)
        comp_type : string
            String dictating whether the companion has a 'He_core', 'CO_core',
            or 'not_giant_companion' (e.g. a compact object or MS star).
        double_CE : bool
            In case we have a double CE situation.
        verbose : bool
            In case we want information about the CEE.

        Returns
        -------
        m1c_f: float
            Donor mass after the complete CE (in Msun)
        r1c_f: float
            Donor radius after the complete CE (in Rsun)
        m2c_f: float
            Companion mass after the complete CE (in Msun)
        r2c_f: float
            Companion radius after the complete CE (in Rsun)
        """
        if donor_type == 'He_core':
            mc1_f = donor.he_core_mass
            rc1_f = donor.he_core_radius
        elif donor_type == 'CO_core':
            mc1_f = donor.co_core_mass
            rc1_f = donor.co_core_radius
        else:
            raise ValueError(
                "donor_type in common envelope is not recognized. "
                "donor_type = {}, don't know how to proceed".
                format(donor_type))
        if mc1_f > mc1_i:
            mc1_f = mc1_i
            Pwarn("The final donor core mass (even after stable, postCEE MT)"
                  " is higher than the postCEE core mass. Now equalizing to "
                  "postCEE mass", "ApproximationWarning")
        if not double_CE:
            mc2_f = mc2_i
            rc2_f = rc2_i
        else:
            if comp_type == 'He_core':
                mc2_f = comp_star.he_core_mass
                rc2_f = donor.he_core_radius
            elif comp_type == 'CO_core':
                mc2_f = comp_star.co_core_mass
                rc2_f = donor.co_core_radius
            elif comp_type == 'not_giant_companion':
                mc2_f = comp_star.mass
                rc2_f = 10.**(comp_star.log_R)
            else:
                raise ValueError(
                    "comp_type in common envelope is not recognized. "
                    "comp_type = {}, don't know how to proceed".
                    format(comp_type))
            if mc2_f > mc2_i:
                mc2_f = mc2_i
                Pwarn("The accretor's final core mass (even after "
                      "non-conservative stable, postCEE MT) is "
                      "higher that postCEE core mass. "
                      "Now equalizing to postCEE mass", "ApproximationWarning")
        if verbose:
            print("m1 core mass (in Msun) change after fast CE phase from: ",
                  mc1_i, " to: ", mc1_f)
            print("r1 core radius (in Rsun) change after fast CE phase "
                  "from: ", rc1_i, " to: ", rc1_f)
            print("m2 core mass (in Msun) change after fast CE phase from: ",
                  mc2_i, " to: ", mc2_f)
            print("r2 core radius (in Rsun) change after fast CE phase "
                  "from: ", rc2_i, " to: ", rc2_f)

        return mc1_f, rc1_f, mc2_f, rc2_f

    def CEE_one_phase_variable_core_definition(self, donor, mc1_i, rc1_i,
                                               comp_star, mc2_i, rc2_i,
                                               separation_postCEE,
                                               verbose=False):
        """Calculate the post-common-envelope parameters upon exiting a CEE.

        This prescription assumes the he_core_mass/radius (or
        co_core_mass/radius for CEE of stripped_He*) are not further stripped
        beyond the core defined given by the core_definition_H/He_fraction.
        Hence, there are no changes on the orbit after successful ejection.
        Instead it is assumed that the defined core will become the remaining
        stripped object with core abundances at its surface.

        Parameters
        ----------
        donor : SingleStar object
            The donor star
        mc1_i : float
            Core mass of the donor after the fast CE phase (in Msun)
        rc1_i : float
            Core radius of the donor after the fast CE phase (in Rsun)
        comp_star : SingleStar object
            The companion star
        mc2_i : float
            Core mass of the companion after the fast CE phase (in Msun)
        rc2_i : float
            Core radius of the companion after the fast CE phase (in Rsun)
        separation_postCEE : float
            Binary's separation after the fast CE phase (in cm)
        verbose : bool
            In case we want information about the CEE.

        Returns
        -------
        m1c_f: float
            Donor mass after the complete CE (in Msun)
        r1c_f: float
            Donor radius after the complete CE (in Rsun)
        m2c_f: float
            Companion mass after the complete CE (in Msun)
        r2c_f: float
            Companion radius after the complete CE (in Rsun)
        separation_f : float
            Binary's final separation upon exiting the CE (in Rsun)
        orbital_period_f : float
            Binary's final orbital period upon exiting the CE (in days)
        merger : bool
            Whether the binary merged in the CE
        """
        mc1_f = mc1_i
        rc1_f = rc1_i
        mc2_f = mc2_i
        rc2_f = rc2_i
        if verbose:
            print("m1 core mass pre CEE (He,CO) / to after CEE : ",
                  donor.he_core_mass, donor.co_core_mass, mc1_f)
            print("m1 core radius pre CEE (He,CO) / to after CEE : ",
                  donor.he_core_radius, donor.co_core_radius, rc1_f)
            print("m2 core mass pre CEE (He,CO) / to after CEE : ",
                  comp_star.he_core_mass, comp_star.co_core_mass, mc2_f)
            print("m2 core radius pre CEE (He,CO) / to after CEE : ",
                  comp_star.he_core_radius, comp_star.co_core_radius, rc2_f)

        # Check to see if the system has merged
        merger = cf.check_for_RLO(mc1_i, rc1_f, mc2_i, rc2_f,
                        separation_postCEE/const.Rsun, self.CEE_tolerance_err)

        if verbose:
            if merger:
                print("system merges within the CEE")
            else:
                print("system survives CEE, the whole CE is ejected")

        # since the binary components' masses do not change, the output
        # separation is just the input separation in different units
        separation_f = separation_postCEE / const.Rsun

        # Calculate the orbital period accordingly
        orbital_period_f = cf.orbital_period_from_separation(separation_f,
                                                             mc1_i, mc2_i)

        return (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
                merger)

    def CEE_two_phases_stableMT(self, donor, mc1_i, rc1_i, donor_type,
                                comp_star, mc2_i, rc2_i, comp_type, double_CE,
                                separation_postCEE, verbose=False):
        """Calculate the post-common-envelope parameters upon exiting a CEE.

        This prescription assumes the he_core_mass/radius (or
        co_core_mass/radius for CEE of stripped_He*) becomes the value
        determined by MESA pre CEE. It stripes the envelope between the core
        defined by the core_definition_H/He_fraction and the MESA core by
        assuming an instantaneous stable MT phase (non-conservative, with mass
        lost from the accretor) after the successful ejection of the outer
        envelope part lost during the fast CE phase.

        Parameters
        ----------
        donor : SingleStar object
            The donor star
        mc1_i : float
            Core mass of the donor after the fast CE phase (in Msun)
        rc1_i : float
            Core radius of the donor after the fast CE phase (in Rsun)
        comp_star : SingleStar object
            The companion star
        mc2_i : float
            Core mass of the companion after the fast CE phase (in Msun)
        rc2_i : float
            Core radius of the companion after the fast CE phase (in Rsun)
        separation_postCEE : float
            Binary's separation after the fast CE phase (in cm)
        verbose : bool
            In case we want information about the CEE.

        Returns
        -------
        m1c_f: float
            Donor mass after the complete CE (in Msun)
        r1c_f: float
            Donor radius after the complete CE (in Rsun)
        m2c_f: float
            Companion mass after the complete CE (in Msun)
        r2c_f: float
            Companion radius after the complete CE (in Rsun)
        separation_f : float
            Binary's final separation upon exiting the CE (in Rsun)
        orbital_period_f : float
            Binary's final orbital period upon exiting the CE (in days)
        merger : bool
            Whether the binary merged in the CE
        """
        if double_CE:
            Pwarn("A double CE cannot have a stable mass transfer afterwards "
                  "as both cores are donors, switch to losing the mass as "
                  "wind.", "ReplaceValueWarning")
            return self.CEE_two_phases_windloss(donor, mc1_i, rc1_i, 
                                                donor_type, comp_star, mc2_i, 
                                                rc2_i, comp_type, double_CE,
                                                separation_postCEE, verbose)
        # First find the post-CE parameters for each star
        mc1_f, rc1_f, mc2_f, rc2_f = self.CEE_adjust_post_CE_core_masses(
            donor, mc1_i, rc1_i, donor_type, comp_star, mc2_i, rc2_i,
            comp_type, double_CE, verbose=verbose)

        # calculate the orbital period after CEE (before adjustment)
        orbital_period_postCEE = cf.orbital_period_from_separation(
            separation_postCEE / const.Rsun, mc1_i, mc2_i)

        # First, check if merger happens before stable MT phase. Note this uses
        # the final core radii as we are beyond the point of radius contraction
        # (the final radii are a better estimate for the radius after
        #  contraction) then the post-CEE value. But the masses are taken
        # post-CEE. Hence, this might not detect all mergers, while not
        # classifying any surviving binary as merger.
        merger = cf.check_for_RLO(mc1_i, rc1_f, mc2_i, rc2_f,
            separation_postCEE / const.Rsun, self.CEE_tolerance_err)

        if merger:
            if verbose:
                print("system merges within the CEE, prior to stable MT")
            return mc1_i, rc1_f, mc2_i, rc2_f, \
                separation_postCEE / const.Rsun, orbital_period_postCEE, merger
        else:
            if verbose:
                print("system survives initial CEE detachment")
                print("donor core mass / core radius:", mc1_f, rc1_i)
                print("companion core mass / core radius:", mc2_f, rc2_i)


        # An assumed stable mass transfer case after postCEE with
        # fully non-conservative MT and mass lost from the vicinity
        # of the accretor:
        orbital_period_f = cf.period_change_stable_MT(
            orbital_period_postCEE, Mdon_i=mc1_i, Mdon_f=mc1_f,
            Macc_i=mc2_i, alpha=0.0, beta=1.0)

        if verbose:
            print("during the assumed stable MT phase after postCE")
            print("the orbit changed from postCEE : ", orbital_period_postCEE)
            print("to : ", orbital_period_f)

        # Calculate the post-CEE separation
        separation_f = cf.orbital_separation_from_period(orbital_period_f,
            mc1_f, mc2_f)

        # Check once more to see if the system has merged during stable MT
        merger = cf.check_for_RLO(mc1_f, rc1_f, mc2_f, rc2_f, separation_f, 
            self.CEE_tolerance_err)

        if verbose:
            if merger:
                print("system merges within the CEE")
            else:
                print("system survives CEE, the whole CE is ejected and the "
                      "orbit is adopted according to the stable MT")

        return (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
                merger)

    def CEE_two_phases_windloss(self, donor, mc1_i, rc1_i, donor_type,
                                comp_star, mc2_i, rc2_i, comp_type,
                                double_CE, separation_postCEE, verbose=False):
        """Calculate the post-common-envelope parameters upon exiting a CEE.

        This prescription assumes the he_core_mass/radius (or
        co_core_mass/radius for CEE of stripped_He*) becomes the value
        determined by MESA pre CEE. It stripes the envelope between the core
        defined by the core_definition_H/He_fraction and the MESA core by
        assuming an instantaneous wind loss phase from the donor (or both
        components in case of a double CE) after the successful ejection of the
        outer envelope part lost during the fast CE phase.

        Parameters
        ----------
        donor : SingleStar object
            The donor star
        mc1_i : float
            Core mass of the donor after the fast CE phase (in Msun)
        rc1_i : float
            Core radius of the donor after the fast CE phase (in Rsun)
        comp_star : SingleStar object
            The companion star
        mc2_i : float
            Core mass of the companion after the fast CE phase (in Msun)
        rc2_i : float
            Core radius of the companion after the fast CE phase (in Rsun)
        separation_postCEE : float
            Binary's separation after the fast CE phase (in cm)
        verbose : bool
            In case we want information about the CEE.

        Returns
        -------
        m1c_f: float
            Donor mass after the complete CE (in Msun)
        r1c_f: float
            Donor radius after the complete CE (in Rsun)
        m2c_f: float
            Companion mass after the complete CE (in Msun)
        r2c_f: float
            Companion radius after the complete CE (in Rsun)
        separation_f : float
            Binary's final separation upon exiting the CE (in Rsun)
        orbital_period_f : float
            Binary's final orbital period upon exiting the CE (in days)
        merger : bool
            Whether the binary merged in the CE
        """
        # First find the post-CE parameters for each star
        mc1_f, rc1_f, mc2_f, rc2_f = self.CEE_adjust_post_CE_core_masses(
            donor, mc1_i, rc1_i, donor_type, comp_star, mc2_i, rc2_i,
            comp_type, double_CE, verbose=verbose)

        # calculate the orbital period after CEE (before adjustment)
        orbital_period_postCEE = cf.orbital_period_from_separation(
            separation_postCEE / const.Rsun, mc1_i, mc2_i)

        # First, check if merger happens before wind loss phase. Note this uses
        # the final core radii as we are beyond the point of radius contraction
        # (the final radii are a better estimate for the radius after
        #  contraction) then the post-CEE value. But the masses are taken
        # post-CEE. Hence, this might not detect all mergers, while not
        # classifying any surviving binary as merger.
        merger = cf.check_for_RLO(mc1_i, rc1_f, mc2_i, rc2_f,
            separation_postCEE / const.Rsun, self.CEE_tolerance_err)

        if merger:
            if verbose:
                print("system merges within the CEE, prior to wind loss")
            return mc1_i, rc1_f, mc2_i, rc2_f, \
                separation_postCEE / const.Rsun, orbital_period_postCEE, merger
        else:
            if verbose:
                print("system survives initial CEE detachment")
                print("donor core mass / core radius:", mc1_f, rc1_i)
                print("companion core mass / core radius:", mc2_f, rc2_i)


        # An assumed wind loss after postCEE with mass lost
        # from the vicinity of the donor
        orbital_period_f = cf.period_change_stable_MT(
            orbital_period_postCEE, Mdon_i=mc1_i, Mdon_f=mc1_f,
            Macc_i=mc2_i, alpha=1.0, beta=0.0)

        if double_CE:
            # a wind mass loss from the 2nd star assumed to be
            # happening at the same time
            orbital_period_f = cf.period_change_stable_MT(
                orbital_period_f, Mdon_i=mc2_i, Mdon_f=mc2_f,
                Macc_i=mc1_f, alpha=1.0, beta=0.0)
        if verbose:
            print("during the assumed windloss phase after postCE")
            print("the orbit changed from postCEE : ", orbital_period_postCEE)
            print("to : ", orbital_period_f)

        # Calculate the post-CEE separation
        separation_f = cf.orbital_separation_from_period(orbital_period_f,
            mc1_f, mc2_f)

        # Check once more to see if the system has merged during stable MT
        merger = cf.check_for_RLO(mc1_f, rc1_f, mc2_f, rc2_f, separation_f,
            self.CEE_tolerance_err)

        if verbose:
            if merger:
                print("system merges within the CEE")
            else:
                print("system survives CEE, the whole CE is ejected and the "
                      "orbit is adopted according to the wind mass loss")

        return (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
                merger)
    
    def calculate_binding_energy(self,core_definition_H_fraction,star,common_envelope_alpha_thermal = 1):
        #Complete the function: Step 4
        pass
    
    
    
    def calculate_lamda_CE(self,core_definition_H_fraction,star,common_envelope_alpha_thermal = 1):
        #Complete the function: Step 4
        pass
    
    def CEE_simple_alpha_prescription(
            self, binary, donor, comp_star, double_CE=False,
            verbose=False, common_envelope_option_after_succ_CEE=MODEL[
                'common_envelope_option_after_succ_CEE'],
            core_definition_H_fraction=MODEL['core_definition_H_fraction'],
            core_definition_He_fraction=MODEL['core_definition_He_fraction'],
            mass_loss_during_CEE_merged=MODEL['mass_loss_during_CEE_merged']):
        """Apply the alpha-lambda common-envelope prescription from Lab1"""
        # Get stars properties: Step 1
     
        # Get binary parameters: Step 2
        
        #Define the profiles: Step 3
        

        lambda1_CE,mc1_i,rc1_i = self.calculate_lamda_CE(core_definition_H_fraction,donor_prof,self.common_envelope_alpha_thermal)
        
        if double_CE:
            lambda2_CE,mc2_i,rc2_i = self.calculate_lamda_CE(core_definition_H_fraction,comp_prof,self.common_envelope_alpha_thermal)
        else:
            mc2_i = comp_star.mass
            rc2_i = 10**comp_star.log_R
            lambda2_CE = np.nan

        # Calculate the binding energy: Step 5
        ebind_i = 

        if double_CE:
            # Calculate the binding energy: Step 5
            ebind_i += 
        
        
        #Calculate the seperation 
        
        separation_i = # Step 6

        eorb_i = # Step 7

        eorb_postCEE = # Step 7
        
        separation_postCEE = #Step 7


        # Check to make sure final orbital separation is positive
        if separation_postCEE < -self.CEE_tolerance_err:
            raise ValueError("CEE problem, negative postCEE separation")

        if verbose:
            print("CEE alpha-lambda prescription")
            print("ebind_i", ebind_i)
            print("eorb_i", eorb_i)
            print("eorb_postCEE", eorb_postCEE)
            print("DEorb", eorb_postCEE - eorb_i)
            print("separation_i in Rsun", separation_i/const.Rsun)
            print("separation_postCEE in Rsun", separation_postCEE/const.Rsun)

            
        def normalize(s):
            return unicodedata.normalize("NFKC", s).strip()

        # normalize states
        state = normalize(donor.state)
        comp_state = normalize(comp_star.state)

        # normalize lists
        STAR_STATE_POST_MS_clean = [normalize(s) for s in STAR_STATE_POST_MS]
        STAR_STATE_POST_HeMS_clean = [normalize(s) for s in STAR_STATE_POST_HeMS]

        # donor type
        if state in STAR_STATE_POST_MS_clean:
            donor_type = 'He_core'
        elif state in STAR_STATE_POST_HeMS_clean:
            donor_type = 'CO_core'
            core_element_fraction_definition = 0.1
        else:
            closest = difflib.get_close_matches(state, STAR_STATE_POST_MS_clean + STAR_STATE_POST_HeMS_clean)
            raise ValueError(f"type = {state!r} of donor not recognized. Closest match: {closest}")
        
        if double_CE:
            # companion type
            if comp_state in STAR_STATE_POST_MS_clean:
                comp_type = 'He_core'
            elif comp_state in STAR_STATE_POST_HeMS_clean:
                comp_type = 'CO_core'
                core_element_fraction_definition = 0.1
            else:
                closest = difflib.get_close_matches(comp_state, STAR_STATE_POST_MS_clean + STAR_STATE_POST_HeMS_clean)
                raise ValueError(f"type = {comp_state!r} of companion not recognized. Closest match: {closest}")
        else:
            comp_type = "not_giant_companion"
            
        # Calculate the post-CE binary properties
        if (common_envelope_option_after_succ_CEE
            == "one_phase_variable_core_definition"):
            (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
             merger) = self.CEE_one_phase_variable_core_definition(donor, 
                        mc1_i, rc1_i, comp_star, mc2_i, rc2_i, 
                        separation_postCEE, verbose=verbose)
        elif (common_envelope_option_after_succ_CEE
              == "two_phases_stableMT"):
            (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
             merger) = self.CEE_two_phases_stableMT(donor, mc1_i, rc1_i,
                                                    donor_type, comp_star,
                                                    mc2_i, rc2_i, comp_type,
                                                    double_CE,
                                                    separation_postCEE,
                                                    verbose=verbose)
        elif (common_envelope_option_after_succ_CEE
              == "two_phases_windloss"):
            (mc1_f, rc1_f, mc2_f, rc2_f, separation_f, orbital_period_f,
             merger) = self.CEE_two_phases_windloss(donor, mc1_i, rc1_i,
                                                    donor_type, comp_star,
                                                    mc2_i, rc2_i, comp_type,
                                                    double_CE,
                                                    separation_postCEE,
                                                    verbose=verbose)
        else:
            raise ValueError("Not accepted option in "
                             "common_envelope_option_after_succ_CEE = "
                             f"{common_envelope_option_after_succ_CEE}, do "
                             "not know how to proceed")

        # Adjust stellar and binary parameters depending on whether the system
        # mergers in the CE or not
        if merger:
            Mejected_donor = 0.0
            Mejected_comp = 0.0

                # Adjust the binary and single star properties due to merger
            self.CEE_adjust_binary_upon_merger(binary, donor, comp_star, m1_i,
                                                   m2_i, donor_type, comp_type,
                                                   Mejected_donor, Mejected_comp,
                                                   verbose)
        else:
            # Adjust the binary and single star properties due to ejection
            self.CEE_adjust_binary_upon_ejection(binary, donor, mc1_f, rc1_f,
                donor_type, comp_star, mc2_f, rc2_f, comp_type, double_CE,
                separation_f, orbital_period_f,
                common_envelope_option_after_succ_CEE,
                core_definition_H_fraction,
                core_definition_He_fraction,
                verbose)

        return



    def CEE_adjust_binary_upon_ejection(self, binary, donor, mc1_f, rc1_f,
                donor_type, comp_star, mc2_f, rc2_f, comp_type, double_CE,
                separation_f, orbital_period_f,
                common_envelope_option_after_succ_CEE,
                core_definition_H_fraction,
                core_definition_He_fraction,
                verbose=False):
        """Update the binary and component stars upon exiting a CEE.

        The binary's parameters (orbital period, separation, state, etc.) are
        updated along with the donor's (and in the case of a double CE the
        companion's) parameters as well. Note that certain parameters are set
        to np.nan if they are undetermined after the CE.

        Parameters
        ----------
        binary: BinaryStar object
            The binary system
        donor : SingleStar object
            The donor star
        mc1_f : float
            Final core mass of the donor (in Msun)
        rc1_f : float
            Final core radius of the donor (in Rsun)
        donor_type : string
            Descriptor for the stellar type of the donor's core
        comp_star : SingleStar object
            The companion star
        mc2_f : float
            Final core mass of the companion (in Msun)
        rc2_f : float
            Final core radius of the companion (in Rsun)
        comp_type : string
            Descriptor for the stellar type of the companion or it's core
        double_CE : bool
            Whether the CEE is a double CE or not
        separation_f : float
            Final binary separation upon exiting the CEE (in Rsun)
        orbital_period_f : float
            Final orbital period upon exiting the CEE (in days)
        common_envelope_option_after_succ_CEE : string
            Which type of post-common envelope evolution is used to remove
            the final layers around the core
        core_definition_H_fraction : float
            The fractional abundance of H defining the He core (0.3, 0.1, or 
            0.01)
        core_definition_He_fraction : float
            The fractional abundance of He defining the CO core (typically 0.1)
        verbose : bool
            In case we want information about the CEE.
        """
        # Adjust binary properties
        binary.separation = separation_f
        binary.orbital_period = orbital_period_f
        binary.eccentricity = 0.0
        binary.state = 'detached'
        binary.event = None

        # If the binary is sufficiently evolved (core helium exhaustion), then
        # don't sent it to the detached step after successful ejection, but
        # send the binary directly to the core collapse step to calculate the
        # explosion
        if donor.state == 'stripped_He_Central_He_depleted':
            if donor == binary.star_1:
                binary.event = 'CC1'
            elif donor == binary.star_2:
                binary.event = 'CC2'

        if verbose:
            print("CEE succesfully ejected")
            print("new orbital period = ", binary.orbital_period)
            print("new orbital separation = ", binary.separation)
            print("binary event : ", binary.event)
            print("double CEE : ", double_CE)

        # Set binary values that are not set in CE step to np.nan
        for key in BINARYPROPERTIES:

            # the binary attributes that are changed in the CE step
            if key in ["separation", "orbital_period",
                       "eccentricity", "state", "event"]:
                continue

            # the binary attributes that keep the same value from the
            # previous step
            if key in ["time", "V_sys", "mass_transfer_case",
                       "nearest_neighbour_distance"]:
                continue

            # the rest become np.nan
            setattr(binary, key, np.nan)

        # Create list of stars that need updated parameters. If normal CE,
        # then only donor has updated properties
        stars = [donor]
        star_types = [donor_type]
        core_masses = [mc1_f]
        core_radii = [rc1_f]

        # If a double CE, then the companion's properties are updated, too
        if double_CE:
            stars.append(comp_star)
            star_types.append(comp_type)
            core_masses.append(mc2_f)
            core_radii.append(rc2_f)

        # Adjust stellar properties
        for star, star_type, core_mass, core_radius in zip(
                stars, star_types, core_masses, core_radii):
            star.mass = core_mass
            star.log_R = np.log10(core_radius)
            attributes_changing = [
                                    "mass",
                                    "log_R",
                                    "state"
                                    ]

            if star_type == 'He_core':
                if common_envelope_option_after_succ_CEE in [
                        "one_phase_variable_core_definition"]:
                    star.surface_h1 = core_definition_H_fraction
                else:
                    star.surface_h1 = 0.01
                star.he_core_mass = core_mass
                star.he_core_radius = core_radius
                if star.metallicity is None:
                    star.surface_he4 = 1 - star.surface_h1 - 0.0142
                else:
                    star.surface_he4 = 1.0-star.surface_h1-star.metallicity
                star.log_LH = -1e99
                attributes_changing.extend([
                                        "surface_h1",
                                        "surface_he4",
                                        "log_LH",
                                        "log_LHe",
                                        'he_core_mass',
                                        'he_core_radius'
                    ])
            elif star_type == 'CO_core':
                star.he_core_mass = core_mass
                star.he_core_radius = core_radius
                star.co_core_mass = core_mass
                star.co_core_radius = core_radius
                star.surface_h1 = 0.0
                if common_envelope_option_after_succ_CEE in [
                        "one_phase_variable_core_definition"]:
                    star.surface_he4 = core_definition_He_fraction
                else:
                    star.surface_he4 = 0.1
                star.log_LH = -1e99
                star.log_LHe = -1e99
                attributes_changing.extend([
                                        "surface_h1",
                                        "surface_he4",
                                        "log_LH",
                                        "log_LHe",
                                        'he_core_mass',
                                        'he_core_radius',
                                        'co_core_mass',
                                        'co_core_radius'

                    ])
            elif star_type == "not_giant_companion":
                continue
            else:
                raise ValueError("Unrecognized star type:", star_type)

            # Update state of star
            state_old = star.state
            star.state = check_state_of_star(star)
            if verbose:
                print("star state of before/ after CE : ",
                        state_old, star.state)

            # Set star values that are unchanged in CE step to np.nan
            for key in STARPROPERTIES:

                # the singlestar attributes that are changed
                # in the CE step
                if key in attributes_changing:
                    continue

                # the singlestar attributes that keep the same value
                # from previous step if not in attributes_changing
                if key in [
                        "surface_h1", "surface_he4", "log_LH",
                        "log_LHe", "he_core_mass", "he_core_radius",
                        "co_core_mass", "co_core_radius", "center_h1",
                        "center_he4", "center_c12", "center_o16",
                        "center_n14", "metallicity", "log_LZ",
                        "log_Lnuc", "c12_c12", "center_gamma",
                        "avg_c_in_c_core"]:
                    continue

                # the rest become NaN's
                setattr(star, key, np.nan)
        return

    

    def CEE_adjust_binary_upon_merger(self, binary, donor, comp_star, m1_i,
                                      m2_i, donor_type, comp_type,
                                      Mejected_donor, Mejected_comp,
                                      verbose=False):
        """Update the binary and component stars upon merging within a CEE.

        The binary's state and event are updated along with the donor and 
        companion star masses and radii corresponding to a merger event.

        Parameters
        ----------
        binary: BinaryStar object
            The binary system
        donor : SingleStar object
            The donor star
        comp_star : SingleStar object
            The companion star
        m1_i : float
            Mass of the donor upon entering a CE (in Msun)
        m2_i : float
            Mass of the companion upon entering a CE (in Msun)
        donor_type : string
            Descriptor for the stellar type of the donor's core
        comp_type : string
            Descriptor for the stellar type of the companion or it's core
        Mejected_donor : float
            How much mass is ejected from the donor upon merger (in Msun)
        Mejected_comp : float
            How much mass is ejected from the companion upon merger (in Msun)
        verbose : bool
            In case we want information about the CEE.
        """
        # system merges
        binary.state = 'merged'
        if binary.event in ["oCE1", "oDoubleCE1"]:
            binary.event = "oMerging1"
        if binary.event in ["oCE2", "oDoubleCE2"]:
            binary.event = "oMerging2"

        if verbose:
            print("system merges due to one of the two star's core filling"
                    "its RL")

        donor.mass = m1_i - Mejected_donor
        donor.log_R = np.nan
        comp_star.mass = m2_i - Mejected_comp
        comp_star.log_R = np.nan
        if donor_type == 'CO_core':
            donor.he_core_mass = m1_i - Mejected_donor
            donor.he_core_radius = np.nan
        if comp_type == 'CO_core':
            comp_star.he_core_mass = m2_i - Mejected_comp
            comp_star.he_core_radius = np.nan

        return
