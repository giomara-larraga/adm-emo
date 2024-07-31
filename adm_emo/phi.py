"""This code implements the PHI (Preference-based Hypervolume Indicator) and related decision assessment
 methods as introduced in the paper "A Performance Indicator for Interactive Evolutionary Multiobjective 
 Optimization Methods." It's designed for analyzing multiobjective optimization problems, taking into 
 account decision-maker preferences. The PHI indicator evaluates the performance of solutions relative
 to a reference point, focusing on the coverage of the desired solution region.

 To run the code to get the phi values you should run get_phi(),and for the decision phase you should run assess_decision_phase()

For inquiries or further details, contact the authors of the original paper.
 When using this code or its methodology in academic or research work, 
 please cite the paper appropriately to acknowledge the original work and its contributors.
 P. Aghaei Pour, S. Bandaru, B. Afsar, M. Emmerich and K. Miettinen, "A Performance Indicator
  for Interactive Evolutionary Multiobjective Optimization Methods," in IEEE Transactions
on Evolutionary Computation, doi: 10.1109/TEVC.2023.3272953.
 """

import numpy as np
from desdeo_tools.utilities.fast_non_dominated_sorting import dominates, fast_non_dominated_sort_indices
from desdeo_tools.utilities.quality_indicator import hypervolume_indicator


class phi():
    def __init__(self, ideal):
        """Initialize with an ideal point for hypervolume calculations."""
        self.name = 'test'
        self.ideal = ideal

    def check_rp_dominated(self, set_of_s, RP):
        """Check if the reference point (RP) is dominated by any solution in set_of_s."""
        r = False
        doms = []
        for s in set_of_s:
            if dominates(s, RP):
                doms.append(True)
                r = True
            else:
                doms.append(False)
        return r, doms

    def RP_dom_cal(self, set_of_s, RP, doms, nadir):
        """Calculate various hypervolume metrics when RP is dominated."""
        ind = np.where(doms)[0]
        nondoms = np.vstack((set_of_s, RP))[fast_non_dominated_sort_indices(np.vstack((set_of_s, RP)))[0][0]]
        max_phv = hypervolume_indicator(np.asanyarray(self.ideal).reshape(1, -1), nadir)
        all_phv = hypervolume_indicator(nondoms, nadir)
        rp_phv = hypervolume_indicator(np.asanyarray(RP).reshape(1, -1), nadir)
        pos_phv = hypervolume_indicator(np.asanyarray(set_of_s[ind]), nadir) - rp_phv
        neg_phv = all_phv - pos_phv - rp_phv
        if all_phv == 0:
            return 0, 0, 0
        else:
            return 1 + (pos_phv / max_phv), (pos_phv + rp_phv) / max_phv, neg_phv / max_phv, rp_phv / max_phv

    def RP_nondom_cal(self, set_of_s, RP, nadir):
        """Calculate various hypervolume metrics when RP is not dominated."""
        nondoms = np.vstack((set_of_s, RP))[fast_non_dominated_sort_indices(np.vstack((set_of_s, RP)))[0][0]]
        all_phv = hypervolume_indicator(nondoms, nadir)
        rp_phv = hypervolume_indicator(np.asanyarray(RP).reshape(1, -1), nadir)
        s_phv = hypervolume_indicator(np.asanyarray(set_of_s), nadir)
        nondom_area = all_phv - s_phv
        pos_phv = rp_phv - nondom_area
        neg_phv = all_phv - rp_phv
        if all_phv == 0:
            return 0, 0, 0
        else:
            return pos_phv / rp_phv, pos_phv / all_phv, neg_phv / all_phv, rp_phv

    def get_phi(self, set_of_s, RP, nadir):
        """Calculates PHI. Requires set of solutions, reference point, and nadir."""
        is_rp_dominated, doms = self.check_rp_dominated(set_of_s, RP)
        if is_rp_dominated:
            results = self.RP_dom_cal(set_of_s, RP, doms, nadir)
        else:
            results = self.RP_nondom_cal(set_of_s, RP, nadir)
        return results


class phi_decision():
    def __init__(self, n_interactions, indicator_values, nadir):
        """Initialize with the number of interactions, indicator values, and nadir for hypervolume calculations."""
        self.name = 'test'
        self.n_interactions = n_interactions
        self.indicator_values = indicator_values
        self.nadir = nadir

    def get_areas(self, rp1, rp2):
        """Calculate the shared hypervolume area between two reference points."""
        # Ensure rp1 and rp2 are 2D arrays
        if rp1.ndim == 1:
            rp1 = rp1.reshape(1, -1)
        if rp2.ndim == 1:
            rp2 = rp2.reshape(1, -1)

        dom21 = dominates(rp2.flatten(), rp1.flatten())
        dom12 = dominates(rp1.flatten(), rp2.flatten())
        hv_rp1 = hypervolume_indicator(rp1, self.nadir_1d)
        hv_rp2 = hypervolume_indicator(rp2, self.nadir_1d)
        hv_rp12 = hypervolume_indicator(np.vstack((rp1, rp2)), self.nadir_1d)
        self.hv_rp12 = hv_rp12
        if dom21:
            shared_area = hv_rp1
        elif dom12:
            shared_area = hv_rp2
        else:
            extra_area_in_rp1 = abs(hv_rp12 - hv_rp2)
            shared_area = hv_rp1 - extra_area_in_rp1
        return shared_area

    def interactions_areas(self, set_of_RPs, main_RP, n_interactions):
        """Calculate interaction areas for a set of reference points and a main reference point."""
        areas = []
        if n_interactions > 2:
            for s in set_of_RPs:
                areas.append(self.get_areas(s, main_RP))
        else:
            areas = self.get_areas(set_of_RPs, main_RP)
        return areas

    def get_weights(self, w, main_w):
        """Calculate the weights for the hypervolume shared areas."""
        return(w/self.hv_rp12)
        #return np.hstack((w / main_w, 1))

    def assess(self, w, assessment_values):
        """Assess the decision phase using weighted mean of assessment values."""
        assessment = np.mean(w * assessment_values)
        return assessment

    def assess_decision_phase(self, set_of_RPs, main_RP):
        """Assess the decision phase for a set of reference points and a main reference point."""
        # Reshape main_RP to 2D array if it is 1D
        if main_RP.ndim == 1:
            main_RP = main_RP.reshape(1, -1)

        # Ensure self.nadir is a 1D array
        self.nadir_1d = self.nadir.flatten()

        main_area = hypervolume_indicator(main_RP, self.nadir_1d)
        shared_areas = self.interactions_areas(set_of_RPs, main_RP, self.n_interactions)
        weights = self.get_weights(np.asarray(shared_areas), main_area)
        results = self.assess(np.asarray(weights), np.asarray(self.indicator_values))
        return results, weights