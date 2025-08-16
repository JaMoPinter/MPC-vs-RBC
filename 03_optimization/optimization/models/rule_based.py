from ..base import BaseOptimizer



class RuleBasedOptimizer(BaseOptimizer):
    """
    Simple rule-based: charge if PV > load and battery not full, discharge if the reverse.
    """

    # def __init__(self, config: dict):
    #     """
    #     Initialize the Rule-Based Optimizer.
    #     """

    #     super().__init__(config=config) # only necessary if this class has its own init method


    def optimize(self, t_now, fc_slice) -> dict:
        """ Nothing to optimize here. """

        return {'action': None} # TODO: Check how to handle rule based


    def update_soe(self, t_now, action, gt) -> float:

        net_load = gt  # TODO: Rewrite net_load to gt

        if net_load <= 0:  # Charge the battery
            eta = self.eta_ch

            # Calculate the available charge of the battery. If available charge is more than net load, charge with net load. 
            # Otherwise, charge with available charge. Thus battery should be fully charged at the end.

            available_cap = self.cap_max - self.soe_now
            needed_cap = - net_load * eta * self.gt_inc 

            if available_cap >= needed_cap:
                pb = max(net_load, self.pb_min)  # Charge with net load, but not below pb_min

            else:  # Not enough capacity avalable to compensate net load
                pb = - available_cap / (eta * self.gt_inc)  # Power to charge the battery with available capacity
                pb = max(pb, self.pb_min)


        else:  # Discharge the battery
            eta = 1 / self.eta_dis

            # Calculate the available discharge of the battery. If available discharge is more than net load, discharge with net load.
            # Otherwise, discharge with available discharge. Thus battery should be fully discharged at the end.

            available_cap = self.soe_now - self.cap_min
            needed_cap = net_load * eta * self.gt_inc

            if available_cap >= needed_cap:
                pb = min(net_load, self.pb_max)  # Discharge with net load, but not above pb_max

            else:  # Not enough capacity avalable to compensate net load
                pb = available_cap / (eta * self.gt_inc)
                pb = min(pb, self.pb_max)


        pg = net_load - pb  # Grid Power
        soe_new = self.soe_now - pb * self.gt_inc * eta


        # check if the new state of charge is within limits
        if round(soe_new, 6) > self.cap_max or round(soe_new, 6) < self.cap_min:  # TODO: Maybe should relax this a bit to tolerate numerical errors?
            raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.")        

        self.results_realization[t_now] = {
            'timestamp': t_now,  # Current timestamp
            'action': pb,  # Power setpoint for the battery at t_now
            'pg': pg,  # Grid power after applying the action
            "gt": net_load,  # Ground truth net load at t_now
            'soe_now': self.soe_now,  # Current state of charge before applying the action
            'soe_new': soe_new  # New state of charge after applying the action
        }

        self.soe_now = soe_new  # Update the current state of charge for the next iteration


        return soe_new 

