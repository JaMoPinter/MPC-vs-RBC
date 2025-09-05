# Rule-Based enhanced deterministic MPC


from .mpc_det import MpcDetOptimizer
import pandas as pd
from pathlib import Path
from utils import map_building_to_pv_num_orientation


class MpcRuleOptimizer(MpcDetOptimizer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def update_soe(self, t_now, decision, gt):
        
        fc_average = self.fc_exp['expected_value'].mean()
        if fc_average > 0:  # Expect prosumption to be positive => Net consumer
            self.update_soe_net_consumer(t_now, decision, gt)
        else:
            self.update_soe_net_producer(t_now, decision, gt)



    def update_soe_net_producer(self, t_now, decision, gt):
        """ Net Producer Mode
        We expect to have on average a PV surplus for this day. Therefore, we should be able to match the local consumption
        via Rule-Based behavior to minimize grid imports.

        To be precise, whenever we would import, instead take from the battery.
        """
        print("\n\nNET PRODUCER MODE\n\n")
        
        pb_mpc = decision.get('pb')
        pg_mpc = gt - pb_mpc

        if pg_mpc > 0:  # IMPORTING
            pb_desired = min(pb_mpc + pg_mpc, self.pb_max)
        else:
            pb_desired = pb_mpc  # follow MPC decision


        if pb_desired <= 0:  # CHARGING
            eta = self.eta_ch
            pb = pb_desired  # Should always be within limits according to MPC logic

        else:  # DISCHARGING
            eta = 1/self.eta_dis

            # Check here if we can discharge more or if we hit our battery limits!
            available_cap = self.soe_now - self.cap_min
            needed_cap = pb_desired * eta * self.gt_inc

            if available_cap >= needed_cap:
                pb = pb_desired  # Discharge with desired power to match local consumption

            else: # Discharge to minimal capacity
                pb = available_cap / (eta * self.gt_inc)

        pg = gt - pb
        soe_new = self.soe_now - pb * self.gt_inc * eta

        if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
            raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.") 
        
        if soe_new > self.cap_max:
            soe_new = self.cap_max
        if soe_new < self.cap_min:
            soe_new = self.cap_min

        self.results_realization[t_now] = {
            'timestamp': t_now,
            'action': pb,       # Power setpoint for the battery at t_now
            'pb': pb,   # Power setpoint for the battery at t_now
            'pg': pg,           # Grid power after applying the action
            'gt': gt,               # Ground truth at t_now
            'soe_now': self.soe_now,  # Current state of energy before applying the action
            'soe_new': soe_new      # New state of energy after applying the action
        }


        self.soe_now = soe_new
        return soe_new



    def update_soe_net_consumer(self, t_now, decision, gt):
            """ Net Consumer Mode
            We expect to have on average a PV deficit for the upcoming day. Importing therefore is a necessity. 
            Whenever we would export, instead charge the battery. The times/volumes at which the Battery system is charged are solved via scheduling. 
            Whenever, the battery is discharged, it is to cover the local consumption. Therefore, enable rule-based behavior for those times.

            So basically, this model is rule-based except for the grid imports that cover the charging of the battery.

            """
            print("\n\nNET CONSUMER MODE\n\n")
            
            pb_mpc = decision.get('pb')
            pg_mpc = gt - pb_mpc

            if pg_mpc > 0:  # IMPORTING
                pb_desired = pb_mpc  # follow MPC decision

            else:  # EXPORTING
                pb_desired = max(pb_mpc + pg_mpc, self.pb_min)



            if pb_desired >= 0:  # DISCHARGING
                eta = 1/self.eta_dis
                pb = pb_desired  # Should always be within limits according to MPC logic

                # We want to cover our load by using the battery

                available_cap = self.soe_now - self.cap_min
                needed_cap = gt * eta * self.gt_inc

                if available_cap >= needed_cap:
                    pb = min(gt, self.pb_max)  # Discharge with gt but not above pb_max

                else:
                    pb = available_cap / (eta * self.gt_inc)
                    pb = min(pb, self.pb_max)

            else:  # CHARGING
                eta = self.eta_ch

                # Check if we can charge more or if we hit our battery limits
                available_cap = self.cap_max - self.soe_now
                needed_cap = - pb_desired * eta * self.gt_inc

                if available_cap >= needed_cap:
                    pb = pb_desired  # Charge with desired power to match local production
                
                else: # Charge to maximal capacity
                    pb = - available_cap / (eta * self.gt_inc)

            pb = max(pb, self.pb_min)
            pb = min(pb, self.pb_max)

            pg = gt - pb
            soe_new = self.soe_now - pb * self.gt_inc * eta

            if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
                raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.") 
            
            if soe_new > self.cap_max:
                soe_new = self.cap_max
            if soe_new < self.cap_min:
                soe_new = self.cap_min

            self.results_realization[t_now] = {
                'timestamp': t_now,
                'action': pb,       # Power setpoint for the battery at t_now
                'pb': pb,   # Power setpoint for the battery at t_now
                'pg': pg,           # Grid power after applying the action
                'gt': gt,               # Ground truth at t_now
                'soe_now': self.soe_now,  # Current state of energy before applying the action
                'soe_new': soe_new      # New state of energy after applying the action
            }


            self.soe_now = soe_new
            return soe_new
    




class IdealRuleOptimizer(MpcRuleOptimizer):

    def __init__(self, *args, **kwargs):
        self.super_fc_long = None  # Store the ground truth to decrease number of file access
        super().__init__(*args, **kwargs)



    def _prepare_forecast(self, forecast: pd.DataFrame) -> pd.DataFrame:
        ''' Instead of using the provided forecast => use ground truth as forecast. '''

        if self.super_fc_long is None:
            # Load the full gt
            num_pv_modules, orientation = map_building_to_pv_num_orientation(self.b)

            path = Path(f'01_data/prosumption_data/{self.mpc_freq}min/prosumption_{self.b}_num_pv_modules_{num_pv_modules}_pv_{orientation}_hp_1.0.csv')
            df = pd.read_csv(path, parse_dates=['index'], index_col='index', usecols=['index', 'P_TOT'])
            df.index.name = 'timestamp'
            self.super_fc_long = df
        
        self.super_fc = self.super_fc_long.copy()
        self.super_fc = self.super_fc.loc[forecast.index.get_level_values('timestamp')]
        self.super_fc['P_TOT'] = self.super_fc['P_TOT'] / 1000.0  # Convert from W to kW

        self.super_fc.rename(columns={'P_TOT': 'expected_value'}, inplace=True)
        return self.super_fc
    






# class MpcRuleConsumerOptimizer(MpcDetOptimizer):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


#     def update_soe(self, t_now, decision, gt):
#         """ Net Consumer Mode
#         We expect to have on average a PV deficit for the upcoming day. Importing therefore is a necessity. 
#         Whenever we would export, instead charge the battery. The times/volumes at which the Battery system is charged are solved via scheduling. 
#         Whenever, the battery is discharged, it is to cover the local consumption. Therefore, enable rule-based behavior for those times.

#         So basically, this model is rule-based except for the grid imports that cover the charging of the battery.

#         """
        
#         pb_mpc = decision.get('pb')
#         pg_mpc = gt - pb_mpc

#         if pg_mpc > 0:  # IMPORTING
#             pb_desired = pb_mpc  # follow MPC decision

#         else:  # EXPORTING
#             pb_desired = max(pb_mpc + pg_mpc, self.pb_min)



#         if pb_desired >= 0:  # DISCHARGING
#             eta = 1/self.eta_dis
#             pb = pb_desired  # Should always be within limits according to MPC logic

#             # We want to cover our load by using the battery

#             available_cap = self.soe_now - self.cap_min
#             needed_cap = gt * eta * self.gt_inc

#             if available_cap >= needed_cap:
#                 pb = min(gt, self.pb_max)  # Discharge with gt but not above pb_max

#             else:
#                 pb = available_cap / (eta * self.gt_inc)
#                 pb = min(pb, self.pb_max)

#         else:  # CHARGING
#             eta = self.eta_ch

#             # Check if we can charge more or if we hit our battery limits
#             available_cap = self.cap_max - self.soe_now
#             needed_cap = - pb_desired * eta * self.gt_inc

#             if available_cap >= needed_cap:
#                 pb = pb_desired  # Charge with desired power to match local production
            
#             else: # Charge to maximal capacity
#                 pb = - available_cap / (eta * self.gt_inc)

#         pb = max(pb, self.pb_min)
#         pb = min(pb, self.pb_max)

#         pg = gt - pb
#         soe_new = self.soe_now - pb * self.gt_inc * eta

#         if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
#             raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.") 
        
#         if soe_new > self.cap_max:
#             soe_new = self.cap_max
#         if soe_new < self.cap_min:
#             soe_new = self.cap_min

#         self.results_realization[t_now] = {
#             'timestamp': t_now,
#             'action': pb,       # Power setpoint for the battery at t_now
#             'pb': pb,   # Power setpoint for the battery at t_now
#             'pg': pg,           # Grid power after applying the action
#             'gt': gt,               # Ground truth at t_now
#             'soe_now': self.soe_now,  # Current state of energy before applying the action
#             'soe_new': soe_new      # New state of energy after applying the action
#         }


#         self.soe_now = soe_new
#         return soe_new
    
    




#     def lalalalaupdate_soe(self, t_now, decision, gt):
#         """ Net Producer Mode
#         We expect to have on average a PV surplus for this day. Therefore, we should be able to match the local consumption
#         via Rule-Based behavior to minimize grid imports.

#         To be precise, whenever we would import, instead take from the battery.
#         """
        
#         pb_mpc = decision.get('pb')
#         pg_mpc = gt - pb_mpc

#         if pg_mpc > 0:  # IMPORTING
#             pb_desired = min(pb_mpc + pg_mpc, self.pb_max)
#         else:
#             pb_desired = pb_mpc  # follow MPC decision


#         if pb_desired <= 0:  # CHARGING
#             eta = self.eta_ch
#             pb = pb_desired  # Should always be within limits according to MPC logic

#         else:  # DISCHARGING
#             eta = 1/self.eta_dis

#             # Check here if we can discharge more or if we hit our battery limits!
#             available_cap = self.soe_now - self.cap_min
#             needed_cap = pb_desired * eta * self.gt_inc

#             if available_cap >= needed_cap:
#                 pb = pb_desired  # Discharge with desired power to match local consumption

#             else: # Discharge to minimal capacity
#                 pb = available_cap / (eta * self.gt_inc)

#         pg = gt - pb
#         soe_new = self.soe_now - pb * self.gt_inc * eta

#         if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
#             raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.") 
        
#         if soe_new > self.cap_max:
#             soe_new = self.cap_max
#         if soe_new < self.cap_min:
#             soe_new = self.cap_min

#         self.results_realization[t_now] = {
#             'timestamp': t_now,
#             'action': pb,       # Power setpoint for the battery at t_now
#             'pb': pb,   # Power setpoint for the battery at t_now
#             'pg': pg,           # Grid power after applying the action
#             'gt': gt,               # Ground truth at t_now
#             'soe_now': self.soe_now,  # Current state of energy before applying the action
#             'soe_new': soe_new      # New state of energy after applying the action
#         }


#         self.soe_now = soe_new
#         return soe_new
    





# class MpcRuleConsumerOptimizer(MpcDetOptimizer):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


#     def update_soe(self, t_now, decision, gt):
#         """ Net Consumer Mode
#         We expect to have on average a PV deficit for the upcoming day. Therefore we should minimize grid exports if possible.

#         To be precise, whenever we would export, instead charge the battery.
#         """
        
#         pb_mpc = decision.get('pb')
#         pg_mpc = gt - pb_mpc

#         if pg_mpc > 0:  # IMPORTING
#             pb_desired = pb_mpc  # follow MPC decision

#         else:  # EXPORTING
#             pb_desired = max(pb_mpc + pg_mpc, self.pb_min)



#         if pb_desired >= 0:  # DISCHARGING
#             eta = 1/self.eta_dis
#             pb = pb_desired  # Should always be within limits according to MPC logic

#         else:  # CHARGING
#             eta = self.eta_ch

#             # Check if we can charge more or if we hit our battery limits
#             available_cap = self.cap_max - self.soe_now
#             needed_cap = - pb_desired * eta * self.gt_inc

#             if available_cap >= needed_cap:
#                 pb = pb_desired  # Charge with desired power to match local production
            
#             else: # Charge to maximal capacity
#                 pb = - available_cap / (eta * self.gt_inc)

#         pb = max(pb, self.pb_min)
#         pb = min(pb, self.pb_max)

#         pg = gt - pb
#         soe_new = self.soe_now - pb * self.gt_inc * eta

#         if round(soe_new, 5) > self.cap_max or round(soe_new, 5) < self.cap_min:
#             raise ValueError(f"State of charge out of bounds: {soe_new} kWh. Should be between {self.cap_min} and {self.cap_max} kWh.") 
        
#         if soe_new > self.cap_max:
#             soe_new = self.cap_max
#         if soe_new < self.cap_min:
#             soe_new = self.cap_min

#         self.results_realization[t_now] = {
#             'timestamp': t_now,
#             'action': pb,       # Power setpoint for the battery at t_now
#             'pb': pb,   # Power setpoint for the battery at t_now
#             'pg': pg,           # Grid power after applying the action
#             'gt': gt,               # Ground truth at t_now
#             'soe_now': self.soe_now,  # Current state of energy before applying the action
#             'soe_new': soe_new      # New state of energy after applying the action
#         }


#         self.soe_now = soe_new
#         return soe_new