import pandas as pd
from datetime import timedelta


class SimulationEngine:
    """
    Given:
      - an optimizer model
      - a ForecastManager
      - a GroundTruthManager
      - a MPC frequency (in minutes)      # TODO: Cant this be derived from the optimizer or something else?
      - a GT base frequency (in minutes)  # TODO: Cant this be derived from the GTManager?
    
    this class runs the full rolling-horizon simulation:
       1.) at every MPC frequency -> rerun an optimization
       2.) for every GT base frequency until next MPC step -> Apply the action from the optimization to the GT
    """


    def __init__(self,
                 optimizer,
                 fc_manager,
                 gt_manager,
                 building: str,
                 mpc_freq: int):
        
        self.opt = optimizer
        self.fc_manager = fc_manager
        self.gt_manager = gt_manager
        self.building = building
        self.mpc_delta = timedelta(minutes=mpc_freq)
        self.gt_delta = timedelta(minutes=self.gt_manager.gt_freq)



    
    def run(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        
        """

        gt_full = self.gt_manager.get_gt(self.building, self.gt_manager.gt_freq)       # TODO: Check if get_gt works like that

        logs = []
        t_now = start
        #soe = self.opt.soe_initial

        while t_now < end:

            # 1. Decision at t_now
            fc_slice = self.fc_manager.get_forecast(t_now)                  # TODO: Check if get_forecasts works like that
            decision = self.opt.optimize(t_now, fc_slice)['action']    # TODO: Check if optimize works like that

            # 2. Determine next decision time
            t_next = t_now + self.mpc_delta

            # 3. Apply decision with GT frequency between [t_now, t_next)
            times = pd.date_range(start=t_now, end=t_next - self.gt_delta, freq=self.gt_delta)  # TODO: Check if this freq works correctly
            for t in times:

                #print("gt_full columns:", gt_full)

                gt = gt_full.loc[t, 'P_TOT']

                self.opt.update_soe(t, decision, gt)  # TODO: Check if update_soe works like that

                # log per gt_freq results (likely every minute)
                logs.append(self.opt.results_realization[t])
                # logs.append({
                #     'timestamp': t,
                #     'soe_old': soe,
                #     'soe_new': soe_new,
                #     'decision': decision,
                #     'grid': result['grid'],
                #     'gt': gt,
                #     'c_buy': result['c_buy'],
                #     'c_sell': result['c_sell'],
                #     'building': self.building
                # })

                #soe = soe_new

            t_now = t_next

        return pd.DataFrame(logs)
                    






