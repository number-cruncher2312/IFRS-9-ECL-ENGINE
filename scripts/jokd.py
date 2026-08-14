def assign_stage(
        pd_origin: float ,
        pd_current: float, 
        default_status: int
        ) -> int:

        if default_status == 1:
                return 3
        elif pd_current/pd_origin >=2:
            if pd_current - pd_origin >= 0.005:
                return 2
        else:
              return 1
            

