import copy


class FedTuningTuner:

    def __init__(self, *, alpha: float, beta: float, gamma: float, initial_M: int, initial_E: float,
                 M_min: int, M_max: int, E_min: float, E_max: float):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.initial_M = initial_M
        self.initial_E = initial_E
        self.M_min = M_min
        self.M_max = M_max
        self.E_min = E_min
        self.E_max = E_max

        self.current_M = self.initial_M
        self.current_E = self.initial_E

        # skip the first training rounds
        self.n_round_skipped = 2

        # FL settings
        self.S_cur = FLSetting()  # Current FL hyper-parameter set
        self.S_prv = FLSetting()  # Previous FL hyper-parameter set

        # decision only made when accuracy is improved by at least eps_accuracy
        self.eps_accuracy = 0.01

    def update(self, *, model_accuracy: float, time_cost: float, computation_cost: float, communication_cost: float) -> tuple[int, float]:

        self.S_cur.add_cost(time=time_cost, computation=computation_cost, communication=communication_cost)

        if model_accuracy - self.S_prv.model_accuracy > self.eps_accuracy:

            self.S_cur.normalize_cost(accuracy_length=model_accuracy-self.S_prv.model_accuracy)

            if self.n_round_skipped <= 0:

                delta_M = self.alpha * abs(self.S_cur.tot_time - self.S_prv.tot_time) / self.S_cur.tot_time \
                          - self.beta * abs(self.S_cur.tot_computation - self.S_prv.tot_computation) / self.S_cur.tot_computation \
                          - self.gamma * abs(self.S_cur.tot_communication - self.S_prv.tot_communication) / self.S_cur.tot_communication

                delta_E = -self.alpha * abs(self.S_cur.tot_time - self.S_prv.tot_time) / self.S_cur.tot_time \
                          - self.beta * abs(self.S_cur.tot_computation - self.S_prv.tot_computation) / self.S_cur.tot_computation \
                          + self.gamma * abs(self.S_cur.tot_communication - self.S_prv.tot_communication) / self.S_cur.tot_communication

                if delta_M > 0:
                    self.current_M += 1
                    self.current_M = min(self.current_M, self.M_max)
                else:
                    self.current_M -= 1
                    self.current_M = max(self.current_M, self.M_min)

                if delta_E > 0:
                    self.current_E += 1
                    self.current_E = min(self.current_E, self.E_max)
                else:
                    self.current_E -= 1
                    self.current_E = max(self.current_E, self.E_min)
            else:
                # skip the first training rounds
                self.n_round_skipped -= 1

            self.S_cur.model_accuracy = model_accuracy
            self.S_prv = copy.deepcopy(self.S_cur)
            self.S_cur = FLSetting()

        return self.current_M, self.current_E


class FLSetting:
    # Store information under a set of FL hyper-parameters
    def __init__(self):
        self.tot_time = 0
        self.tot_computation = 0
        self.tot_communication = 0
        self.model_accuracy = 0

    def add_cost(self, *, time, computation, communication) -> None:
        self.tot_time += time
        self.tot_computation += computation
        self.tot_communication += communication

    def normalize_cost(self, *, accuracy_length) -> None:
        self.tot_time /= accuracy_length
        self.tot_computation /= accuracy_length
        self.tot_communication /= accuracy_length

    def __str__(self):
        return f'self.tot_time = {self.tot_time}, ' \
               f'self.tot_computation = {self.tot_computation}, ' \
               f'self.tot_communication = {self.tot_communication}'

