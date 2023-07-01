if config.STOCHASTIC:
    from Models.StochasticMuZero_torch_agent import StochasticMuZeroNetwork as TFTNetwork
else:
    from Models.MuZero_torch_agent import MuZeroNetwork as TFTNetwork

class Checkpoint:
    def __init__(self, epoch, q_score):
        self.epoch = epoch
        self.q_score = q_score
        self.live = True

    def get_model(self):
        model = TFTNetwork()
        if epoch == 0:
            return model.get_weights()
        else:
            return model.tft_load_model(self.epoch).get_weights()

    def get_live(self):
        return self.live

    def set_live(self, live):
        self.live = live

    def update_q_score(self, episode, prob):
        self.q_score = self.q_score - (0.01 / (episode * prob))

