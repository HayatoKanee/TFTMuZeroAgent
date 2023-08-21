class CheckWinWrapper():
    def __init__(self, type):
        if type == "standard":
            self.checker = Standard(type)
        elif type == "level":
            self.checker = Level(type)
    
    def run(self, env):
        result = self.checker.run_Check(env)
        return result
    

class CheckWin():
    def __init__(self, wincon):
        self.wincon = wincon
        self.env = None
    
    def cleanup(self):
        ...
    
    def get_Id(self):
        ...

    def check(self):
        ...
    
    def run_Check(self, env):
        self.env = env 
        gameOver = self.check()
        if gameOver == True:
            
            winner = self.get_Id() 
            reward = self.reward 
            self.runCleanup() 
            return reward, winner 
        return None

class Standard(CheckWin):
    def __init__(self, wincon):
        super().__init__(wincon)
        self.reward = 40 
        
    def check(self):
        num_alive = 0
        for key, player in self.env.PLAYERS.items():
            if player:
                if player.health <= 0:
                    self.env.NUM_DEAD += 1
                    self.env.game_round.NUM_DEAD = self.env.NUM_DEAD
                    self.env.pool_obj.return_hero(player)
                    self.env.kill_list.append(key)
                else:
                    num_alive += 1
        if num_alive <= 1 or self.env.game_round.current_round > 48:
            return True
    
    def getId(self):
        for player_id in self.env.agents:
            if self.env.PLAYERS[player_id] and self.env.PLAYERS[player_id].health > 0:
                return player_id
    
    def runCleanup(self):
        for player_id in self.env.agents:
            if self.env.PLAYERS[player_id] and self.env.PLAYERS[player_id].health > 0: 
                self.env.PLAYERS[player_id].won_game()
                self.env.rewards[player_id] = self.reward + self.env.PLAYERS[player_id].reward
                self.env._cumulative_rewards[player_id] = self.env.rewards[player_id]
                self.env.PLAYERS[player_id] = None  # Without this the reward is res
        self.env.terminations = {a: True for a in self.env.agents}

class Level(CheckWin):
    def __init__(self, wincon):
        super().__init__(wincon)
        self.reward = 40 
    
    def check(self): 
        for key, player in self.env.PLAYERS.items():
            if player:
                if player.level >= 6:
                    return True 
                
    def getId(self):
        for key, player in self.env.PLAYERS.items():
            if player:
                if player.level >= 6:
                    return key 
                
    def runCleanup(self):
        for player_id in self.env.agents:
            if self.env.PLAYERS[player_id] and self.env.PLAYERS[player_id].level >= 6: 
                self.env.PLAYERS[player_id].won_game()
                self.env.rewards[player_id] = self.reward + self.env.PLAYERS[player_id].reward
                self.env._cumulative_rewards[player_id] = self.env.rewards[player_id]
                self.env.PLAYERS[player_id] = None  # Without this the reward is res
            else: 
                self.env.NUM_DEAD += 1
                self.env.game_round.NUM_DEAD = self.env.NUM_DEAD
                self.env.pool_obj.return_hero(self.env.PLAYERS[player_id])
                self.env.kill_list.append(player_id)

        self.env.terminations = {a: True for a in self.env.agents}
