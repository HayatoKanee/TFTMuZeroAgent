class checkWinWrapper():
    def __init__(self, type):
        if type == "standard":
            self.checker = standard(type)
        elif type == "level":
            self.checker = level(type)
    
    def run(self, env):
        result = self.checker.runCheck(env)
        return result

class checkWin():
    def __init__(self, wincon):
        self.wincon = wincon

    def runCheck(self, env):
        gameOver = self.check(env)
        if gameOver == True:
            
            winner = self.getId(env) 
            reward = self.reward 
            self.runCleanup(env) 
            return reward, winner 
        return None

class standard(checkWin):
    def __init__(self, wincon):
        super().__init__(wincon)
        self.reward = 40 
        
    def check(self, env):
        num_alive = 0
        for key, player in env.PLAYERS.items():
            if player:
                if player.health <= 0:
                    env.NUM_DEAD += 1
                    env.game_round.NUM_DEAD = env.NUM_DEAD
                    env.pool_obj.return_hero(player)
                    env.kill_list.append(key)
                else:
                    num_alive += 1
        if num_alive <= 1 or env.game_round.current_round > 48:
            return True
    
    def getId(self, env):
        for player_id in env.agents:
            if env.PLAYERS[player_id] and env.PLAYERS[player_id].health > 0:
                return player_id
    
    def runCleanup(self, env):
        for player_id in env.agents:
            if env.PLAYERS[player_id] and env.PLAYERS[player_id].health > 0: 
                env.PLAYERS[player_id].won_game()
                env.rewards[player_id] = 40 + env.PLAYERS[player_id].reward
                env._cumulative_rewards[player_id] = env.rewards[player_id]
                env.PLAYERS[player_id] = None  # Without this the reward is res
        env.terminations = {a: True for a in env.agents}

class level(checkWin):
    def __init__(self, wincon):
        super().__init__(wincon)
        self.reward = 40 
    
    def check(self, env): 
        for key, player in env.PLAYERS.items():
            if player:
                if player.level >= 7:
                    return True 
                
    def getId(self, env):
        for key, player in env.PLAYERS.items():
            if player:
                if player.level >= 7:
                    return key 
                
    def runCleanup(self, env):
        for player_id in env.agents:
            if env.PLAYERS[player_id] and env.PLAYERS[player_id].level >= 7: 
                env.PLAYERS[player_id].won_game()
                env.rewards[player_id] = 40 + env.PLAYERS[player_id].reward
                env._cumulative_rewards[player_id] = env.rewards[player_id]
                env.PLAYERS[player_id] = None  # Without this the reward is res
            else: 
                env.NUM_DEAD += 1
                env.game_round.NUM_DEAD = env.NUM_DEAD
                env.pool_obj.return_hero(env.PLAYERS[player_id])
                env.kill_list.append(player_id)

            print(player_id)
        env.terminations = {a: True for a in env.agents}
