# -*- coding: utf-8 -*-
"""
強化学習エージェントの学習用プログラム
"""
import numpy as np
import torch

########################################################
# 環境Envの定義 (この部分を学習したい問題にあわせて自作してください)
########################################################
class Env(object):

    def __init__(self):
        ########################################
        # init で一番最初の初期化を実行
        # (この情報を元にAgentも初期化されるので注意)
        #######################################
        self.state_dim = 3           # 状態(観測)の次元
        self.actions   = [-1, 0, 1]  # 入力の選択肢のリスト
        self.dt        = 0.1         # 時間ステップ
        
    def reset(self):
        ################################
        # reset でエピソード毎の初期化を実行
        ################################
        s  = np.sign( np.random.randn(2) )
        r  = np.random.rand(2)
        X1  = s[0]*r[0]*1.5
        X2  = s[1]*r[1]*1.0
        T   = 2.0
        self.state = np.array([X1, X2,T])
        return self.state

    def step(self, action):       
        # Current state
        X1 = self.state[0]
        X2 = self.state[1]
        T  = self.state[2]
        U  = action
        # next state
        next_state = np.zeros_like(self.state)
        next_state[0]  =  X1 + self.dt*( - X1**3- X2)
        next_state[1]  =  X2 + self.dt*( X1   + X2 + U )
        next_state[2]  =  T -self.dt
        
        # Check terminal conditios 
        isTimeOver = (T <= self.dt)
        isUnsafe   = abs( X2 ) > 1
        done       = isTimeOver or isUnsafe

        # Reward
        if done and (not isUnsafe):
            reward = 1
        else:
            reward = 0

        self.state = next_state
        
        return next_state, reward, done

    def action_from_index(self, action_idx):
        return self.actions[action_idx] 


##########################################################
# Main     
##########################################################
if __name__ == "__main__":

    #######################
    # 実行時のオプションの指定
    #######################
    import argparse        
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",  default=0, type=int)  # Sets PyTorch and Numpy seeds
    args   = parser.parse_args()

    #################    
    # 乱数のシードを設定
    #################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    
    #########################################
    # 環境のオブジェクトを作成
    #########################################
    rlenv      = Env()
    state_dim  = rlenv.state_dim
    action_num = len(rlenv.actions)    
    
    #########################################
    # Agent setting
    #########################################
    from agent import DQN    
    
    agent = DQN.Agent(state_dim, action_num,
                      # 以下エージェントのオプション(必要に応じて追加すること)
                      CRITIC_LEARN_RATE   = 5e-3
                      )
    
    
    #########################################
    # Training
    #########################################

    LOG_DIR = 'logs/test'
    
    agent.train(rlenv, 
                EPISODES      = 3000, 
                SHOW_PROGRESS = True, 
                LOG_DIR       = LOG_DIR,
                SAVE_AGENTS   = True, 
                SAVE_FREQ     = 500,
                )
    
    