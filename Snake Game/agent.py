import json

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import deque
from torch import optim
import os
from snake_game import Game
class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Replay_Memory:
    def __init__(self , capacity):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.capacity = capacity
        self.memory = []
    def push(self ,event):
        ''' event is a tuple that consists of (state , action , reward , next_state , done) '''
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            self.memory.pop(0)

    def sample(self, batch_size):
        import random
        actual_batch_size = min(batch_size, len(self.memory))  # ensures safe sampling

        experience = random.sample(self.memory, actual_batch_size)

        states = torch.from_numpy(np.vstack([e[0] for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experience if e is not None])).long().to(self.device)
        reward = torch.from_numpy(np.vstack([e[2] for e in experience if e is not None])).float().to(self.device)
        next_state = torch.from_numpy(np.vstack([e[3] for e in experience if e is not None])).float().to(self.device)
        done = torch.from_numpy(np.vstack([e[4] for e in experience if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, reward, next_state, done


#HyperParameneters
number_of_episodes = 10000
maximum_number_of_steps_per_episode = 200000
epsilon = 1.0
epsilon_ending_value = 0.001
epsilon_decay = 0.998
learning_rate = 0.001
mini_batch_size = 100
gamma = 0.95
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-2
state_size = 16
action_size = 4
scores_on_100_episodes = deque(maxlen = 100)
folder = 'model'

class Agent():
    def __init__(self , state_size , action_size ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_size = state_size
        self.action_size = action_size
        self.local_network = ANN(state_size , action_size).to(self.device)
        self.target_network = ANN(state_size , action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters() , lr=learning_rate)
        self.memory = Replay_Memory(replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon  = -1
    def Get_state(self , game):
        ''' the 16 size state is comprised of these things
        [left up , right up , left down , right down , left , down , up , right , left_dir_true , right_dir_true , up_dir_true , down_dir_true , food_left , food_right , food_down , food_up]

        '''
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        point_left = [head_x - game.WIDTH , head_y]
        point_right = [head_x + game.WIDTH , head_y]
        point_up = [head_x, head_y - game.HEIGHT]
        point_down = [head_x, head_y + game.HEIGHT]
        point_left_up = [(head_x- game.WIDTH) , (head_y - game.WIDTH)]
        point_right_up = [(head_x + game.WIDTH) , (head_y - game.WIDTH)]
        point_left_down = [(head_x - game.WIDTH), (head_y+game.WIDTH)]
        point_right_down = [(head_x + game.WIDTH), (head_y + game.WIDTH)]

        state = [
            game.isDanger(point_left) ,
            game.isDanger(point_right),
            game.isDanger(point_up),
            game.isDanger(point_down),
            game.isDanger(point_left_up),
            game.isDanger(point_right_up),
            game.isDanger(point_left_down),
            game.isDanger(point_right_down),

            game.snake.direction == 'left',
            game.snake.direction == 'right',
            game.snake.direction == 'up',
            game.snake.direction == 'down',


            game.food.x > head_x ,  #food right
            game.food.x < head_x ,  #food left
            game.food.y > head_y ,  # food down
            game.food.y < head_y ,  #food up

        ]
        return np.array(state , dtype = int)
    def step(self, state , action , reward , next_state , done):
        self.memory.push((state , action , reward , next_state , done))
        self.t_step  = (self.t_step + 1) % 4
        if self.t_step == 0 :
            experience = self.memory.sample(mini_batch_size)
            self.learn(experience)
    def get_action(self , state  ,epsilon ):
        import random
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_network.eval()
        with torch.no_grad():
            actions = self.local_network(state)
        self.local_network.train()
        if random.random() > epsilon:
            move = torch.argmax(actions).item()
        else :
            move = random.randint(0 , 3)
        return move
    def learn(self , experiences):
        states, actions, rewards, next_states, dones = experiences
        next_q_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_target * (1 - dones))
        q_epected = self.local_network(states).gather(1 , actions)
        loss = F.mse_loss(q_epected , q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network)
    def soft_update(self  , local_network , target_network):
        for target_param, local_param in zip(target_network.parameters() , local_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - interpolation_parameter) + local_param.data * interpolation_parameter)
    def load_model(self , file_name = 'model.pth'):
        file_path = os.path.join(folder , file_name)
        if os.path.exists(file_path):
            self.local_network.load_state_dict(torch.load(file_path))
            print('model loaded')
            self.retrieve_data()
    def save_model(self , file_name = 'model.pth'):
        os.makedirs("model", exist_ok=True)
        filename = os.path.join(folder , file_name)
        torch.save(self.local_network.state_dict() , filename)
    def retrieve_data(self):
        filename = 'data.json'
        model_data_path = os.path.join(folder , filename)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as f:
                data = json.load(f)

                if data is not None:
                    self.record = data['record']
                    self.epsilon = data['epsilon']
    def save_data(self , record , epsilon):
        filename = 'data.json'
        if not os.path.exists(folder):
            os.makedirs(folder)
        complete_path = os.path.join(folder , filename)
        data = {'record' : self.record, 'epsilon' : self.epsilon}
        with open(complete_path, 'w') as f:
            json.dump(data, f ,indent=4)
if __name__ == '__main__':
    game = Game()
    max_score = 0
    agent = Agent(state_size , action_size)
    agent.load_model()
    epsilon = epsilon
    if agent.epsilon != -1:
        epsilon = agent.epsilon
        max_score = max(max_score , agent.record)
    for episode in range(number_of_episodes):
        game.reset()
        score = 0
        for t in range(maximum_number_of_steps_per_episode):
            state_old = agent.Get_state(game)
            action = agent.get_action(state_old , epsilon)
            move = [0 , 0 , 0 , 0]
            move[action] = 1
            reward , done , score = game.run(move)
            state_new = agent.Get_state(game)
            agent.step(state_old, action, reward , state_new , done)
            if done :
                break
        max_score = max(score, max_score)
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_ending_value , epsilon*epsilon_decay)
        agent.save_model()
        agent.save_data(max_score , epsilon)
        if episode % 50 == 0:
            print(f'Episodes :{episode} \t score :{score} \t Max Score :{max_score} \t epsilon :{epsilon}')


