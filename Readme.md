🐍 Reinforcement Learning Snake Game
This project implements the classic Snake game using a Reinforcement Learning (RL) agent that learns to play and improve over time. The game logic is inspired by the traditional Snake game, and the agent is trained using Deep Q-Learning.

🚀 Getting Started
▶️ Run the Game
To run the Snake game with the trained RL agent:
python agent.py

The agent will start playing the game automatically.




📁 Project Structure

Snake Game/
├── agent.py # Main entry point: runs the game with the RL agent
├── base.py # Core logic
├── snake_game.py # Game mechanics (snake movement, collisions, food)
├── game_no_ui.py # Headless version of the game for faster training
├── model/ # Trained model weights and architecture (optional)
├── assets/ # Fonts and visuals
├── apple.jpg # Apple image used in the game
├── block.jpg # Block image used in the game
└── Readme.md # Project documentation





🧠 Reinforcement Learning Approach
✅ Algorithm: Deep Q-Learning

🎯 State Space: Direction of snake, location of food, and dangers around the head

🏆 Reward Function:

+10 for eating food

-10 for dying

-0.1 for every move (to encourage faster eating)

🧠 Neural Network: Simple feed-forward network with fully connected layers




📦 Requirements
Install dependencies using:

pip install -r requirements.txt



# For visual mode (with UI)
from snake_game import Game

# For headless mode (faster, no UI)
from game_no_ui import Game




🙌 Acknowledgements
Inspired by the original Snake game.

RL approach based on Deep Q-Learning tutorials and custom modifications.

📄 License
This project is open source under the MIT License.