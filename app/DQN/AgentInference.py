import torch 
from .QNetwork import *

class AgentInference:
    def __init__(self, action_size, device):
        self.action_size = action_size 
        self.device = device
        self.qnet = QNetwork(self.action_size).to(device) 

    def get_action(self, state):
        """Select the action using the trained Q-network."""
        self.qnet.eval()  # Ensure evaluation mode for inference
        with torch.no_grad():  # Disable gradient calculation
            state = state.detach().unsqueeze(0).float().to(self.device)
            qs = self.qnet(state)  # Get Q-values
        return qs.argmax().item()  # Return action with the highest Q-value

    # Load model for inference only
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.qnet.load_state_dict(checkpoint['qnet_state_dict'])  # Load the trained Q-network
        self.qnet.eval()  # Set Q-network to evaluation mode for inference
        print(f"Model loaded for inference from {path}")
