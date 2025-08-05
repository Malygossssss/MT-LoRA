import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

# Mapping from task name to (metric key, higher_is_better)
TASK_METRIC_INFO = {
    'semseg': ('mIoU', True),
    'human_parts': ('mIoU', True),
    'sal': ('mIoU', True),
    'edge': ('loss', False),
    'depth': ('rmse', False),
    'normals': ('rmse', False),
}


def extract_task_metrics(eval_results, tasks):
    """Extract a single scalar per task from evaluation results.

    The returned values are adjusted such that higher is always better by
    negating metrics where lower values indicate better performance.
    """
    metrics = {}
    for t in tasks:
        key, higher_is_better = TASK_METRIC_INFO.get(t, ('mIoU', True))
        if t in eval_results and key in eval_results[t]:
            val = eval_results[t][key]
            metrics[t] = val if higher_is_better else -val
    return metrics


def build_state(tasks, loss_dict=None, metric_dict=None):
    """Build a state vector from task losses and metrics."""
    state = []
    for t in tasks:
        if loss_dict is not None and t in loss_dict:
            state.append(loss_dict[t])
        else:
            state.append(0.0)
        if metric_dict is not None and t in metric_dict:
            state.append(metric_dict[t])
        else:
            state.append(0.0)
    return torch.tensor(state, dtype=torch.float32)


def compute_reward(curr_metrics, prev_metrics):
    """Compute reward as improvement in average metric across tasks."""
    if prev_metrics is None:
        return 0.0
    curr = torch.tensor([curr_metrics[t] for t in curr_metrics])
    prev = torch.tensor([prev_metrics[t] for t in prev_metrics])
    return (curr.mean() - prev.mean()).item()


class TaskWeightAgent(nn.Module):
    """Simple policy network to output task weights.

    The policy outputs the concentration parameters of a Dirichlet
    distribution from which task weight vectors are sampled.
    """

    def __init__(self, state_dim, num_tasks, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.log_prob = None

    def forward(self, state):
        logits = self.policy(state)
        alpha = F.softplus(logits) + 1e-3  # ensure positivity
        dist = Dirichlet(alpha)
        weights = dist.rsample()
        self.log_prob = dist.log_prob(weights)
        return weights

    def select_weights(self, state):
        state = state.to(next(self.parameters()).device)
        weights = self.forward(state)
        return weights.detach().cpu()

    def update(self, reward):
        if self.log_prob is None:
            return
        loss = -self.log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.log_prob = None