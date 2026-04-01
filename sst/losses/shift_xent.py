import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ShiftInvariantCrossEntropy(nn.Module):
    def __init__(self, num_classes=300, tempo_min=20, tempo_max=300, device=None):
        """
        Args:
            num_classes: number of output classes / bins
            tempo_min: minimum tempo to represent
            tempo_max: maximum tempo to represent
        """
        super(ShiftInvariantCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.device = device if device is not None else torch.device("cpu")
        
        # Calculate log bin properties
        self.log_tempo_min = math.log(self.tempo_min)
        self.log_tempo_max = math.log(self.tempo_max)
        self.log_bin_width = (self.log_tempo_max - self.log_tempo_min) / self.num_classes
        
    def forward(self, z_i, z_j, ts_rate_i, ts_rate_j):
        """
        Computes shift-invariant cross entropy loss.
        z_i, z_j: logits from TCN, shape (batch_size, num_classes)
        ts_rate_i, ts_rate_j: time stretch rates applied, shape (batch_size, 1)
        """
        batch_size = z_i.size(0)
        
        # tempo(j) = tempo(i) * (ts_rate_j / ts_rate_i)
        # Therefore, z_j's distribution should be shifted right by log(ratio)/bin_width
        ratio = ts_rate_j / ts_rate_i
        
        shift_logs = torch.log(ratio) / self.log_bin_width
        shift_bins = torch.round(shift_logs).long().view(-1)
        
        log_prob_i = F.log_softmax(z_i, dim=-1)
        prob_j = F.softmax(z_j, dim=-1)
        
        shifted_prob_j = torch.zeros_like(prob_j)
        
        for b in range(batch_size):
            sb = shift_bins[b].item()
            # If sb > 0, j is at a higher tempo than i.
            # To match i, we must shift j's distribution to the LEFT by sb.
            shift_to_match_i = -sb
            
            # Roll the tensor
            shifted_prob_j[b] = torch.roll(prob_j[b], shifts=shift_to_match_i, dims=-1)
            
            # Mask out wrapped-around probabilities
            if shift_to_match_i > 0:
                shifted_prob_j[b, :shift_to_match_i] = 0.0
            elif shift_to_match_i < 0:
                shifted_prob_j[b, shift_to_match_i:] = 0.0
                
        # Do NOT re-normalize: the masked distribution is used as-is so that
        # samples whose shift pushes most mass out of range contribute less loss,
        # which is the correct behaviour (less valid signal = less penalty).

        # Cross entropy: H(P_target, P_pred) = - \sum P_target * \log P_pred
        # Using shifted_prob_j as target for log_prob_i
        loss = -torch.sum(shifted_prob_j * log_prob_i, dim=-1).mean()
        
        return loss
