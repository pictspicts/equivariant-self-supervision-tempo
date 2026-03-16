import torch
import torchaudio

def generate_click_track(bpm, sr, num_samples):
    audio = torch.zeros(1, num_samples)
    interval_samples = int(sr * 60.0 / bpm)
    spike_len = int(sr * 0.05)
    t = torch.arange(spike_len).float() / sr
    spike = torch.sin(2 * 3.141592 * 1000.0 * t) * torch.exp(-t * 100)
    
    idx = 0
    while idx + spike_len < num_samples:
        audio[0, idx:idx+spike_len] += spike
        idx += interval_samples
    
    return audio

audio = generate_click_track(120, 44100, 44100*5)
torchaudio.save('click.wav', audio, 44100)
print('saved')
