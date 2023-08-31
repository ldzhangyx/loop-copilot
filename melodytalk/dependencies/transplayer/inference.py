import librosa as lr
import resampy
import os
import torch
import numpy as np
from math import ceil
from model import Generator

device = 'cuda:0'


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0]) / base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0, len_pad), (0, 0)), 'constant'), len_pad

def inference(input_file_path,
              output_file_path,
              org='piano', trg='guitar',
              feature_len=2400,
              cp_path='weights.pth'):
    G = Generator(dim_neck=32,
                  dim_emb=4,
                  dim_pre=512,
                  freq=32).eval().to(device)
    if os.path.exists(cp_path):
        save_info = torch.load(cp_path)
        G.load_state_dict(save_info["model"])

    # process input
    audio, sr = lr.load(input_file_path)
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    cqt_representation = lr.cqt(audio, sr=sr, hop_length=256)
    cqt_magnitude = np.abs(cqt_representation)

    # one-hot
    ins_list = ['harp', 'trumpet', 'epiano', 'viola', 'piano', 'guitar', 'organ', 'flute']
    ins_org = org
    ins_trg = trg
    emb_org = ins_list.index(ins_org)
    emb_trg = ins_list.index(ins_trg)
    emb_org = torch.unsqueeze(torch.tensor(emb_org), dim=0).to(device)
    emb_trg = torch.unsqueeze(torch.tensor(emb_trg), dim=0).to(device)

    x_org = np.log(cqt_magnitude.T)[:feature_len]
    # x_org = np.load(config.spectrogram_path).T
    x_org, len_pad = pad_seq(x_org)
    x_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_trg)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    # save output
    waveform = lr.icqt(x_trg, sr=sr, hop_length=256)
    lr.output.write_wav(output_file_path, waveform, sr)


if __name__ == '__main__':
    inference('/home/intern-2023-02/melodytalk/melodytalk/music/2d2c_piano_2333801f_2333801f.wav', 'output.wav')