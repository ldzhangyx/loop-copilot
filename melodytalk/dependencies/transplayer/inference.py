import librosa
import resampy

def transform(filepath):
    audio, sr = librosa.load(filepath)
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)
    cqt_representation = lr.cqt(audio, sr=sr, hop_length=256)

    cqt_magnitude = np.abs(cqt_representation)


import os
import argparse
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
              org='piano', trg='piano',
              cp_path=None):
    G = Generator(dim_neck=32,
                  dim_emb=4,
                  dim_pre=512,
                  freq=32).eval().to(device)
    if os.path.exists(cp_path):
        save_info = torch.load(cp_path)
        G.load_state_dict(save_info["model"])

    # one-hot
    ins_list = ['harp', 'trumpet', 'epiano', 'viola', 'piano', 'guitar', 'organ', 'flute']
    ins_org = org
    ins_trg = trg
    emb_org = ins_list.index(ins_org)
    emb_trg = ins_list.index(ins_trg)
    # emb_org = [i == ins_org for i in ins_list]
    # emb_trg = [i == ins_trg for i in ins_list]
    emb_org = torch.unsqueeze(torch.tensor(emb_org), dim=0).to(device)
    emb_trg = torch.unsqueeze(torch.tensor(emb_trg), dim=0).to(device)

    x_org = np.log(np.load(config.feature_path).T)[:config.feature_len]
    # x_org = np.load(config.spectrogram_path).T
    x_org, len_pad = pad_seq(x_org)
    x_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_org)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    np.save(os.path.basename(config.feature_path)[:-4] + "_" + ins_org + "_" + ins_org + ".npy", x_trg.T)
    print("result saved.")

    with torch.no_grad():
        _, x_identic_psnt, _ = G(x_org, emb_org, emb_trg)
        if len_pad == 0:
            x_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            x_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

    np.save(os.path.basename(config.feature_path)[:-4] + "_" + ins_org + "_" + ins_trg + ".npy", x_trg.T)
    print("result saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=0, help='weight for hidden code loss')
    # Training configuration.
    parser.add_argument('--feature_path', type=str, default='../../data_syn/cropped/piano_all_00.wav_cqt.npy')
    parser.add_argument('--feature_len', type=int, default=2400)
    # parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    # parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')

    # Miscellaneous.
    parser.add_argument('--cp_path', type=str,
                        default="../../autovc_cp/weights_log_cqt_down32_neck32_onehot4_withcross")

    config = parser.parse_args()
    print(config)
    inference(config)