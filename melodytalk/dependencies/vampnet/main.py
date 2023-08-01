from pathlib import Path
from typing import Tuple
import yaml
import tempfile
import uuid
import shutil
from dataclasses import dataclass, asdict

import numpy as np
import audiotools as at
import argbind
import torch

import gradio as gr
from melodytalk.dependencies.vampnet.interface import Interface
from melodytalk.dependencies.vampnet import mask as pmask


def vamp(input_audio_path=None,
         output_audio_path=None,
         interface=None,
         top_p=0,
         prefix_s=0,  # inpainting
         suffix_s=0,  # inpainting
         rand_mask_intensity=1,
         num_steps=36,
         periodic_p=0,  # periodic mask
         periodic_w=0,  # periodic mask
         onset_mask_width=0,  # onset mask
         beat_mask_width=0,  # beat mask
         dropout=0,  # dropout
         beat_mask_downbeats=False,
         n_conditioning_codebooks=0,
         seed=0,
         masktemp=1.5,
         sampletemp=1.0,
         typical_filtering=False,
         typical_mass=0.15,
         typical_min_tokens=64,
         use_coarse2fine=True):
    # preprocess files
    # trim to 10s

    # sig = at.AudioSignal(input_audio_path)
    sig = at.AudioSignal(input_audio_path, duration=10)

    z = interface.encode(sig)

    ncc = n_conditioning_codebooks

    # build the mask
    mask = pmask.linear_random(z, rand_mask_intensity)
    mask = pmask.mask_and(
        mask, pmask.inpaint(
            z,
            interface.s2t(prefix_s),
            interface.s2t(suffix_s)
        )
    )
    mask = pmask.mask_and(
        mask, pmask.periodic_mask(
            z,
            periodic_p,
            periodic_w,
            random_roll=True
        )
    )
    if onset_mask_width > 0:
        mask = pmask.mask_or(
            mask, pmask.onset_mask(sig, z, interface, width=onset_mask_width)
        )
    if beat_mask_width > 0:
        beat_mask = interface.make_beat_mask(
            sig,
            after_beat_s=(beat_mask_width / 1000),
            mask_upbeats=not beat_mask_downbeats,
        )
        mask = pmask.mask_and(mask, beat_mask)

    # these should be the last two mask ops
    mask = pmask.dropout(mask, dropout)
    mask = pmask.codebook_unmask(mask, ncc)
    _top_p = top_p if top_p > 0 else None

    _seed = seed if seed > 0 else None
    zv, mask_z = interface.coarse_vamp(
        z,
        mask=mask,
        sampling_steps=num_steps,
        mask_temperature=masktemp * 10,
        sampling_temperature=sampletemp,
        return_mask=True,
        typical_filtering=typical_filtering,
        typical_mass=typical_mass,
        typical_min_tokens=typical_min_tokens,
        top_p=_top_p,
        gen_fn=interface.coarse.generate,
        seed=_seed,
    )

    if use_coarse2fine:
        zv = interface.coarse_to_fine(
            zv,
            mask_temperature=masktemp * 10,
            sampling_temperature=sampletemp,
            mask=mask,
            sampling_steps=num_steps,
            seed=_seed,
        )

    sig = interface.to_signal(zv).cpu()
    sig.write(output_audio_path)
