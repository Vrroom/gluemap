"""Load an iPhone+360 video take: phone frames (ar_frames.jsonl) + stabilized
equirect video (panorama_export.mp4).

Clock sync: phone frames and phone audio share one clock (per-frame timestamp,
firstAudioPTSSeconds); the pano's audio and video share the mp4 timeline. The
audio-measured sync_offset (see measure_audio_offset) bridges the two devices:
t_pano = (timestamp - first_audio_pts) + sync_offset, frame = t_pano * fps.
"""

import json
import os
import subprocess
import tempfile

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, correlate, hilbert, sosfilt


def extract_wav(src, out, sr=16000):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1",
                    "-ar", str(sr), out], check=True)
    rate, x = wavfile.read(out)
    assert rate == sr, f"(extract_wav): got {rate} Hz, wanted {sr}"
    return x.astype(np.float64)


def onset_envelope(x, sr=16000, out_sr=1000):
    sos = butter(4, [300, 4000], btype="band", fs=sr, output="sos")
    e = np.abs(hilbert(sosfilt(sos, x - x.mean())))
    step = sr // out_sr
    e = e[: len(e) // step * step].reshape(-1, step).mean(axis=1)
    return e - e.mean()


def measure_audio_offset(run_dir, cache_path, take="take_000", env_sr=1000):
    """Constant clock offset (s): t_pano = (t_clock - firstAudioPTSSeconds) + offset.

    * Detect salient events in both audio tracks using onset envelopes, which
      measure loudness over time. Shared events (footsteps, voices, door thuds)
      become spikes at the same instants in both envelopes.
    * Cross-correlate the two full envelopes. The peak gives a coarse lag.
    * Slide a window over the phone envelope with a fixed stride.
    * For each phone window, use the coarse lag to cut the pano slice that
      should contain the same events, widened by a search margin on both sides.
    * Correlate the window against its slice. The peak inside the margin is
      that window's own, refined lag estimate.
    * Keep estimates with a sharp correlation peak that agree with the median
      of all estimates. The offset is the median of the survivors.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            offset = json.load(f)["offset_seconds"]
        print(f"(measure_audio_offset): {offset:+.4f} s (cached, {cache_path})")
        return offset
    phone_m4a = os.path.join(run_dir, "video", "takes", take, "phone", "ar_audio.m4a")
    pano_mp4 = os.path.join(run_dir, "insta360", "takes", take, "panorama_export.mp4")
    with tempfile.TemporaryDirectory() as tmp:
        ep = onset_envelope(extract_wav(phone_m4a, os.path.join(tmp, "phone.wav")), out_sr=env_sr)
        ev = onset_envelope(extract_wav(pano_mp4, os.path.join(tmp, "pano.wav")), out_sr=env_sr)
    xc = correlate(ev, ep, mode="full", method="fft")
    lag = np.arange(-len(ep) + 1, len(ev))[xc.argmax()] / env_sr
    win, hop, search = 8 * env_sr, 2 * env_sr, 2 * env_sr
    offs = []
    for s in range(0, len(ep) - win, hop):
        seg = ep[s:s + win]
        ctr = s + int(lag * env_sr)
        lo, hi = max(0, ctr - search), min(len(ev), ctr + win + search)
        xcw = correlate(ev[lo:hi], seg, mode="valid", method="fft")
        z = (xcw.max() - xcw.mean()) / (xcw.std() + 1e-12)
        offs.append(((lo + xcw.argmax() - s) / env_sr, z))
    offs = np.array(offs)
    med = np.median(offs[:, 0])
    good = offs[(offs[:, 1] > 3.5) & (np.abs(offs[:, 0] - med) < 0.05), 0]
    assert len(good) >= 5, f"(measure_audio_offset): only {len(good)}/{len(offs)} consistent windows"
    offset = float(np.median(good))
    print(f"(measure_audio_offset): {offset:+.4f} s from {len(good)}/{len(offs)} windows, "
          f"spread {good.max() - good.min():.4f} s")
    with open(cache_path, "w") as f:
        json.dump({"offset_seconds": offset, "n_windows": int(len(good))}, f)
    return offset


class VideoTake:
    def __init__(self, run_dir, sync_offset, take="take_000"):
        self.phone_dir = os.path.join(run_dir, "video", "takes", take, "phone")
        self.video_path = os.path.join(run_dir, "insta360", "takes", take, "panorama_export.mp4")
        assert os.path.isfile(self.video_path), f"(VideoTake::__init__): missing {self.video_path}"
        with open(os.path.join(self.phone_dir, "ar_frames.jsonl")) as f:
            self.frames = [json.loads(line) for line in f]
        self.timestamps = np.array([f["timestamp"] for f in self.frames])
        assert len(self.frames) > 1, f"(VideoTake::__init__): {len(self.frames)} phone frames"
        with open(os.path.join(self.phone_dir, "phone_manifest.json")) as f:
            self.first_audio_pts = json.load(f)["firstAudioPTSSeconds"]
        self.sync_offset = sync_offset
        cap = cv2.VideoCapture(self.video_path)
        self.n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        assert self.n_video_frames > 1, f"(VideoTake::__init__): {self.n_video_frames} video frames"
        assert self.fps > 0, f"(VideoTake::__init__): bad fps {self.fps}"

    def video_frame_index(self, phone_index):
        t_pano = self.timestamps[phone_index] - self.first_audio_pts + self.sync_offset
        return int(round(t_pano * self.fps))

    def read_equirects(self, video_indices):
        wanted = sorted(set(int(i) for i in video_indices))
        assert 0 <= wanted[0] and wanted[-1] < self.n_video_frames, \
            f"(VideoTake::read_equirects): indices {wanted[0]}..{wanted[-1]} out of range"
        cap = cv2.VideoCapture(self.video_path)
        out, pos = {}, 0
        for idx in wanted:
            while pos <= idx:
                ok = cap.grab()
                assert ok, f"(VideoTake::read_equirects): grab failed at frame {pos}"
                pos += 1
            ok, img = cap.retrieve()
            assert ok, f"(VideoTake::read_equirects): retrieve failed at frame {idx}"
            out[idx] = img
        cap.release()
        return out

    def sample_instants(self, n):
        valid = [i for i in range(len(self.frames))
                 if 0 <= self.video_frame_index(i) < self.n_video_frames]
        assert valid, "(VideoTake::sample_instants): no phone frame overlaps the pano video"
        if len(valid) < len(self.frames):
            print(f"(VideoTake::sample_instants): {len(self.frames) - len(valid)} phone frames "
                  f"fall outside the pano video, sampling the remaining {len(valid)}")
        phone_idx = np.round(np.linspace(valid[0], valid[-1], n)).astype(int)
        video_idx = [self.video_frame_index(i) for i in phone_idx]
        equirects = self.read_equirects(video_idx)
        return [{"phone_index": int(p), "video_index": v, "meta": self.frames[p],
                 "equirect": equirects[v]}
                for p, v in zip(phone_idx, video_idx)]
