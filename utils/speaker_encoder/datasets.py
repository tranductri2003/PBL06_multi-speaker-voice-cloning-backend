from typing import List
from pathlib import Path
import pickle
import logging
from multiprocessing import Process, JoinableQueue
import time
import os
import random

import torch
import numpy as np

from utils.objects import Speaker, Utterance


class AudioDataset(object):
    def __init__(self, id: str, speakers: List[Speaker] = []):
        self.id = id
        self.speakers = speakers

    def add_speaker(self, speaker: Speaker):
        """Add a speaker to this dataset."""

        self.speakers.append(speaker)

    def random_speakers(self, n):
        """Return n speakers randomly."""

        return [self.speakers[idx] for idx in np.random.randint(0, len(self.speakers), n)]

    def serialize_speaker(self, queue: JoinableQueue, counter_queue: JoinableQueue):
        while True:
            speaker, root, overwrite = queue.get()

            if not root.exists():
                root.mkdir(parents=True)

            dsdir = root / self.id
            if not dsdir.exists():
                dsdir.mkdir()

            spkdir = dsdir / speaker.id
            if not spkdir.exists():
                spkdir.mkdir()

            for uttrn_idx, uttrn in enumerate(speaker.utterances):
                uttrnpath = spkdir / (uttrn.id + '.pkl')
                is_overwrite = False
                is_empty = False
                if uttrnpath.exists():
                    if os.path.getsize(uttrnpath) == 0:
                        logging.debug(f'overrite empty file {uttrnpath}')
                    elif not overwrite:
                        logging.debug(f'{uttrnpath} already exists, skip')
                        counter_queue.put(1)
                        continue
                    is_overwrite = True
                try:
                    mel = uttrn.melspectrogram()
                    with uttrnpath.open(mode='wb') as f:
                        pickle.dump(mel, f)
                    if is_overwrite:
                        logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(speaker.utterances)}), overwrite')
                    else:
                        logging.debug(f'dump pickle object to {uttrnpath} ({uttrn_idx+1}/{len(speaker.utterances)})')
                except Exception as err:
                    logging.warning(f'failed to dump mel features for file {uttrnpath}: {err}')
                counter_queue.put(1)
            queue.task_done()

    def serialization_counter(self, total_count, queue: JoinableQueue):
        count = 0
        while True:
            start_time = time.time()
            done = queue.get()
            duration = time.time() - start_time
            count += 1
            logging.debug(f'serialization progress {count}/{total_count}, {int(duration*1000)}ms/item')
            queue.task_done()

    def serialize_mel_feature(self, root: Path, overwrite=False):
        """Serialize melspectrogram features for all utterances of all speakers to the disk."""

        num_processes = 8
        queue = JoinableQueue()
        counter_queue = JoinableQueue()
        processes = []
        for i in range(num_processes):
            p = Process(target=self.serialize_speaker, args=(queue, counter_queue))
            processes.append(p)
            p.start()
        total_count = sum([len(spk.utterances) for spk in self.speakers])
        counter_process = Process(target=self.serialization_counter, args=(total_count, counter_queue))
        counter_process.start()
        # add tasks to queue
        logging.debug(f'total {len(self.speakers)} speakers')
        for spk in self.speakers:
            queue.put((spk, root, overwrite)) 
        # wait for all task done
        queue.join() 
        counter_queue.join()
        for p in processes:
            p.terminate()
        counter_process.terminate()

class MultiAudioDataset(object):
    def __init__(self, datasets: List[AudioDataset]):
        self.id = ''
        self.speakers = []
        ids = []
        for ds in datasets:
            ids.append(ds.id)
            self.speakers.extend(ds.speakers)
        self.id = '+'.join(ids)

class SpeakerDataset(object):
    def __init__(self, speakers, utterances_per_speaker, seq_len):
        self.speakers = speakers
        n_speakers = len(self.speakers)
        n_utterances = sum([len(spk.utterances) for spk in self.speakers])
        logging.info(f'total {n_speakers} speakers, {n_utterances} utterances')
        self.utterances_per_speaker = utterances_per_speaker
        self.seq_len = seq_len

    def random_utterance_segment(self, speaker_idx, seq_len):
        """Must return an utterance segment as long as the speaker has at least
        one effective utterance."""

        while True:
            try:
                utterance = self.speakers[speaker_idx].random_utterances(1)[0]
                return utterance.random_mel_segment(seq_len)
            except Exception as err:
                logging.debug(f'failed to load utterances of speaker idx {speaker_idx}: {err}')
                continue

    def __getitem__(self, idx):
        """Return random segments of random utterances for the specified speaker."""
        seq_len = 0
        if isinstance(self.seq_len, int):
            seq_len = self.seq_len
        elif isinstance(self.seq_len, list):
            seq_len = self.seq_len[random.randint(0, len(self.seq_len)-1)]
        else:
            raise ValueError('seq_len must be int or int list')

        segments = np.array([self.random_utterance_segment(idx, seq_len) for _ in range(self.utterances_per_speaker)])
        return torch.tensor(segments)

    def __len__(self):
        return len(self.speakers)



from pathlib import Path

def load_librispeech360_dataset(root: Path):
    """Load the LibriSpeech train-clean-360 dataset into an AudioDataset.

    The dataset can be downloaded from: https://www.openslr.org/12

    Args:
        root (Path): Path to the root directory of the LibriSpeech dataset.
        mel_feature_root (Path, optional): Path to the root directory where the precomputed mel features are stored.

    Returns:
        AudioDataset: A dataset object containing the loaded speakers and their utterances.
    """

    dataset_id = 'librispeech360'
    id2speaker = dict()

    # Recursively find all .flac files in the dataset
    wav_files = root.rglob('*.flac')
    
    for f in wav_files:
        # LibriSpeech files are typically structured as: <root>/<speaker_id>/<chapter_id>/<utterance_id>.flac
        speaker_id = f.parent.parent.name  # Extract speaker ID from the parent folder
        chapter_id = f.parent.name  # Extract chapter ID from the immediate parent folder
        utterance_id = f.stem  # Use the file stem as the utterance ID (without .flac extension)

        uttrn = Utterance(utterance_id, raw_file=f)

        if speaker_id in id2speaker:
            id2speaker[speaker_id].add_utterance(uttrn)
        else:
            spk = Speaker(speaker_id)
            spk.add_utterance(uttrn)
            id2speaker[speaker_id] = spk

    dataset = AudioDataset(dataset_id, speakers=list(id2speaker.values()))
    return dataset
