from pathlib import Path
import numpy as np
from core.utils.objects.utterance import Utterance


class Speaker(object):
    def __init__(self, _id: str, utterances: list = []):
        self._id = _id
        self.utterances = utterances

    def add_utterance(self, utterance: Utterance):
        self.utterances.append(utterance)

    def random_utterances(self, num_utterances):
        if num_utterances > len(self.utterances):
            raise ValueError(
                "Requested number of utterances exceeds available utterances."
            )

        return [
            self.utterances[idx]
            for idx in np.random.randint(0, len(self.utterances), num_utterances)
        ]

    @classmethod
    def load_from_root_folder(cls, root: Path | str, file_extension: str):
        root = root if isinstance(root, Path) else Path(root)
        utterances = [
            Utterance(_id=audio_file.stem, raw_file=audio_file)
            for audio_file in root.rglob(f"*{file_extension}")
        ]
        return Speaker(_id=root.name, utterances=utterances)
