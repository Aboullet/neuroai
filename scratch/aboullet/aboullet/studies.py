# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
import typing as tp
import warnings
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download, utils
from neuralset import utils as nutils
from neuralset.events import etypes, study

mne.set_log_level(False)


class Boullet2026Petit(study.Study):
    """Boullet2026Petit dataset.

    This dataset contains 128-channel EEG recordings from X participants
    performing an auditory and covert production task.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("ICM")
    bibtex: tp.ClassVar[str] = """
    @article{XXX}
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/dsXXXX"
    description: tp.ClassVar[str] = (
        "EEG recordings from X participants listening to Le Petit Prince in French."
    )

    task: tp.ClassVar[str] = "listen"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.version = "v1"


    def _download(self) -> None:
        raise NotImplementedError(
            "This function is meant to be called on each independent study, Bel2026PetitRead and Bel2026PetitListen,"
            "which have different dataset IDs. See their implementations for details."
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self.path.exists():
            raise RuntimeError(f"Missing folder {self.path}")
        for run in range(1, 3):

            yield dict(subject="01", session="session01", task=self.task, run=run)


    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load events from the events.tsv file for the listen task."""
        tl = timeline
        error_msg_prefix = (
            f"subject {tl['subject']}, session {tl['session']}, run {tl['run']}\n"
        )
        meg = self._make_meg_event(tl)
        raw = meg.read()
        sound_triggers = mne.find_events(raw, stim_channel="STI001", shortest_event=1)

        # Load events from TSV file
        events: list[dict[str, tp.Any]] = []
        tsv = str(meg.filepath).replace("meg.fif", "events.tsv")
        words_df = pd.read_csv(tsv, delimiter="\t")

        for _, row in words_df.iterrows():
            trial_type = row["trial_type"]
            if "BAD_ACQ_SKIP" in str(trial_type):
                continue

            events.append(
                {
                    "condition": "sentence",
                    "type": (
                        trial_type.capitalize() if isinstance(trial_type, str) else "Word"
                    ),
                    "start": row["onset"],
                    "duration": row["duration"],
                    "stop": row["onset"] + row["duration"],
                    "language": "french",
                    "text": row.get("stimulus", row.get("word", "")),
                }
            )

        # extract sound annotation
        sound_triggers = sound_triggers[sound_triggers[:, 2] == 5]
        if len(sound_triggers) != 2:
            warnings.warn(
                f"No sound triggers found for subject {tl['subject']}, run {tl['run']}"
            )
        else:
            start, stop = sound_triggers[:, 0] / raw.info["sfreq"]
            events.append(
                dict(
                    type="Audio",
                    start=start,
                    duration=stop - start,
                    filepath=self.path
                    / "sourcedata/stimuli/audio"
                    / CHAPTER_PATHS[int(tl["run"]) - 1],
                )
            )

        events_df = pd.DataFrame(events)

        # Remove empty words that were included in the metadata files...
        events_df = events_df[events_df["text"] != " "]

        metadata = pd.read_csv(self._get_seq_id_path(tl))
        rows_events, rows_metadata = nutils.match_list(
            [str(word) for word in events_df["text"].values],
            [str(word) for word in metadata["word"].values],
        )

        assert len(rows_events) / len(events_df) > 0.95, (
            error_msg_prefix
            + f"only {len(rows_events) / len(events_df)} of the words were found in the metadata"
        )
        events_idx, metadata_idx = (
            events_df.index[rows_events],
            metadata.index[rows_metadata],
        )

        # Adding the information about sequence_id and n_closing
        events_df["word"] = events_df["text"]
        for col in ["sequence_id", "n_closing", "is_last_word", "pos"]:
            events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col]

        # add train/test/val splits
        events_df = utils.add_sentences(events_df)

        words = events_df.loc[events_df.type == "Word"]

        # Get the word triggers from STI008, as a step so we can get the offset
        word_triggers = mne.find_stim_steps(raw, stim_channel="STI008")
        word_triggers = word_triggers[word_triggers[:, 2] == 0]

        abs_tol, max_missing = TOL_MISSING_DICT.get(
            (int(tl["subject"]), int(tl["run"])), (10, 5)
        )
        i, j = nutils.approx_match_samples(
            (words.start * 1000).tolist(),
            word_triggers[:, 0],
            abs_tol=abs_tol,
            max_missing=max_missing,
        )

        words = words.iloc[i, :]

        events_df.loc[:, "unaligned_start"] = events_df.loc[:, "start"]
        events_df.loc[words.index, "start"] = word_triggers[j, 0] / raw.info["sfreq"]

        events_df = pd.concat([pd.DataFrame([meg.to_dict()]), events_df])
        events_df = self._add_text_context(tl, events_df)

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "heard"
        )
        return events_df.sort_values(by="start").reset_index(drop=True)


