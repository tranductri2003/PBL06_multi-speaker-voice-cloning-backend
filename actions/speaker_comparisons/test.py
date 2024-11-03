import os
import random
import torch
from settings import TEST_FOLDER, SPEAKER_TRANSFORMER_ENCODER, SPEAKER_LSTM_ENCODER
from actions.speaker_comparisons.comparison import calculate_cosine_similarity


def test_transformer_speaker_verification():
    speakers = os.listdir(TEST_FOLDER)
    utterances = []
    for speaker in speakers:
        folder = os.path.join(TEST_FOLDER, speaker)
        utterances.extend([os.path.join(TEST_FOLDER, speaker, file) for file in random.sample(os.listdir(folder), 2)])
        
    results = {}
    for utterance in utterances:
        similarity_scores = []
        
        for other_utterance in utterances:
            similarity_scores.append(calculate_cosine_similarity(SPEAKER_TRANSFORMER_ENCODER, utterance, other_utterance))
            
        results[utterance] = similarity_scores
        
    return results


def test_lstm_speaker_verification():
    speakers = os.listdir(TEST_FOLDER)
    utterances = []
    for speaker in speakers:
        folder = os.path.join(TEST_FOLDER, speaker)
        utterances.extend([os.path.join(TEST_FOLDER, speaker, file) for file in random.sample(os.listdir(folder), 2)])
        
    results = {}
    for utterance in utterances:
        similarity_scores = []
        
        for other_utterance in utterances:
            similarity_scores.append(calculate_cosine_similarity(SPEAKER_LSTM_ENCODER, utterance, other_utterance))
            
        results[utterance] = similarity_scores
        
    return results
