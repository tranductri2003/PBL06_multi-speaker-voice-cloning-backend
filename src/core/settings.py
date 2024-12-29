import os
from dotenv import load_dotenv

load_dotenv()
DEVICE = "cpu"
MODEL_PATHS = {
    "LstmSpeakerEncoder": os.getenv("SPEAKER_LSTM_ENCODER_MODEL"),
    "TransformerSpeakerEncoder": os.getenv("SPEAKER_TRANSFORMER_ENCODER_MODEL"),
    "ModifiedUNet": os.getenv("VOICE_ENHANCEMENT_MODIFIED_UNET_MODEL"),
    "UNet": os.getenv("VOICE_ENHANCEMENT_UNET_MODEL"),
    "UNetPlusPlus": os.getenv("VOICE_ENHANCEMENT_UNET_PLUS_PLUS_MODEL"),
    "UNet100": os.getenv("VOICE_ENHANCEMENT_UNET100_MODEL"),
    "CNN50": os.getenv("VOICE_ENHANCEMENT_CNN50_MODEL"),
    "CNN100": os.getenv("VOICE_ENHANCEMENT_CNN100_MODEL"),
    "EN_TACOTRON": os.getenv("EN_TACOTRON"),
    "VI_TACOTRON": os.getenv("VI_TACOTRON"),
    "Mel2Mag": os.getenv("MEL2MAG_MODEL"),
}

ACRONYMS_FILEPATH = "core/utils/text2sequence/vn/text_bank/acronyms.json"
BASE_NUMBERS_FILEPATH = "core/utils/text2sequence/vn/text_bank/base_numbers.json"
DATE_PREFIXES_FILEPATH = "core/utils/text2sequence/vn/text_bank/date_prefixes.json"
FINAL_CONSONANTS_FILEPATH = "core/utils/text2sequence/vn/text_bank/final_consonants.json"
HEAD_CONSONANTS_FILEPATH = "core/utils/text2sequence/vn/text_bank/head_consonants.json"
LETTERS_FILEPATH = "core/utils/text2sequence/vn/text_bank/letters.json"
NUMBER_LEVELS_FILEPATH = "core/utils/text2sequence/vn/text_bank/number_levels.json"
SAME_PHONEMES_FILEPATH = "core/utils/text2sequence/vn/text_bank/same_phonemes.json"
SYMBOLS_FILEPATH = "core/utils/text2sequence/vn/text_bank/symbols.json"
TONES_FILEPATH = "core/utils/text2sequence/vn/text_bank/tones.json"
UNITS_FILEPATH = "core/utils/text2sequence/vn/text_bank/units.json"
VOWELS_FILEPATH = "core/utils/text2sequence/vn/text_bank/vowels.json"

TTS_STOP_THRESHOLD = -3.4

EN_TACOTRON_PARAMS = {
    "embed_dims": 512, 
    "num_chars": 66, 
    "encoder_dims": 256, 
    "decoder_dims": 128, 
    "n_mels": 80, 
    "fft_bins": 80, 
    "postnet_dims": 512, 
    "encoder_K": 5, 
    "lstm_dims": 1024, 
    "postnet_K": 5, 
    "num_highways": 4,
    "dropout": 0.5, 
    "stop_threshold": TTS_STOP_THRESHOLD, 
    "speaker_embedding_size": 256
}

VI_TACOTRON_PARAMS = {
    "embed_dims": 512, 
    "num_chars": 93, 
    "encoder_dims": 256, 
    "decoder_dims": 128, 
    "n_mels": 80, 
    "fft_bins": 80, 
    "postnet_dims": 512, 
    "encoder_K": 5, 
    "lstm_dims": 1024, 
    "postnet_K": 5, 
    "num_highways": 4,
    "dropout": 0.5, 
    "stop_threshold": TTS_STOP_THRESHOLD, 
    "speaker_embedding_size": 256
}