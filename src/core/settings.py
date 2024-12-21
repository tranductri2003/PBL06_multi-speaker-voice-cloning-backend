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
    "CNN50": os.getenv("VOICE_ENHANCEMENT_CNN50_MODEL"),
    "CNN100": os.getenv("VOICE_ENHANCEMENT_CNN100_MODEL"),
    "OriginTextToSpeech": os.getenv("ORIGINAL_TEXT_TO_SPEECH_MODEL"),
    "Mel2Mag": os.getenv("MEL2MAG_MODEL"),
}

ACRONYMS_FILEPATH = "core/utils/tokenizer/text_bank/acronyms.json"
BASE_NUMBERS_FILEPATH = "core/utils/tokenizer/text_bank/base_numbers.json"
DATE_PREFIXES_FILEPATH = "core/utils/tokenizer/text_bank/date_prefixes.json"
FINAL_CONSONANTS_FILEPATH = "core/utils/tokenizer/text_bank/final_consonants.json"
HEAD_CONSONANTS_FILEPATH = "core/utils/tokenizer/text_bank/head_consonants.json"
LETTERS_FILEPATH = "core/utils/tokenizer/text_bank/letters.json"
NUMBER_LEVELS_FILEPATH = "core/utils/tokenizer/text_bank/number_levels.json"
SAME_PHONEMES_FILEPATH = "core/utils/tokenizer/text_bank/same_phonemes.json"
SYMBOLS_FILEPATH = "core/utils/tokenizer/text_bank/symbols.json"
TONES_FILEPATH = "core/utils/tokenizer/text_bank/tones.json"
UNITS_FILEPATH = "core/utils/tokenizer/text_bank/units.json"
VOWELS_FILEPATH = "core/utils/tokenizer/text_bank/vowels.json"



