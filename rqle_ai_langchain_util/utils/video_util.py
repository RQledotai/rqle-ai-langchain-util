import json

import moviepy.editor as mpe
from pydub import AudioSegment
import speech_recognition as sr

from rqle_ai_langchain_util.settings import AUDIO_TMP_FOLDER, LOG_CONFIG

# logging configuration
import logging
from logging.config import dictConfig
dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

SUPPORTED_SPEECH_RECOGNITION_ENGINES = ['sphinx', 'vosk']


def video_to_wav(video_path: str) -> str:
    """
    :param video_path: path to the video for which audio track needs to be extracted
    :return: path to the file containing the audio track of the video
    TODO add support for retrieving videos from YouTube
    """
    try:
        audio_filename = f'{AUDIO_TMP_FOLDER}/extracted_audio.wav'
        video = mpe.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_filename)
        video.close()
        return audio_filename
    except FileNotFoundError:
        raise FileNotFoundError(f'Video file at {video_path} not found')
    except Exception as e:
        raise IOError(f'Error converting {video_path} to audio: {e}')


def chunk_audio(audio_path: str, segment_length: int, overlap: int) -> list:
    """
    :param audio_path: path to the audio file to be chunked
    :param segment_length: the length of each chunked segment
    :param overlap: the overlap between audio segments
    :return: a list of audio segment files that were generated
    """
    # initialize the variable for storing list of segment files
    tmp_segment_files = list()
    # initialize AudioSegment object based on file
    audio = AudioSegment.from_wav(audio_path)
    audio_length = len(audio)

    # determine whether audio files need to be chunked in smaller chunks
    if audio_length < segment_length:
        tmp_segment_files.append(audio_path)
    else:
        logger.info(f'Length of {audio_path} is bigger than {segment_length} and needs to be chunked.')
        # initialize the variables required for chunking
        counter = 1
        end = 0
        eof_flag = False

        # iterate through the audio file to generate sub-segments for processing
        for i in range(0, 2 * len(audio), segment_length):
            # set the indices for the first iteration
            if i == 0:
                start = 0
                end = segment_length
            else:
                start = end - overlap
                end = start + segment_length

            # check that the end index is not out of bounds
            if end > len(audio):
                end = len(audio)
                eof_flag = True

            logger.debug(f'Creating segment {counter} with start: {start} and end: {end}')
            # create a new audio segment
            segment = audio[start:end]
            # create a filename for the segment and store to tmp storage
            segment_filename = f'{AUDIO_TMP_FOLDER}/segment_{counter}.wav'
            segment.export(segment_filename, format='wav')
            tmp_segment_files.append(segment_filename)

            counter += 1

            # break if the end of the initial audio file is reached
            if eof_flag:
                break

        logger.info(f'Chunking process created {len(tmp_segment_files)} segments of audio.')

    return tmp_segment_files


def transcribe_audio(audio_path: str, segment_size: int = 60000, segment_overlap: int = 1000, language: str = 'en-US',
                     speech_recognition_engine: str = 'sphinx'):
    """
    :param audio_path: path to the audio to be transcribed
    :param segment_size: the optimal size of audio segment for processing
    :param segment_overlap: the overlap between audio segments
    :param language: the language if the audio track
    :param speech_recognition_engine: the speech recognition engine to be used for extraction text
    :return: a text version of the audio
    """
    logger.info(f'Transcribe audio from {audio_path} with segment size {segment_size} and overlap {segment_overlap}')

    # check whether the speech recognition engine for transcription is supported
    if speech_recognition_engine not in SUPPORTED_SPEECH_RECOGNITION_ENGINES:
        raise NotImplementedError(f'Speech recognition engine {speech_recognition_engine} is not supported.')

    # check whether the audio file needs to be chunked and if so, chunk it
    audio_segment_files = chunk_audio(audio_path, segment_size, segment_overlap)

    # initialize the variable for storing the transcribed text
    recognized_text = ''

    # initialize speech recognition engine
    recognizer_engine = sr.Recognizer()

    try:
        for segment_audio_file in audio_segment_files:
            logger.debug(f'Transcribing {segment_audio_file} with {speech_recognition_engine}')
            # open and read the audio segment file
            with sr.AudioFile(segment_audio_file) as audio_file:
                # read the audio track from the file
                audio_track = recognizer_engine.listen(audio_file)
                # transcribe the audio track based on the speech recognition engine
                if speech_recognition_engine == 'sphinx':
                    #TODO to be tested to determine that it works as expected
                    recognized_text += f'{json.loads(recognizer_engine.recognize_sphinx(audio_track, language=language))}\n'
                elif speech_recognition_engine == 'vosk':
                    recognized_text += f'{json.loads(recognizer_engine.recognize_vosk(audio_track, language=language))["text"]}\n'

        logger.info(f'Transcription process completed for {audio_path}.')

        # TODO create a temporary file to store the transcribed text
        # TODO create langchain document with text + metadata
        return recognized_text
    except FileNotFoundError:
        raise FileNotFoundError(f'Audio file not found at {audio_path}')
    except sr.UnknownValueError:
        raise RuntimeError(f'Audio file ({audio_path}) does not include understandable audio.')
    except Exception as e:
        raise IOError(f'Error transcribing data from {audio_path}: {e}')


if __name__ == '__main__':
    #video_to_audio('C:/Users/quent/Videos/interviews/TalkLab_Podcast_EP22_AI_Episode-1.3.mp4')
    #TODO implement support for YouTube video retrieval
    #video_to_audio('https://youtube.com/shorts/D8Gz_uOBtG0?si=Rbi2eyAB57kGx04Z', 'youtube')
    transcribed_text = transcribe_audio(audio_path=f'{AUDIO_TMP_FOLDER}/extracted_audio.wav',
                                        speech_recognition_engine='vosk')
    print(transcribed_text)
