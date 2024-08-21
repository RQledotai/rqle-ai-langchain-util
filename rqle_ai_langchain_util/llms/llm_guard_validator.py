import re

from llm_guard.input_scanners import (
    Language as InputLanguage,
    PromptInjection as InputPromptInjection,
    Toxicity as InputToxicity)
from llm_guard.input_scanners.language import MatchType as InputLanguageMatchType
from llm_guard.input_scanners.prompt_injection import MatchType as InputPromptInjectionMatchType
from llm_guard.input_scanners.toxicity import MatchType as InputToxicityMatchType
from llm_guard.output_scanners import (
    Gibberish as OutputGibberish,
    ReadingTime,
    Relevance as OutputRelevance,
    Sentiment as OutputSentiment,
    Toxicity as OutputToxicity
)
from llm_guard.output_scanners.gibberish import MatchType as OutputGibberishMatchType
from llm_guard.output_scanners.toxicity import MatchType as OutputToxicityMatchType


def validate_llm_input(input: str, languages: list[str] = None) -> (bool, str):
    '''
    :param input: the input text from the LLM
    :return: a tuple of boolean and an error message
    '''
    # check whether language is valid
    if languages is None:
        languages = ['en']
    language_scanner = InputLanguage(valid_languages=languages, match_type=InputLanguageMatchType.FULL)
    lang_sanitized_prompt, lang_is_valid, lang_risk_score = language_scanner.scan(prompt=input)
    if not lang_is_valid:
        return (False, f'Input is in a language not supported by application with {lang_risk_score} confidence.'
                       f'The list of supported languages is {languages}.')
    # check whether the input includes prompt injection
    prompt_injection_scanner = InputPromptInjection(threshold=0.5, match_type=InputPromptInjectionMatchType.FULL)
    prompt_injection_sanitized_prompt, prompt_injection_is_valid, prompt_injection_risk_score = prompt_injection_scanner.scan(prompt=input)
    if not prompt_injection_is_valid:
        return (False, f'Input contains risk of prompt injection with {prompt_injection_risk_score} confidence.')
    # check whether the input includes toxic content
    toxicity_scanner = InputToxicity(threshold=0.5, match_type=InputToxicityMatchType.SENTENCE)
    toxic_sanitized_prompt, toxic_is_valid, toxic_risk_score = toxicity_scanner.scan(prompt=input)
    if not toxic_is_valid:
        return (False, f'Input contains toxic content with {toxic_risk_score} confidence.')
    return (True, '')


def validate_llm_output(prompt: str, output: str) -> (bool, str):
    """
    :param prompt: the prompt text used by LLM
    :param output: the output text from the LLM
    :return: a tuple of boolean and an error message
    """
    # check whether the output is gibberish or nonsensical content
    gibberish_scanner = OutputGibberish(threshold=0.5, match_type=OutputGibberishMatchType.FULL)
    gibberish_sanitized_output, gibberish_is_valid, gibberish_risk_score = gibberish_scanner.scan(prompt=prompt, output=output)
    if not gibberish_is_valid:
        return (False, f'Output is gibberish with {gibberish_risk_score} confidence.')
    # check whether the output includes text with negative sentiment
    sentiment_scanner = OutputSentiment(threshold=0)
    sentiment_sanitized_output, sentiment_is_valid, sentiment_risk_score = sentiment_scanner.scan(prompt=prompt, output=output)
    if not sentiment_is_valid:
        return (False, f'Output contains content with negative sentiment with {sentiment_risk_score} confidence.')
    # check whether the output includes toxic content
    toxicity_scanner = OutputToxicity(threshold=0.5, match_type=OutputToxicityMatchType.SENTENCE)
    toxic_sanitized_output, toxic_is_valid, toxic_risk_score = toxicity_scanner.scan(prompt=prompt, output=output)
    if not toxic_is_valid:
        return (False, f'Output contains toxic content with {toxic_risk_score} confidence.')
    return (True, '')


def validate_llm_output_relevance(prompt: str, output: str) -> float:
    '''
    :param prompt: the prompt text used by LLM
    :param output: the output text from the LLM
    :return: the relevancy score of the text compaired to prompt
    '''
    # check whether the output is relevant to the prompt
    relevance_scanner = OutputRelevance(threshold=0.0)
    relevance_sanitized_output, relevance_is_valid, relevance_risk_score = relevance_scanner.scan(prompt=prompt,
                                                                                                  output=output)
    return relevance_risk_score


def validate_llm_output_reading_time(prompt: str, output: str, reading_time: int = 5) -> (bool, str):
    '''
    :param prompt: the prompt text used by LLM
    :param output: the output text from the LLM
    :param reading_time: the expected reading time in minutes
    :return: a tuple of boolean and a message
    '''
    scanner = ReadingTime(max_time=reading_time, truncate=True)
    sanitized_output, is_valid, risk_score = scanner.scan(prompt=prompt, output=output)
    if not is_valid:
        return (False, sanitized_output)
    else:
        return (True, output)


def estimate_reading_time(text: str, WPM: int = 200) -> int:
    """
    Calculate the estimated reading time for a given text based on the average words per minute.

    :param text: The input text for which the reading time needs to be estimated.
    :param WPM: The average words per minute for calculating the reading time. Default is 200.
    :return: The estimated reading time in minutes as an integer
    """
    total_words = len(re.findall(r'\w+', text))
    time_minute = total_words // WPM + 1
    if time_minute == 0:
        time_minute = 1
    return time_minute
