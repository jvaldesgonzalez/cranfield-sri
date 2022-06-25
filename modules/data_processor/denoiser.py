import contractions


def expand_contractions(text: str):
    """
    It will divide words like didn't into did not,
    whick is better for the tokenizer
    """
    return contractions.fix(text)
