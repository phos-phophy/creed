def get_tokenizer_len_attribute(tokenizer):
    for attr in tokenizer.__dict__:
        if 'max_len' in attr:
            return attr

    raise ValueError("Can not find max_length attribute in tokenizer object")
