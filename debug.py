from transformers import BertTokenizer


if __name__ == '__main__':
    ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    print(tokenizer.tokenize("中华<e1>人民</e1>共和国"))
