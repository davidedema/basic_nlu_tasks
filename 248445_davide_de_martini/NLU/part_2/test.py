from transformers import BertTokenizer

text = "'d"

tknz = BertTokenizer.from_pretrained('bert-base-uncased')
res = tknz(text)

print(tknz.convert_ids_to_tokens(res['input_ids']))