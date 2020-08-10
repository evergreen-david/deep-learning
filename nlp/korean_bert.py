from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base")

tokens = tokenizer.encode("안녕 난 피자를 좋아해")

for token in tokens:
    print(tokenizer.decode(token))


