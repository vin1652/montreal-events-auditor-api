from fastembed.text import TextEmbedding
#print(TextEmbedding.list_supported_models())

model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
texts = ["Montreal events", "Jazz festival", "AI meetup"]

e = model.embed(texts)
for item in e:
    print(item[:5])  # print first 5 values of each embedding