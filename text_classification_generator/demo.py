
import turicreate as tc

file_name = "SAD.csv"
model_name = "Sentiment"
author = "By Cyril"

data = tc.SFrame(file_name)

model = tc.sentence_classifier.create(data, 'Sentiment', features=['SentimentText'])

model.author = author

model.save(model_name + '.model')

model.export_coreml(model_name + '.mlmodel')

