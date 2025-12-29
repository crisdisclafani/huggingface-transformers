from transformers import pipeline
import spacy
from spacy import displacy

#sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")
result1 = classifier("I love using Hugging Face transformers!")
print(result1)

result2 = classifier("This is the worst experience I've ever had.")
print(result2)

zsc=pipeline("zero-shot-classification")
result3 = zsc("Hugging Face is creating a tool that the community uses to solve NLP tasks.", 
    candidate_labels=["technology", "education", "politics"])
print(result3)

result4 = zsc("Juventus won yesterday", 
    candidate_labels=["technology", "sports", "politics"])
print(result4)

generator = pipeline("text-generation", model="gpt2")
result5 = generator("In a distant future, humanity has", 
                    max_new_tokens=30,max_length=30, num_return_sequences=2)

print(result5)

ner=pipeline("ner", grouped_entities=True)
result6 = ner("My name is Cris, i live in Turin and I work for CrisAndCris Inc.")
print(result6)

#spacy for visualization
nlp = spacy.load("en_core_web_sm")
doc = nlp("My name is Cris, i live in Turin and I work for CrisAndCris Inc.")
displacy.serve(doc, style="ent")

#use spacy to show result6 values from transformers
ents = []
for ent in result6:
    ents.append({
        "start": ent['start'],
        "end": ent['end'],
        "label": ent['entity_group']
    })
doc2 = spacy.tokens.Doc(nlp.vocab, words=list("My name is Cris, i live in Turin and I work for CrisAndCris Inc.".replace(",", "").split(" ")))
doc2.ents = [spacy.tokens.Span(doc2,  ent['start'], ent['end'], label=ent['label']) for ent in ents]
displacy.serve(doc2, style="ent")


translate=pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

result7 = translate("Hugging Face is creating a tool that the community uses to solve NLP tasks.")
print(result7)

translate_it=pipeline("translation", model="Helsinki-NLP/opus-mt-en-it")
result8 = translate_it("Hugging Face is creating a tool that the community uses to solve NLP tasks.")
print(result8)