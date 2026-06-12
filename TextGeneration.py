from transformers import pipeline

p = pipeline("text-generation",model="gpt2")

a=input("Enter text : ")

r=p(a,max_length=100)

print("\n")

for i in r:
  print(i['generated_text'])
