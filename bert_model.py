'''from transformers import pipeline

# Load sentiment model
classifier = pipeline("sentiment-analysis")

# Test
texts = [
    "Boy I love this music",
    "Wow man you can talk",
    "Bro seriously you left your friend alone."
]

for t in texts:
    print(t, "->", classifier(t))
'''
