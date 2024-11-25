from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def summarize_text(text, max_length=150, min_length=50):
    
    # Prepare input with a "summarize:" prefix
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
    )

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# Example  article
article = """
Football, also known as soccer in many parts of the world, is a sport that captivates millions with its simplicity and excitement. Played by two teams of eleven players on a rectangular field, the objective is to score goals by getting the ball into the opposing team's net. The game is governed by the Laws of the Game, which include rules on offside, fouls, and the use of yellow and red cards for misconduct. Football's rich history dates back to ancient civilizations, but the modern version began in England in the 19th century. Today, it is the most popular sport globally, with major tournaments like the FIFA World Cup and UEFA Champions League drawing massive audiences. Beyond the pitch, football has a profound cultural and social impact, uniting people and fostering community spirit.
"""

# Summarize the  article
summary = summarize_text(article)
print("Summary:", summary)
