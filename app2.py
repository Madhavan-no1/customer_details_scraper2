import spacy
from transformers import pipeline
import json
import re

# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Define categories
customer_requirements = {
    "Car Type": ["Hatchback", "SUV", "Sedan"],
    "Fuel Type": ["Petrol", "Diesel", "Electric"],
    "Color": [],
    "Distance Travelled": [],
    "Make Year": [],
    "Transmission Type": ["Automatic", "Manual"]
}

company_policies = [
    "Free RC Transfer",
    "5-Day Money Back Guarantee",
    "Free RSA for One Year",
    "Return Policy"
]

customer_objections = [
    "Refurbishment Quality",
    "Car Issues",
    "Price Issues",
    "Customer Experience Issues"
]

# Zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", clean_up_tokenization_spaces = "True")

def extract_customer_requirements(transcript):
    result = {}
    
    for category, options in customer_requirements.items():
        if options:
            prediction = classifier(transcript, options)
            result[category] = prediction['labels'][0] if prediction['scores'][0] > 0.5 else None
        else:
            doc = nlp(transcript)
            if category == "Color":
                colors = [ent.text for ent in doc.ents if ent.label_ == "COLOR"]
                result[category] = colors if colors else None
            elif category == "Make Year":
                years = [ent.text for ent in doc.ents if re.match(r"\b(19|20)\d{2}\b", ent.text)]
                result[category] = years if years else None
            elif category == "Distance Travelled":
                distances = [ent.text for ent in doc.ents if re.match(r"\d+(,\d{3})*(\.\d+)?\s*(km|kilometers|miles)", ent.text)]
                result[category] = distances if distances else None
    
    return result

def extract_company_policies(transcript):
    result = []
    prediction = classifier(transcript, company_policies)
    for i, policy in enumerate(prediction['labels']):
        if prediction['scores'][i] > 0.5:
            result.append(policy)
    return result

def extract_customer_objections(transcript):
    result = []
    prediction = classifier(transcript, customer_objections)
    for i, objection in enumerate(prediction['labels']):
        if prediction['scores'][i] > 0.5:
            result.append(objection)
    return result

def process_transcript(transcript, conversation_id):
    customer_requirements = extract_customer_requirements(transcript)
    company_policies = extract_company_policies(transcript)
    customer_objections = extract_customer_objections(transcript)

    return {
        "conversation_id": conversation_id,
        "customer_requirements": customer_requirements,
        "company_policies_discussed": company_policies,
        "customer_objections": customer_objections
    }

def main():
    # Example transcripts (replace with file reading logic)
    transcripts = {
        "001": """
        I am looking for a red SUV, preferably automatic. It should not have traveled more than 50,000 kilometers,
        and should be from 2018 or later. I'm concerned about the refurbishment quality and the high price.
        Also, is the 5-day money-back guarantee applicable?
        """,
        "002": """
        I'm interested in a hatchback with manual transmission. It should be a diesel car, preferably blue in color. 
        I'm not sure about the price, and I'm worried about the long wait times.
        """
    }

    results = []
    for conversation_id, transcript in transcripts.items():
        result = process_transcript(transcript, conversation_id)
        results.append(result)
    
    # Convert results to JSON format
    output = {
        "transcripts": results
    }
    
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
