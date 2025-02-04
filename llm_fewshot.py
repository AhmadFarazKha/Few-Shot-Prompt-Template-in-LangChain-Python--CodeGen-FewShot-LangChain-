from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate  # No langchain_core needed here
from langchain.prompts.example_selector import LengthBasedExampleSelector
from decouple import config
import os

# Load environment variables
os.environ["GOOGLE_API_KEY"] = config("GOOGLE_API_KEY")

# Initialize Google model
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Define few-shot examples
examples = [
    {"input": "happy", "output": "The sun was shining brightly, birds were singing..."},
    {"input": "mysterious", "output": "The fog clung to the ancient stones..."},
    {"input": "tense", "output": "The clock's ticking echoed..."},
    {"input": "joyful", "output": "Laughter filled the air, children played..."}, # Added more examples
    {"input": "serene", "output": "The calm lake reflected the pastel sky..."},
    {"input": "chaotic", "output": "Sirens wailed, people rushed in all directions..."}
]

# Create prompt template for examples
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Create example selector
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=300 # Adjust this as needed
)

# Create dynamic prompt template (using the selector)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
    example_separator="\n\n"
)



def generate_description(adjective):
    formatted_prompt = dynamic_prompt.format(adjective=adjective)
    response = llm.invoke(formatted_prompt)
    return response.content

if __name__ == "__main__":
    adjective = input("Enter an adjective to generate a scene: ")
    result = generate_description(adjective)
    print("\nGenerated Scene:")
    print(result)