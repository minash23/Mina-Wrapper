from dotenv import load_dotenv  # Module to load environment variables from a .env file
import os  # Module for interacting with the operating system
from openai import OpenAI  # OpenAI library for interacting with their API
import sys  # Module for interacting with the Python runtime environment

# Load environment variables from the .env file (e.g., API keys, configurations)
load_dotenv()

# Retrieve the OpenAI API key and configuration settings from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")  # Your API key to authenticate requests to OpenAI
API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")  # Base URL for the API
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")  # Default model to use
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))  # Default creativity level
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", 150))  # Default max token limit

# Check if the API key is provided; if not, raise an error and stop the program
if not API_KEY:
    raise EnvironmentError("Please set your OPENAI_API_KEY in the .env file.")

# Initialize the OpenAI client using the provided API key and base URL
client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def get_response(prompt, model="gpt-3.5-turbo", temperature=0.7, max_tokens=150):
    """
    Sends a prompt to the ChatGPT API and retrieves a response.

    Args:
        prompt (str): The user's input to send to the API.
        model (str): The language model to use (default: gpt-3.5-turbo).
        temperature (float): Creativity level (0.0 = deterministic, 1.0 = more creative).
        max_tokens (int): Maximum length of the response.

    Returns:
        str: The response from the ChatGPT model.
    """
    try:
        # Send the request to the API and get the response
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],  # User's input is passed here
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Extract and return the main content of the API's response
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Handle errors (e.g., API issues, network problems) and inform the user
        print(f"An error occurred: {e}")
        return f"Error: {e}"


def main():
    """
    Main function to interact with the user and get responses from the ChatGPT API.
    """
    print("Welcome to the ChatGPT Wrapper Application!")
    print("Type 'exit' to quit.")  # Instructions for the user to end the program

    while True:
        # Prompt the user for input
        user_input = input("\nYou: ")

        # Exit the application if the user types 'exit'
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Validate that the user input is not empty or just whitespace
        if not user_input.strip():
            print("Invalid input. Please try again.")
            continue

        # Get the response from ChatGPT using the input and user-defined settings
        response = get_response(user_input)

        # Display the response to the user
        print(f"\nChatGPT: {response}")


# Entry point of the program
if __name__ == "__main__":
    main()  # Start the application
