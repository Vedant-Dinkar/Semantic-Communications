import os
import tiktoken

def tiktoken_test():
    """
    A simple function to demonstrate the core functionalities of the tiktoken library.
    """
    try:
        # 1. Get the tokenizer for a specific OpenAI model (e.g., "gpt-4o")
        # This is the recommended way to get the correct encoding for a model.
        encoding = tiktoken.encoding_for_model("gpt-4o")
        print(f"✅ Successfully loaded tokenizer: {encoding.name}")

    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("Please make sure you have installed tiktoken with 'pip install tiktoken'")
        return

    print("\n" + "="*50 + "\n")
    
    enc_gpt4o = tiktoken.encoding_for_model("gpt-4o")
    print(f"GPT-4o ('o200k_base') Vocabulary Size: {enc_gpt4o.n_vocab}")

    print("\n" + "="*50 + "\n")
    
    file_path = './text_input.txt'
    
    # 2. Define a sample text to work with
    f = open(file_path, 'r')
    sample_text = f.read()
    f.close()
    print(f"Original Text:\n'{sample_text[:50]}'")

    print("\n" + "-"*50 + "\n")

    # 3. Encode the text into a list of token integers
    encoded_tokens = encoding.encode(sample_text)
    print(f"Type: {type(encoded_tokens), type(encoded_tokens[0])}")
    print(f"Encoded Tokens (list of integers):\n{encoded_tokens}")

    # 4. Count the number of tokens
    token_count = len(encoded_tokens)
    print(f"\nNumber of tokens: {token_count}")

    print("\n" + "-"*50 + "\n")

    # 5. Decode the list of tokens back into a string
    decoded_text = encoding.decode(encoded_tokens)
    print(f"Decoded Text:\n'{decoded_text[:50]}'")

    print("\n" + "="*50)
    
    print(f"Compression Ratio: {os.path.getsize(file_path) * 8.0 / (token_count * 18)}") 

    # 6. Verify that the decoded text matches the original
    assert sample_text == decoded_text
    print("\n✅ Verification successful: Decoded text matches the original text.")


if __name__ == "__main__":
    tiktoken_test()