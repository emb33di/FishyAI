from agent import PropertyLawAgent
import os

def main():
    # Initialize the agent with the PDFs directory
    pdf_directory = "pdfs"
    agent = PropertyLawAgent(pdf_directory)
    
    print("Loading PDFs...")
    agent.load_pdfs()
    print("PDFs loaded successfully!")
    
    print("\nProperty Law AI Assistant")
    print("Type 'quit' to exit")
    
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        answer = agent.ask_question(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
