from agent import PropertyLawAgent
import os
import time

def print_welcome():
    print("\n" + "="*50)
    print("Welcome to the Property Law AI Assistant!")
    print("="*50)
    print("\nCommands:")
    print("- Type your question and press Enter to get an answer")
    print("- Type 'reload' to reload PDFs")
    print("- Type 'clear' to clear chat history")
    print("- Type 'quit' to exit")
    print("\nNote: Your chat history will be maintained until you clear it or quit.")
    print("="*50 + "\n")

def main():
    # Initialize the agent with the PDFs directory
    pdf_directory = "pdfs"
    agent = PropertyLawAgent(pdf_directory)
    
    print_welcome()
    
    # Initial PDF loading
    success, message = agent.load_pdfs()
    print(message)
    if not success:
        print("\nPlease add your PDF files to the 'pdfs' directory and type 'reload' when ready.")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
                
            if question.lower() == 'quit':
                print("\nThank you for using the Property Law AI Assistant. Goodbye!")
                break
                
            if question.lower() == 'reload':
                print("\nReloading PDFs...")
                success, message = agent.load_pdfs()
                print(message)
                continue
                
            if question.lower() == 'clear':
                agent.chat_history = []
                print("\nChat history cleared.")
                continue
            
            print("\nThinking...")
            start_time = time.time()
            
            result = agent.ask_question(question)
            
            print("\nAnswer:", result["answer"])
            
            if result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"- {source}")
            
            end_time = time.time()
            print(f"\nResponse time: {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
