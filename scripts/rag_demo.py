"""Run the full pipeline on a fixed list of claims (was test_rag.py's __main__)."""
from claimscan.pipeline import fact_check, load_services

if __name__ == "__main__":

    services = load_services()

    print("Loading complete \n")

    test_claims = [
    # Clear factual claims
    "Water boils at 100 degrees Celsius",
    "Tokyo is the capital of Japan",
    "Albert Einstein developed the theory of relativity",
    
    # Current/temporal claims (test temporal filtering)
    "Trump is the president of the United States",
    "Joe Biden is the current president of the United States",
    "Barack Obama is the former president of the United States",
    "Canada was a colony of the UK",
    "India was also a colony of the UK",
    "Socrates was a real person",
    
    # Sports/records (test comparative + biographical)
    "Lebron James is the all time points leader in the NBA",
    "Michael Jordan played for the Chicago Bulls",
    "Tom Brady won seven Super Bowls",
    "Lionel Messi has won a world cup or seven Super Bowls",
    
    # Clear false claims
    "The moon is made of cheese",
    "Cats can fly naturally",
    "The Pacific Ocean is the smallest ocean",
    "The Earth is flat",
    "The Eiffel Tower is in London",
    "Australia is the largest country in the world",
    "The sun orbits the Earth",
    "Steph Curry is the all time 3-pt leader in the WNBA",
    
    # Authorship claims (test authorship filtering)
    "Charles Dickens wrote Oliver Twist",
    "J.K. Rowling wrote Harry Potter",
    "William Shakespeare authored Hamlet",
    "Hamlet is a play by Shakespeare",
    "George Orwell wrote 1984",
    "Jane Austen wrote Pride and Prejudice",
    "Ernest Hemingway wrote The Old Man and the Sea",
    
    # Geographic claims (test geographic filtering)
    "Rome is the capital of Italy",
    "Moscow is located in Russia",
    "The Amazon rainforest is in South America",
    
    # Biographical claims (test biographical filtering)
    "Steve Jobs founded Apple",
    "Nelson Mandela was president of South Africa",
    "Martin Luther King Jr. was born in Atlanta",
    "Leonardo da Vinci painted the Mona Lisa",
    "Drake made Gods Plan",
    
    # Edge cases that might lack evidence
    "The weather is nice today",
    "This coffee tastes good"
    ]
    
    for claim in test_claims:
      pred, response = fact_check(claim, services)
      print(f"\nClaim: {claim}")
      print(f"Result: {response}")
