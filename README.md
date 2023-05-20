# HACKRU 2023 Hackathon

![image](https://github.com/Orbitoly/hackru_23_hackathon/assets/17669444/a225d059-f5f7-47c8-a89f-d243c9f7f804)


# Improving NICE Customer Support

This project was developed as part of the NICE company challenge aimed at improving the customer support system. The objective was to use AI to improve its current system.

## Problem Statement

During the hackathon, we interacted with product managers to gain a deeper understanding of the current situation.
We discovered that it there is a high customer churn rate during bad customer support - one of the problems are recurring issues that had not been resolved through past conversations - are causing frustration as the agent / Bot tries to solve them for the client not knowing that statisticlly such problem was not resolved.
To tackle this problem, we proposed an algorithm that leverages natural language processing (NLP) models.

## Solution

Our solution involves using the Robustly Optimized BERT Approach (RoBERTa) NLP model to extract the model of the problem mentioned by the user and the specific problem itself. this done by using a "QA Task" - providing the model with the conversation as a context.
In addition we've used sentiment analysis on past conversations - analysing how the conversations ended - and only the lines of the client.
By analyzing past conversations, we apply statistical analysis to identify product-related problems that have not been resolved (bad sentiment in the end of conversation from client side). We've developed a frontend-backend system which simulates Chat with agent and alerts Realtime to the agent side that this product with this problem were statisticlly not resolved in the past.
This allows the Agent to transfer the call to someone more proffesional, or if it is A bot - this allows the bot to transfer the chat to a human agent.

## Algorithm Overview

Train phase:
1. Input: User's conversations text about a product and a problem.
2. Extract problem model and problem using RoBERTa NLP model, and sentiment of the end of conversation from client side - assuming that tells if the problem was resolved or not.
3. Create DB - products -> problems -> resolved %.

Production phase:
1. Input: User's conversation text about a product - realtime.
2. Extract problem model and problem using RoBERTa NLP model.
3. Query DB - Identify statistically unresolved problems by comparing sentiment analysis results.
4. Provide recommendations to customer support teams to prioritize unresolved issues.
5. Update DB in the end of conversation

## Technologies Used

- RoBERTa: We utilized the RoBERTa NLP model for extracting problem models and problems from user conversations.
- Statistical Analysis: We applied statistical techniques to determine unresolved problems based on sentiment analysis of past conversations.

## Demo
<img width="816" alt="image" src="https://github.com/Orbitoly/hackru_23_hackathon/assets/17669444/a4964576-ec27-40ec-b16a-2c3c7859f29d">
<img width="809" alt="image" src="https://github.com/Orbitoly/hackru_23_hackathon/assets/17669444/a7a5d35c-080e-445c-a8e0-93a416746bce">
<img width="866" alt="image" src="https://github.com/Orbitoly/hackru_23_hackathon/assets/17669444/9c9be7a4-fa92-49c1-95de-0b28626941ca">

## Assumptions
The conversations are 'tagged' with the speaker identity and the side which ended the conversation<br/>
Agent: ...<br/>
Customer: ...<br/>
Agent: ...<br/>
Customer: ...<br/>
-- Agent / Client ended the conversation --<br/>


## Getting Started

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/your-repo.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the application: `python model.py`


## Conclusion

Our project provides a solution to improve the customer support system by automatically identifying and prioritizing unresolved product-related problems. By leveraging NLP models and sentiment analysis, customer support teams can identify problematic conversations and allocate their resources more effectively, resulting in less client frustration and churn rate.

## Acknowledgments

We would like to express our gratitude to Reichman University and NICE company for organizing this hackathon and providing us with an opportunity to work on such a challenging problem. Special thanks to the product managers who provided valuable insights and feedback throughout the development process.

## Meet the Team

<img width="741" alt="image" src="https://github.com/Orbitoly/hackru_23_hackathon/assets/17669444/455f58a4-4bbc-4fc6-92d5-00874aa6fbcf">
<br/>Ronny Baturov, Omri Gal Shenhav, Lior Alon, Or Virt
