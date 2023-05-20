from transformers import AutoModelForSequenceClassification,AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch


def predict_sentiment(conversation):
    last_lines_num = 7
    tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
    model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
    lines = conversation.split('\n')
    customer_lines = [line for line in lines if line.startswith("Customer:")]
    last_half_chat = '\n'.join(customer_lines[-last_lines_num:])
    encoded_prompt = tokenizer(last_half_chat, return_tensors="pt")
    output = model(**encoded_prompt)
    predicted_label = torch.argmax(output.logits).item()
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    predicted_sentiment = sentiment_labels[predicted_label]

    return predicted_sentiment

def query_conversation(conversation,question):
    qa_model_name = "deepset/roberta-base-squad2"

    # a) Get predictions
    nlp = pipeline('question-answering', model=qa_model_name, tokenizer=qa_model_name)
    QA_input = {
        'question': question,
        'context': conversation
    }
    res = nlp(QA_input)

    # b) Load model & tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    # c) Tokenize input
    input_dict = tokenizer(QA_input["question"], QA_input["context"], return_tensors="pt")

    # d) Feed input_dict to model
    output = model(**input_dict)

    # e) Extract answer
    start_index = output.start_logits.argmax().item()
    end_index = output.end_logits.argmax().item()
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_dict["input_ids"][0][start_index:end_index + 1]))

    return answer


def diagnose_conversation(conversation):
    query_product_problem = "What is the problem of the product?"
    query_product_name = "What is the name of the user's product?"

    product_name = query_conversation(promptNeg, query_product_name)
    product_problem = query_conversation(promptNeg, query_product_problem)
    conversation_sentiment = predict_sentiment(conversation)
    return product_name, product_problem, conversation_sentiment
promptNeg = """
Customer: Hi, I'm having trouble with my electric Toshiba X8672 washing machine.
Agent: Hi there, I'm sorry to hear that. What seems to be the problem?
Customer: It makes a horrible noise when starting up. I've tried a few things, but it's still not working.
Agent: Okay, let's try some troubleshooting steps. Have you tried unplugging the machine and plugging it back in?
Customer: Yes, I have. It didn't help.
Agent: Alright. Have you checked to make sure the circuit breaker for the washing machine isn't tripped?
Customer: Yes, I did that too. Still no luck.
Agent: I see. Have you tried resetting the machine?
Customer: Yes, I did that as well. It's still not working.
Agent: I'm sorry to hear that. Let me check if there are any known issues with this model. Can you please give me a moment?
Agent: Thank you for waiting. Unfortunately, it seems like there are no known issues with this model. I suggest you schedule a service appointment.
Customer: This is so frustrating. I'm really disappointed that this machine isn't working properly.
Agent: I understand your frustration. We want our customers to be happy with our products. I apologize for the inconvenience caused. Is there anything else I can do to help?
Customer: No, that's all. I'll just schedule a service appointment.
Agent: Alright. I hope the issue gets resolved soon. Have a good day.
"""

promptNeg2 = """

Customer: Hi, I'm having trouble with my electric Toshiba X8672 washing machine.
Agent: Hi there, I'm sorry to hear that. What seems to be the problem?
Customer: It's not working properly. I've tried a few things, but it's still not working.
Agent: Okay, let's try some troubleshooting steps. Have you tried unplugging the machine and plugging it back in?
Customer: Yes, I have. It didn't help.
Agent: Alright. Have you checked to make sure the circuit breaker for the washing machine isn't tripped?
Customer: Yes, I did that too. Still no luck.
Agent: I see. Have you tried resetting the machine?
Customer: Yes, I did that as well. It's still not working.
Agent: I'm sorry to hear that. Let's try one more thing before we schedule a service appointment. Can you please try cleaning the filter? Sometimes a dirty filter can cause issues with the machine.
Customer: I didn't think of that. Let me try it.
Customer: It didnt work!
(Client leaves the chat)

"""

prompt = """

Customer: Hi, I'm having trouble with my electric Toshiba X8672 washing machine.

Agent: Hi there, I'm sorry to hear that. What seems to be the problem?

Customer: It's not working properly. I've tried a few things, but it's still not working.

Agent: Okay, let's try some troubleshooting steps. Have you tried unplugging the machine and plugging it back in?

Customer: Yes, I have. It didn't help.

Agent: Alright. Have you checked to make sure the circuit breaker for the washing machine isn't tripped?

Customer: Yes, I did that too. Still no luck.

Agent: I see. Have you tried resetting the machine?

Customer: Yes, I did that as well. It's still not working.

Agent: I'm sorry to hear that. Let's try one more thing before we schedule a service appointment. Can you please try cleaning the filter? Sometimes a dirty filter can cause issues with the machine.

Customer: I didn't think of that. Let me try it.

Customer: Wow, it worked! The machine is working perfectly fine now.

Agent: That's great news! I'm glad the issue is resolved. Is there anything else I can do for you?

Customer: No, that's all. Thank you so much for your help. I really appreciate it.

Agent: You're welcome! We're always here to help. If you have any more issues, don't hesitate to contact us. Have a great day!

"""

product_name, product_problem, conversation_sentiment = diagnose_conversation(promptNeg)

print("Product name:", product_name)
print("Product problem:", product_problem)
print("Predicted sentiment:", predict_sentiment(promptNeg))
