import datetime
import time
import openai
import backoff


API_KEY = ""
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

@backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=8)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(args, input):
    
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    # time.sleep(1)
    time.sleep(args.api_time_interval)
    
    # https://beta.openai.com/account/api-keys
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = API_KEY
    #print(openai.api_key)
    
    # Specify engine ...
    # Instruct GPT3
    if args.model == "gpt3":
        engine = "text-ada-001"
    elif args.model == "gpt3-medium":
        engine = "text-babbage-001"
    elif args.model == "gpt3-large":
        engine = "text-curie-001"
    elif args.model == "gpt3-xl":
        engine = "text-davinci-002"
    else:
        raise ValueError("model is not properly defined ...")
        
    # response = openai.Completion.create(
    #   engine=engine,
    #   prompt=input,
    #   max_tokens=max_length,
    #   temperature=0,
    #   stop=None
    # )
    response = completions_with_backoff(
        model=engine,  # text-davinci-002  code-davinci-002
        prompt=input, 
        temperature=0, 
        max_tokens=args.max_length,
    )
    
    return response["choices"][0]["text"]

class Decoder():
    def __init__(self, args):
        print_now()
 
    def decode(self, args, input):
        response = decoder_for_gpt3(args, input)
        return response
