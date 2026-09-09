"""
Generates a synthetic sample of customer<->brand tweets for "AmazonHelp" that
matches the EXACT column schema of the real Kaggle "Customer Support on
Twitter" dataset:

    tweet_id, author_id, inbound, created_at, text,
    response_tweet_id, in_response_to_tweet_id

WHY SYNTHETIC: this sandbox's network is restricted to pypi/github/anthropic
domains and cannot reach kaggle.com. To keep the pipeline runnable and the
results honest, every downstream script is written against this exact schema
so a real download is a drop-in replacement:

    1. Download from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
    2. Filter to one brand, e.g.:
         df = pd.read_csv("twcs.csv")
         df = df[(df.author_id == "AmazonHelp") | (df.in_response_to_tweet_id.isin(
                 df[df.author_id == "AmazonHelp"].tweet_id))]
    3. Save as data/raw/tweets.csv with the same columns and re-run
       `python src/pipeline.py --rebuild-golden` to regenerate everything on
       real data. No other code changes needed.

The generator below builds ~600 (customer_tweet, agent_reply) pairs across
the 10 intents in src/intents.py, with randomized entities (order numbers,
product names, day counts) so surface text varies even within an intent -
this is what makes the classification task non-trivial rather than a
keyword lookup.
"""
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)

BRAND = "AmazonHelp"
PRODUCTS = ["headphones", "the blender", "my phone case", "the desk lamp",
            "the router", "my shoes", "the backpack", "the monitor",
            "the vacuum", "the coffee maker"]
ORDER_FMT = lambda: f"#{random.randint(100000000, 999999999)}"
DAYS = lambda: random.randint(2, 14)

# Each entry: (intent, list of customer text templates, list of agent reply templates)
TEMPLATES = {
    "order_status": (
        ["Hi, can you tell me where order {order} is? Placed it last week.",
         "@{brand} any update on order {order}? Tracking hasn't moved in 3 days.",
         "Where's my package for order {order}? No tracking info showing up.",
         "Just checking in on order {order}, when will it ship?"],
        ["Hi there! Let's take a look. Please DM us your order number {order} and zip code so we can pull up tracking. ^AB",
         "Thanks for reaching out! Could you send us the order number {order} via DM so we can check the latest tracking status? ^KT",
         "We hear you! Send us a DM with order {order} and we'll get you the most current tracking update. ^DM"],
    ),
    "delivery_delay": (
        ["My order {order} was supposed to arrive {days} days ago and it's still not here!",
         "This is ridiculous, order {order} is {days} days late with no explanation.",
         "@{brand} order {order} missed its delivery window again. What's going on?",
         "Order {order} says delivered but I never got it, and it was already {days} days late."],
        ["We're really sorry for the delay on order {order}. Please DM your order number and address so we can investigate with the carrier. ^JS",
         "Apologies for the wait on order {order}! Send us a DM with your order details and we'll escalate with the delivery partner. ^RL",
         "That's not the experience we want for you. DM us order {order} and we'll look into the carrier delay right away. ^AB"],
    ),
    "damaged_or_wrong_item": (
        ["{product} arrived completely broken from order {order}. Not happy.",
         "I got the wrong item for order {order}, I ordered {product} and got something else.",
         "Order {order} arrived damaged, {product} was smashed in the box.",
         "This is the second time {product} came defective, order {order}."],
        ["Oh no, we're sorry to hear that! Please DM us a photo along with order {order} and we'll get a replacement started. ^KT",
         "That's not okay, we'll fix this. DM order {order} with a photo of the damage so we can process a replacement. ^JS",
         "Sorry about that! Send a DM with order {order} and a picture and we'll sort out a replacement or refund. ^DM"],
    ),
    "refund_request": (
        ["I want a full refund for order {order}, this is unacceptable.",
         "Please refund order {order} immediately, item never arrived.",
         "@{brand} I've asked twice already, I need a refund on order {order}.",
         "Cancel and refund order {order}, I don't want it anymore after this experience."],
        ["We completely understand. Please DM your order number {order} so we can review and process a refund. ^RL",
         "So sorry for the trouble. DM us order {order} and we'll get the refund process started for you. ^AB",
         "We'd like to make this right. Send order {order} via DM and our team will look into the refund. ^KT"],
    ),
    "cancel_order": (
        ["Can I cancel order {order}? It hasn't shipped yet I think.",
         "I need to cancel order {order} before it ships, how do I do that?",
         "Please cancel my order {order}, I found it cheaper elsewhere.",
         "Is it too late to cancel order {order}?"],
        ["We can help with that. Please DM order {order} and we'll check if it can still be cancelled before shipping. ^JS",
         "Sure thing, DM us order {order} and we'll see what options are available to cancel. ^DM",
         "Let's take a look, DM order {order} and we'll confirm the cancellation status. ^RL"],
    ),
    "account_access": (
        ["I can't log into my account, keeps saying wrong password even after reset.",
         "@{brand} my account got locked out of nowhere, need help getting back in.",
         "Password reset email never arrived, I've tried 3 times.",
         "Someone else may have accessed my account, I can't log in anymore."],
        ["Sorry for the trouble! Please DM us the email on the account (no passwords) and we'll help you regain access. ^AB",
         "We can help secure your account. DM us the email on file so our team can assist. ^KT",
         "Let's get this fixed. Please send a DM with the account email (never the password) so we can investigate. ^JS"],
    ),
    "billing_dispute": (
        ["I was charged twice for order {order}, please fix this.",
         "There's a charge on my card I don't recognize, might be related to order {order}.",
         "Why was I billed for a subscription I cancelled months ago?",
         "@{brand} charged me the wrong amount for order {order}."],
        ["We're sorry about that! Please DM order {order} and the charge amount so our billing team can investigate. ^RL",
         "That shouldn't happen, let's fix it. DM us the details of the charge and order {order}. ^DM",
         "Apologies for the confusion. Send a DM with your account email and the charge date so we can look into it. ^AB"],
    ),
    "general_complaint": (
        ["Been a customer for years and the service has really gone downhill lately.",
         "This is the third bad experience in a row with {brand}, so frustrating.",
         "Customer service wait times are unacceptable, been trying to get help for an hour.",
         "Really disappointed with how {brand} has handled things lately."],
        ["We're sorry to hear that and want to understand more. Could you DM us more detail so we can help? ^KT",
         "That's not the experience we want you to have. Please DM us so we can look into what happened. ^JS",
         "We hear your frustration and want to make it right, please DM us the details. ^DM"],
    ),
    "positive_feedback": (
        ["Just wanted to say the replacement I got was super fast, thank you!",
         "@{brand} support just resolved my issue in minutes, really appreciate it!",
         "Great experience with customer service today, thanks for the quick help.",
         "Shoutout to the support team for sorting out my order so quickly."],
        ["That means a lot, thank you for letting us know! ^AB",
         "So glad we could help, thanks for the kind words! ^KT",
         "Awesome to hear, thank you for the feedback! ^RL"],
    ),
    "other": (
        ["following you now, love the brand",
         "does anyone even work here lol",
         "@{brand} check your DMs",
         "random question, are you hiring right now?"],
        ["Thanks for following us! ^DM",
         "We're here! How can we help? ^AB",
         "Thanks for the DM, we'll take a look! ^KT",
         "We are! Check our careers page for openings. ^RL"],
    ),
}

def fill(template):
    return template.format(order=ORDER_FMT(), product=random.choice(PRODUCTS),
                            days=DAYS(), brand=BRAND)

def generate(n_per_intent=60):
    rows = []
    tweet_id = 1
    base_time = datetime(2024, 3, 1)
    for intent, (cust_templates, agent_templates) in TEMPLATES.items():
        for i in range(n_per_intent):
            cust_text = fill(random.choice(cust_templates))
            agent_text = fill(random.choice(agent_templates))
            t_cust = base_time + timedelta(minutes=tweet_id * 7)
            t_agent = t_cust + timedelta(minutes=random.randint(5, 90))

            cust_id = tweet_id
            agent_id = tweet_id + 1

            rows.append({
                "tweet_id": cust_id,
                "author_id": f"cust_{random.randint(10000,99999)}",
                "inbound": True,
                "created_at": t_cust.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": cust_text,
                "response_tweet_id": str(agent_id),
                "in_response_to_tweet_id": "",
                "_gold_intent": intent,  # kept with leading underscore = not part of real schema
            })
            rows.append({
                "tweet_id": agent_id,
                "author_id": BRAND,
                "inbound": False,
                "created_at": t_agent.strftime("%a %b %d %H:%M:%S +0000 %Y"),
                "text": agent_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": str(cust_id),
                "_gold_intent": intent,
            })
            tweet_id += 2
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate(n_per_intent=60)  # 10 intents * 60 * 2 rows = 1200 rows, 600 threads
    # Public-schema file (what a real pipeline would ingest) drops the gold label
    public_cols = ["tweet_id", "author_id", "inbound", "created_at", "text",
                    "response_tweet_id", "in_response_to_tweet_id"]
    df[public_cols].to_csv("data/raw/tweets.csv", index=False)
    # Internal file that keeps the gold intent, used ONLY to build the golden eval set,
    # never fed to the classifier at inference time.
    df.to_csv("data/raw/tweets_with_gold_intent_INTERNAL.csv", index=False)
    print(f"Wrote {len(df)} rows ({len(df)//2} threads) to data/raw/tweets.csv")
