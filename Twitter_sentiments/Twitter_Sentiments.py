from textblob import TextBlob
import tweepy
import csv


consumer_key= 'wA7aAgh2K8TH42IyrZhNVstVT'
consumer_secret= '4pM0ZS6GIPbEhjNVW7xmMZlnRBK7R3z5yDk0IaLPndEwQqthBG'

access_token='143696051-JRp7w0Pd3mBaiVMM1luMwQkqP9YwBuyzYVGgsih8'
access_token_secret='TLz9r6JsKYV8MCqa4ONH2vrdIKqJ0i9ril008lw4pGt5t'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Formula1')

#for tweet in public_tweets:
#	print(tweet.text)
#
#	analysis = TextBlob(tweet.text)
#	print(analysis.sentiment)
#	print("")


with open('Sentiments_on_twitter.csv', 'w', newline = '') as csvfile:
	writer = csv.writer(csvfile)
	for tweet in public_tweets:
		analysis = TextBlob(tweet.text)
		emotion = analysis.polarity
		if emotion > 0:
			writer.writerow([tweet.text,"positive",analysis])
		else:
			writer.writerow([tweet.text,"negative", analysis])
