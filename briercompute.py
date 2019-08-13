import numpy as np
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
from scipy.stats import rankdata

defaultusername = "__default"
default_score = 0.5

def brier(optionprobs, correct, ordered=False):
	"""
		This computes the brier score for a user on
		a *single* day (daily averages are handled in a different function).

		optionprobs: a LIST of answer probabilities. Should sum to 1.
			In other words, it assumes that *all* possibilites are accounted for.
			e.g. [0.64, 0.36] for a binary question.
			In the case of an ordered multinomial, it assumes that the ordering is correct.

		correct: the index of the correct answer. Zero-based.

		ordered: a boolean flag determining whether or not the answer is *ordered* multinomial.
			If so, the formula is substantially different.
	"""

	# make a one-hot vector for the true outcome.
	true_outcomes = [0] * len(optionprobs)
	true_outcomes[correct] = 1

	brier = 0.
	if ordered:
		# the the original categories (A-B-C-D) and break them up into a set of cumulative,
		# binary categories (A-BCD; AB-CD; ABC-D)
		for split in range(1, len(optionprobs)):
			# group into two binary categories
			lside = sum(optionprobs[0:split])
			rside = sum(optionprobs[split:])
			# figure out if the left side or right side happened,
			# making a one-hot vector for this pseudo case.
			pseudo_true = [1, 0] if sum(true_outcomes[0:split]) > 0 else [0, 1]
			# compute the brier of the pseudo split
			brier += sum([(i - j)**2. for i, j in zip([lside, rside], pseudo_true)])
		# take the average of the splits (n - 1)
		brier /= len(optionprobs) - 1

	else:
		brier = sum([(i - j)**2. for i, j in zip(optionprobs, true_outcomes)])

	return brier


def mmdb(optionprobs_by_day, correct, ordered=False):
	"""
		This computes the mean daily brier score.

		optionprobs_by_day: This is a vector with the user's forecast at the end of each
		day. This will be an array of arrays. Each entry in the array represents one day's
		forecast.

		correct: the index of the correct answer. Zero-based.

		ordered: a boolean flag determining whether or not the answer is *ordered* multinomial.
			If so, the formula is substantially different.
	"""
	brierscore = sum([brier(i, correct, ordered) for i in optionprobs_by_day])
	return brierscore / len(optionprobs_by_day)


if __name__ == "__main__":
	print(brier([0.2, 0.8], 1, False))
	'''
	print('the T&E example from page 51. This is D1.')
	print(brier([.25, .25, .50, .0], 1, True))
	print('the T&E example from page 51. This is "D2".')
	print(brier([.25, .25, .30, .20], 1, True))

	print('the T&E example from page 52. This is D1.')
	print(brier([.0, .50, .25, .25], 1, True))
	print('the T&E example from page 52. This is "D2".')
	print(brier([.20, .30, .40, .10], 1, True))

	# some funny example to test the multi-day function.
	print(mmdb([
		[.25, .25, .50, .0],
		[.25, .25, .30, .20]
		], 1, True))

	'''



def adjustday(dt):
	"""
	convert the forecasting time to an daystr
	"""
	dt = dt.strftime("%Y-%m-%d")
	dt = datetime.strptime(dt, "%Y-%m-%d")

	return dt

def computebrier(questioncondition, firstday, lastday, question_is_ordered, correct_answer, 
		userfieldfn=lambda x: x[2],
		datefieldfn=lambda x: x[0]
		):

	firstday = adjustday(firstday)
	lastday = adjustday(lastday)

	questioncondition = sorted(questioncondition, key=lambda x: x[0])

	"""
			Take the question's predictions and compute the brier.
	"""
	userset = set([userfieldfn(i) for i in questioncondition])

	# figure out the question's options and make a default 
	# uniform forecast for the default
	f_uniform = questioncondition[0]
	n_options = f_uniform[4]
	predfieldfn=lambda x: x[5:5+n_options]
	prob = 1. / n_options
	for i in range(5,5+n_options):
		 f_uniform[i] = prob


	# now organize the forecasts into days (adjust for HFC midnight)
	# this will only take the latest forecast per day
	dayuserforecast = defaultdict(dict)
	for forecast in questioncondition:
		dt = max(adjustday(datefieldfn(forecast)), firstday) # to avoid these super fast dudes.
		dayuserforecast[dt][userfieldfn(forecast)] = forecast

	if firstday not in dayuserforecast:
		dayuserforecast[firstday][defaultusername] = f_uniform

	# fill in the empty days by propagating forward
	oneday = timedelta(days=1)
	mday = firstday
	xday = lastday

	cday = mday
	while cday <= xday:
		if cday not in dayuserforecast:
				dayuserforecast[cday] = dayuserforecast[cday - oneday]
		cday += oneday

	# now compute the brier by day
	userswhohaveforecasted = set([])
	dayuserbrier = defaultdict(dict)
	lastday = None

	dayuserforecast = sorted(dayuserforecast.items(), key=lambda x: datefieldfn(x) )
	for day, userforecasts in dayuserforecast:

		dayroster = set([])
		for user, forecast in userforecasts.items():
			dayroster.add(user)
			userswhohaveforecasted.add(user)

			# pull out the probabilities and the outcome
			bscore = brier(predfieldfn(forecast), correct_answer, question_is_ordered)
			dayuserbrier[day][user] = bscore

		# COND 1
		# make a brier for:
		# all of the users who did NOT forecast, but DID forecast before
		for u in userswhohaveforecasted - dayroster:
			dayuserbrier[day][u] = lastday[u]

		# we are about to do condition 2, let's see the average brier so far
		useravg = np.mean([b for u, b in dayuserbrier[day].items()])
		
		# COND 2
		# make a forecast for:
		# all of the users who did NOT forecast and DID NOT forecast before
		# useravg = np.mean([dayuserbrier[day][i] for i in dayroster])
		if np.isnan(useravg):
			useravg = default_score
		dayuserbrier[day][defaultusername] = useravg
		for u in userset - userswhohaveforecasted:
			dayuserbrier[day][u] = useravg

		lastday = dayuserbrier[day]

	return dayuserbrier





def get_user_brier(data, db_dates, db_answer, brier_type="mean_sum"):
	raw_question2user2brier = defaultdict(dict)
	question2user2brier = defaultdict(list)

	for ifp in data.keys():
		user2dailyscores = defaultdict(list)
		user2scores = {}
		firstday = min(db_dates[ifp])
		lastday = max(db_dates[ifp])
		dayuserbrier = computebrier(data[ifp], firstday, lastday, db_answer[ifp][1], db_answer[ifp][0])

		
		for d, ub in dayuserbrier.items():
			for u, b in ub.items():
				user2dailyscores[u].append(b)

		ifp_brier = []
		for u,b in user2dailyscores.items():
			brier_user = np.nanmean(b)
			user2scores[u] = brier_user
			ifp_brier.append(brier_user)
			raw_question2user2brier[u][ifp] = brier_user


		#normalize z scores
		ifp_mean = np.mean(ifp_brier)
		ifp_std = np.std(ifp_brier)
		for u,b in user2scores.items():
			question2user2brier[u].append( (b-ifp_mean)/ifp_std)

	if brier_type == "mean":
		final = dict([(i, np.mean(j)) for i, j in question2user2brier.items()])
	elif brier_type == "sum":
		final = dict([(i, np.sum(j)) for i, j in question2user2brier.items()])
	else: # This is the new default: the average of mean and sum score  
		final = dict([(i, ((np.sum(j) + np.mean(j))/2 )) for i, j in question2user2brier.items()])

	ranked_brier = {}
	keys = final.keys()
	values = rankdata(final.values())    
	ranked_brier = dict(zip(keys, values))

	return defaultdict(lambda: 0.0,ranked_brier), question2user2brier

