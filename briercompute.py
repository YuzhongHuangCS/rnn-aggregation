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
