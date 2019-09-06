with open('machine_rctb.csv', 'w') as fout:
	fout.write('machine_model,created_at,question_id,hfc_id,discover_id,num_options,option_1,option_2,option_3,option_4,option_5\n')
	for model in ('Auto ARIMA', 'M4-Meta', 'Arithmetic RW', 'DS-Holt', 'DS-Holt-damped', 'DS-RW', 'DS-SES', 'ETS', 'Geometric RW', 'M4-Comb', 'Mean', 'NNETAR', 'RW', 'RW-DRIFT', 'RW-SEAS', 'STLM-AR', 'TBATS', 'THETA'):
		fin = open('machine_forecasts_{}.csv'.format(model))
		fin.readline()
		for line in fin:
			fout.write('{},{}'.format(model, line))
		fin.close()
